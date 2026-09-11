"""MP data-source helpers: CHGCAR discovery, mp-id extraction, exact-inputs writing."""
import json
import os
import re


class MissingMPTaskDocError(RuntimeError):
    """MP has no CHGCAR / TaskDoc for this mp-id."""


def find_chgcars(root_dir):
    """Recursively find MP CHGCAR(.lz4) files (must contain 'mp-' in basename)."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            lower = fn.lower()
            if "mp-" not in lower:
                continue
            if (
                lower.endswith("chgcar")
                or lower.endswith("chgcar.lz4")
                or lower.endswith("chgcar.gz")
            ):
                yield os.path.join(dirpath, fn)


def extract_mpid_from_path(path):
    """Return 'mp-XXXXXX' (lowercased) extracted from the basename, or None."""
    base = os.path.basename(path)
    m = re.match(r"^(mp-\d+)\.", base, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower()


def filter_by_json_test(chgcar_paths, json_path):
    with open(json_path, "r") as f:
        splits = json.load(f)
    raw_test_ids = splits.get("test", [])
    if not raw_test_ids:
        print(f"⚠️  No 'test' entries found in {json_path}; returning empty list.")
        return []
    test_ids = set()
    for x in raw_test_ids:
        if isinstance(x, int):
            test_ids.add(f"mp-{x}")
        else:
            test_ids.add(str(x).lower())
    return [p for p in chgcar_paths
            if (mpid := extract_mpid_from_path(p)) is not None and mpid in test_ids]


def derive_mpid(row, chgcar_path: str) -> str:
    """Pull MP_ID from a CSV row, or fall back to regex on the path."""
    for col in ("MP_ID", "mp_id", "material_id", "materialId"):
        if col in row.index:
            val = str(row[col]).strip()
            if val.startswith("mp-"):
                return val.lower()
    m = re.search(r"(mp-\d+)", chgcar_path, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    raise RuntimeError(f"Could not infer MP ID for CHGCAR_PATH={chgcar_path}")


def get_exact_chg_taskdoc(mpid: str):
    """Retrieve the exact VASP TaskDoc that produced this CHGCAR.

    Wraps mpr.get_charge_density_from_material_id(..., inc_task_doc=True).

    Any failure to obtain the density/taskdoc — a ``None`` result, a missing S3
    object ("No object found: s3://materialsproject-parsed/chgcars/<id>.json.gz"),
    or a transient API/JSON/connection error — is normalized to
    ``MissingMPTaskDocError``. Every runner already treats that as "skip this
    structure (exit 3)", so a per-material data gap or hiccup can never escape
    as an unhandled exception and kill the whole slice job.

    The call itself is watchdogged: ``MPRester`` sets no request timeout, and a
    stalled fetch has held a node at 0% CPU indefinitely. ``WatchdogTimeout`` is
    deliberately re-raised rather than folded into ``MissingMPTaskDocError``,
    because the latter is recorded as a *terminal* status — a network stall must
    stay retryable on the next invocation, not permanently skip the structure.
    """
    from mp_api.client import MPRester
    from ..failsafe import WatchdogTimeout, watchdog
    try:
        with watchdog(900, f"MP charge-density fetch for {mpid}"):
            with MPRester() as mpr:
                result = mpr.get_charge_density_from_material_id(mpid, inc_task_doc=True)
    except (MissingMPTaskDocError, WatchdogTimeout):
        raise
    except Exception as e:
        raise MissingMPTaskDocError(
            f"Could not retrieve charge density / task document for {mpid}: "
            f"{type(e).__name__}: {e}"
        ) from e
    if result is None:
        raise MissingMPTaskDocError(f"No charge density / task document for {mpid}")
    _chgcar, taskdoc = result
    return taskdoc


def write_mp_inputs_for_mpid(mpid: str, workdir: str):
    """Write POSCAR/INCAR/KPOINTS/POTCAR for `mpid` into `workdir` using
    the exact TaskDoc that produced its CHGCAR.
    """
    from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints, Potcar

    os.makedirs(workdir, exist_ok=True)
    tdoc = get_exact_chg_taskdoc(mpid)
    vinput = tdoc["input"] if isinstance(tdoc, dict) else tdoc.input

    structure = vinput["structure"] if isinstance(vinput, dict) else vinput.structure
    Poscar(structure).write_file(os.path.join(workdir, "POSCAR"))

    # Prefer input.incar (the original submitted INCAR tags). input.parameters
    # is the OUTCAR-expanded effective set, which carries per-ion POMASS lists
    # ("POMASS = 12.011 ...") that VASP 5.4.4 cannot re-read from INCAR
    # ("Error reading item 'POMASS' from file INCAR. Found N=1 data."). This
    # mirrors what run_default_mp.py:160-166 already does.
    if isinstance(vinput, dict):
        incar_dict = vinput.get("incar", None)
        if not incar_dict:
            incar_dict = vinput.get("parameters", {}) or {}
    else:
        incar_dict = getattr(vinput, "incar", None) or vinput.parameters or {}
    Incar(incar_dict).write_file(os.path.join(workdir, "INCAR"))

    if isinstance(vinput, dict):
        k_raw = vinput.get("kpoints", None)
    else:
        k_raw = getattr(vinput, "kpoints", None)
    kpoints = None
    if isinstance(k_raw, Kpoints):
        kpoints = k_raw
    elif isinstance(k_raw, dict):
        kpoints = Kpoints.from_dict(k_raw)
    if kpoints is not None:
        kpoints.write_file(os.path.join(workdir, "KPOINTS"))

    if isinstance(vinput, dict):
        potcar_spec = vinput.get("potcar_spec", None)
    else:
        potcar_spec = getattr(vinput, "potcar_spec", None)
    if not potcar_spec:
        raise RuntimeError(f"POTCAR metadata missing for {mpid} (no potcar_spec)")

    symbols = []
    for ps in potcar_spec:
        if hasattr(ps, "symbol") and ps.symbol is not None:
            symbols.append(str(ps.symbol))
        elif isinstance(ps, dict) and "symbol" in ps:
            symbols.append(str(ps["symbol"]))
        elif hasattr(ps, "titel"):
            parts = str(ps.titel).split()
            if len(parts) > 1:
                symbols.append(parts[1])
            else:
                raise RuntimeError(f"Unexpected POTCAR titel format: {ps.titel}")
        elif isinstance(ps, dict) and "titel" in ps:
            parts = str(ps["titel"]).split()
            if len(parts) > 1:
                symbols.append(parts[1])
            else:
                raise RuntimeError(f"Unexpected POTCAR titel format (dict): {ps['titel']}")
        else:
            raise RuntimeError(f"Unrecognized potcar_spec entry type: {ps!r}")
    Potcar(symbols).write_file(os.path.join(workdir, "POTCAR"))


# Header used by make_tasks_mp.py
TASK_HEADER = [
    "CHGCAR_PATH",
    "MP_ID",
    "performed_default",
    "performed_true_init",
    "performed_ml_init",
]


def task_row(path: str):
    mpid = extract_mpid_from_path(path)
    if mpid is None:
        return None
    return [path, mpid, "False", "False", "False"]
