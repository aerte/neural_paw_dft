"""Harvest MP relaxed per-site magmoms for the test set.

For each MP_ID in the input CSV:
  1. materials.search -> calc_types: split tasks into Structure Optimization
     (relax) and Static.
  2. Latest static task (same selection get_exact_chg_taskdoc ends up with):
     its input structure defines the site order of the POSCAR our runners
     write, and its input MAGMOM is recorded for reference.
  3. Pick the latest relax task (preferring the static's run-type flavor,
     e.g. GGA+U) whose output structure matches the static input structure,
     and read per-site moments from output.outcar["magnetization"][i]["tot"].
  4. Remap relax sites onto static site order (identity if species+coords
     already line up, else StructureMatcher.get_mapping) and write one CSV row.

Output columns:
  MP_ID, status, static_task_id, relax_task_id, n_sites, mapping,
  magmom_static_input, magmom_relaxed
MAGMOM lists are space-separated floats in static-POSCAR site order,
directly usable as an INCAR MAGMOM string.

Resumable: existing rows in the output CSV are skipped on rerun.

Usage:
  python exps/harvest_relaxed_magmoms.py                # full test set
  python exps/harvest_relaxed_magmoms.py --limit 10     # smoke test
"""
import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

FIELDNAMES = [
    "MP_ID", "status", "static_task_id", "relax_task_id", "relax_flavor",
    "n_sites", "mapping", "cell_dev_abc", "cell_dev_angle",
    "magmom_static_input", "magmom_relaxed",
]

# The CHGCAR task selection in mp_api's get_charge_density_from_material_id
# is the latest task among EXACTLY these calc types — mirror it, or the site
# order reference can be a different (e.g. HSE06) static.
STATIC_TYPES = ("GGA Static", "GGA+U Static")

COORD_TOL = 1e-3  # frac-coord tolerance for the identity site match


def fmt_list(vals):
    return " ".join(f"{float(v):.4f}" for v in vals)


def identity_match(static_struct, relax_struct):
    """True if sites line up index-for-index (species + frac coords, pbc)."""
    if len(static_struct) != len(relax_struct):
        return False
    for a, b in zip(static_struct, relax_struct):
        if a.species != b.species:
            return False
        d = np.abs(a.frac_coords - b.frac_coords)
        d = np.minimum(d, 1.0 - d)
        if np.any(d > COORD_TOL):
            return False
    return True


def moments_from_task(doc):
    """Per-site 'tot' moments from a task doc's output OUTCAR, or None."""
    out = doc.output
    oc = getattr(out, "outcar", None) if out is not None else None
    if not isinstance(oc, dict):
        return None
    mag = oc.get("magnetization")
    if not mag:
        return None
    try:
        return [float(m["tot"]) for m in mag]
    except (KeyError, TypeError, ValueError):
        return None


def cell_deviation(static_struct, relax_struct):
    """(max relative abc dev, max angle dev in deg) between Niggli-reduced
    lattices — orientation/setting-invariant, flags loose matches."""
    try:
        a = static_struct.lattice.get_niggli_reduced_lattice()
        b = relax_struct.lattice.get_niggli_reduced_lattice()
    except Exception:
        a, b = static_struct.lattice, relax_struct.lattice
    abc_a, abc_b = np.array(sorted(a.abc)), np.array(sorted(b.abc))
    ang_a, ang_b = np.array(sorted(a.angles)), np.array(sorted(b.angles))
    return (float(np.max(np.abs(abc_a - abc_b) / abc_a)),
            float(np.max(np.abs(ang_a - ang_b))))


def pick_relax(static_struct, static_flavor, relax_docs, matcher):
    """Choose a relax task and return (doc, moments_in_static_order, mapping).

    Preference order: same run-type flavor as the static before other
    flavors, newest first within each group; identity site order before a
    StructureMatcher-derived mapping.
    """
    def sort_key(d):
        same_flavor = d._flavor == static_flavor
        return (not same_flavor, -d.last_updated.timestamp())

    for doc in sorted(relax_docs, key=sort_key):
        moments = moments_from_task(doc)
        rs = doc.output.structure if doc.output is not None else None
        if moments is None or rs is None or len(moments) != len(rs):
            continue
        if identity_match(static_struct, rs):
            return doc, moments, "identity"
        if len(rs) != len(static_struct):
            continue
        try:
            mapping = matcher.get_mapping(rs, static_struct)
        except ValueError:
            mapping = None
        if mapping is not None:
            return doc, [moments[j] for j in mapping], "matched"
    return None, None, None


def process(mpid, mpr, matcher):
    """Return one output row (dict) for `mpid`."""
    row = {k: "" for k in FIELDNAMES}
    row["MP_ID"] = mpid

    mats = mpr.materials.search(material_ids=[mpid],
                                fields=["material_id", "calc_types"])
    if not mats:
        row["status"] = "no_material_doc"
        return row
    calc_types = {str(k): str(v) for k, v in (mats[0].calc_types or {}).items()}
    static_tids = [t for t, c in calc_types.items() if c in STATIC_TYPES]
    relax_tids = [t for t, c in calc_types.items()
                  if "Structure Optimization" in c]
    if not static_tids:
        row["status"] = "no_static_task"
        return row
    if not relax_tids:
        row["status"] = "no_relax_task"
        return row

    docs = mpr.materials.tasks.search(
        task_ids=static_tids + relax_tids,
        fields=["task_id", "input", "output", "last_updated"],
    )
    by_id = {str(d.task_id).replace("mp-", ""): d for d in docs}

    statics = sorted((by_id[t] for t in static_tids if t in by_id),
                     key=lambda d: d.last_updated)
    if not statics:
        row["status"] = "static_taskdoc_missing"
        return row
    static = statics[-1]
    row["static_task_id"] = str(static.task_id)
    if static.input is None or static.input.structure is None:
        row["status"] = "static_no_input_structure"
        return row
    sstruct = static.input.structure
    row["n_sites"] = len(sstruct)
    incar = getattr(static.input, "incar", None) or {}
    mm_in = incar.get("MAGMOM")
    if isinstance(mm_in, (list, tuple)):
        row["magmom_static_input"] = fmt_list(mm_in)

    static_flavor = calc_types[str(static.task_id).replace("mp-", "")]
    static_flavor = static_flavor.replace(" Static", "")
    relax_docs = []
    for t in relax_tids:
        d = by_id.get(t)
        if d is not None:
            d._flavor = calc_types[t].replace(" Structure Optimization", "")
            relax_docs.append(d)
    if not relax_docs:
        row["status"] = "relax_taskdoc_missing"
        return row

    relax, moments, mapping = pick_relax(sstruct, static_flavor, relax_docs,
                                         matcher)
    if relax is None:
        row["status"] = "no_matching_relax"
        return row
    dev_abc, dev_ang = cell_deviation(sstruct, relax.output.structure)
    row.update(status="ok", relax_task_id=str(relax.task_id),
               relax_flavor=relax._flavor, mapping=mapping,
               cell_dev_abc=f"{dev_abc:.4f}", cell_dev_angle=f"{dev_ang:.2f}",
               magmom_relaxed=fmt_list(moments))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = os.path.join(os.path.dirname(__file__), "..")
    ap.add_argument("--input", default=os.path.join(root, "MP_SAD_WITHMAG_2.csv"))
    ap.add_argument("--output", default=os.path.join(root, "MP_RELAXED_MAGMOMS.csv"))
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N unprocessed mp-ids.")
    args = ap.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    assert api_key, "MP_API_KEY not found (load from .env failed?)"

    from mp_api.client import MPRester
    from pymatgen.analysis.structure_matcher import StructureMatcher

    mpids = pd.read_csv(args.input)["MP_ID"].str.lower().tolist()

    done = set()
    if os.path.exists(args.output):
        done = set(pd.read_csv(args.output)["MP_ID"].str.lower())
    todo = [m for m in mpids if m not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(mpids)} test ids, {len(done)} already done, "
          f"processing {len(todo)}")

    matcher = StructureMatcher(primitive_cell=False, allow_subset=True,
                               attempt_supercell=False)
    write_header = not os.path.exists(args.output)
    n_err = 0
    with MPRester(api_key, mute_progress_bars=True) as mpr, \
            open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for i, mpid in enumerate(todo, 1):
            try:
                row = process(mpid, mpr, matcher)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                row = {k: "" for k in FIELDNAMES}
                row.update(MP_ID=mpid, status=f"error:{type(e).__name__}")
                n_err += 1
                print(f"  !! {mpid}: {type(e).__name__}: {e}", file=sys.stderr)
            writer.writerow(row)
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {mpid}: {row['status']}", flush=True)
    print(f"done ({n_err} errors)")


if __name__ == "__main__":
    main()
