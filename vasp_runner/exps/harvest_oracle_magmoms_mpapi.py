"""Oracle per-site MAGMOM for MP straight from the Materials Project API.

The reference CHGCAR of every MP structure comes from one specific static task
(the latest "GGA Static"/"GGA+U Static", the same selection mp_api's
get_charge_density_from_material_id makes -- mirrored from
exps/harvest_relaxed_magmoms.py). That task doc carries its own converged
site-projected moments in output.outcar["magnetization"][i]["tot"], in the site
order of the static's input structure, i.e. exactly the POSCAR order our
runners write. So the oracle MAGMOM is a database lookup, not something to be
reconstructed.

This replaces exps/extract_oracle_magmoms_chgcar.py (Voronoi integration of the
reference CHGCAR's diff channel) for MP: same quantity VASP itself would print
under LORBIT, no cell-partitioning approximation, and no dependency on the
CHGCARs being reachable. The GNoME half has no MP database behind it and keeps
using the LORBIT sidecars harvested from our own runs
(exps/oracle_magmoms_gnome_lorbit.csv, exps/harvest_oracle_magmoms.py).

Output columns match what vasp_runner.relaxmag.load_relaxed_magmoms wants
(`magmom_oracle`), so it drops straight into any runner's --relaxed_magmoms.

Statuses: ok | ispin1 (task ran non-spin-polarised: no magnetization block,
all-zero moments emitted) | no_material_doc | no_static_task |
static_taskdoc_missing | static_no_input_structure | nsite_mismatch.

Resumable: ids already in the output CSV are skipped.

  python exps/harvest_oracle_magmoms_mpapi.py \
      --input exps/relaxmag_full_1943.csv \
      --output exps/oracle_magmoms_mp_mpapi.csv
"""
import argparse
import csv
import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

FIELDNAMES = ["MP_ID", "status", "static_task_id", "run_type", "n_sites",
              "magmom_oracle", "total_mag", "magmom_static_input"]

STATIC_TYPES = ("GGA Static", "GGA+U Static")


def fmt_list(vals):
    return " ".join(f"{float(v):.4f}" for v in vals)


def process(mpid, mpr):
    row = {k: "" for k in FIELDNAMES}
    row["MP_ID"] = mpid

    mats = mpr.materials.search(material_ids=[mpid],
                                fields=["material_id", "calc_types"])
    if not mats:
        row["status"] = "no_material_doc"
        return row
    calc_types = {str(k): str(v) for k, v in (mats[0].calc_types or {}).items()}
    static_tids = [t for t, c in calc_types.items() if c in STATIC_TYPES]
    if not static_tids:
        row["status"] = "no_static_task"
        return row

    docs = mpr.materials.tasks.search(
        task_ids=static_tids,
        fields=["task_id", "input", "output", "last_updated"],
    )
    if not docs:
        row["status"] = "static_taskdoc_missing"
        return row
    static = sorted(docs, key=lambda d: d.last_updated)[-1]
    tid = str(static.task_id).replace("mp-", "")
    row["static_task_id"] = str(static.task_id)
    row["run_type"] = calc_types.get(tid, "").replace(" Static", "")

    if static.input is None or static.input.structure is None:
        row["status"] = "static_no_input_structure"
        return row
    nsites = len(static.input.structure)
    row["n_sites"] = nsites
    incar = getattr(static.input, "incar", None) or {}
    mm_in = incar.get("MAGMOM")
    if isinstance(mm_in, (list, tuple)):
        row["magmom_static_input"] = fmt_list(mm_in)

    out = static.output
    oc = getattr(out, "outcar", None) if out is not None else None
    mag = oc.get("magnetization") if isinstance(oc, dict) else None
    if not mag:
        # ISPIN=1 static: no spin channel at all, moments are identically zero.
        row.update(status="ispin1", magmom_oracle=fmt_list([0.0] * nsites),
                   total_mag="0.0000")
        return row
    try:
        moments = [float(m["tot"]) for m in mag]
    except (KeyError, TypeError, ValueError):
        row["status"] = "unparsable_magnetization"
        return row
    if len(moments) != nsites:
        row["status"] = "nsite_mismatch"
        return row
    row.update(status="ok", magmom_oracle=fmt_list(moments),
               total_mag=f"{sum(moments):.4f}")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    root = os.path.join(os.path.dirname(__file__), "..")
    ap.add_argument("--input", default=os.path.join(root, "exps",
                                                    "relaxmag_full_1943.csv"))
    ap.add_argument("--output", default=os.path.join(root, "exps",
                                                     "oracle_magmoms_mp_mpapi.csv"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    api_key = os.environ.get("MP_API_KEY")
    assert api_key, "MP_API_KEY not found (load from .env failed?)"
    from mp_api.client import MPRester

    mpids = pd.read_csv(args.input)["MP_ID"].str.lower().tolist()
    done = set()
    if os.path.exists(args.output):
        done = set(pd.read_csv(args.output, dtype=str)["MP_ID"].str.lower())
    todo = [m for m in mpids if m not in done]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(mpids)} ids, {len(done)} already done, processing {len(todo)}",
          flush=True)

    write_header = not os.path.exists(args.output)
    counts = {}
    with MPRester(api_key, mute_progress_bars=True) as mpr, \
            open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for i, mpid in enumerate(todo, 1):
            try:
                row = process(mpid, mpr)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                row = {k: "" for k in FIELDNAMES}
                row.update(MP_ID=mpid, status=f"error:{type(e).__name__}")
            writer.writerow(row)
            f.flush()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {counts}", flush=True)
    print(f"wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
