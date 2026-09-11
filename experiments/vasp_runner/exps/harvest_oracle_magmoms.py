"""Harvest the converged ("oracle") per-site moments from a finished variant.

Every runner appends the OUTCAR per-site magnetization of each OK run to a
`<results_csv>_magnetization.jsonl` sidecar (one record per structure, sites in
POSCAR order, `tot` = sphere-integrated moment). This collects them for one
variant into a MAGMOM CSV that `vasp_runner.relaxmag.load_relaxed_magmoms`
accepts (column `magmom_oracle`), so the runners' `--relaxed_magmoms` path can
seed the spin channel with the converged moments: Oracle total + Oracle MAGMOM.

Site order matches the INCAR MAGMOM the runners write (same POSCAR), and the
values are signed, unlike CHGNet's. Structures whose reference ran ISPIN=1
have no site-resolved magnetization and are simply absent (they then keep the
reference MAGMOM, `magmom_source=default_fallback`).

Run on the machine holding the full results/ tree:
  python exps/harvest_oracle_magmoms.py --variant true_init \
      --out exps/oracle_magmoms_mp_lorbit.csv
  python exps/harvest_oracle_magmoms.py --variant gnome_true_init \
      --id-col GNOME_ID --out exps/oracle_magmoms_gnome.csv
"""
import argparse
import glob
import json
import os

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def results_dir(variant: str) -> str:
    # MP baselines live under results/<variant>_scf, GNoME under results/<tag>.
    for d in (f"results/{variant}_scf", f"results/{variant}"):
        if os.path.isdir(os.path.join(REPO, d)):
            return os.path.join(REPO, d)
    raise SystemExit(f"no results dir for variant {variant!r} under {REPO}/results")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True,
                    help="Variant whose sidecars to read (true_init, default, "
                         "gnome_true_init, ...).")
    ap.add_argument("--id-col", default="MP_ID", choices=("MP_ID", "GNOME_ID"),
                    help="Id column of the output CSV (the sidecars always key "
                         "on MP_ID, GNoME runners included).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(results_dir(args.variant),
                                          "*_magnetization.jsonl")))
    moments, conflicts, unparsed = {}, 0, 0
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = r.get("magnetization")
            if not isinstance(m, list) or not m:
                unparsed += 1          # ISPIN=1 (empty) or str-serialized
                continue
            try:
                tots = [float(s["tot"]) for s in m]
            except (KeyError, TypeError, ValueError):
                unparsed += 1
                continue
            sid = str(r["MP_ID"])
            if sid in moments and moments[sid] != tots:
                conflicts += 1       # re-run landed elsewhere; last one wins
            moments[sid] = tots

    rows = [{args.id_col: sid, "n_sites": len(t),
             "magmom_oracle": " ".join(f"{x:.4f}" for x in t),
             "source_variant": args.variant}
            for sid, t in sorted(moments.items())]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    n_mag = sum(any(abs(x) >= 0.001 for x in t) for t in moments.values())
    print(f"{args.out}: {len(rows)} ids from {len(files)} sidecars "
          f"({n_mag} with a site |m| >= 0.001; {conflicts} duplicate ids "
          f"with differing moments, {unparsed} records without site data)")


if __name__ == "__main__":
    main()
