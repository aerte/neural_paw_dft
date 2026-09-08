#!/usr/bin/env python3
"""Build the GNOME_2 task CSV: non-magnetic, all predictions present.

The population is fixed here rather than gated at run time (as the MP runners do
via `scf_stats.magnetic_map` over runs.db), because GNOME has no `true_init` rows
in our manifest to classify against — the magnetic verdict comes from the
reference `GNOME_*_WITHMAG.csv`, produced from the reference `run_default`
OUTCARs with the ChargE3Net criterion (|M_tot| < 0.1 µB AND |M_i| < 0.1 µB for
every atom). Baking it into the CSV means all four experiments run on exactly the
same rows and no runner needs a magnetic gate.

CHGCAR_PATH is the converged density from the reference run — the output of the
very input set the runners copy — NOT the ECD-paper `gnome_ecd_from_ecd_paper/`
archive. The two were produced with different POTCAR sets and their PAW
augmentation block lengths disagree on ~23% of structures (e.g. `Li` vs `Li_sv`,
15 vs 33 entries). AugNet's predictions are aligned with the reference set, so the
ECD-paper file is both the wrong Oracle for these inputs and unusable as the
augmentation template. Their FFT grids are identical, so the ELECTRAFI /
Charge3Net predicted grids are compatible with either.

Exclusions, in order:
  * no magnetic classification (no parseable reference OUTCAR)
  * magnetic
  * no converged CHGCAR in the reference run
  * missing from any of the three prediction dirs

There is NO element filter. An earlier version dropped every Yb-containing
structure because the first AugNet export (gnome2_eval) had predictions for 0
of the 75 of them; the shared-head export below covers 71/75, as do all the
ELECTRAFI GNoME grids, so the exclusion was retired on 2026-08-28 and must not
be reintroduced — a structure is dropped only when a prediction or the
reference itself is genuinely missing, which the funnel prints per reason.

Usage (on the cluster, where the paths resolve):
    python exps/make_gnome2_tasks.py --out exps/gnome2_nonmag.csv
    python exps/make_gnome2_tasks.py --out exps/gnome2_full.csv --include-magnetic
    python exps/make_gnome2_tasks.py --out /dev/null --report-only
"""
import argparse
import csv
import os
import sys


# Site-specific paths; override via the environment.
ECD_DIR = os.environ.get("GNOME_ECD_DIR", "/path/to/gnome_ecd_from_ecd_paper")
REF_RUNS = os.environ.get("GNOME_REF_RUNS_ROOT", "/path/to/GNOME_REFERENCE_RUNS")
MAG_CSV = os.environ.get("GNOME_MAG_CSV", "/path/to/GNOME_WITHMAG.csv")

# The ICLR shared-head export (AugNet). The superseded gnome2_eval export had
# no Yb predictions at all, which is what motivated the old element filter.
AUG_DIR = os.environ.get("GNOME_AUG_PRED_DIR", "/path/to/augnet_gnome/aug_pred_dir_test_total")
ELECTRAFI_DIR = os.environ.get("GNOME_ELECTRAFI_DIR", "/path/to/electrafi_ref/gnome_ecd")
CHARGE3NET_DIR = os.environ.get("GNOME_CHARGE3NET_DIR", "/path/to/charge3net/gnome")


def gid_of(path):
    return os.path.basename(path).split(".")[0]


def read_mag_map(path):
    """{gid: is_magnetic} from the reference WITHMAG CSV (rows with a True/False verdict)."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            verdict = (row.get("is_magnetic") or "").strip()
            if verdict in ("True", "False"):
                out[gid_of(row["CHGCAR_PATH"])] = (verdict == "True")
    return out


def ref_file(gid, name):
    return os.path.join(REF_RUNS, f"{gid}.chgcar", "run_default", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="exps/gnome2_nonmag.csv",
                    help="Output task CSV (default: exps/gnome2_nonmag.csv).")
    ap.add_argument("--report-only", action="store_true",
                    help="Print the funnel and write nothing.")
    ap.add_argument("--include-magnetic", action="store_true",
                    help="Keep magnetic structures (drop only the 'magnetic' "
                         "exclusion) — the full test set for the "
                         "per-site-MAGMOM (chgnetmag) experiments.")
    args = ap.parse_args()

    all_ids = sorted(gid_of(f) for f in os.listdir(ECD_DIR)
                     if f.endswith(".chgcar.lz4"))
    print(f"converged reference CHGCARs: {len(all_ids)}")

    mag = read_mag_map(MAG_CSV)
    print(f"  with a magnetic classification: {len(mag)} "
          f"({sum(mag.values())} magnetic)")

    aug_ids = {n[: -len("_total_aug.npz")] for n in os.listdir(AUG_DIR)
               if n.endswith("_total_aug.npz")}
    efi_ids = {n.split("_", 1)[0] for n in os.listdir(ELECTRAFI_DIR)
               if n.endswith(".CHGCAR")}
    c3n_ids = {d for d in os.listdir(CHARGE3NET_DIR)
               if os.path.isfile(os.path.join(CHARGE3NET_DIR, d, "CHGCAR"))}
    print(f"  predictions — AugNet {len(aug_ids)}, ELECTRAFI {len(efi_ids)}, "
          f"Charge3Net {len(c3n_ids)}")

    kept, dropped = [], {"unclassified": 0, "magnetic": 0,
                         "no_ref_chgcar": 0, "no_augnet": 0,
                         "no_electrafi": 0, "no_charge3net": 0}
    for gid in all_ids:
        if gid not in mag:
            dropped["unclassified"] += 1
            continue
        if mag[gid] and not args.include_magnetic:
            dropped["magnetic"] += 1
            continue
        if not os.path.isfile(ref_file(gid, "CHGCAR")):
            dropped["no_ref_chgcar"] += 1
            continue
        if gid not in aug_ids:
            dropped["no_augnet"] += 1
            continue
        if gid not in efi_ids:
            dropped["no_electrafi"] += 1
            continue
        if gid not in c3n_ids:
            dropped["no_charge3net"] += 1
            continue
        kept.append(gid)

    print("dropped:")
    for k, v in dropped.items():
        print(f"  {k:>14}: {v}")
    print(f"kept: {len(kept)}")

    if args.report_only:
        return

    if not kept:
        print("FATAL: empty task set.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["CHGCAR_PATH", "GNOME_ID"])
        for gid in kept:
            w.writerow([ref_file(gid, "CHGCAR"), gid])
    print(f"wrote {len(kept)} rows to {args.out}")


if __name__ == "__main__":
    main()
