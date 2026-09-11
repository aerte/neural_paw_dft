"""
Score a CJLM MoS2 run against Focassio et al. Fig. 2(b), with the filters that
make the comparison honest.

The published number lives on the L <= 2 subset (20,115 of 27,540 rows): the CJLM
descriptor is identically zero for L > LMAXMIX, so L=3/L=4 are unmodelable for it
by construction while the DFT values there are nonzero.

A second filter matters on this dataset and is NOT in the paper: the 10 `ts_`
frames were run at LMAXMIX=2, so their Mo L>=3 blocks are literal 0.0 (see
the paper). Scoring a model against those inflates apparent
L>=3 skill. `--rows written` drops them; `--rows all` keeps them, reproducing the
older (over-optimistic) unfiltered numbers.

Metrics are pooled over the selected coefficients exactly as the paper pools them
(flat vector), in raw physical occupancy units.

Usage:
    python cjlm_paper_comparison.py --run-dir <checkpoints/run_name> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv as _csv
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import yaml

# Bootstrap: repo root (parent of this script's dir) must be importable for
# the src.* / scripts.* imports below when this file is run directly.
import os as _os, sys as _sys
from scripts.analyze_l_channels import SYMBOL, read_truth, slot_labels
from neural_init.augnet.paw_basis_transform import set_l_channel_overrides
from neural_init.augnet.augnet_model import Z_TO_SCHEMA
from neural_init.augnet.run_paw_chgcar import detect_lmaxmix, resolve_lmaxmix

# Published reference rows, for the table header (MACE_COMPARISON_NOTE.md).
REFERENCE = [
    ("paper Fig. 2(b)", 20115, 0.998490, 0.012981, 0.045906, 0.913685),
    ("CJLM regenerated (shipped models)", 20115, 0.998540, 0.020823, 0.054820, 1.070408),
]


def metrics(pred: np.ndarray, truth: np.ndarray) -> tuple:
    """Pooled (n, R2, MAE, RMSE, MaxAE) over a flat vector, as sklearn would."""
    if pred.size == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    err = pred - truth
    ss_res = float((err ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return (int(err.size), r2, float(np.abs(err).mean()),
            float(np.sqrt((err ** 2).mean())), float(np.abs(err).max()))


def collect(run_dir: Path) -> dict:
    """Flatten every test coefficient into aligned arrays plus their labels."""
    cfg = yaml.safe_load((run_dir / "config_used.yaml").read_text())
    overrides = (cfg.get("data") or {}).get("potcar_l_channels") or {}
    if overrides:
        # Also what makes the LMAXMIX detection below correct -- see §8 of the note.
        set_l_channel_overrides(overrides)
        print(f"POTCAR projector-order overrides: {overrides}")

    npzs = sorted((run_dir / "aug_pred_dir_test_total").glob("*_total_aug.npz"))
    if not npzs:
        raise SystemExit(f"no predictions in {run_dir / 'aug_pred_dir_test_total'}")

    pred, truth, Ls, Zs, written, frames = [], [], [], [], [], []
    for npz_path in npzs:
        d = np.load(npz_path, allow_pickle=True)
        zs = d["atomic_numbers"]
        p = d["aug_sanvito_padded"].astype(np.float64)
        source = str(d["source_path"])
        t = read_truth(source, zs).astype(np.float64)

        # What this frame's own LMAXMIX was: coefficients above it were never
        # written by VASP, so scoring them measures nothing.
        from neural_init.augnet.run_paw_chgcar import parse_aug_from_file
        lm = resolve_lmaxmix(detect_lmaxmix(zs, parse_aug_from_file(source, zs)))
        mid = Path(npz_path).name.replace("_total_aug.npz", "")

        for a, z in enumerate(zs.tolist()):
            schema = Z_TO_SCHEMA[int(z)]
            labs = slot_labels(int(z))
            pred.append(p[a, :schema])
            truth.append(t[a, :schema])
            Ls.append(np.array([lab["L"] for lab in labs]))
            Zs.append(np.full(schema, int(z)))
            written.append(np.array([lm is None or lab["L"] <= lm for lab in labs]))
            frames.append(np.full(schema, mid))

    return {
        "pred": np.concatenate(pred), "truth": np.concatenate(truth),
        "L": np.concatenate(Ls), "Z": np.concatenate(Zs),
        "written": np.concatenate(written), "frame": np.concatenate(frames),
        "n_frames": len(npzs),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    d = collect(args.run_dir)
    L, Z, w = d["L"], d["Z"], d["written"]
    n_dropped = int((~w).sum())
    print(f"{d['n_frames']} test frames, {L.size} coefficients; "
          f"{n_dropped} of them ({100 * n_dropped / L.size:.1f}%) were never written "
          f"by VASP (LMAXMIX truncation)")

    selections = [
        ("MACE, L<=2 (paper-comparable)", L <= 2),
        ("MACE, no L filter, written only", w),
        ("MACE, no L filter, ALL rows", np.ones_like(w)),
        ("MACE, L>=3, written only", (L >= 3) & w),
        ("MACE, L>=3, ALL rows", L >= 3),
        ("MACE, S only", Z == 16),
        ("MACE, Mo only (L<=2)", (Z == 42) & (L <= 2)),
    ]

    rows = []
    print(f"\n{'':38s} {'n':>7} {'R2':>10} {'MAE':>10} {'RMSE':>10} {'MaxAE':>10}")
    for name, n, r2, mae, rmse, maxae in REFERENCE:
        print(f"{name:38s} {n:7d} {r2:10.6f} {mae:10.6f} {rmse:10.6f} {maxae:10.6f}")
    for name, sel in selections:
        m = metrics(d["pred"][sel], d["truth"][sel])
        rows.append({"selection": name, "n": m[0], "R2": m[1], "MAE": m[2],
                     "RMSE": m[3], "MaxAE": m[4]})
        print(f"{name:38s} {m[0]:7d} {m[1]:10.6f} {m[2]:10.6f} {m[3]:10.6f} {m[4]:10.6f}")

    print("\n'written only' drops the coefficients this frame's LMAXMIX truncated;\n"
          "'ALL rows' keeps them (exact zeros in the target) and is the older,\n"
          "over-optimistic convention. The L<=2 row is unaffected by either.")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            wtr = _csv.DictWriter(f, fieldnames=list(rows[0]))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
