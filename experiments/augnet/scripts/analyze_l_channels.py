"""
Break the test-set augmentation-occupancy error down by angular-momentum channel.

The 33 / 138 numbers VASP writes per atom in the CHGCAR "augmentation occupancies"
block are not 33 / 138 independent scalars: they are the Clebsch-Gordan recoupling
of the on-site density matrix D_ij into (radial pair, L, M) blocks,

    rho_i(r) = sum_{i<=j} sum_L sum_M  c^{(ij)}_{LM}  Y_LM(r-hat) f_ij(|r|)

with L running |l_i-l_j| .. l_i+l_j in steps of 2. So each stored number carries a
definite angular momentum L, and the L it carries is what decides how an error in
it propagates into an SCF restart:

    L = 0  monopole  -> on-site electron count -> Hartree potential, band filling
    L = 1  dipole    -> on-site polarization
    L = 2  quadrupole-> crystal-field splitting of the d/p shell
    L >= 3            -> high multipoles; for LMAXMIX = 2 (the VASP default) VASP
                         does not even feed these into the density mixer on restart

This script reports, per element and per L (and per radial shell-pair within L):
  * RMSE / MAE in the raw occupancy units VASP writes
  * the physical spread of that channel across the test frames (its own std)
  * a skill score vs. the train-set mean, 1 - MSE_model / MSE_trainmean, i.e.
    "does the model beat just writing the average occupancy?" -- the average is
    roughly what a superposition-of-atomic-densities start gives you
  * the share of the total squared error that lands in that L

Plus the error in Tr(D), the trace of the reconstructed on-site PAW density matrix.
It is not literally the valence electron count (the partial waves are not
orthonormal, so it overcounts), but it is the scalar invariant of the on-site
occupancy, and it is what the monopole channel controls.

Usage:
    python analyze_l_channels.py --run-dir checkpoints/<run_name> [--csv out.csv]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch
import yaml

# Bootstrap: repo root (parent of this script's dir) must be importable for
# the src.* / scripts.* imports below when this file is run directly.
import os as _os, sys as _sys
from neural_paw_dft.augnet.paw_basis_transform import set_l_channel_overrides, sanvito_to_e3nn_with_basis_padded
from neural_paw_dft.augnet.augnet_model import Z_TO_SCHEMA, SCHEMA_L_CHANNELS, build_sanvito_blocks
from neural_paw_dft.augnet.paw_moments import reconstruct_D

SYMBOL = {16: "S", 42: "Mo"}
SHELL = {0: "s", 1: "p", 2: "d", 3: "f"}


def physical_l_channels(z: int) -> list[int]:
    """The projector-l order these CHGCARs actually use (after any override)."""
    from neural_paw_dft.augnet.paw_basis_transform import _L_CHANNEL_OVERRIDES
    from neural_paw_dft.augnet.mp_potcar_map import MP_POTCAR_BY_Z

    if int(z) in _L_CHANNEL_OVERRIDES:
        return list(_L_CHANNEL_OVERRIDES[int(z)])
    entry = MP_POTCAR_BY_Z.get(int(z))
    if entry is not None:
        return list(entry["l_channels"])
    return list(SCHEMA_L_CHANNELS[Z_TO_SCHEMA[int(z)]])


def slot_labels(z: int) -> list[dict]:
    """One label per stored coefficient: which (pair, l1, l2, L, M) it is.

    Built in Sanvito/CHGCAR order, so it indexes the raw block verbatim. Per-(pair,L)
    squared error is basis-independent (Q_L is orthogonal within an L), so working
    in Sanvito order rather than e3nn order changes nothing about the numbers below.
    """
    l_phys = physical_l_channels(z)
    labels: list[dict] = []
    for b in build_sanvito_blocks(l_phys):
        for m in b["M"]:
            labels.append(
                {"pair": b["pair"], "l1": b["l1"], "l2": b["l2"], "L": b["L"], "M": m}
            )
    assert len(labels) == Z_TO_SCHEMA[int(z)]
    return labels


def _atomic_numbers(chgcar_path: str) -> np.ndarray:
    from neural_paw_dft.augnet.run_paw_chgcar import read_text_maybe_lz4, _atomic_numbers_from_chgcar_header

    return _atomic_numbers_from_chgcar_header(read_text_maybe_lz4(chgcar_path))


def read_truth(chgcar_path: str, atomic_numbers: np.ndarray) -> np.ndarray:
    """Ground-truth augmentation blocks, padded to [n_atoms, 390], Sanvito order."""
    from neural_paw_dft.augnet.run_paw_chgcar import parse_aug_from_file, pack_aug_padded_sanvito

    vecs = parse_aug_from_file(chgcar_path, atomic_numbers)
    y, _ = pack_aug_padded_sanvito(vecs, atomic_numbers)
    return y.numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True,
                   help="checkpoints/<run_name>, containing config_used.yaml and "
                        "aug_pred_dir_test_total/")
    p.add_argument("--csv", type=Path, default=None,
                   help="also write the per-(element, pair, L) table here")
    args = p.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config_used.yaml").read_text())
    overrides = (cfg.get("data") or {}).get("potcar_l_channels") or {}
    if overrides:
        set_l_channel_overrides(overrides)
        print(f"POTCAR projector-order overrides: {overrides}")

    pred_dir = args.run_dir / "aug_pred_dir_test_total"
    npzs = sorted(pred_dir.glob("*_total_aug.npz"))
    if not npzs:
        raise SystemExit(f"no predictions in {pred_dir}")

    split = json.loads(Path(cfg["data"]["split_file"]).read_text())
    data_dir = Path(cfg["project"]["data_dir"])

    # --- train-set per-slot mean: the "just write the average" baseline ---------
    train_sum: dict[int, np.ndarray] = {}
    train_n: dict[int, int] = defaultdict(int)
    for mid in split["train"]:
        path = str(data_dir / f"{mid}.chgcar.lz4")
        zs = _atomic_numbers(path)
        y = read_truth(path, zs)
        for z in np.unique(zs):
            sel = zs == z
            train_sum.setdefault(int(z), np.zeros(390))
            train_sum[int(z)] += y[sel].sum(0)
            train_n[int(z)] += int(sel.sum())
    train_mean = {z: train_sum[z] / train_n[z] for z in train_sum}
    print(f"train baseline from {len(split['train'])} frames: "
          + ", ".join(f"{SYMBOL.get(z, z)} n={train_n[z]}" for z in sorted(train_n)))

    # --- accumulate squared errors per (element, pair, L) ----------------------
    acc: dict[tuple, dict] = defaultdict(
        lambda: {"se": 0.0, "se_mean": 0.0, "ae": 0.0, "n": 0,
                 "sum_y": 0.0, "sum_y2": 0.0}
    )
    trd_err, trd_ref = defaultdict(list), defaultdict(list)

    for npz_path in npzs:
        d = np.load(npz_path, allow_pickle=True)
        zs = d["atomic_numbers"]
        pred = d["aug_sanvito_padded"].astype(np.float64)
        truth = read_truth(str(d["source_path"]), zs).astype(np.float64)

        for a, z in enumerate(zs.tolist()):
            schema = Z_TO_SCHEMA[int(z)]
            labs = slot_labels(int(z))
            e_model = pred[a, :schema] - truth[a, :schema]
            e_mean = train_mean[int(z)][:schema] - truth[a, :schema]
            for k, lab in enumerate(labs):
                key = (int(z), lab["pair"], lab["l1"], lab["l2"], lab["L"])
                s = acc[key]
                s["se"] += e_model[k] ** 2
                s["se_mean"] += e_mean[k] ** 2
                s["ae"] += abs(e_model[k])
                s["n"] += 1
                s["sum_y"] += truth[a, k]
                s["sum_y2"] += truth[a, k] ** 2

        # Tr(D): the on-site electron count. Needs e3nn order.
        zt = torch.as_tensor(zs, dtype=torch.long)
        for name, arr in (("pred", pred), ("true", truth)):
            y = torch.as_tensor(arr, dtype=torch.float64)
            y_e3, _ = sanvito_to_e3nn_with_basis_padded(
                y, torch.ones_like(y, dtype=torch.bool), zt
            )
            for schema, (idx, D) in reconstruct_D(y_e3, zt).items():
                tr = torch.diagonal(D, dim1=1, dim2=2).sum(-1)
                for i, row in enumerate(idx.tolist()):
                    (trd_ref if name == "true" else trd_err)[int(zs[row])].append(
                        float(tr[i])
                    )

    # --- report ---------------------------------------------------------------
    rows = []
    for (z, pair, l1, l2, L), s in acc.items():
        mse = s["se"] / s["n"]
        mse_mean = s["se_mean"] / s["n"]
        mu = s["sum_y"] / s["n"]
        var = max(s["sum_y2"] / s["n"] - mu**2, 0.0)
        rows.append({
            "element": SYMBOL.get(z, str(z)),
            "Z": z,
            "pair": f"{pair[0]}-{pair[1]}",
            "shells": f"{SHELL[l1]}{SHELL[l2]}",
            "L": L,
            "n_coeff": s["n"],
            "rmse": mse**0.5,
            "mae": s["ae"] / s["n"],
            "test_std": var**0.5,
            "test_rms": (s["sum_y2"] / s["n"]) ** 0.5,
            "rmse_baseline_mean": mse_mean**0.5,
            "skill_vs_mean": 1.0 - mse / mse_mean if mse_mean > 0 else float("nan"),
            "sse": s["se"],
        })

    total_sse = {el: sum(r["sse"] for r in rows if r["element"] == el)
                 for el in {r["element"] for r in rows}}

    for el in sorted(total_sse, key=lambda e: -total_sse[e]):
        sub = [r for r in rows if r["element"] == el]
        print(f"\n{'='*92}\n{el}: error by angular momentum L "
              f"(units = raw CHGCAR augmentation occupancies)\n{'='*92}")
        print(f"{'L':>2} {'n_coeff':>8} {'RMSE':>11} {'MAE':>11} {'signal RMS':>11} "
              f"{'RMSE(mean)':>11} {'skill':>7} {'%SSE':>6}")
        for L in sorted({r["L"] for r in sub}):
            g = [r for r in sub if r["L"] == L]
            n = sum(r["n_coeff"] for r in g)
            sse = sum(r["sse"] for r in g)
            sse_m = sum(r["rmse_baseline_mean"] ** 2 * r["n_coeff"] for r in g)
            rms = (sum(r["test_rms"] ** 2 * r["n_coeff"] for r in g) / n) ** 0.5
            mae = sum(r["mae"] * r["n_coeff"] for r in g) / n
            print(f"{L:>2} {n:>8} {(sse/n)**0.5:>11.3e} {mae:>11.3e} {rms:>11.3e} "
                  f"{(sse_m/n)**0.5:>11.3e} {1-sse/sse_m:>7.3f} "
                  f"{100*sse/total_sse[el]:>6.1f}")

        print(f"\n  by radial shell pair:")
        print(f"  {'pair':>5} {'shells':>7} {'L':>2} {'n':>6} {'RMSE':>11} "
              f"{'signal RMS':>11} {'var(test)':>11} {'skill':>7} {'%SSE':>6}")
        for r in sorted(sub, key=lambda r: -r["sse"]):
            print(f"  {r['pair']:>5} {r['shells']:>7} {r['L']:>2} {r['n_coeff']:>6} "
                  f"{r['rmse']:>11.3e} {r['test_rms']:>11.3e} {r['test_std']:>11.3e} "
                  f"{r['skill_vs_mean']:>7.3f} {100*r['sse']/total_sse[el]:>6.1f}")

    print(f"\n{'='*92}\nTr(D): scalar invariant of the on-site PAW density matrix "
          f"(what the L=0 channel controls)\n{'='*92}")
    for z in sorted(trd_ref):
        t = np.array(trd_ref[z])
        p_ = np.array(trd_err[z])
        print(f"  {SYMBOL.get(z, z):>3}: Tr(D)_true = {t.mean():8.4f} "
              f"+/- {t.std():.4f}   MAE = {np.abs(p_-t).mean():.4e}   "
              f"max = {np.abs(p_-t).max():.4e}   "
              f"({100*np.abs(p_-t).mean()/abs(t.mean()):.3f} % of Tr(D))")

    if args.csv:
        import csv as _csv

        with open(args.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: (r["element"], r["L"], r["pair"])))
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
