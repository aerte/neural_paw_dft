#!/usr/bin/env python3
"""Recompute Sanvito-space test metrics from exported predictions (no model, no GPU).

Runs finished before the test_step mask fix report a biased Sanvito number; the
exported <id>_<channel>_aug.npz files are unaffected, so the corrected metrics are a
pure post-processing job. Reports MAE / RMSE / MaxAE over all structures and over
the non-magnetic subset. The e3nn columns are recomputed as a check: they must
reproduce the run's published test_component_* exactly.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.paw_basis_transform import (
    sanvito_schema_and_lmaxmix_masks,
    sanvito_to_e3nn_with_basis_padded,
)
from src.augnet_model import lmaxmix_component_mask
from src.paw_moments import moments_from_coeffs
from src.run_paw_chgcar import build_stats_example_from_chgcar, resolve_lmaxmix


class Acc:
    """Streaming MAE / RMSE / MaxAE."""

    def __init__(self):
        self.n = 0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.max_abs = 0.0

    def add(self, n, sum_abs, sum_sq, max_abs):
        self.n += n
        self.sum_abs += sum_abs
        self.sum_sq += sum_sq
        self.max_abs = max(self.max_abs, max_abs)

    def as_dict(self):
        if self.n == 0:
            return {"n_coeff": 0, "mae": None, "rmse": None, "maxae": None}
        return {
            "n_coeff": self.n,
            "mae": self.sum_abs / self.n,
            "rmse": math.sqrt(self.sum_sq / self.n),
            "maxae": self.max_abs,
        }


def structure_stats(err: torch.Tensor):
    if err.numel() == 0:
        return 0, 0.0, 0.0, 0.0
    a = err.abs()
    return int(err.numel()), float(a.sum()), float((err ** 2).sum()), float(a.max())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--export-dir", action="append", required=True,
                   help="aug_pred_dir_test_* directory. Repeatable; dirs sharing a "
                        "test set parse each CHGCAR once.")
    p.add_argument("--label", action="append", default=None,
                   help="Name for each --export-dir, in the same order.")
    p.add_argument("--channel", default="total")
    p.add_argument("--mag-min", type=float, default=0.1,
                   help="sum_atoms |Tr(D_mag)| below this is non-magnetic "
                        "(loss.spin_mag_min).")
    p.add_argument("--out-dir", default="results/sanvito_fixed")
    args = p.parse_args()

    dirs = [Path(d) for d in args.export_dir]
    labels = args.label or [d.parent.name for d in dirs]
    if len(labels) != len(dirs):
        raise SystemExit("--label must be given once per --export-dir")

    by_material: dict[str, dict[str, Path]] = defaultdict(dict)
    for label, d in zip(labels, dirs):
        files = sorted(d.glob(f"*_{args.channel}_aug.npz"))
        if not files:
            raise SystemExit(f"No *_{args.channel}_aug.npz in {d}")
        for f in files:
            by_material[f.name[: -len(f"_{args.channel}_aug.npz")]][label] = f
        print(f"{label}: {len(files)} exported structures in {d}")

    acc = {l: {"all_s": Acc(), "nm_s": Acc(), "all_e": Acc(), "nm_e": Acc()} for l in labels}
    rows = {l: [] for l in labels}
    n_mag = {l: 0 for l in labels}

    for i, (mid, per_label) in enumerate(sorted(by_material.items())):
        first = np.load(next(iter(per_label.values())), allow_pickle=True)
        src = str(first["source_path"])

        z, targets = build_stats_example_from_chgcar(src, target_order="sanvito",
                                                    with_lmaxmix=True)
        # mag: physical spin-occupancy target. A structure without a mag CHGCAR
        # channel has no mag target and contributes 0 coefficients, the same
        # convention the trainer's test metrics use (its mag_mask is all-False;
        # for structures with the channel the mask equals the schema mask).
        if args.channel == "mag":
            target_sanvito = targets["mag_target"]
            chan_absent = not bool(targets["has_mag"])
        else:
            target_sanvito = targets["total_target"]
            chan_absent = False
        full_schema_mask, _ = sanvito_schema_and_lmaxmix_masks(z, None)
        target_e3nn, _ = sanvito_to_e3nn_with_basis_padded(
            target_sanvito, full_schema_mask, z,
        )
        detected_lmaxmix = resolve_lmaxmix(targets.get("lmaxmix"))

        if bool(targets["has_mag"]):
            mag_e3nn, _ = sanvito_to_e3nn_with_basis_padded(
                targets["mag_target"], full_schema_mask, z,
            )
            mag_tr1, _ = moments_from_coeffs(mag_e3nn, z, [1])
            mag_moment = float(mag_tr1[:, 0].abs().sum())
        else:
            mag_moment = 0.0
        magnetic = mag_moment >= args.mag_min

        for label, npz_path in per_label.items():
            d = np.load(npz_path, allow_pickle=True)
            pred_sanvito = torch.from_numpy(d["aug_sanvito_padded"]).to(target_sanvito.dtype)

            if not torch.equal(torch.from_numpy(d["atomic_numbers"]).long(), z):
                raise SystemExit(f"{mid}: atomic numbers in {npz_path} do not match {src}")

            # Older exports lack schema_mask/lmaxmix_mask; rebuild from the re-detected LMAXMIX.
            if "schema_mask" in d.files:
                schema_mask = torch.from_numpy(d["schema_mask"]).bool()
                lmaxmix_mask = torch.from_numpy(d["lmaxmix_mask"]).bool()
                lmaxmix = int(d["lmaxmix"])
            else:
                lmaxmix = -1 if detected_lmaxmix is None else int(detected_lmaxmix)
                schema_mask, lmaxmix_mask = sanvito_schema_and_lmaxmix_masks(
                    z, None if lmaxmix < 0 else lmaxmix,
                )

            mask_sanvito = schema_mask & lmaxmix_mask
            err_mask_sanvito = mask_sanvito if not chan_absent else torch.zeros_like(mask_sanvito)
            err_s = (pred_sanvito - target_sanvito)[err_mask_sanvito]

            pred_e3nn, _ = sanvito_to_e3nn_with_basis_padded(pred_sanvito, mask_sanvito, z)
            mask_e3nn = schema_mask.clone()
            if lmaxmix >= 0:
                mask_e3nn = mask_e3nn & lmaxmix_component_mask(z, lmaxmix)
            if chan_absent:
                mask_e3nn = torch.zeros_like(mask_e3nn)
            # npz["mask"] is e3nn-ordered in old exports and Sanvito-ordered in new ones.
            stored = torch.from_numpy(d["mask"]).bool()
            reconstructed = mask_sanvito if "schema_mask" in d.files else mask_e3nn
            if not torch.equal(reconstructed, stored):
                raise SystemExit(
                    f"{mid} ({label}): reconstructed mask does not match the mask "
                    f"stored in {npz_path}"
                )
            err_e = (pred_e3nn - target_e3nn)[mask_e3nn]

            ns, sa, sq, ma = structure_stats(err_s)
            ne, ea, eq, em = structure_stats(err_e)

            acc[label]["all_s"].add(ns, sa, sq, ma)
            acc[label]["all_e"].add(ne, ea, eq, em)
            if not magnetic:
                acc[label]["nm_s"].add(ns, sa, sq, ma)
                acc[label]["nm_e"].add(ne, ea, eq, em)
            else:
                n_mag[label] += 1

            rows[label].append({
                "material_id": mid, "n_atoms": int(z.numel()), "n_coeff": ns,
                "lmaxmix": lmaxmix, "mag_moment": round(mag_moment, 8),
                "is_magnetic": int(magnetic),
                "sanvito_mae": sa / max(ns, 1), "sanvito_mse": sq / max(ns, 1),
                "sanvito_maxae": ma,
                "e3nn_mae": ea / max(ne, 1), "e3nn_mse": eq / max(ne, 1),
                "e3nn_maxae": em,
            })

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(by_material)} structures", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for label in labels:
        n_struct = len(rows[label])
        summary[label] = {
            "n_structures": n_struct,
            "n_magnetic": n_mag[label],
            "n_non_magnetic": n_struct - n_mag[label],
            "mag_min": args.mag_min,
            "all": {"sanvito": acc[label]["all_s"].as_dict(), "e3nn": acc[label]["all_e"].as_dict()},
            "non_magnetic": {"sanvito": acc[label]["nm_s"].as_dict(), "e3nn": acc[label]["nm_e"].as_dict()},
        }

        import csv
        csv_path = out_dir / f"{label}_sanvito_per_structure.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[label][0]))
            w.writeheader()
            w.writerows(rows[label])
        print(f"wrote {csv_path}")

    (out_dir / "sanvito_metrics_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
