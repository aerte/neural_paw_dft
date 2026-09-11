"""Test-set evaluation of the spin channel with an external magnetization constraint.

The trained "constraint" (`_build_spin_weights`) rescales the signed spin weights so
that int m dv == m_total, a single signed net moment per structure. During training
and in `test_step`, m_total is derived from the ground-truth cd_diff grid -- an
oracle at inference time. This script substitutes a per-structure value from a CSV
(e.g. CHGNet-predicted magmoms) so the constraint is deployable, and scores the spin
prediction against the ground-truth grid exactly as `_apply_spin_loss` does.

CHGNet caveat: its per-site moments are UNSIGNED, so their sum is a ferromagnetic
assumption. For an antiferromagnet the true net moment is ~0 (the oracle skips the
rescale via spin_mag_min) while the unsigned sum is large -- the csv arm is expected
to be actively hurt there. m_abs_sum in the output CSV (int|m| dv from the target
grid) lets the analysis separate those regimes.

Constraint modes:
  csv     m_total looked up by mp-id (missing id -> None, i.e. rescale skipped)
  oracle  m_total from the target grid, as in training (reference arm)
  none    m_total = None everywhere (isolates what the constraint contributes)

Usage (see eval_runs/eval_spin_chgnet_3090.sh for the SLURM wrapper):
  python eval_spin_constrained.py --config <yaml> --ckpt <file-or-dir> \
      --constraint csv --constraint_csv eval_runs/chgnet_magmom_constraints_mp.csv \
      --out_csv spin_eval_csv.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import os
import time

import lz4.frame
import numpy as np
import torch
import yaml
from neural_init._resources import resolve_data_path

from neural_init.spin_electrafi.model.ELECTRAFI import ELECTRAFI
from neural_init.spin_electrafi.model.model_utils import PWGrid
from neural_init.spin_electrafi.utils.train_helper_funcs import get_files, set_all_paths


def resolve_ckpt(path: str) -> str:
    """A directory means: the newest checkpoint in it (last.ckpt / hpc_ckpt_*.ckpt),
    mirroring train.py's resume-from-newest-by-mtime rule."""
    if os.path.isdir(path):
        candidates = [
            p
            for p in [
                os.path.join(path, "last.ckpt"),
                *glob.glob(os.path.join(path, "hpc_ckpt_*.ckpt")),
            ]
            if os.path.isfile(p)
        ]
        if not candidates:
            raise FileNotFoundError(f"no *.ckpt in {path}")
        return max(candidates, key=os.path.getmtime)
    return path


def load_constraints(csv_path: str, id_col: str, value_col: str) -> dict[str, float]:
    with open(csv_path) as f:
        return {row[id_col]: float(row[value_col]) for row in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint file, .model_state_dict file, or checkpoint dir")
    ap.add_argument("--constraint", choices=["csv", "oracle", "none"], default="csv")
    ap.add_argument("--constraint_csv", default=None)
    ap.add_argument("--id_col", default="MP_ID")
    ap.add_argument("--value_col", default="chgnet_magmom_sum")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--save_grids", default=None,
                    help="directory to dump the predicted spin grid per structure "
                         "as <mp_id>_spin.npy.lz4 (float32, shape (nx, ny, nz))")
    ap.add_argument("--ood_path", default=None,
                    help="directory of .lz4 CHGCARs; evaluate these instead of the test split")
    ap.add_argument("--ood_name", default="gnome_ecd")
    ap.add_argument("--limit", type=int, default=0, help="stop after N structures (smoke test)")
    args = ap.parse_args()

    if args.save_grids:
        os.makedirs(args.save_grids, exist_ok=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if config.get("spin_type") != "total_diff":
        raise SystemExit("config has no spin channel (spin_type != total_diff)")
    config["wandb"] = False
    # Mirror the backbone-config injection in train.run(); these values are part of the trained architecture.
    with open(resolve_data_path(config["escaip_cfg_path"], "spin_electrafi")) as f:
        config["escaip_config"] = yaml.safe_load(f)
    if torch.cuda.is_available() and str(config["data_split"]).startswith("mpfull2025"):
        backbone = config["escaip_config"]["model"]["backbone"]
        backbone["max_num_nodes_per_batch"] = 154
        backbone["use_compile"] = True
        backbone["max_neighbors"] = 300
    config = set_all_paths(config, None)

    constraints: dict[str, float] = {}
    if args.constraint == "csv":
        if not args.constraint_csv:
            raise SystemExit("--constraint csv requires --constraint_csv")
        constraints = load_constraints(args.constraint_csv, args.id_col, args.value_col)
        print(f"[constraints] {len(constraints)} ids from {args.constraint_csv}")

    _, test_files, _ = get_files(config)
    model = ELECTRAFI(
        train_files=[], test_files=test_files, validation_files=[],
        model_handler=None, config=config,
    )

    ckpt_path = resolve_ckpt(args.ckpt)
    print(f"[ckpt] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.ood_path:
        # Same file discovery as get_files_ood, minus the shuffle. DensityDataset substitutes
        # the next file on a load failure, so dedupe the output by mp_id.
        files = sorted(
            os.path.join(args.ood_path, f)
            for f in os.listdir(args.ood_path)
            if f.endswith(".lz4")
        )
        print(f"[ood] {len(files)} files from {args.ood_path}")
        loader = model.ood_dataloader(files=files, path=args.ood_path, name=args.ood_name)
    else:
        loader = model.test_dataloader()

    fieldnames = [
        "mp_id", "n_atoms", "n_elec", "constraint_mode", "m_constraint",
        "m_oracle_net", "m_abs_sum", "m_pred_net",
        "spin_nmae", "spin_abs_err_e", "charge_err_pct", "seconds",
    ]
    write_header = not os.path.exists(args.out_csv)
    out_f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    n_done = n_missing = n_magnetic = 0
    nmae_sum = charge_sum = 0.0
    for batch in loader:
        density, struc, n_elec, grid_dict, filename, ood_name, cd_diff = batch[0]
        mp_id = os.path.basename(str(filename)).split(".")[0]
        struc, rot_meta = model._maybe_rotate_structure(struc, split="test", filename=str(filename))
        density = model._maybe_transform_density_target(density, rot_meta)

        nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
        dv = float(struc.get_volume()) / (nx * ny * nz)
        Ne_true = float(density.sum().item() * dv)
        m_oracle, m_abs_sum = model._spin_stats(cd_diff, dv)

        if args.constraint == "oracle":
            m_used = m_oracle
        elif args.constraint == "csv":
            m_used = constraints.get(mp_id)
            if m_used is None:
                n_missing += 1
        else:
            m_used = None

        if (model.pw.nx, model.pw.ny, model.pw.nz) != (nx, ny, nz):
            model.pw = PWGrid((nx, ny, nz), device=model.device, dtype=model.precision)

        t0 = time.time()
        with torch.inference_mode():
            out = model(struc, n_elec=Ne_true, sampled_points=None, m_total=m_used)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0

        _, charge_err = model.loss.compute_total_loss(
            sys=struc, pred_cd=out["rho"], true_cd_tens=density, grid_dict=grid_dict,
            n_valence_electrons=Ne_true, sampled_points=None, training=False, volume=None,
        )
        charge_pct = float(charge_err) * 100.0

        spin_nmae = spin_abs_err = m_pred = None
        if out["rho_spin"] is not None and cd_diff is not None:
            _, nmae, abs_err = model.spin_loss.compute_spin_loss(
                out["rho_spin"], cd_diff, dv, None
            )
            spin_nmae = float(nmae) if nmae is not None else None
            spin_abs_err = float(abs_err)
            m_pred = float(out["rho_spin"].sum()) * dv

        if args.save_grids and out["rho_spin"] is not None:
            buf = io.BytesIO()
            np.save(buf, out["rho_spin"].detach().float().cpu().numpy())
            with open(os.path.join(args.save_grids, f"{mp_id}_spin.npy.lz4"), "wb") as gf:
                gf.write(lz4.frame.compress(buf.getvalue()))

        writer.writerow({
            "mp_id": mp_id,
            "n_atoms": len(struc),
            "n_elec": round(Ne_true, 4),
            "constraint_mode": args.constraint,
            "m_constraint": "" if m_used is None else round(m_used, 4),
            "m_oracle_net": "" if m_oracle is None else round(m_oracle, 4),
            "m_abs_sum": round(m_abs_sum, 4),
            "m_pred_net": "" if m_pred is None else round(m_pred, 4),
            "spin_nmae": "" if spin_nmae is None else round(spin_nmae, 6),
            "spin_abs_err_e": "" if spin_abs_err is None else round(spin_abs_err, 4),
            "charge_err_pct": round(charge_pct, 4),
            "seconds": round(dt, 3),
        })
        out_f.flush()

        n_done += 1
        charge_sum += charge_pct
        if spin_nmae is not None:
            n_magnetic += 1
            nmae_sum += spin_nmae
        if n_done % 25 == 0:
            print(
                f"[{n_done}] charge {charge_sum / n_done:.3f}% | "
                f"spin nmae {nmae_sum / max(n_magnetic, 1):.4f} "
                f"({n_magnetic} magnetic, {n_missing} ids missing from csv)",
                flush=True,
            )
        if args.limit and n_done >= args.limit:
            break

    out_f.close()
    print(
        f"\n[done] {n_done} structures -> {args.out_csv}\n"
        f"  mean charge err:        {charge_sum / max(n_done, 1):.4f} %\n"
        f"  mean spin nmae:         {nmae_sum / max(n_magnetic, 1):.4f} over {n_magnetic} magnetic\n"
        f"  ids missing from csv:   {n_missing}"
    )


if __name__ == "__main__":
    main()
