#!/usr/bin/env python3
"""ML-initialized augmentation-charge SCF run.

For one MP structure per invocation, seeds VASP with:
  - pseudo density grid (total + diff) from the CONVERGED CHGCAR, and
  - augmentation occupancies (total + optional diff data_aug) from ML
    predictions stored as per-mpid `.npz` files in `--aug_pred_dir`.

Mirrors run_conv_grid_sad_aug_mp.py (grid=converged) but the augmentation
comes from an ML model's predictions instead of a SAD CHGCAR. The variant tag
is auto-derived from the prediction directory (e.g. `.../10k/...` -> ml_aug_10k)
so results for the 1k/10k/50k models never collide.

Prediction files (one set per non-magnetic mpid):
  {aug_pred_dir}/{mpid}_total_aug.npz      (always required)
  {aug_pred_dir}/{mpid}_magnetic_aug.npz   (used only if present)

Exit codes (match the SBATCH wrapper expectations):
  0 = OK              2 = slice exhausted     3 = per-structure failure
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import lz4.frame
import pandas as pd
import wandb

from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Outcar

import scf_stats
from neural_paw_dft.vasp_runner.chgcar import (build_ml_aug_chgcar,
                                build_total_pseudo_grid_chgcar,
                                get_chgcar_grid_dims_textparse)
from neural_paw_dft.vasp_runner.oszicar import count_scf_breakdown_from_oszicar
from neural_paw_dft.vasp_runner.spin_npy import index_spin_dir
from neural_paw_dft.vasp_runner.sources.mp import derive_mpid, write_mp_inputs_for_mpid, MissingMPTaskDocError
from neural_paw_dft.vasp_runner import runs_db, results
from neural_paw_dft.vasp_runner.failsafe import run_guarded, watchdog
from neural_paw_dft.vasp_runner.scf import prune_workdir_keep_outputs


ICHARG = 1
REPO_ROOT = Path(__file__).resolve().parent
MAG_VARIANT = "true_init"          # DB variant carrying the magnetic truth
# Prediction filename pattern. channel is 'total' or 'magnetic'. Keep this in
# one place so the magnetic naming convention is trivial to change later.
AUG_NPZ = "{mpid}_{channel}_aug.npz"
# The AugNet spin export (--mag_aug_pred_dir) names its files {mpid}_mag_aug.npz.
MAG_AUG_NPZ = "{mpid}_mag_aug.npz"


def derive_variant(aug_pred_dir: str, override: str = None) -> str:
    """Auto-derive the variant tag from the prediction directory.

    Scans the path components for one matching `^\\d+k$` (1k/10k/50k) or `full`
    and returns `ml_aug_<that>`. Falls back to `override` (if given) or a
    sanitized basename.
    """
    if override:
        return override if override.startswith("ml_aug") else f"ml_aug_{override}"
    parts = [p for p in os.path.normpath(aug_pred_dir).split(os.sep) if p]
    for p in parts:
        if re.fullmatch(r"\d+k|full", p):
            return f"ml_aug_{p}"
    base = re.sub(r"[^0-9A-Za-z]+", "_", os.path.basename(os.path.normpath(aug_pred_dir)))
    return f"ml_aug_{base}".rstrip("_")


PRED_RE = re.compile(r"^(mp-\d+)_")


def _pred_error(name: str) -> float:
    """Error float embedded before `_test_` in a predicted CHGCAR filename.

    Mirrors `the total-pseudo-grid runner (not included)::_pred_error`; +inf on parse failure so
    such files sort last and are never preferred.
    """
    try:
        return float(name.split("_test_")[0].rsplit("_", 1)[1])
    except Exception:
        return float("inf")


def index_pred_dir(pred_dir: str) -> dict:
    """Map mp-id -> chosen predicted CHGCAR filename (lowest error)."""
    groups: dict = {}
    for name in os.listdir(pred_dir):
        if not name.endswith(".CHGCAR"):
            continue
        m = PRED_RE.match(name)
        if m:
            groups.setdefault(m.group(1), []).append(name)
    return {mp: min(files, key=_pred_error) for mp, files in groups.items()}


def _materialize_chgcar(chgcar_path: str, dest: str):
    """Copy or lz4-decompress `chgcar_path` into `dest`."""
    if chgcar_path.endswith(".lz4"):
        with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy(chgcar_path, dest)


def run_variant(mpid: str, workdir: str, chgcar_path: str, vasp_cmd,
                ispin1=False, magmom=None, relax_magmom=None):
    os.makedirs(workdir, exist_ok=True)
    write_mp_inputs_for_mpid(mpid, workdir)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM"):
        incar.pop(bad, None)
    incar["ICHARG"] = ICHARG
    incar["ISTART"] = 0
    incar["LCHARG"] = True
    if ispin1:
        # Strictly non-magnetic: collapse to a single spin channel and drop the
        # spin-2-only moment tags so VASP does not attempt to consult them.
        incar["ISPIN"] = 1
        for spin_tag in ("MAGMOM", "NUPDOWN"):
            incar.pop(spin_tag, None)
    if magmom is not None:
        # Uniform per-ion moment replacing MP's own MAGMOM; the seed carries no
        # diff channel, so this is what VASP starts the spin channel from.
        existing = incar.get("MAGMOM")
        natoms = len(existing) if isinstance(existing, (list, tuple)) else None
        if natoms is None:
            from pymatgen.core import Structure
            natoms = len(Structure.from_file(os.path.join(workdir, "POSCAR")))
        incar["MAGMOM"] = [magmom] * natoms
    elif relax_magmom is not None and not ispin1:
        # Per-site relaxed moments (MP relax-task output) replacing MP's own
        # MAGMOM; see vasp_runner/relaxmag.py.
        from neural_paw_dft.vasp_runner.relaxmag import apply_relaxed_magmom
        apply_relaxed_magmom(incar, relax_magmom)
    incar.write_file(incar_path)

    _materialize_chgcar(chgcar_path, os.path.join(workdir, "CHGCAR"))

    start = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
        vasp_ok = True
    except subprocess.CalledProcessError:
        vasp_ok = False
    wall = time.time() - start

    osz_path = os.path.join(workdir, "OSZICAR")
    dav, rmm, total, other_counts = count_scf_breakdown_from_oszicar(osz_path)
    other_total = sum(other_counts.values()) if other_counts else 0

    energy = total_mag = magnetization = spin = None
    outcar_path = os.path.join(workdir, "OUTCAR")
    if vasp_ok and os.path.isfile(outcar_path):
        try:
            outcar = Outcar(outcar_path)
            energy = outcar.final_energy
            total_mag = getattr(outcar, "total_mag", None)
            magnetization = getattr(outcar, "magnetization", None)
            spin = getattr(outcar, "spin", None)
        except Exception:
            pass

    return {
        "status": "ok" if energy is not None else "failed",
        "energy": energy,
        "total_mag": total_mag,
        "magnetization": magnetization,
        "spin": spin,
        "dav": dav,
        "rmm": rmm,
        "other": other_total,
        "total": total,
        "wall": wall,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the converged-grid + ML-augmentation SCF variant for one MP structure."
    )
    parser.add_argument("--csv", "-C", dest="CSV_PATH", required=True,
                        help="Task CSV with at least ['CHGCAR_PATH','MP_ID'].")
    parser.add_argument("--outdir", "-o", dest="BASE_DIR", required=True,
                        help="Base directory for per-structure work dirs.")
    parser.add_argument("--results_csv", "-R", dest="RESULTS_CSV", required=True,
                        help="Narrow-format CSV to append per-structure rows to.")
    parser.add_argument("--aug_pred_dir", dest="AUG_PRED_DIR", default=None,
                        help="Directory holding ML augmentation predictions "
                             "({mpid}_total_aug.npz, optional {mpid}_magnetic_aug.npz). "
                             "May be omitted only with --mag_aug_pred_dir (the "
                             "total augmentation then stays converged).")
    parser.add_argument("--mag_aug_pred_dir", dest="MAG_AUG_PRED_DIR", default=None,
                        help="Directory of ML DIFF-augmentation predictions "
                             "({mpid}_mag_aug.npz, the AugNet spin export). "
                             "Replaces the converged diff augmentation; required "
                             "per structure when given. Mutually exclusive with "
                             "--total_only/--ispin1.")
    parser.add_argument("--spin_pred_dir", dest="SPIN_PRED_DIR", default=None,
                        help="Directory of ML spin-density grids "
                             "(<mpid>_spin.npy.lz4, e/A^3; volume-scaled on load) "
                             "replacing the converged diff GRID. Mutually "
                             "exclusive with --total_only/--ispin1.")
    parser.add_argument("--variant", dest="VARIANT_OVERRIDE", default=None,
                        help="Override the auto-derived variant tag (default: "
                             "ml_aug_<size> parsed from --aug_pred_dir).")
    parser.add_argument("--grid_pred_dir", "--grid-pred-dir", dest="grid_pred_dir",
                        default=None,
                        help="Directory of ML-predicted total-density CHGCARs "
                             "(same layout as the total-pseudo-grid runner (not included) "
                             "--pred_dir). When given, the total pseudo grid "
                             "comes from the model too, so the whole total "
                             "density is ML: ML grid + ML total augmentation. "
                             "Without it the grid stays converged.")
    parser.add_argument("--mag_only", action="store_true",
                        help="Run ONLY magnetic structures (DB criterion over "
                             "true_init); non-magnetic ones are recorded as "
                             "skipped_nonmag. Mutually exclusive with --nonmag_only.")
    parser.add_argument("--nonmag_only", action="store_true",
                        help="Skip magnetic structures using the DB criterion "
                             "(scf_stats over true_init), as the total-pseudo-grid runner does. "
                             "Without this the full test set is run.")
    parser.add_argument("--total_only", action="store_true",
                        help="Drop the diff pseudo grid and diff augmentation: "
                             "converged TOTAL valence density + ML total "
                             "augmentation only, spin channel from MAGMOM.")
    parser.add_argument("--magmom", type=float, default=None,
                        help="Uniform per-ion INCAR MAGMOM (e.g. --magmom 0.001). "
                             "Intended for use with --total_only.")
    parser.add_argument("--relaxed_magmoms", default=None,
                        help="Harvest CSV from exps/harvest_relaxed_magmoms.py. "
                             "When given, INCAR MAGMOM is replaced by the "
                             "per-site MP relax-task moments; mp-ids without a "
                             "usable entry keep MP's own MAGMOM and are flagged "
                             "magmom_source=default_fallback. Mutually "
                             "exclusive with --magmom/--ispin1.")
    parser.add_argument("--pbe-only", dest="pbe_only",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="With --relaxed_magmoms: only GGA/GGA+U relax "
                             "moments (default). --no-pbe-only admits "
                             "r2SCAN/SCAN/PBEsol too.")
    parser.add_argument("--ispin1", action="store_true",
                        help="Force ISPIN=1 in the INCAR (drop MAGMOM/NUPDOWN) "
                             "and build a strictly total-only seed: total pseudo "
                             "grid + ML total augmentation, no diff grid or diff "
                             "aug. Implies --total_only for the seed build. "
                             "Mutually exclusive with --magmom.")
    parser.add_argument("--renorm", action="store_true",
                        help="Scale the ML-predicted total grid so its integrated "
                             "charge matches the converged reference CHGCAR's, i.e. "
                             "NELECT. Only meaningful with --grid_pred_dir (the "
                             "converged grid is already normalized). ChargE3Net "
                             "predictions are not normalized in advance; ElectraFi's "
                             "are, so this should be a no-op there.")
    parser.add_argument("--vasp_cmd", nargs="+", default=["vasp_std"],
                        help="Command used to run VASP (e.g. --vasp_cmd srun vasp_std).")
    parser.add_argument("--runs_db", dest="RUNS_DB", default="runs/runs.db",
                        help="SQLite manifest used for the cross-experiment skip-check.")
    parser.add_argument("--start_row", type=int, default=0,
                        help="Inclusive start row index in the task CSV slice.")
    parser.add_argument("--end_row", type=int, default=None,
                        help="Exclusive end row index in the task CSV slice. "
                             "Defaults to len(df).")
    args = parser.parse_args()

    if args.ispin1 and args.magmom is not None:
        print("❌ --ispin1 and --magmom are mutually exclusive.")
        sys.exit(1)
    if args.relaxed_magmoms and (args.ispin1 or args.magmom is not None):
        print("❌ --relaxed_magmoms is mutually exclusive with "
              "--magmom/--ispin1.")
        sys.exit(1)

    if args.AUG_PRED_DIR is None and args.MAG_AUG_PRED_DIR is None:
        print("❌ Need --aug_pred_dir and/or --mag_aug_pred_dir.")
        sys.exit(1)
    if (args.MAG_AUG_PRED_DIR or args.SPIN_PRED_DIR) and (args.total_only or args.ispin1):
        print("❌ --mag_aug_pred_dir/--spin_pred_dir are mutually exclusive with "
              "--total_only/--ispin1 (those drop the diff channel).")
        sys.exit(1)
    if args.mag_only and args.nonmag_only:
        print("❌ --mag_only and --nonmag_only are mutually exclusive.")
        sys.exit(1)

    relax_map = None
    if args.relaxed_magmoms:
        from neural_paw_dft.vasp_runner.relaxmag import load_relaxed_magmoms
        relax_map = load_relaxed_magmoms(args.relaxed_magmoms, args.pbe_only)
        print(f"Relaxed magmoms: {len(relax_map)} usable mp-ids from "
              f"{args.relaxed_magmoms} (others keep MP's own MAGMOM)")

    VARIANT = derive_variant(args.AUG_PRED_DIR or args.MAG_AUG_PRED_DIR,
                             args.VARIANT_OVERRIDE)
    PERFORMED_COL = f"performed_{VARIANT}"
    SUBDIR = f"run_{VARIANT}"
    WANDB_PROJECT = "VASP_SCF_MlAug_MP"
    print(f"Variant={VARIANT} (aug_pred_dir={args.AUG_PRED_DIR}, "
          f"mag_aug_pred_dir={args.MAG_AUG_PRED_DIR})")
    grid_index = None
    if args.grid_pred_dir:
        grid_index = index_pred_dir(args.grid_pred_dir)
        print(f"  grid source: ML total density from {args.grid_pred_dir} "
              f"({len(grid_index)} mp-ids indexed)")
    spin_index = None
    if args.SPIN_PRED_DIR:
        spin_index = index_spin_dir(args.SPIN_PRED_DIR)
        print(f"  diff grid source: ML spin density from {args.SPIN_PRED_DIR} "
              f"({len(spin_index)} mp-ids indexed)")

    for label, d in (("Prediction", args.AUG_PRED_DIR),
                     ("Mag-aug prediction", args.MAG_AUG_PRED_DIR)):
        if d is not None and not os.path.isdir(d):
            print(f"❌ {label} dir not found: {d}")
            sys.exit(1)

    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str, "MP_ID": str})
    n_rows = len(df)
    start = max(args.start_row, 0)
    end = n_rows if args.end_row is None else min(args.end_row, n_rows)
    if start >= end:
        print(f"❌ Invalid slice [{start}:{end}) for CSV with {n_rows} rows.")
        sys.exit(1)
    print(f"Using slice [{start}:{end}) out of {n_rows}")

    if PERFORMED_COL not in df.columns:
        df[PERFORMED_COL] = False

    df_slice = df.iloc[start:end]

    # Manifest is the source of truth: overwrite slice CSV's performed flag
    # from runs.db (any recorded status counts as "attempted" → skip). On
    # manifest read failure, leave the slice CSV alone.
    conn = None
    try:
        conn = runs_db.get_connection(args.RUNS_DB)
        done = runs_db.terminal_set(conn, VARIANT, df_slice["MP_ID"].astype(str))
        # runs.db is the SOLE skip-truth: hard-overwrite the local performed
        # flag from the manifest's terminal set. A dropped runs.db write (NFS
        # lock contention) now causes at worst a cheap idempotent re-run of one
        # structure, never a permanent skip. terminal_set excludes in_progress
        # so walltime-killed rows stay retryable.
        df.loc[df_slice.index, PERFORMED_COL] = df_slice["MP_ID"].astype(str).isin(done)
        df_slice = df.iloc[start:end]
    except Exception as e:
        print(f"⚠️ Could not read runs manifest {args.RUNS_DB}: {e!r}. "
              f"Falling back to slice CSV performed flags.")

    pending = df_slice[df_slice[PERFORMED_COL] == False]
    if pending.empty:
        print(f"No pending rows in slice [{start}:{end}).")
        sys.exit(2)

    # Scan the pending rows for the first runnable structure.
    #   - missing reference CHGCAR  -> TERMINAL skip: mark performed (it will
    #     never appear), record `missing_chgcar`, move on.
    #   - missing ML prediction npz -> NON-TERMINAL skip: leave performed=False
    #     and write nothing to runs.db, so a later submission re-picks the row
    #     once predictions exist. We just step past it here to keep the slice
    #     making forward progress (otherwise the wrapper would re-select the
    #     same first-pending row every iteration and spin forever).
    # DB magnetic verdict per mp-id (scf_stats criterion over true_init), only
    # consulted with --nonmag_only.
    mag_map = {}
    if (args.nonmag_only or args.mag_only) and conn is not None:
        try:
            mag_map = scf_stats.magnetic_map(conn, REPO_ROOT, MAG_VARIANT,
                                             scf_stats.DEFAULT_MAG_THRESHOLD)
        except Exception as e:
            print(f"⚠️ Could not build magnetic map from {args.RUNS_DB}: {e!r}.")
        print(f"Magnetic map: {len(mag_map)} classified "
              f"({sum(mag_map.values())} magnetic).")

    idx = chgcar_path = mpid = total_npz = mag_npz = None
    use_mag = False
    for cand in pending.index:
        cand_chgcar = df.at[cand, "CHGCAR_PATH"]
        cand_mpid = derive_mpid(df.loc[cand], cand_chgcar)

        if not os.path.isfile(cand_chgcar):
            print(f"⚠️ CHGCAR not found for {cand_mpid}: {cand_chgcar}. "
                  f"Marking performed and skipping.")
            df.at[cand, PERFORMED_COL] = True
            df.to_csv(args.CSV_PATH, index=False)
            runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                              chgcar_path=cand_chgcar, status="missing_chgcar",
                              source_csv=args.CSV_PATH, source_row=int(cand))
            continue

        if args.nonmag_only:
            if cand_mpid not in mag_map:
                print(f"⚠️ No {MAG_VARIANT} classification for {cand_mpid}; cannot "
                      f"confirm non-magnetic. Marking performed "
                      f"(skipped_unknown_mag).")
                df.at[cand, PERFORMED_COL] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar,
                                  status="skipped_unknown_mag",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue
            if mag_map[cand_mpid]:
                print(f"↷ {cand_mpid} is magnetic (DB criterion); skipping.")
                df.at[cand, PERFORMED_COL] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar,
                                  status="skipped_magnetic",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue

        if args.mag_only:
            if cand_mpid not in mag_map:
                print(f"⚠️ No {MAG_VARIANT} classification for {cand_mpid}; cannot "
                      f"confirm magnetic. Marking performed "
                      f"(skipped_unknown_mag).")
                df.at[cand, PERFORMED_COL] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar,
                                  status="skipped_unknown_mag",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue
            if not mag_map[cand_mpid]:
                print(f"↷ {cand_mpid} is non-magnetic (DB criterion); skipping.")
                df.at[cand, PERFORMED_COL] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar,
                                  status="skipped_nonmag",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue

        cand_total = None
        if args.AUG_PRED_DIR is not None:
            cand_total = os.path.join(args.AUG_PRED_DIR,
                                      AUG_NPZ.format(mpid=cand_mpid, channel="total"))
            if not os.path.isfile(cand_total):
                print(f"⚠️ No total prediction for {cand_mpid}: {cand_total}. "
                      f"Leaving row PENDING (retryable) and scanning past it.")
                continue
        if args.MAG_AUG_PRED_DIR is not None:
            cand_mag = os.path.join(args.MAG_AUG_PRED_DIR,
                                    MAG_AUG_NPZ.format(mpid=cand_mpid))
            if not os.path.isfile(cand_mag):
                print(f"⚠️ No mag-aug prediction for {cand_mpid}: {cand_mag}. "
                      f"Leaving row PENDING (retryable) and scanning past it.")
                continue
        else:
            # Legacy layout: an optional {mpid}_magnetic_aug.npz next to the
            # total npz in --aug_pred_dir.
            cand_mag = os.path.join(args.AUG_PRED_DIR,
                                    AUG_NPZ.format(mpid=cand_mpid, channel="magnetic"))
            if not os.path.isfile(cand_mag):
                cand_mag = None

        idx = cand
        chgcar_path = cand_chgcar
        mpid = cand_mpid
        total_npz = cand_total
        mag_npz = cand_mag
        use_mag = cand_mag is not None
        break

    if idx is None:
        print(f"No runnable rows in slice [{start}:{end}) — all pending rows "
              f"lack a reference CHGCAR or an ML prediction. Leaving "
              f"missing-prediction rows pending for a later submission.")
        sys.exit(2)

    print(f"Selected idx={idx}, CHGCAR={chgcar_path}, MPID={mpid}")
    print(f"Predictions: total={total_npz or '(converged)'} "
          f"magnetic={mag_npz if use_mag else '(none)'}")

    # Manifest in_progress only — do NOT set the local performed flag yet. If
    # the job is killed mid-run, this row has no terminal outcome and must stay
    # pending so the next sbatch iteration re-picks it. PERFORMED_COL is flipped
    # True only after a terminal result is written below.
    runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                      chgcar_path=chgcar_path, status="in_progress",
                      source_csv=args.CSV_PATH, source_row=int(idx))

    base = os.path.splitext(os.path.basename(chgcar_path))[0]
    run_base = os.path.join(args.BASE_DIR, base)
    os.makedirs(run_base, exist_ok=True)

    try:
        grid_dims = get_chgcar_grid_dims_textparse(chgcar_path)
    except Exception:
        grid_dims = None

    converged_local_dir = os.path.join(run_base, "_converged")
    os.makedirs(converged_local_dir, exist_ok=True)
    converged_chg = os.path.join(converged_local_dir, "CHGCAR")
    _materialize_chgcar(chgcar_path, converged_chg)

    hybrid_dir = os.path.join(run_base, "_hybrids")
    os.makedirs(hybrid_dir, exist_ok=True)
    hybrid_chg = os.path.join(hybrid_dir, f"CHGCAR_{VARIANT}")

    # With --grid_pred_dir the total pseudo grid comes from a density model as
    # well, so the complete total density is ML. Built in two stages from the
    # existing builders: first graft the ML total grid onto the converged
    # CHGCAR (keeping converged total aug + converged diff grid/aug), then
    # replace the total augmentation with the ML prediction.
    # --spin_pred_dir rides on the same grid stage: the ML spin grid replaces
    # the converged diff grid (the total grid is ML or converged as above).
    grid_src_chg = converged_chg
    pred_chg = spin_grid = None
    if grid_index is not None:
        pred_name = grid_index.get(mpid)
        if pred_name is None:
            print(f"⚠️ No predicted grid for {mpid} in {args.grid_pred_dir}; "
                  f"skipping structure.")
            runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                              chgcar_path=chgcar_path,
                              status="missing_pred_grid",
                              source_csv=args.CSV_PATH, source_row=int(idx))
            df.at[idx, PERFORMED_COL] = True
            df.to_csv(args.CSV_PATH, index=False)
            sys.exit(3)
        pred_chg = os.path.join(args.grid_pred_dir, pred_name)
    if spin_index is not None:
        spin_grid = spin_index.get(mpid)
        if spin_grid is None:
            print(f"⚠️ No spin grid for {mpid} in {args.SPIN_PRED_DIR}; "
                  f"skipping structure.")
            runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                              chgcar_path=chgcar_path,
                              status="missing_spin_grid",
                              source_csv=args.CSV_PATH, source_row=int(idx))
            df.at[idx, PERFORMED_COL] = True
            df.to_csv(args.CSV_PATH, index=False)
            sys.exit(3)
    if pred_chg is not None or spin_grid is not None:
        grid_src_chg = os.path.join(hybrid_dir, f"CHGCAR_{VARIANT}_gridstage")
        try:
            build_total_pseudo_grid_chgcar(
                grid_src=pred_chg if pred_chg is not None else converged_chg,
                aug_src=converged_chg,
                out_path=grid_src_chg,
                # Under --ispin1 the diff channel is dropped at the next stage
                # anyway; a total-only grid stage also works for references that
                # carry no diff channel at all.
                conv_spin=not args.ispin1,
                renorm=args.renorm,
                spin_grid_src=spin_grid,
            )
        except Exception as e:
            print(f"⚠️ ML grid graft failed for {mpid}: {e!r}. "
                  f"Marked performed; skipping structure.")
            runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                              chgcar_path=chgcar_path,
                              status="failed_grid_build",
                              source_csv=args.CSV_PATH, source_row=int(idx))
            df.at[idx, PERFORMED_COL] = True
            df.to_csv(args.CSV_PATH, index=False)
            sys.exit(3)

    src_desc = "ML" if grid_index is not None else "converged"
    if spin_grid is not None:
        src_desc += ", diff grid=ML spin"
    aug_desc = ("ML total" if total_npz else "converged total") + (
        ", ML diff" if use_mag else "")
    print(f"Building {VARIANT} CHGCAR (grid={src_desc}, aug={aug_desc}) …")
    try:
        build_ml_aug_chgcar(
            grid_src=grid_src_chg,
            total_aug_npz=total_npz,
            magnetic_aug_npz=(None if (args.total_only or args.ispin1)
                              else (mag_npz if use_mag else None)),
            out_path=hybrid_chg,
            total_only=args.total_only or args.ispin1,
        )
    except Exception as e:
        print(f"⚠️ ML-aug CHGCAR build failed for {mpid}: {e!r}. "
              f"Marked performed; skipping structure.")
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_ml_build",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        sys.exit(3)

    variant_dir = os.path.join(run_base, SUBDIR)
    if os.path.exists(variant_dir):
        shutil.rmtree(variant_dir)
    os.makedirs(variant_dir, exist_ok=True)

    relax_magmom = relax_map.get(mpid) if relax_map is not None else None
    magmom_source = ""
    if relax_map is not None:
        magmom_source = "relaxed" if relax_magmom is not None else "default_fallback"
        if relax_magmom is None:
            print(f"ℹ️ No usable relaxed magmoms for {mpid}; keeping MP's own "
                  f"MAGMOM (default_fallback).")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{VARIANT}_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "mpid": mpid,
            "workdir": variant_dir,
            "grid_dims": grid_dims,
            "aug_pred_dir": args.AUG_PRED_DIR,
            "mag_aug_pred_dir": args.MAG_AUG_PRED_DIR,
            "spin_pred_dir": args.SPIN_PRED_DIR,
            "used_magnetic": use_mag,
            "ispin1": bool(args.ispin1),
            "magmom_source": magmom_source,
            "slice_start": int(start),
            "slice_end": int(end),
        },
    )

    print(f"--- Running variant: {VARIANT} (ICHARG={ICHARG}, CHGCAR={hybrid_chg}) ---")
    try:
        r = run_variant(
            mpid=mpid,
            workdir=variant_dir,
            chgcar_path=hybrid_chg,
            vasp_cmd=tuple(args.vasp_cmd),
            ispin1=args.ispin1,
            magmom=args.magmom,
            relax_magmom=relax_magmom,
        )
    except MissingMPTaskDocError as e:
        # MP has no CHGCAR/TaskDoc for this mp-id: close W&B cleanly and
        # skip with code 3 so the SBATCH wrapper continues.
        print(f"⚠️ {e}. Marked performed; skipping structure.")
        wandb.log({f"status_{VARIANT}": "failed_no_taskdoc"})
        wandb.finish()
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_no_taskdoc",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        sys.exit(3)

    wandb.log({
        f"status_{VARIANT}": r["status"],
        f"energy_{VARIANT}": r["energy"],
        f"total_mag_{VARIANT}": r["total_mag"],
        f"spin_{VARIANT}": r["spin"],
        f"scf_steps_dav_{VARIANT}": r["dav"],
        f"scf_steps_rmm_{VARIANT}": r["rmm"],
        f"scf_steps_other_{VARIANT}": r["other"],
        f"scf_steps_total_{VARIANT}": r["total"],
        f"time_{VARIANT}": r["wall"],
        f"magmom_source_{VARIANT}": magmom_source,
    })
    wandb.finish()

    # Every write below lands on the NFS-backed share; a hung lock here once
    # held a node for 19h at 0% CPU after the SCF had already succeeded. If the
    # watchdog fires the manifest row stays 'in_progress', which terminal_set
    # treats as retryable, so the next invocation re-runs this structure cleanly.
    with watchdog(600, f"post-run bookkeeping for {mpid}"):
        mags_jsonl = os.path.splitext(args.RESULTS_CSV)[0] + "_magnetization.jsonl"
        if r["magnetization"] is not None:
            try:
                mag_serialized = [
                    dict(m) if not isinstance(m, dict) else m for m in r["magnetization"]
                ]
            except Exception:
                mag_serialized = str(r["magnetization"])
            record = {
                "CHGCAR_PATH": chgcar_path,
                "MP_ID": mpid,
                "csv_index": int(idx),
                "variant": VARIANT,
                "magnetization": mag_serialized,
            }
            with open(mags_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")

        header = [
            "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
            f"status_{VARIANT}",
            f"energy_{VARIANT}",
            f"total_mag_{VARIANT}",
            f"spin_{VARIANT}",
            f"scf_steps_dav_{VARIANT}",
            f"scf_steps_rmm_{VARIANT}",
            f"scf_steps_other_{VARIANT}",
            f"scf_steps_total_{VARIANT}",
            f"time_{VARIANT}",
            f"magmom_source_{VARIANT}",
        ]
        row_out = [
            chgcar_path, mpid, int(idx), int(start), int(end),
            r["status"],
            "" if r["energy"] is None else r["energy"],
            "" if r["total_mag"] is None else r["total_mag"],
            "" if r["spin"] is None else r["spin"],
            r["dav"], r["rmm"], r["other"], r["total"], r["wall"],
            magmom_source,
        ]
        results.upsert_result(args.RESULTS_CSV, header, row_out)

        # Terminal result recorded (ok or failed): safe to mark performed now. A
        # 'failed' row is retried only via reconcile_csvs.py --clear-status failed.
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)

        runs_db.write_run(
            conn, mp_id=mpid, variant=VARIANT, chgcar_path=chgcar_path, status=r["status"],
            result={
                f"energy_{VARIANT}": r["energy"],
                f"total_mag_{VARIANT}": r["total_mag"],
                f"spin_{VARIANT}": r["spin"],
                f"scf_steps_dav_{VARIANT}": r["dav"],
                f"scf_steps_rmm_{VARIANT}": r["rmm"],
                f"scf_steps_other_{VARIANT}": r["other"],
                f"scf_steps_total_{VARIANT}": r["total"],
                f"time_{VARIANT}": r["wall"],
                f"magmom_source_{VARIANT}": magmom_source,
            },
            source_csv=args.CSV_PATH, source_row=int(idx),
        )

        prune_workdir_keep_outputs(variant_dir)
        shutil.rmtree(converged_local_dir, ignore_errors=True)
        shutil.rmtree(hybrid_dir, ignore_errors=True)

    print("Done.")
    print(f"  {VARIANT}: status={r['status']} energy={r['energy']} "
          f"DAV={r['dav']} RMM={r['rmm']} OTHER={r['other']} "
          f"TOTAL={r['total']} wall={r['wall']:.1f}s")

    if r["energy"] is None:
        sys.exit(3)


if __name__ == "__main__":
    run_guarded(main)
