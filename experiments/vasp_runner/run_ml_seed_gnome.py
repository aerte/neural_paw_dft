#!/usr/bin/env python3
"""GNOME_2 fully-ML seed SCF run: ML total grid + ML augmentation + uniform MAGMOM.

For one GNOME structure per invocation, seeds VASP with a strictly total-only
CHGCAR in which *both* halves of the charge density are predictions:

  * the ``total`` pseudo density grid from a predicted CHGCAR in ``--pred_dir``
    (ELECTRAFI-ref or Charge3Net), and
  * the PAW augmentation occupancies from AugNet's ``{gid}_total_aug.npz``.

There is no ``diff`` grid and no ``diff`` augmentation, so under ISPIN=2 VASP
builds the initial magnetization from the INCAR MAGMOM — set here to a small
uniform value on every ion (``--magmom``, default 0.001). On MP that recipe was
worth ~96% of the achievable SCF saving on non-magnetic structures, beating every
trained augmentation model that instead guessed a spin channel.

Nothing in the seed comes from the converged reference except the per-atom
augmentation block lengths (a POTCAR property, used to validate the npz) and the
per-ion moment line the CHGCAR format requires.

Both prediction directory layouts are supported:
  flat    {pred_dir}/{gid}_{formula}_{err}_test_{i}.CHGCAR   (ELECTRAFI-ref)
  nested  {pred_dir}/{gid}/CHGCAR                            (Charge3Net)
Duplicate flat entries for one gid resolve to the file with the lowest embedded
error float.

Exit codes (match the SBATCH wrapper expectations):
  0 = OK              2 = slice exhausted     3 = per-structure failure
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import pandas as pd
import wandb

from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Outcar

from neural_paw_dft.vasp_runner import runs_db, results
from neural_paw_dft.vasp_runner.chgcar import (
    build_ml_aug_chgcar,
    build_ml_grid_ml_aug_chgcar,
    build_total_pseudo_grid_chgcar,
    get_chgcar_grid_dims_textparse,
)
from neural_paw_dft.vasp_runner.failsafe import run_guarded, watchdog
from neural_paw_dft.vasp_runner.spin_npy import index_spin_dir
from neural_paw_dft.vasp_runner.oszicar import count_scf_breakdown_from_oszicar
from neural_paw_dft.vasp_runner.scf import prune_workdir_keep_outputs
from neural_paw_dft.vasp_runner.sources.gnome2 import (
    GID_RE,
    MissingGnomeInputsError,
    derive_gid,
    materialize_chgcar,
    write_gnome_inputs_for_gid,
)


ICHARG = 1
WANDB_PROJECT = "VASP_SCF_MlSeed_GNOME2"
AUG_NPZ = "{gid}_total_aug.npz"
# AugNet spin export naming (diff augmentation).
MAG_AUG_NPZ = "{gid}_mag_aug.npz"


def _pred_error(name: str) -> float:
    """Parse the error float embedded before `_test_` in a flat predicted filename.

    Returns +inf on any parse failure so such files sort last (never preferred).
    """
    try:
        head = name.split("_test_")[0]
        return float(head.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return float("inf")


def index_pred_dir(pred_dir: str) -> dict:
    """Map gid -> predicted CHGCAR path, covering both layouts.

    Nested (`{gid}/CHGCAR`) wins over flat if a directory somehow holds both,
    since it is unambiguous.
    """
    flat: dict[str, list[str]] = {}
    nested: dict[str, str] = {}
    for name in os.listdir(pred_dir):
        m = GID_RE.match(name)
        if not m:
            continue
        gid = m.group(1)
        path = os.path.join(pred_dir, name)
        if os.path.isdir(path):
            inner = os.path.join(path, "CHGCAR")
            if os.path.isfile(inner):
                nested[gid] = inner
        elif name.endswith(".CHGCAR"):
            flat.setdefault(gid, []).append(name)

    out = {gid: os.path.join(pred_dir, min(files, key=_pred_error))
           for gid, files in flat.items()}
    out.update(nested)
    return out


def run_variant(gid, workdir, ref_runs_root, seed_chgcar, magmom, vasp_cmd,
                relax_magmom=None, conv_spin=False):
    write_gnome_inputs_for_gid(gid, workdir, ref_runs_root)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM"):
        incar.pop(bad, None)
    incar["ICHARG"] = ICHARG
    incar["ISTART"] = 0
    incar["LCHARG"] = True
    # The seed carries no diff channel, so this is what VASP initializes the
    # magnetization from — replacing the reference's element-dependent MAGMOM
    # (0.6 per ion, 5.0 for transition metals) with either the predicted
    # per-site moments (--relaxed_magmoms) or a small uniform moment.
    natoms = len(Poscar.from_file(os.path.join(workdir, "POSCAR")).structure)
    if conv_spin:
        # The seed carries the converged diff grid + diff augmentation, so
        # ICHARG=1 never consults MAGMOM; leave the reference INCAR's own
        # value in place rather than pretending a spin guess was made.
        magmom_source = "conv_spin"
    elif relax_magmom is not None and len(relax_magmom) == natoms:
        incar["MAGMOM"] = list(relax_magmom)
        magmom_source = "chgnet"
    else:
        if relax_magmom is not None:
            print(f"⚠️ predicted MAGMOM length {len(relax_magmom)} != natoms "
                  f"{natoms} for {gid}; falling back to uniform {magmom}.")
        incar["MAGMOM"] = [magmom] * natoms
        magmom_source = "uniform_fallback"
    incar.write_file(incar_path)

    shutil.copy(seed_chgcar, os.path.join(workdir, "CHGCAR"))

    start = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
        vasp_ok = True
    except subprocess.CalledProcessError:
        vasp_ok = False
    wall = time.time() - start

    dav, rmm, total, other_counts = count_scf_breakdown_from_oszicar(
        os.path.join(workdir, "OSZICAR"))
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
        "magmom_source": magmom_source,
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
        description="Run the ML-grid + ML-aug + MAGMOM seed SCF for one GNOME structure."
    )
    parser.add_argument("--csv", "-C", dest="CSV_PATH", required=True,
                        help="Task CSV with ['CHGCAR_PATH','GNOME_ID'] "
                             "(CHGCAR_PATH = converged reference CHGCAR).")
    parser.add_argument("--outdir", "-o", dest="BASE_DIR", required=True,
                        help="Base directory for per-structure work dirs.")
    parser.add_argument("--results_csv", "-R", dest="RESULTS_CSV", required=True,
                        help="Narrow-format CSV to append per-structure rows to.")
    parser.add_argument("--pred_dir", dest="PRED_DIR", default=None,
                        help="Directory of predicted total-grid CHGCARs "
                             "(flat {gid}_..._test_{i}.CHGCAR or nested "
                             "{gid}/CHGCAR). Omit under --conv_spin to keep the "
                             "CONVERGED total grid (augmentation-only seed).")
    parser.add_argument("--aug_pred_dir", dest="AUG_PRED_DIR", default=None,
                        help="Directory holding AugNet's {gid}_total_aug.npz. "
                             "Omit under --conv_spin to keep the CONVERGED total "
                             "augmentation (grid-only seed).")
    parser.add_argument("--ref_runs_root", dest="REF_RUNS_ROOT", required=True,
                        help="Root of the reference runs "
                             "(<root>/<gid>.chgcar/run_default/ supplies the inputs).")
    parser.add_argument("--variant", dest="VARIANT", required=True,
                        help="Variant tag, e.g. gnome_mlaug_electrafi_magmom0p001.")
    parser.add_argument("--spin_pred_dir", dest="SPIN_PRED_DIR", default=None,
                        help="Directory of ML spin-density grids "
                             "(<gid>_spin.npy.lz4, e/A^3; volume-scaled on "
                             "load) replacing the converged diff GRID. "
                             "Requires --conv_spin.")
    parser.add_argument("--mag_aug_pred_dir", dest="MAG_AUG_PRED_DIR", default=None,
                        help="Directory of ML DIFF-augmentation predictions "
                             "({gid}_mag_aug.npz, the AugNet spin export) "
                             "replacing the converged diff augmentation. "
                             "Requires --conv_spin.")
    parser.add_argument("--conv_spin", action="store_true",
                        help="Keep the CONVERGED spin channel: the diff grid "
                             "and diff augmentation come verbatim from the "
                             "reference CHGCAR, so the seed is ML in the total "
                             "grid and total augmentation only and MAGMOM is "
                             "never consulted. Mutually exclusive with "
                             "--relaxed_magmoms.")
    parser.add_argument("--magmom", type=float, default=0.001,
                        help="Uniform MAGMOM on every ion (default: 0.001). "
                             "Ignored under --conv_spin.")
    parser.add_argument("--relaxed_magmoms", default=None,
                        help="CSV of per-site predicted MAGMOMs (e.g. "
                             "chgnet_preds/chgnet_pred_gnome.csv, keyed by "
                             "GNOME_ID). Gids missing from the CSV fall back "
                             "to the uniform --magmom and are flagged "
                             "magmom_source=uniform_fallback.")
    parser.add_argument("--renorm", action="store_true",
                        help="Scale the predicted total grid so its integrated "
                             "charge matches the converged reference CHGCAR's, i.e. "
                             "NELECT. Off by default (every run so far used the "
                             "prediction verbatim). The predicted augmentation "
                             "blocks are never rescaled.")
    parser.add_argument("--vasp_cmd", nargs="+", default=["vasp_std"],
                        help="Command used to run VASP (e.g. --vasp_cmd srun vasp_std).")
    parser.add_argument("--runs_db", dest="RUNS_DB", default="runs/runs_gnome.db",
                        help="SQLite manifest used for the skip-check.")
    parser.add_argument("--start_row", type=int, default=0,
                        help="Inclusive start row index in the task CSV slice.")
    parser.add_argument("--end_row", type=int, default=None,
                        help="Exclusive end row index. Defaults to len(df).")
    args = parser.parse_args()

    if not args.PRED_DIR and not args.AUG_PRED_DIR:
        print("FATAL: give --pred_dir, --aug_pred_dir, or both — a seed with "
              "neither is just the converged reference (that is true_init).",
              file=sys.stderr)
        sys.exit(2)
    if (not args.PRED_DIR or not args.AUG_PRED_DIR) and not args.conv_spin:
        print("FATAL: omitting --pred_dir or --aug_pred_dir requires "
              "--conv_spin; the partial seeds are only defined against a "
              "converged remainder.", file=sys.stderr)
        sys.exit(2)
    if (args.SPIN_PRED_DIR or args.MAG_AUG_PRED_DIR) and not args.conv_spin:
        print("FATAL: --spin_pred_dir/--mag_aug_pred_dir require --conv_spin "
              "(the seed then carries a full diff channel and the INCAR MAGMOM "
              "is left untouched).")
        sys.exit(1)
    if args.conv_spin and args.relaxed_magmoms:
        print("FATAL: --conv_spin and --relaxed_magmoms are mutually exclusive "
              "(a converged spin channel makes MAGMOM irrelevant).",
              file=sys.stderr)
        sys.exit(2)

    VARIANT = args.VARIANT
    PERFORMED_COL = f"performed_{VARIANT}"
    SUBDIR = f"run_{VARIANT}"
    print(f"Variant={VARIANT} (pred_dir={args.PRED_DIR}, "
          f"aug_pred_dir={args.AUG_PRED_DIR}, magmom={args.magmom})")

    for label, path in (("Prediction dir", args.PRED_DIR),
                        ("Augmentation prediction dir", args.AUG_PRED_DIR),
                        ("Mag-augmentation prediction dir", args.MAG_AUG_PRED_DIR),
                        ("Reference runs root", args.REF_RUNS_ROOT)):
        if path is None:
            continue
        if not os.path.isdir(path):
            print(f"❌ {label} not found: {path}")
            sys.exit(1)

    pred_index = {}
    if args.PRED_DIR:
        pred_index = index_pred_dir(args.PRED_DIR)
        print(f"Indexed {len(pred_index)} unique gids in {args.PRED_DIR}")
    else:
        print("No --pred_dir: the total pseudo grid stays CONVERGED.")
    if not args.AUG_PRED_DIR:
        print("No --aug_pred_dir: the total augmentation stays CONVERGED.")
    spin_index = None
    if args.SPIN_PRED_DIR:
        spin_index = index_spin_dir(args.SPIN_PRED_DIR)
        print(f"Indexed {len(spin_index)} spin grids in {args.SPIN_PRED_DIR}")

    relax_map = None
    if args.relaxed_magmoms:
        from neural_paw_dft.vasp_runner.relaxmag import load_relaxed_magmoms
        relax_map = load_relaxed_magmoms(args.relaxed_magmoms)
        print(f"Loaded per-site MAGMOM guesses for {len(relax_map)} gids from "
              f"{args.relaxed_magmoms} (others fall back to uniform "
              f"{args.magmom})")

    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str, "GNOME_ID": str})
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

    conn = None
    try:
        conn = runs_db.get_connection(args.RUNS_DB)
        done = runs_db.terminal_set(conn, VARIANT, df_slice["GNOME_ID"].astype(str))
        df.loc[df_slice.index, PERFORMED_COL] = df_slice["GNOME_ID"].astype(str).isin(done)
        df_slice = df.iloc[start:end]
    except Exception as e:
        print(f"⚠️ Could not read runs manifest {args.RUNS_DB}: {e!r}. "
              f"Falling back to slice CSV performed flags.")

    pending = df_slice[df_slice[PERFORMED_COL] == False]
    if pending.empty:
        print(f"No pending rows in slice [{start}:{end}).")
        sys.exit(2)

    # Scan pending rows for the first runnable structure.
    #   - missing reference CHGCAR   -> TERMINAL: missing_chgcar
    #   - missing prediction (grid or aug) -> NON-TERMINAL: leave pending, scan past
    #     so the row is re-picked once the prediction export lands.
    idx = chgcar_path = gid = pred_chg = total_npz = spin_grid = mag_npz = None
    for cand in pending.index:
        cand_chgcar = df.at[cand, "CHGCAR_PATH"]
        cand_gid = derive_gid(df.loc[cand], cand_chgcar)

        if not os.path.isfile(cand_chgcar):
            print(f"⚠️ Reference CHGCAR not found for {cand_gid}: {cand_chgcar}. "
                  f"Marking performed and skipping.")
            df.at[cand, PERFORMED_COL] = True
            df.to_csv(args.CSV_PATH, index=False)
            runs_db.write_run(conn, mp_id=cand_gid, variant=VARIANT,
                              chgcar_path=cand_chgcar, status="missing_chgcar",
                              source_csv=args.CSV_PATH, source_row=int(cand))
            continue

        cand_pred = None
        if args.PRED_DIR:
            cand_pred = pred_index.get(cand_gid)
            if cand_pred is None:
                print(f"⚠️ No predicted grid for {cand_gid} in {args.PRED_DIR}. "
                      f"Leaving row PENDING (retryable) and scanning past it.")
                continue

        cand_spin = None
        if spin_index is not None:
            cand_spin = spin_index.get(cand_gid)
            if cand_spin is None:
                print(f"⚠️ No spin grid for {cand_gid} in {args.SPIN_PRED_DIR}. "
                      f"Leaving row PENDING (retryable) and scanning past it.")
                continue

        cand_mag = None
        if args.MAG_AUG_PRED_DIR:
            cand_mag = os.path.join(args.MAG_AUG_PRED_DIR,
                                    MAG_AUG_NPZ.format(gid=cand_gid))
            if not os.path.isfile(cand_mag):
                print(f"⚠️ No mag-aug prediction for {cand_gid}: {cand_mag}. "
                      f"Leaving row PENDING (retryable) and scanning past it.")
                continue

        cand_npz = None
        if args.AUG_PRED_DIR:
            cand_npz = os.path.join(args.AUG_PRED_DIR,
                                    AUG_NPZ.format(gid=cand_gid))
            if not os.path.isfile(cand_npz):
                print(f"⚠️ No augmentation prediction for {cand_gid}: "
                      f"{cand_npz}. Leaving row PENDING (retryable) and "
                      f"scanning past it.")
                continue

        idx = cand
        chgcar_path = cand_chgcar
        gid = cand_gid
        pred_chg = cand_pred
        total_npz = cand_npz
        spin_grid = cand_spin
        mag_npz = cand_mag
        break

    if idx is None:
        print(f"No runnable rows in slice [{start}:{end}) — all pending rows are "
              f"skipped or lack a prediction. Leaving missing-prediction rows "
              f"pending for a later submission.")
        sys.exit(2)

    print(f"Selected idx={idx}, ref_CHGCAR={chgcar_path}, GID={gid}")
    print(f"Predicted grid: {pred_chg or '(converged)'}")
    print(f"Predicted augmentation: {total_npz or '(converged)'}")
    print(f"Predicted spin grid: {spin_grid or '(converged)'}  "
          f"spin aug: {mag_npz or '(converged)'}")

    runs_db.write_run(conn, mp_id=gid, variant=VARIANT,
                      chgcar_path=chgcar_path, status="in_progress",
                      source_csv=args.CSV_PATH, source_row=int(idx))

    run_base = os.path.join(args.BASE_DIR, gid)
    os.makedirs(run_base, exist_ok=True)

    try:
        grid_dims = get_chgcar_grid_dims_textparse(chgcar_path)
    except Exception:
        grid_dims = None

    # The reference is an lz4-compressed tar; pymatgen needs it unwrapped on disk.
    converged_local_dir = os.path.join(run_base, "_converged")
    os.makedirs(converged_local_dir, exist_ok=True)
    converged_chg = os.path.join(converged_local_dir, "CHGCAR")
    materialize_chgcar(chgcar_path, converged_chg)

    seed_dir = os.path.join(run_base, "_seed")
    os.makedirs(seed_dir, exist_ok=True)
    seed_chg = os.path.join(seed_dir, f"CHGCAR_{VARIANT}")

    spin_note = ("spin=converged" if args.conv_spin
                 and not (spin_grid or mag_npz) else
                 f"spin grid={'ML' if spin_grid else 'conv'}/"
                 f"spin aug={'ML' if mag_npz else 'conv'}"
                 if args.conv_spin else "no diff channel")
    print(f"Building {VARIANT} CHGCAR "
          f"(grid={'ML total pseudo' if pred_chg else 'converged'}, "
          f"aug={'ML' if total_npz else 'converged'}, {spin_note}) …")
    try:
        if args.conv_spin:
            # Two-stage composition mirroring run_ml_seed_mp.py.
            # Stage 1 (grid channels): ML total grid and/or ML spin grid onto
            # the converged reference (skipped when both stay converged).
            # Stage 2 (aug blocks): swap in the ML total and/or diff
            # augmentation (skipped when both stay converged).
            grid_stage_src = converged_chg
            if pred_chg or spin_grid:
                grid_stage_src = os.path.join(
                    seed_dir, f"CHGCAR_{VARIANT}_gridstage"
                ) if (total_npz or mag_npz) else seed_chg
                build_total_pseudo_grid_chgcar(
                    grid_src=pred_chg if pred_chg else converged_chg,
                    aug_src=converged_chg,
                    out_path=grid_stage_src,
                    conv_spin=True,
                    renorm=args.renorm,
                    spin_grid_src=spin_grid,
                )
            if total_npz or mag_npz:
                build_ml_aug_chgcar(
                    grid_src=grid_stage_src,
                    total_aug_npz=total_npz,
                    magnetic_aug_npz=mag_npz,
                    out_path=seed_chg,
                )
        else:
            build_ml_grid_ml_aug_chgcar(
                grid_src=pred_chg,
                aug_ref_src=converged_chg,
                total_aug_npz=total_npz,
                out_path=seed_chg,
                renorm=args.renorm,
            )
    except Exception as e:
        print(f"⚠️ ML seed CHGCAR build failed for {gid}: {e!r}. "
              f"Marked performed; skipping structure.")
        runs_db.write_run(conn, mp_id=gid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_ml_build",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        shutil.rmtree(converged_local_dir, ignore_errors=True)
        shutil.rmtree(seed_dir, ignore_errors=True)
        sys.exit(3)

    workdir = os.path.join(run_base, SUBDIR)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{VARIANT}_{gid}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "gnome_id": gid,
            "workdir": workdir,
            "grid_dims": grid_dims,
            "pred_dir": args.PRED_DIR,
            "pred_chgcar": pred_chg,
            "aug_pred_dir": args.AUG_PRED_DIR,
            "spin_pred_dir": args.SPIN_PRED_DIR,
            "mag_aug_pred_dir": args.MAG_AUG_PRED_DIR,
            "magmom": args.magmom,
            "relaxed_magmoms": args.relaxed_magmoms,
            "slice_start": int(start),
            "slice_end": int(end),
        },
    )

    print(f"--- Running variant: {VARIANT} (ICHARG={ICHARG}, CHGCAR={seed_chg}) ---")
    try:
        r = run_variant(
            gid=gid,
            workdir=workdir,
            ref_runs_root=args.REF_RUNS_ROOT,
            seed_chgcar=seed_chg,
            magmom=args.magmom,
            conv_spin=args.conv_spin,
            vasp_cmd=tuple(args.vasp_cmd),
            relax_magmom=relax_map.get(gid) if relax_map is not None else None,
        )
    except MissingGnomeInputsError as e:
        print(f"⚠️ {e}. Marked performed; skipping structure.")
        wandb.log({f"status_{VARIANT}": "missing_inputs"})
        wandb.finish()
        runs_db.write_run(conn, mp_id=gid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="missing_inputs",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        shutil.rmtree(converged_local_dir, ignore_errors=True)
        shutil.rmtree(seed_dir, ignore_errors=True)
        sys.exit(3)

    wandb.log({
        f"status_{VARIANT}": r["status"],
        f"magmom_source_{VARIANT}": r["magmom_source"],
        f"energy_{VARIANT}": r["energy"],
        f"total_mag_{VARIANT}": r["total_mag"],
        f"spin_{VARIANT}": r["spin"],
        f"scf_steps_dav_{VARIANT}": r["dav"],
        f"scf_steps_rmm_{VARIANT}": r["rmm"],
        f"scf_steps_other_{VARIANT}": r["other"],
        f"scf_steps_total_{VARIANT}": r["total"],
        f"time_{VARIANT}": r["wall"],
    })
    wandb.finish()

    with watchdog(600, f"post-run bookkeeping for {gid}"):
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
                "MP_ID": gid,
                "csv_index": int(idx),
                "variant": VARIANT,
                "magnetization": mag_serialized,
            }
            with open(mags_jsonl, "a") as f:
                f.write(json.dumps(record) + "\n")

        header = [
            "CHGCAR_PATH", "GNOME_ID", "csv_index", "slice_start", "slice_end",
            f"status_{VARIANT}",
            f"magmom_source_{VARIANT}",
            f"energy_{VARIANT}",
            f"total_mag_{VARIANT}",
            f"spin_{VARIANT}",
            f"scf_steps_dav_{VARIANT}",
            f"scf_steps_rmm_{VARIANT}",
            f"scf_steps_other_{VARIANT}",
            f"scf_steps_total_{VARIANT}",
            f"time_{VARIANT}",
        ]
        row_out = [
            chgcar_path, gid, int(idx), int(start), int(end),
            r["status"],
            r["magmom_source"],
            "" if r["energy"] is None else r["energy"],
            "" if r["total_mag"] is None else r["total_mag"],
            "" if r["spin"] is None else r["spin"],
            r["dav"], r["rmm"], r["other"], r["total"], r["wall"],
        ]
        results.upsert_result(args.RESULTS_CSV, header, row_out, key_col="GNOME_ID")

        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)

        runs_db.write_run(
            conn, mp_id=gid, variant=VARIANT, chgcar_path=chgcar_path,
            status=r["status"],
            result={
                f"magmom_source_{VARIANT}": r["magmom_source"],
                f"energy_{VARIANT}": r["energy"],
                f"total_mag_{VARIANT}": r["total_mag"],
                f"spin_{VARIANT}": r["spin"],
                f"scf_steps_dav_{VARIANT}": r["dav"],
                f"scf_steps_rmm_{VARIANT}": r["rmm"],
                f"scf_steps_other_{VARIANT}": r["other"],
                f"scf_steps_total_{VARIANT}": r["total"],
                f"time_{VARIANT}": r["wall"],
            },
            source_csv=args.CSV_PATH, source_row=int(idx),
        )

        prune_workdir_keep_outputs(workdir)
        shutil.rmtree(converged_local_dir, ignore_errors=True)
        shutil.rmtree(seed_dir, ignore_errors=True)

    print("Done.")
    print(f"  {VARIANT}: status={r['status']} energy={r['energy']} "
          f"DAV={r['dav']} RMM={r['rmm']} OTHER={r['other']} "
          f"TOTAL={r['total']} wall={r['wall']:.1f}s")

    if r["energy"] is None:
        sys.exit(3)


if __name__ == "__main__":
    run_guarded(main)
