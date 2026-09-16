#!/usr/bin/env python3
"""GNOME_2 baseline SCF runs: `default` (from scratch) and `true_init` (Oracle).

For one GNOME structure per invocation:
  * `--mode default`   ICHARG=2, no CHGCAR seed — the from-scratch baseline.
  * `--mode true_init`  ICHARG=1 seeded with the converged reference CHGCAR
                        verbatim (both spin channels, converged augmentation) —
                        the Oracle / best-case bound.
  * `--mode true_init_total`  ICHARG=1 seeded with the converged reference
                        CHGCAR stripped to its total grid + total augmentation
                        (no diff channel); the spin init comes from the INCAR
                        MAGMOM (pair with --relaxed_magmoms, e.g. CHGNet).

The two modes differ only in those two lines, which is the point: everything
else — POSCAR/INCAR/KPOINTS/POTCAR — is copied verbatim from the reference run
directory by `sources.gnome2`, so a step-count difference is attributable to the
seed alone. This is the GNOME counterpart of run_default_mp.py and
run_true_init_mp.py, which instead re-derive their inputs from the MP API.

There is no magnetic gate: the task CSV is already restricted to the
non-magnetic, Yb-free population by `exps/make_gnome2_tasks.py`.

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
    build_total_only_chgcar,
    get_chgcar_grid_dims_textparse,
)
from neural_paw_dft.vasp_runner.failsafe import run_guarded, watchdog
from neural_paw_dft.vasp_runner.oszicar import count_scf_breakdown_from_oszicar
from neural_paw_dft.vasp_runner.scf import prune_workdir_keep_outputs
from neural_paw_dft.vasp_runner.sources.gnome2 import (
    MissingGnomeInputsError,
    derive_gid,
    materialize_chgcar,
    write_gnome_inputs_for_gid,
)


WANDB_PROJECT = "VASP_SCF_Baselines_GNOME2"
MODE_ICHARG = {"default": 2, "true_init": 1, "true_init_total": 1}


class SeedBuildError(Exception):
    """Stripping the reference CHGCAR to a total-only seed failed."""


def run_variant(gid, workdir, ref_runs_root, icharg, seed_chgcar, vasp_cmd,
                relax_magmom=None, total_only=False):
    write_gnome_inputs_for_gid(gid, workdir, ref_runs_root)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    # Parallelization tags are node-layout specific; the reference INCARs carry
    # none, but strip them defensively as every MP runner does.
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM"):
        incar.pop(bad, None)
    incar["ICHARG"] = icharg
    incar["ISTART"] = 0
    incar["LCHARG"] = True
    # Optional per-site MAGMOM replacement (--relaxed_magmoms, e.g. CHGNet
    # predictions). Unlike run_ml_seed_gnome.py there is no uniform fallback: a gid without
    # a usable prediction keeps the reference INCAR's own MAGMOM, mirroring
    # the MP runners' default_fallback semantics.
    magmom_source = ""
    if relax_magmom is not None:
        natoms = len(Poscar.from_file(os.path.join(workdir, "POSCAR")).structure)
        if len(relax_magmom) == natoms:
            incar["MAGMOM"] = list(relax_magmom)
            magmom_source = "chgnet"
        else:
            print(f"⚠️ predicted MAGMOM length {len(relax_magmom)} != natoms "
                  f"{natoms} for {gid}; keeping the reference MAGMOM.")
            magmom_source = "default_fallback"
    incar.write_file(incar_path)

    if seed_chgcar is not None:
        dest = os.path.join(workdir, "CHGCAR")
        materialize_chgcar(seed_chgcar, dest)
        if total_only:
            # Strip the converged reference seed to its total grid + total
            # augmentation (no diff channel): the spin init then comes from
            # the INCAR MAGMOM, same construction as run_true_init_mp.py --total_only.
            converged = dest + "_converged"
            os.rename(dest, converged)
            try:
                build_total_only_chgcar(converged, dest, spin="none")
            except Exception as e:
                raise SeedBuildError(f"total-only seed build failed for {gid}: "
                                     f"{e!r}") from e
            finally:
                if os.path.exists(converged):
                    os.remove(converged)

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
        "energy": energy,
        "total_mag": total_mag,
        "magnetization": magnetization,
        "spin": spin,
        "dav": dav,
        "rmm": rmm,
        "other": other_total,
        "total": total,
        "wall": wall,
        "magmom_source": magmom_source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a GNOME_2 baseline SCF (default or true_init) for one structure."
    )
    parser.add_argument("--csv", "-C", dest="CSV_PATH", required=True,
                        help="Task CSV with ['CHGCAR_PATH','GNOME_ID'] "
                             "(CHGCAR_PATH = converged reference CHGCAR).")
    parser.add_argument("--outdir", "-o", dest="BASE_DIR", required=True,
                        help="Base directory for per-structure work dirs.")
    parser.add_argument("--results_csv", "-R", dest="RESULTS_CSV", required=True,
                        help="Narrow-format CSV to append per-structure rows to.")
    parser.add_argument("--mode", choices=tuple(MODE_ICHARG), required=True,
                        help="'default' = ICHARG=2 from scratch; 'true_init' = "
                             "ICHARG=1 seeded with the converged CHGCAR; "
                             "'true_init_total' = ICHARG=1 seeded with the "
                             "converged CHGCAR stripped to total grid + total "
                             "augmentation (no diff channel) — the spin init "
                             "comes from the MAGMOM (use --relaxed_magmoms).")
    parser.add_argument("--ref_runs_root", dest="REF_RUNS_ROOT", required=True,
                        help="Root of the reference runs "
                             "(<root>/<gid>.chgcar/run_default/ supplies the inputs).")
    parser.add_argument("--variant", dest="VARIANT", default=None,
                        help="Variant tag (default: gnome_<mode>).")
    parser.add_argument("--relaxed_magmoms", default=None,
                        help="Per-site MAGMOM CSV (e.g. "
                             "chgnet_preds/chgnet_pred_gnome.csv, keyed by "
                             "GNOME_ID). Gids without a usable prediction "
                             "keep the reference INCAR's own MAGMOM and are "
                             "flagged magmom_source=default_fallback.")
    parser.add_argument("--vasp_cmd", nargs="+", default=["vasp_std"],
                        help="Command used to run VASP (e.g. --vasp_cmd srun vasp_std).")
    parser.add_argument("--runs_db", dest="RUNS_DB", default="runs/runs_gnome.db",
                        help="SQLite manifest used for the skip-check.")
    parser.add_argument("--start_row", type=int, default=0,
                        help="Inclusive start row index in the task CSV slice.")
    parser.add_argument("--end_row", type=int, default=None,
                        help="Exclusive end row index. Defaults to len(df).")
    args = parser.parse_args()

    VARIANT = args.VARIANT or f"gnome_{args.mode}"
    icharg = MODE_ICHARG[args.mode]

    relax_map = None
    if args.relaxed_magmoms:
        from neural_paw_dft.vasp_runner.relaxmag import load_relaxed_magmoms
        relax_map = load_relaxed_magmoms(args.relaxed_magmoms)
        print(f"Per-site magmoms: {len(relax_map)} usable gids from "
              f"{args.relaxed_magmoms} (others keep the reference MAGMOM)")
    PERFORMED_COL = f"performed_{VARIANT}"
    SUBDIR = f"run_{VARIANT}"
    print(f"Variant={VARIANT} (mode={args.mode}, ICHARG={icharg})")

    if not os.path.isdir(args.REF_RUNS_ROOT):
        print(f"❌ Reference runs root not found: {args.REF_RUNS_ROOT}")
        sys.exit(1)

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

    # runs.db is the SOLE skip-truth: hard-overwrite the local performed flag
    # from the manifest's terminal set. terminal_set excludes in_progress so
    # walltime-killed rows stay retryable.
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

    # Scan pending rows for the first runnable structure. A missing reference
    # CHGCAR is TERMINAL (it will never appear) — for `default` the seed is
    # unused, but the row is still dropped so both baselines cover exactly the
    # same population.
    idx = chgcar_path = gid = None
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

        idx = cand
        chgcar_path = cand_chgcar
        gid = cand_gid
        break

    if idx is None:
        print(f"No runnable rows in slice [{start}:{end}).")
        sys.exit(2)

    print(f"Selected idx={idx}, CHGCAR={chgcar_path}, GID={gid}")

    # Manifest in_progress only — do NOT set the local performed flag yet. If the
    # job is killed mid-run this row has no terminal outcome and must stay
    # pending so the next sbatch iteration re-picks it.
    runs_db.write_run(conn, mp_id=gid, variant=VARIANT,
                      chgcar_path=chgcar_path, status="in_progress",
                      source_csv=args.CSV_PATH, source_row=int(idx))

    run_base = os.path.join(args.BASE_DIR, gid)
    workdir = os.path.join(run_base, SUBDIR)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    try:
        grid_dims = get_chgcar_grid_dims_textparse(chgcar_path)
    except Exception:
        grid_dims = None

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{VARIANT}_{gid}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "gnome_id": gid,
            "mode": args.mode,
            "icharg": icharg,
            "workdir": workdir,
            "grid_dims": grid_dims,
            "slice_start": int(start),
            "slice_end": int(end),
        },
    )

    print(f"--- Running variant: {VARIANT} (ICHARG={icharg}, "
          f"seed={chgcar_path if icharg == 1 else 'none'}) ---")
    try:
        r = run_variant(
            gid=gid,
            workdir=workdir,
            ref_runs_root=args.REF_RUNS_ROOT,
            icharg=icharg,
            seed_chgcar=chgcar_path if icharg == 1 else None,
            vasp_cmd=tuple(args.vasp_cmd),
            relax_magmom=relax_map.get(gid) if relax_map is not None else None,
            total_only=(args.mode == "true_init_total"),
        )
    except (MissingGnomeInputsError, SeedBuildError) as e:
        status = ("failed_seed_build" if isinstance(e, SeedBuildError)
                  else "missing_inputs")
        print(f"⚠️ {e}. Marked performed; skipping structure.")
        wandb.log({f"status_{VARIANT}": status})
        wandb.finish()
        runs_db.write_run(conn, mp_id=gid, variant=VARIANT,
                          chgcar_path=chgcar_path, status=status,
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        sys.exit(3)

    # A gid absent from the CSV never reaches run_variant's length check.
    if relax_map is not None and not r["magmom_source"]:
        r["magmom_source"] = "default_fallback"

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

    # Every write below lands on the NFS-backed share; a hung lock here once held
    # a node for 19h at 0% CPU after the SCF had already succeeded. If the
    # watchdog fires the manifest row stays 'in_progress', which terminal_set
    # treats as retryable, so the next invocation re-runs this structure cleanly.
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
            chgcar_path, gid, int(idx), int(start), int(end),
            r["status"],
            "" if r["energy"] is None else r["energy"],
            "" if r["total_mag"] is None else r["total_mag"],
            "" if r["spin"] is None else r["spin"],
            r["dav"], r["rmm"], r["other"], r["total"], r["wall"],
            r["magmom_source"],
        ]
        results.upsert_result(args.RESULTS_CSV, header, row_out, key_col="GNOME_ID")

        # Terminal result recorded (ok or failed): safe to mark performed now.
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)

        runs_db.write_run(
            conn, mp_id=gid, variant=VARIANT, chgcar_path=chgcar_path,
            status=r["status"],
            result={
                f"energy_{VARIANT}": r["energy"],
                f"total_mag_{VARIANT}": r["total_mag"],
                f"spin_{VARIANT}": r["spin"],
                f"scf_steps_dav_{VARIANT}": r["dav"],
                f"scf_steps_rmm_{VARIANT}": r["rmm"],
                f"scf_steps_other_{VARIANT}": r["other"],
                f"scf_steps_total_{VARIANT}": r["total"],
                f"time_{VARIANT}": r["wall"],
                "magmom_source": r["magmom_source"],
            },
            source_csv=args.CSV_PATH, source_row=int(idx),
        )

        prune_workdir_keep_outputs(workdir)

    print("Done.")
    print(f"  {VARIANT}: status={r['status']} energy={r['energy']} "
          f"DAV={r['dav']} RMM={r['rmm']} OTHER={r['other']} "
          f"TOTAL={r['total']} wall={r['wall']:.1f}s")

    if r["energy"] is None:
        sys.exit(3)


if __name__ == "__main__":
    run_guarded(main)
