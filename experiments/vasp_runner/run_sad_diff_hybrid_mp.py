#!/usr/bin/env python3
"""Run a single MP SCF initialized from the `sad_diff_hybrid` CHGCAR.

Hybrid CHGCAR = converged total + SAD diff (built per-row from the row's
converged CHGCAR_PATH and a cached/extracted SAD CHGCAR for the same MP id).
Mirrors `run_default_mp.py` but with ICHARG=1.
"""
import os
import json
import shutil
import time
import argparse
import csv
import subprocess

import pandas as pd
import lz4.frame
import wandb

from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Outcar

from neural_init.vasp_runner import runs_db, results
from neural_init.vasp_runner.failsafe import run_guarded
from neural_init.vasp_runner.scf import prune_workdir_keep_outputs
from neural_init.vasp_runner.chgcar import (
    build_hybrid_chgcar,
    get_chgcar_grid_dims_textparse,
)
from neural_init.vasp_runner.oszicar import count_scf_breakdown_from_oszicar
from neural_init.vasp_runner.sad_extract import extract_sad_chgcar
from neural_init.vasp_runner.sources.mp import (
    derive_mpid,
    write_mp_inputs_for_mpid,
    MissingMPTaskDocError,
)

VARIANT_DEFAULT = "sad_diff_hybrid"


def _materialize_chgcar(src: str, dest: str):
    if src.endswith(".lz4"):
        with lz4.frame.open(src, "rb") as s, open(dest, "wb") as d:
            shutil.copyfileobj(s, d)
    else:
        shutil.copy(src, dest)


def run_vasp_mp(mpid, workdir, icharg, chgcar_path, vasp_cmd):
    os.makedirs(workdir, exist_ok=True)
    write_mp_inputs_for_mpid(mpid, workdir)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM"):
        incar.pop(bad, None)
    incar["ICHARG"] = icharg
    incar["ISTART"] = 0
    incar["LCHARG"] = True
    incar.write_file(incar_path)

    if chgcar_path is not None:
        # Hybrid CHGCAR is already a plain (non-lz4) file we built ourselves.
        shutil.copy(chgcar_path, os.path.join(workdir, "CHGCAR"))

    start = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError:
        wall = time.time() - start
        dav, rmm, total, others = count_scf_breakdown_from_oszicar(
            os.path.join(workdir, "OSZICAR"))
        return None, None, None, None, dav, rmm, total, others, wall

    wall = time.time() - start

    dav, rmm, total, others = count_scf_breakdown_from_oszicar(
        os.path.join(workdir, "OSZICAR"))

    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        return None, None, None, None, dav, rmm, total, others, wall

    energy = total_mag = magnetization = spin = None
    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
        total_mag = getattr(outcar, "total_mag", None)
        magnetization = getattr(outcar, "magnetization", None)
        spin = getattr(outcar, "spin", None)
    except Exception:
        pass

    return energy, total_mag, magnetization, spin, dav, rmm, total, others, wall


def main():
    parser = argparse.ArgumentParser(
        description="Run MP SCF with ICHARG=1 + sad_diff_hybrid CHGCAR."
    )
    parser.add_argument("--csv", "-C", dest="CSV_PATH", required=True)
    parser.add_argument("--outdir", "-o", dest="BASE_DIR", required=True)
    parser.add_argument("--results_csv", "-R", dest="RESULTS_CSV", required=True)
    parser.add_argument("--sad_cache", dest="SAD_CACHE", required=True,
                        help="Directory holding cached SAD CHGCARs (one per MP id).")
    parser.add_argument("--vasp_cmd", nargs="+", default=["vasp_std"])
    parser.add_argument("--variant", dest="VARIANT", default=VARIANT_DEFAULT,
                        help=f"Variant tag "
                             f"(default: {VARIANT_DEFAULT}). Use a suffixed tag "
                             f"for a versioned rerun.")
    parser.add_argument("--runs_db", dest="RUNS_DB", default="runs/runs.db")
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--end_row", type=int, default=None)
    args = parser.parse_args()
    VARIANT = args.VARIANT
    PERFORMED_COL = f"performed_{VARIANT}"
    SUBDIR = f"run_{VARIANT}"
    print(f"Variant={VARIANT}")

    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str, "MP_ID": str})
    n_rows = len(df)
    start = max(args.start_row, 0)
    end = n_rows if args.end_row is None else min(args.end_row, n_rows)
    if start >= end:
        print(f"❌ Invalid slice [{start}:{end}) for CSV with {n_rows} rows.")
        raise SystemExit(1)
    print(f"Using slice [{start}:{end}) out of {n_rows}")

    performed_col = f"performed_{VARIANT}"
    if performed_col not in df.columns:
        df[performed_col] = False

    df_slice = df.iloc[start:end]

    # Manifest is the source of truth: overwrite slice CSV's performed flag
    # from runs.db. See run_default_mp.py for rationale.
    conn = None
    try:
        conn = runs_db.get_connection(args.RUNS_DB)
        done = runs_db.terminal_set(conn, VARIANT, df_slice["MP_ID"].astype(str))
        # runs.db is the SOLE skip-truth: hard-overwrite the local performed
        # flag from the manifest's terminal set. A dropped runs.db write (NFS
        # lock contention) now causes at worst a cheap idempotent re-run of one
        # structure, never a permanent skip. terminal_set excludes in_progress
        # so walltime-killed rows stay retryable.
        df.loc[df_slice.index, performed_col] = df_slice["MP_ID"].astype(str).isin(done)
        df_slice = df.iloc[start:end]
    except Exception as e:
        print(f"⚠️ Could not read runs manifest {args.RUNS_DB}: {e!r}. "
              f"Falling back to slice CSV performed flags.")

    pending = df_slice[df_slice[performed_col] == False]
    if pending.empty:
        print(f"No pending rows in slice [{start}:{end}).")
        raise SystemExit(2)

    idx = pending.index[0]
    chgcar_path = df.at[idx, "CHGCAR_PATH"]
    row = df.loc[idx]
    mpid = derive_mpid(row, chgcar_path)
    print(f"Selected idx={idx}, CHGCAR={chgcar_path}, MPID={mpid}")

    if not os.path.isfile(chgcar_path):
        print(f"⚠️ CHGCAR file not found: {chgcar_path}. Marking as performed and skipping.")
        df.at[idx, performed_col] = True
        df.to_csv(args.CSV_PATH, index=False)
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="missing_chgcar",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        raise SystemExit(3)

    # Manifest in_progress only — do NOT set the local performed flag yet. If
    # the job is killed mid-run, this row has no terminal outcome and must stay
    # pending so the next sbatch iteration re-picks it. performed_<variant> is
    # flipped True only after a terminal result is written below.
    runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                      chgcar_path=chgcar_path, status="in_progress",
                      source_csv=args.CSV_PATH, source_row=int(idx))

    base = os.path.splitext(os.path.basename(chgcar_path))[0]
    run_base = os.path.join(args.BASE_DIR, base)
    workdir = os.path.join(run_base, f"run_{VARIANT}")
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    try:
        grid_dims = get_chgcar_grid_dims_textparse(chgcar_path)
    except Exception:
        grid_dims = None

    # Decompress converged CHGCAR locally so pymatgen can read it for the hybrid.
    converged_local_dir = os.path.join(run_base, "_converged")
    os.makedirs(converged_local_dir, exist_ok=True)
    converged_chg = os.path.join(converged_local_dir, "CHGCAR")
    _materialize_chgcar(chgcar_path, converged_chg)

    # Extract (cached) SAD CHGCAR + build the hybrid.
    print(f"Extracting SAD CHGCAR for {mpid} (cache={args.SAD_CACHE}) …")
    try:
        sad_chg = extract_sad_chgcar(
            mpid=mpid,
            sad_cache_dir=args.SAD_CACHE,
            vasp_cmd=tuple(args.vasp_cmd),
        )
    except MissingMPTaskDocError as e:
        print(f"⚠️ {e}. Marked performed; skipping structure.")
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_no_taskdoc",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, performed_col] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    hybrid_dir = os.path.join(run_base, "_hybrids")
    os.makedirs(hybrid_dir, exist_ok=True)
    hybrid_chg = os.path.join(hybrid_dir, f"CHGCAR_{VARIANT}")
    print(f"Building {VARIANT} CHGCAR (total=converged, diff=SAD) …")
    build_hybrid_chgcar(
        total_src=converged_chg,
        diff_src=sad_chg,
        out_path=hybrid_chg,
    )

    wandb.init(
        project="VASP_SCF_SadDiffHybrid_MP",
        name=f"{VARIANT}_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "mpid": mpid,
            "workdir": workdir,
            "grid_dims": grid_dims,
            "slice_start": int(start),
            "slice_end": int(end),
        },
    )

    try:
        energy, total_mag, magnetization, spin, dav, rmm, total, others, wall = run_vasp_mp(
            mpid=mpid,
            workdir=workdir,
            icharg=1,
            chgcar_path=hybrid_chg,
            vasp_cmd=tuple(args.vasp_cmd),
        )
    except MissingMPTaskDocError as e:
        print(f"⚠️ {e}. Marked performed; skipping structure.")
        wandb.log({"status": "failed_no_taskdoc"})
        wandb.finish()
        header = [
            "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
            f"status_{VARIANT}", f"energy_{VARIANT}", f"total_mag_{VARIANT}",
            f"spin_{VARIANT}",
            f"scf_steps_dav_{VARIANT}", f"scf_steps_rmm_{VARIANT}",
            f"scf_steps_other_{VARIANT}", f"scf_steps_total_{VARIANT}",
            f"time_{VARIANT}",
        ]
        row_out = [chgcar_path, mpid, int(idx), int(start), int(end),
                   "failed_no_taskdoc", "", "", "", 0, 0, 0, 0, 0.0]
        results.upsert_result(args.RESULTS_CSV, header, row_out)
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_no_taskdoc",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        df.at[idx, performed_col] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    other_total = sum(others.values()) if others else 0
    status = "ok" if energy is not None else "failed"

    mags_jsonl = os.path.splitext(args.RESULTS_CSV)[0] + "_magnetization.jsonl"
    if magnetization is not None:
        try:
            mag_serialized = [
                dict(m) if not isinstance(m, dict) else m for m in magnetization
            ]
        except Exception:
            mag_serialized = str(magnetization)
        record = {
            "CHGCAR_PATH": chgcar_path,
            "MP_ID": mpid,
            "csv_index": int(idx),
            "magnetization": mag_serialized,
        }
        with open(mags_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")

    wandb.log({
        "status": status,
        f"energy_{VARIANT}": energy,
        f"total_mag_{VARIANT}": total_mag,
        f"spin_{VARIANT}": spin,
        f"scf_steps_dav_{VARIANT}": dav,
        f"scf_steps_rmm_{VARIANT}": rmm,
        f"scf_steps_other_{VARIANT}": other_total,
        f"scf_steps_total_{VARIANT}": total,
        f"time_{VARIANT}": wall,
    })
    wandb.finish()

    header = [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        f"status_{VARIANT}", f"energy_{VARIANT}", f"total_mag_{VARIANT}",
        f"spin_{VARIANT}",
        f"scf_steps_dav_{VARIANT}", f"scf_steps_rmm_{VARIANT}",
        f"scf_steps_other_{VARIANT}", f"scf_steps_total_{VARIANT}",
        f"time_{VARIANT}",
    ]
    row_out = [
        chgcar_path, mpid, int(idx), int(start), int(end),
        status,
        "" if energy is None else energy,
        "" if total_mag is None else total_mag,
        "" if spin is None else spin,
        dav, rmm, other_total, total, wall,
    ]
    results.upsert_result(args.RESULTS_CSV, header, row_out)

    # Terminal result recorded (ok or failed): safe to mark performed now. A
    # 'failed' row is retried only via reconcile_csvs.py --clear-status failed.
    df.at[idx, performed_col] = True
    df.to_csv(args.CSV_PATH, index=False)

    runs_db.write_run(
        conn, mp_id=mpid, variant=VARIANT, chgcar_path=chgcar_path, status=status,
        result={
            f"energy_{VARIANT}": energy,
            f"total_mag_{VARIANT}": total_mag,
            f"spin_{VARIANT}": spin,
            f"scf_steps_dav_{VARIANT}": dav,
            f"scf_steps_rmm_{VARIANT}": rmm,
            f"scf_steps_other_{VARIANT}": other_total,
            f"scf_steps_total_{VARIANT}": total,
            f"time_{VARIANT}": wall,
        },
        source_csv=args.CSV_PATH, source_row=int(idx),
    )

    prune_workdir_keep_outputs(workdir)
    shutil.rmtree(converged_local_dir, ignore_errors=True)
    shutil.rmtree(hybrid_dir, ignore_errors=True)

    print("Done.")
    print(f"Energy: {energy}")
    print(f"Total mag: {total_mag}")
    print(f"DAV: {dav}, RMM: {rmm}, OTHER: {other_total}, TOTAL: {total}")
    print(f"Time: {wall:.2f}s")

    if energy is None:
        raise SystemExit(3)


if __name__ == "__main__":
    run_guarded(main)
