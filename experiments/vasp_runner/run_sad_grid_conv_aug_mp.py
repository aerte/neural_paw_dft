#!/usr/bin/env python3
"""Grid-vs-augmentation decomposition — Experiment B.

For one MP structure per invocation, seeds VASP with:
  - pseudo density grid (total + diff) from the SAD CHGCAR, and
  - augmentation occupancies (total + diff data_aug) from the CONVERGED CHGCAR.

Pairs with run_conv_grid_sad_aug_mp.py (Experiment A, swapped).

Exit codes (match runs/mp_sad_grid_conv_aug.sh expectations):
  0 = OK              2 = slice exhausted     3 = per-structure VASP failure
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time

import lz4.frame
import pandas as pd
import wandb

from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Outcar

from neural_init.vasp_runner.chgcar import build_aug_swap_chgcar, get_chgcar_grid_dims_textparse
from neural_init.vasp_runner.oszicar import count_scf_breakdown_from_oszicar
from neural_init.vasp_runner.sad_extract import extract_sad_chgcar
from neural_init.vasp_runner.sources.mp import derive_mpid, write_mp_inputs_for_mpid, MissingMPTaskDocError
from neural_init.vasp_runner import runs_db, results
from neural_init.vasp_runner.failsafe import run_guarded
from neural_init.vasp_runner.scf import prune_workdir_keep_outputs


VARIANT_DEFAULT = "sad_grid_conv_aug"
ICHARG = 1
WANDB_PROJECT = "VASP_SCF_SadGridConvAug_MP"


def _materialize_chgcar(chgcar_path: str, dest: str):
    """Copy or lz4-decompress `chgcar_path` into `dest`."""
    if chgcar_path.endswith(".lz4"):
        with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy(chgcar_path, dest)


def run_variant(mpid: str, workdir: str, chgcar_path: str, vasp_cmd):
    os.makedirs(workdir, exist_ok=True)
    write_mp_inputs_for_mpid(mpid, workdir)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM"):
        incar.pop(bad, None)
    incar["ICHARG"] = ICHARG
    incar["ISTART"] = 0
    incar["LCHARG"] = True
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
        description="Run the SAD-grid + conv-aug SCF variant for one MP structure."
    )
    parser.add_argument("--csv", "-C", dest="CSV_PATH", required=True,
                        help="Task CSV with at least ['CHGCAR_PATH','MP_ID'].")
    parser.add_argument("--outdir", "-o", dest="BASE_DIR", required=True,
                        help="Base directory for per-structure work dirs.")
    parser.add_argument("--results_csv", "-R", dest="RESULTS_CSV", required=True,
                        help="Narrow-format CSV to append per-structure rows to.")
    parser.add_argument("--sad_cache", dest="SAD_CACHE", required=True,
                        help="Directory holding cached SAD CHGCARs (one per MP id). "
                             "Safe to share with the existing decomposition cache.")
    parser.add_argument("--vasp_cmd", nargs="+", default=["vasp_std"],
                        help="Command used to run VASP (e.g. --vasp_cmd srun vasp_std).")
    parser.add_argument("--variant", dest="VARIANT", default=VARIANT_DEFAULT,
                        help=f"Variant tag "
                             f"(default: {VARIANT_DEFAULT}). Use a suffixed tag "
                             f"for a versioned rerun.")
    parser.add_argument("--runs_db", dest="RUNS_DB", default="runs/runs.db",
                        help="SQLite manifest used for the cross-experiment skip-check. "
                             "Built by update_index.py.")
    parser.add_argument("--start_row", type=int, default=0,
                        help="Inclusive start row index in the task CSV slice.")
    parser.add_argument("--end_row", type=int, default=None,
                        help="Exclusive end row index in the task CSV slice. "
                             "Defaults to len(df).")
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

    idx = pending.index[0]
    chgcar_path = df.at[idx, "CHGCAR_PATH"]
    row = df.loc[idx]
    mpid = derive_mpid(row, chgcar_path)
    print(f"Selected idx={idx}, CHGCAR={chgcar_path}, MPID={mpid}")

    if not os.path.isfile(chgcar_path):
        print(f"⚠️ CHGCAR file not found: {chgcar_path}. "
              f"Marking as performed and skipping.")
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="missing_chgcar",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        sys.exit(3)

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

    print(f"Extracting SAD CHGCAR for {mpid} (cache={args.SAD_CACHE}) …")
    try:
        sad_chg = extract_sad_chgcar(
            mpid=mpid,
            sad_cache_dir=args.SAD_CACHE,
            vasp_cmd=tuple(args.vasp_cmd),
        )
    except Exception as e:
        print(f"⚠️ SAD extraction failed for {mpid}: {e!r}. "
              f"Marked performed; skipping structure.")
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_sad_extract",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        sys.exit(3)

    hybrid_dir = os.path.join(run_base, "_hybrids")
    os.makedirs(hybrid_dir, exist_ok=True)
    hybrid_chg = os.path.join(hybrid_dir, f"CHGCAR_{VARIANT}")

    print(f"Building {VARIANT} CHGCAR (grid=SAD, aug=converged) …")
    try:
        build_aug_swap_chgcar(
            grid_src=sad_chg,
            aug_src=converged_chg,
            out_path=hybrid_chg,
        )
    except Exception as e:
        print(f"⚠️ Hybrid CHGCAR build failed for {mpid}: {e!r}. "
              f"Marked performed; skipping structure.")
        runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                          chgcar_path=chgcar_path, status="failed_hybrid_build",
                          source_csv=args.CSV_PATH, source_row=int(idx))
        sys.exit(3)

    variant_dir = os.path.join(run_base, SUBDIR)
    if os.path.exists(variant_dir):
        shutil.rmtree(variant_dir)
    os.makedirs(variant_dir, exist_ok=True)

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{VARIANT}_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "mpid": mpid,
            "workdir": variant_dir,
            "grid_dims": grid_dims,
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
    })
    wandb.finish()

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
    ]
    row_out = [
        chgcar_path, mpid, int(idx), int(start), int(end),
        r["status"],
        "" if r["energy"] is None else r["energy"],
        "" if r["total_mag"] is None else r["total_mag"],
        "" if r["spin"] is None else r["spin"],
        r["dav"], r["rmm"], r["other"], r["total"], r["wall"],
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
