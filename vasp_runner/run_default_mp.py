#!/usr/bin/env python3
import os
import re
import json
import shutil
import time
import argparse
import csv
import subprocess
import sys
import tempfile

from pathlib import Path

import pandas as pd
import lz4.frame
import wandb

from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Oszicar, Outcar

import scf_stats
from vasp_runner import runs_db, results
from vasp_runner.failsafe import run_guarded
from vasp_runner.scf import prune_workdir_keep_outputs
from vasp_runner.sources.mp import (
    MissingMPTaskDocError,
    derive_mpid,
    write_mp_inputs_for_mpid,
)

REPO_ROOT = Path(__file__).resolve().parent
VARIANT_DEFAULT = "default"
MAG_VARIANT = "true_init"          # DB variant that carries the magnetic truth


# --------------------------------------------------------------------------------------
# Helpers: CHGCAR grid (we still log this, but do NOT use it to override INCAR anymore)
# --------------------------------------------------------------------------------------

def get_chgcar_grid_dims(chgcar_path):
    """
    Parse (NGXF, NGYF, NGZF) from a CHGCAR or CHGCAR.lz4.

    These grid dims are logged to W&B and CSV, but no longer inserted into INCAR.
    """
    path = chgcar_path
    tmp_created = False

    if chgcar_path.endswith(".lz4"):
        tmpfd, tmppath = tempfile.mkstemp(prefix="tmp_chgcar_grid_")
        os.close(tmpfd)
        with lz4.frame.open(chgcar_path, "rb") as src, open(tmppath, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = tmppath
        tmp_created = True

    nx = ny = nz = None
    try:
        with open(path, "r") as f:
            # Skip header + atom counts
            for _ in range(7):
                try:
                    next(f)
                except StopIteration:
                    break

            # First line with exactly 3 integers
            for line in f:
                toks = line.split()
                if len(toks) == 3 and all(t.isdigit() for t in toks):
                    nx, ny, nz = map(int, toks)
                    break
    finally:
        if tmp_created:
            os.remove(path)

    if nx is None:
        raise RuntimeError(f"Could not find NGXF/NGYF/NGZF in {chgcar_path}")

    return nx, ny, nz


# --------------------------------------------------------------------------------------
# Parse SCF breakdown from OSZICAR
# --------------------------------------------------------------------------------------

def count_scf_breakdown_from_oszicar(osz_path: str):
    """
    Count DAV, RMM, and any other iteration types in OSZICAR.

    Returns:
        dav (int)
        rmm (int)
        total (int)  # DAV + RMM + all other tags
        other_counts (dict[str, int])  # e.g. {"HMM": 3, "CG": 1}
    """
    dav = 0
    rmm = 0
    other_counts: dict[str, int] = {}

    try:
        with open(osz_path, "r") as f:
            for line in f:
                ls = line.lstrip()
                # Match leading ALLCAPS tag with colon, e.g. "DAV:", "RMM:", "HMM:", "CG:"
                m = re.match(r"([A-Z]+):", ls)
                if not m:
                    continue
                tag = m.group(1)

                if tag == "DAV":
                    dav += 1
                elif tag == "RMM":
                    rmm += 1
                else:
                    other_counts[tag] = other_counts.get(tag, 0) + 1
    except FileNotFoundError:
        pass

    total = dav + rmm + sum(other_counts.values())
    return dav, rmm, total, other_counts


# --------------------------------------------------------------------------------------
# Run VASP using MP inputs (no NG override)
# --------------------------------------------------------------------------------------

def run_vasp_mp(
    mpid: str,
    workdir: str,
    icharg: int,
    chgcar_path: str | None,
    vasp_cmd,
    grid_dims,
    ispin1: bool = False,
    incar_overrides=None,
    relax_magmom=None,
):
    os.makedirs(workdir, exist_ok=True)

    # Write exact MP input files (POSCAR/INCAR/KPOINTS/POTCAR)
    write_mp_inputs_for_mpid(mpid, workdir)

    # Modify INCAR minimally
    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)

    # Remove parallelization hints
    for bad in ["NPAR", "NCORE", "KPAR", "NSIM"]:
        if bad in incar:
            incar.pop(bad)

    # Only touch what we need for the SCF experiment
    incar["ICHARG"] = icharg   # 2 = SAD, 1 = CHGCAR init
    incar["ISTART"] = 0        # always fresh SCF
    incar["LCHARG"] = True     # always write CHGCAR
    if ispin1:
        # Strictly non-magnetic: collapse to a single spin channel and drop the
        # spin-2-only moment tags so VASP does not attempt to consult them.
        incar["ISPIN"] = 1
        for spin_tag in ("MAGMOM", "NUPDOWN"):
            incar.pop(spin_tag, None)
    elif relax_magmom is not None:
        # Per-site relaxed moments (MP relax-task output) replacing MP's own
        # MAGMOM; see vasp_runner/relaxmag.py.
        from vasp_runner.relaxmag import apply_relaxed_magmom
        apply_relaxed_magmom(incar, relax_magmom)
    # Arbitrary INCAR overrides (e.g. --incar-set LMAXMIX=4), applied last.
    if incar_overrides:
        for k, v in incar_overrides.items():
            incar[k] = v
    incar.write_file(incar_path)

    # If CHGCAR init is requested
    if chgcar_path is not None:
        dest = os.path.join(workdir, "CHGCAR")
        if chgcar_path.endswith(".lz4"):
            with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(chgcar_path, dest)

    # Run VASP
    start = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError:
        wall = time.time() - start
        osz_path = os.path.join(workdir, "OSZICAR")
        dav, rmm, total, other_counts = count_scf_breakdown_from_oszicar(osz_path)
        return None, None, None, None, dav, rmm, total, other_counts, wall

    wall = time.time() - start

    # Parse SCF steps
    osz_path = os.path.join(workdir, "OSZICAR")
    dav, rmm, total, other_counts = count_scf_breakdown_from_oszicar(osz_path)

    # Parse energy + magnetism
    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        return None, None, None, None, dav, rmm, total, other_counts, wall

    energy = None
    total_mag = None
    magnetization = None
    spin = None
    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
        total_mag = getattr(outcar, "total_mag", None)
        magnetization = getattr(outcar, "magnetization", None)
        spin = getattr(outcar, "spin", None)
    except Exception:
        pass

    return energy, total_mag, magnetization, spin, dav, rmm, total, other_counts, wall


# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run VASP SCF for one MP CHGCAR using exact MP inputs."
    )

    parser.add_argument(
        "--csv", "-C",
        dest="CSV_PATH",
        required=True,
        help="Task CSV with at least ['CHGCAR_PATH','MP_ID']"
    )
    parser.add_argument(
        "--outdir", "-o",
        dest="BASE_DIR",
        required=True,
        help="Base directory under which per-structure folders will be created",
    )
    parser.add_argument(
        "--results_csv", "-R",
        dest="RESULTS_CSV",
        required=True,
        help="CSV file to append default-run results to",
    )
    parser.add_argument(
        "--variant",
        dest="VARIANT",
        default=VARIANT_DEFAULT,
        help=f"Variant tag (default: {VARIANT_DEFAULT}).",
    )
    parser.add_argument(
        "--ispin1",
        action="store_true",
        help="Run strictly non-magnetic: force ISPIN=1 in the INCAR (and drop "
             "MAGMOM/NUPDOWN). Magnetic structures are skipped using the "
             "database criterion, so the population matches the other ISPIN=1 "
             "variants.",
    )
    parser.add_argument(
        "--include-magnetic",
        action="store_true",
        help="With --ispin1: do not skip magnetic structures — run every row "
             "under ISPIN=1. Off by default so the population matches the "
             "other ISPIN=1 variants.",
    )
    parser.add_argument(
        "--incar-set",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Arbitrary INCAR override, repeatable: --incar-set LMAXMIX=4.",
    )
    parser.add_argument(
        "--relaxed_magmoms",
        default=None,
        help="Harvest CSV from exps/harvest_relaxed_magmoms.py. When given, "
             "INCAR MAGMOM is replaced by the per-site MP relax-task moments; "
             "mp-ids without a usable entry keep MP's own MAGMOM and are "
             "flagged magmom_source=default_fallback.",
    )
    parser.add_argument(
        "--pbe-only",
        dest="pbe_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="With --relaxed_magmoms: only use GGA/GGA+U relax moments "
             "(default). --no-pbe-only admits r2SCAN/SCAN/PBEsol too.",
    )
    parser.add_argument(
        "--save_chgcar_dir",
        default=None,
        help="After an ok run, lz4-compress the converged CHGCAR into this "
             "directory as <mpid>.chgcar.lz4 (workdirs are pruned afterwards, "
             "so this is the only surviving copy). Failed runs save nothing.",
    )
    parser.add_argument(
        "--vasp_cmd",
        nargs="+",
        default=["vasp_std"],
        help="Command used to run VASP (e.g. --vasp_cmd mpirun -np 32 vasp_std).",
    )
    parser.add_argument(
        "--runs_db",
        dest="RUNS_DB",
        default="runs/runs.db",
        help="SQLite manifest used for the cross-experiment skip-check. "
             "Built by update_index.py.",
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
        help="Inclusive start row index in the task CSV slice."
    )
    parser.add_argument(
        "--end_row",
        type=int,
        default=None,
        help="Exclusive end row index in the task CSV slice. Defaults to len(df)."
    )

    args = parser.parse_args()

    VARIANT = args.VARIANT
    PERFORMED_COL = f"performed_{VARIANT}"

    # Parse --incar-set KEY=VAL overrides, coercing VAL int->float->str.
    incar_overrides = {}
    for item in args.incar_set:
        if "=" not in item:
            print(f"❌ --incar-set expects KEY=VAL, got {item!r}.")
            raise SystemExit(1)
        k, v = item.split("=", 1)
        k, v = k.strip(), v.strip()
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                continue
        incar_overrides[k] = v
    SUBDIR = f"run_{VARIANT}"
    print(f"Variant={VARIANT} (ispin1={bool(args.ispin1)})")

    relax_map = None
    if args.relaxed_magmoms:
        from vasp_runner.relaxmag import load_relaxed_magmoms
        relax_map = load_relaxed_magmoms(args.relaxed_magmoms, args.pbe_only)
        print(f"Relaxed magmoms: {len(relax_map)} usable mp-ids from "
              f"{args.relaxed_magmoms} (others keep MP's own MAGMOM)")

    # Load task CSV
    df = pd.read_csv(args.CSV_PATH, dtype={"CHGCAR_PATH": str, "MP_ID": str})
    n_rows = len(df)

    # Clamp slice
    start = max(args.start_row, 0)
    end = n_rows if args.end_row is None else min(args.end_row, n_rows)

    if start >= end:
        print(f"❌ Invalid slice [{start}:{end}) for CSV with {n_rows} rows.")
        raise SystemExit(1)

    print(f"Using slice [{start}:{end}) out of {n_rows}")

    # Ensure columns exist
    for col in ["performed_default", "performed_true_init", "performed_ml_init",
                PERFORMED_COL]:
        if col not in df.columns:
            df[col] = False

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

    # DB magnetic verdict per mp-id (scf_stats criterion over true_init), only
    # needed for --ispin1: an ISPIN=1 run of a magnetic structure is a different
    # physical calculation, so those rows are skipped as in the total-pseudo-grid runner (not included).
    mag_map = {}
    if args.ispin1 and not args.include_magnetic and conn is not None:
        try:
            mag_map = scf_stats.magnetic_map(conn, REPO_ROOT, MAG_VARIANT,
                                             scf_stats.DEFAULT_MAG_THRESHOLD)
        except Exception as e:
            print(f"⚠️ Could not build magnetic map from {args.RUNS_DB}: {e!r}.")
        print(f"Magnetic map: {len(mag_map)} classified "
              f"({sum(mag_map.values())} magnetic).")

    pending = df_slice[df_slice[PERFORMED_COL] == False]
    if pending.empty:
        print(f"No pending rows in slice [{start}:{end}).")
        raise SystemExit(2)

    # Scan pending rows for the first runnable structure.
    #   - missing reference CHGCAR   -> TERMINAL: missing_chgcar
    #   - magnetic (DB criterion)    -> TERMINAL: skipped_magnetic  (--ispin1 only)
    #   - unknown magnetic status    -> TERMINAL: skipped_unknown_mag (--ispin1 only)
    idx = chgcar_path = mpid = None
    for cand in pending.index:
        cand_chgcar = df.at[cand, "CHGCAR_PATH"]
        cand_mpid = derive_mpid(df.loc[cand], cand_chgcar)

        # Skip rows whose CHGCAR file is missing on disk
        if not os.path.isfile(cand_chgcar):
            print(f"⚠️ CHGCAR file not found: {cand_chgcar}. Marking as performed "
                  f"and skipping.")
            df.at[cand, PERFORMED_COL] = True
            df.to_csv(args.CSV_PATH, index=False)
            runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                              chgcar_path=cand_chgcar, status="missing_chgcar",
                              source_csv=args.CSV_PATH, source_row=int(cand))
            continue

        if args.ispin1 and not args.include_magnetic:
            if cand_mpid not in mag_map:
                print(f"⚠️ No {MAG_VARIANT} classification for {cand_mpid}; cannot "
                      f"confirm non-magnetic. Marking performed (skipped_unknown_mag).")
                df.at[cand, PERFORMED_COL] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar, status="skipped_unknown_mag",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue

            if mag_map[cand_mpid]:
                print(f"↷ {cand_mpid} is magnetic (DB criterion); skipping.")
                df.at[cand, PERFORMED_COL] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar, status="skipped_magnetic",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue

        idx = cand
        chgcar_path = cand_chgcar
        mpid = cand_mpid
        break

    if idx is None:
        print(f"No runnable rows in slice [{start}:{end}) — all pending rows are "
              f"skipped.")
        raise SystemExit(2)

    print(f"Selected idx={idx}, CHGCAR={chgcar_path}, MPID={mpid}")

    # Manifest in_progress only — do NOT set the local performed flag yet. If
    # VASP is killed mid-run (walltime/OOM), this row has no terminal outcome
    # and must stay pending so the next sbatch iteration re-picks it.
    # performed_<variant> is flipped True only after a terminal result below.
    runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                      chgcar_path=chgcar_path, status="in_progress",
                      source_csv=args.CSV_PATH, source_row=int(idx))

    # Directory setup
    base = os.path.splitext(os.path.basename(chgcar_path))[0]
    run_base = os.path.join(args.BASE_DIR, base)
    default_dir = os.path.join(run_base, SUBDIR)

    if os.path.exists(default_dir):
        shutil.rmtree(default_dir)
    os.makedirs(default_dir, exist_ok=True)

    # Get CHGCAR grid dims (for logging only). Must never be fatal: a CHGCAR
    # whose header this parser can't read would otherwise raise an uncaught
    # RuntimeError and kill the whole SBATCH slice, stranding every later row.
    # Mirrors the guarded grid-dims read in run_true_init_mp.py / run_ml_seed_mp.py.
    try:
        grid_dims = get_chgcar_grid_dims(chgcar_path)
    except Exception:
        grid_dims = None

    relax_magmom = relax_map.get(mpid) if relax_map is not None else None
    magmom_source = ""
    if relax_map is not None and not args.ispin1:
        magmom_source = "relaxed" if relax_magmom is not None else "default_fallback"
        if relax_magmom is None:
            print(f"ℹ️ No usable relaxed magmoms for {mpid}; keeping MP's own "
                  f"MAGMOM (default_fallback).")

    # W&B init
    wandb.init(
        project="VASP_SCF_Default_MP",
        name=f"{VARIANT}_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "mpid": mpid,
            "workdir": default_dir,
            "grid_dims": grid_dims,
            "magmom_source": magmom_source,
            "slice_start": int(start),
            "slice_end": int(end),
        },
    )

    # Run VASP (ICHARG = 2 = SAD)
    try:
        energy, total_mag, magnetization, spin, dav, rmm, total, other_counts, wall = run_vasp_mp(
            mpid=mpid,
            workdir=default_dir,
            icharg=2,
            chgcar_path=None,
            vasp_cmd=tuple(args.vasp_cmd),
            grid_dims=grid_dims,
            ispin1=args.ispin1,
            incar_overrides=incar_overrides,
            relax_magmom=relax_magmom,
        )
    except MissingMPTaskDocError as e:
        # MP has no CHGCAR/TaskDoc for this mp-id: skip this structure cleanly
        # so the SBATCH wrapper `continue`s to the next row instead of aborting.
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
        df.at[idx, PERFORMED_COL] = True
        df.to_csv(args.CSV_PATH, index=False)
        raise SystemExit(3)

    other_total = sum(other_counts.values()) if other_counts else 0

    # Record results
    status = "ok" if energy is not None else "failed"

    # Per-atom magnetization is stored in a sibling JSONL file
    # (one record per structure), not inline in the CSV — magnetization
    # is a list of per-atom dicts and bloats the CSV.
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
        f"magmom_source_{VARIANT}": magmom_source,
    })
    wandb.finish()

    # Append to results CSV
    header = [
        "CHGCAR_PATH",
        "MP_ID",
        "csv_index",
        "slice_start",
        "slice_end",
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
        chgcar_path,
        mpid,
        int(idx),
        int(start),
        int(end),
        status,
        "" if energy is None else energy,
        "" if total_mag is None else total_mag,
        "" if spin is None else spin,
        dav,
        rmm,
        other_total,
        total,
        wall,
        magmom_source,
    ]

    results.upsert_result(args.RESULTS_CSV, header, row_out)

    # Terminal result recorded (ok or failed): safe to mark performed now. A
    # 'failed' row is retried only via reconcile_csvs.py --clear-status failed.
    df.at[idx, PERFORMED_COL] = True
    df.to_csv(args.CSV_PATH, index=False)

    # Final manifest write: promotes the in_progress row to its real status.
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
            f"magmom_source_{VARIANT}": magmom_source,
        },
        source_csv=args.CSV_PATH, source_row=int(idx),
    )

    # Archive the converged CHGCAR before the prune below deletes it. Written
    # via a tmp file + rename so a walltime kill mid-compress never leaves a
    # truncated .lz4 that later poisons a true_init seed. Never fatal.
    if args.save_chgcar_dir and status == "ok":
        src_chg = os.path.join(default_dir, "CHGCAR")
        dest_chg = os.path.join(args.save_chgcar_dir, f"{mpid}.chgcar.lz4")
        try:
            os.makedirs(args.save_chgcar_dir, exist_ok=True)
            tmp_chg = dest_chg + ".tmp"
            with open(src_chg, "rb") as fsrc, \
                    lz4.frame.open(tmp_chg, "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst)
            os.replace(tmp_chg, dest_chg)
            print(f"Archived converged CHGCAR → {dest_chg}")
        except Exception as e:
            print(f"⚠️ Failed to archive CHGCAR for {mpid} → {dest_chg}: {e!r}")

    prune_workdir_keep_outputs(default_dir)

    print("Done.")
    print(f"Energy: {energy}")
    print(f"Total mag: {total_mag}")
    print(f"Magnetization: {magnetization}")
    print(f"Spin: {spin}")
    print(f"DAV: {dav}, RMM: {rmm}, OTHER: {other_total}, TOTAL: {total}")
    print(f"Time: {wall:.2f}s")

    # Signal per-structure failure to the SBATCH wrapper so it `continue`s
    # instead of recording the run as OK.
    if energy is None:
        raise SystemExit(3)


if __name__ == "__main__":
    run_guarded(main)
