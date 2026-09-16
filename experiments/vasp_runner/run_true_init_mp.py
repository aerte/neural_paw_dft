#!/usr/bin/env python3
"""Run a single MP SCF with the converged CHGCAR as initialization (`true_init`).

Mirrors `run_default_mp.py` but with ICHARG=1 + the converged CHGCAR
from `df.at[idx, "CHGCAR_PATH"]` dropped into the workdir. Used to measure
the "best-case" SCF convergence cost when starting from the true density.

With `--ispin1` the run is strictly non-magnetic: the reference CHGCAR is
stripped to a total-only seed (no `diff` grid, no `diff` augmentation) and
ISPIN=1 is forced in the INCAR — the ISPIN=1 counterpart of the Oracle, and the
exact control for `total_ml_pg_ispin1` (same construction, converged pseudo grid
in place of the ML-predicted one). Magnetic structures are skipped.

`--zero_mag`, `--noise_mag` and `--magmom` keep ISPIN=2 and the converged total
pseudo grid + converged total augmentation, and vary only the spin-difference
stand-in — the "what is a good spin-difference guess when running ISPIN=2 on a
non-magnetic structure?" experiment. Like `--ispin1` they skip magnetic
structures, so all four share the Oracle's non-magnetic population:

  * `--zero_mag`   explicit all-zero `diff` grid + `diff` augmentation.
  * `--noise_mag`  near-zero symmetry-broken `diff` (log-uniform 1e-7..1e-5 µB/Å³
                   with random sign; the grid is volume-scaled, see
                   `build_total_only_chgcar`).
  * `--magmom X`   total-only seed (no `diff` channel at all) plus a uniform
                   `MAGMOM = X` on every ion, replacing MP's own MAGMOM.
  * `--total_only` total-only seed like `--magmom`, but the MAGMOM comes
                   per-site from `--relaxed_magmoms` (e.g. CHGNet predictions;
                   ids without a prediction keep MP's own MAGMOM). Pair with
                   `--include-magnetic` to run the full population.
"""
import os
import json
import shutil
import time
import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd
import lz4.frame
import wandb

from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Outcar

import scf_stats
from neural_paw_dft.vasp_runner import runs_db, results
from neural_paw_dft.vasp_runner.failsafe import run_guarded, watchdog
from neural_paw_dft.vasp_runner.chgcar import (
    build_total_only_chgcar,
    get_chgcar_grid_dims_textparse,
)
from neural_paw_dft.vasp_runner.scf import prune_workdir_keep_outputs
from neural_paw_dft.vasp_runner.oszicar import count_scf_breakdown_from_oszicar
from neural_paw_dft.vasp_runner.sources.mp import (
    derive_mpid,
    write_mp_inputs_for_mpid,
    MissingMPTaskDocError,
)

REPO_ROOT = Path(__file__).resolve().parent
VARIANT_DEFAULT = "true_init"
MAG_VARIANT = "true_init"          # DB variant that carries the magnetic truth


def _materialize_chgcar(src: str, dest: str):
    """Copy or lz4-decompress `src` into `dest`."""
    if src.endswith(".lz4"):
        with lz4.frame.open(src, "rb") as s, open(dest, "wb") as d:
            shutil.copyfileobj(s, d)
    else:
        shutil.copy(src, dest)


def run_vasp_mp(mpid, workdir, icharg, chgcar_path, vasp_cmd, ispin1=False,
                magmom=None, magmom_pattern="uniform", incar_overrides=None,
                relax_magmom=None):
    os.makedirs(workdir, exist_ok=True)
    write_mp_inputs_for_mpid(mpid, workdir)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM"):
        incar.pop(bad, None)
    incar["ICHARG"] = icharg
    incar["ISTART"] = 0
    incar["LCHARG"] = True
    if ispin1:
        # Strictly non-magnetic: collapse to a single spin channel and drop the
        # spin-2-only moment tags so VASP does not attempt to consult them.
        incar["ISPIN"] = 1
        for spin_tag in ("MAGMOM", "NUPDOWN"):
            incar.pop(spin_tag, None)
    elif magmom is not None:
        # ISPIN=2 with a uniform small starting moment on every ion, replacing
        # MP's own MAGMOM (0.6 per ion, 5.0 for transition metals). The seed
        # CHGCAR carries no diff channel, so this is what VASP initializes the
        # magnetization from. NUPDOWN is left as MP set it.
        natoms = len(Poscar.from_file(os.path.join(workdir, "POSCAR")).structure)
        if magmom_pattern == "alternating":
            # +X, -X, +X, ... by site index — a net-zero, symmetry-broken seed
            # to probe whether local spin *shape* (not just magnitude) helps.
            incar["MAGMOM"] = [magmom * (1.0 if i % 2 == 0 else -1.0)
                               for i in range(natoms)]
        else:
            incar["MAGMOM"] = [magmom] * natoms
    elif relax_magmom is not None:
        # Per-site relaxed moments (MP relax-task output) replacing MP's own
        # MAGMOM; see vasp_runner/relaxmag.py. Orthogonal to the seed: with the
        # full converged CHGCAR the initial density still comes from the seed.
        from neural_paw_dft.vasp_runner.relaxmag import apply_relaxed_magmom
        apply_relaxed_magmom(incar, relax_magmom)
    # Arbitrary INCAR overrides (e.g. spin-mixing tags NUPDOWN/AMIX_MAG/BMIX_MAG)
    # applied last so they win over the mode defaults above.
    if incar_overrides:
        for k, v in incar_overrides.items():
            incar[k] = v
    incar.write_file(incar_path)

    if chgcar_path is not None:
        _materialize_chgcar(chgcar_path, os.path.join(workdir, "CHGCAR"))

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
        description="Run MP SCF with ICHARG=1 + converged CHGCAR (true_init)."
    )
    parser.add_argument("--csv", "-C", dest="CSV_PATH", required=True)
    parser.add_argument("--outdir", "-o", dest="BASE_DIR", required=True)
    parser.add_argument("--results_csv", "-R", dest="RESULTS_CSV", required=True)
    parser.add_argument("--variant", dest="VARIANT", default=VARIANT_DEFAULT,
                        help=f"Variant tag (default: {VARIANT_DEFAULT}).")
    parser.add_argument("--ispin1", action="store_true",
                        help="Run strictly non-magnetic: strip the reference "
                             "CHGCAR to a total-only seed and force ISPIN=1 in "
                             "the INCAR (MAGMOM/NUPDOWN dropped). Magnetic "
                             "structures are skipped using the database criterion.")
    parser.add_argument("--zero_mag", action="store_true",
                        help="ISPIN=2 seed from the converged total grid + total "
                             "augmentation with an explicit ZERO diff grid and "
                             "zero diff augmentation. Magnetic structures skipped.")
    parser.add_argument("--noise_mag", action="store_true",
                        help="Like --zero_mag but with a near-zero symmetry-broken "
                             "diff channel (log-uniform 1e-7..1e-5 µB/Å³, random "
                             "sign) instead of strict zeros.")
    parser.add_argument("--magmom", type=float, default=None,
                        help="ISPIN=2 seed from a total-only converged CHGCAR (no "
                             "diff channel) with MAGMOM set to this value on every "
                             "ion, replacing MP's own MAGMOM. Magnetic structures "
                             "skipped.")
    parser.add_argument("--magmom-pattern", choices=("uniform", "alternating"),
                        default="uniform",
                        help="Shape of the --magmom seed: 'uniform' = +X on every "
                             "ion (default); 'alternating' = +X,-X,... by site "
                             "index (net-zero symmetry-broken seed).")
    parser.add_argument("--total_only", action="store_true",
                        help="Total-only converged seed (converged total grid + "
                             "total augmentation, no diff channel) with the "
                             "per-site MAGMOM from --relaxed_magmoms as the "
                             "spin init (requires --relaxed_magmoms). Same seed "
                             "construction as --magmom, per-site moments "
                             "instead of a uniform value.")
    parser.add_argument("--relaxed_magmoms", default=None,
                        help="Harvest CSV from exps/harvest_relaxed_magmoms.py. "
                             "When given, INCAR MAGMOM is replaced by the "
                             "per-site MP relax-task moments; mp-ids without a "
                             "usable entry keep MP's own MAGMOM and are flagged "
                             "magmom_source=default_fallback.")
    parser.add_argument("--pbe-only", dest="pbe_only",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="With --relaxed_magmoms: only GGA/GGA+U relax "
                             "moments (default). --no-pbe-only admits "
                             "r2SCAN/SCAN/PBEsol too.")
    parser.add_argument("--include-magnetic", action="store_true",
                        help="With a spin-init mode: do not skip magnetic "
                             "structures — run every row. The seed construction "
                             "is unchanged (total-only / zero / noise).")
    parser.add_argument("--incar-set", action="append", default=[],
                        metavar="KEY=VAL",
                        help="Extra INCAR override applied last (repeatable), e.g. "
                             "--incar-set NUPDOWN=0 --incar-set AMIX_MAG=0.6. "
                             "VAL is coerced int→float→str.")
    parser.add_argument("--vasp_cmd", nargs="+", default=["vasp_std"])
    parser.add_argument("--runs_db", dest="RUNS_DB", default="runs/runs.db")
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--end_row", type=int, default=None)
    args = parser.parse_args()

    VARIANT = args.VARIANT

    # Parse --incar-set KEY=VAL overrides, coercing VAL int→float→str.
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
    if args.magmom_pattern == "alternating" and args.magmom is None:
        print("❌ --magmom-pattern requires --magmom.")
        raise SystemExit(1)

    # Every spin-init mode builds its seed from the converged CHGCAR and is only
    # defined for non-magnetic structures, so they share one gate. Plain
    # `true_init` (no mode) keeps the full 1880-structure population.
    spin_modes = (args.ispin1, args.zero_mag, args.noise_mag,
                  args.magmom is not None, args.total_only)
    if sum(bool(m) for m in spin_modes) > 1:
        print("❌ --ispin1, --zero_mag, --noise_mag, --magmom and --total_only "
              "are mutually exclusive.")
        raise SystemExit(1)
    spin_mode = any(spin_modes)
    # The magnetic skip and the seed construction are separate concerns: with
    # --include-magnetic the gate is lifted but the seed is built the same way.
    nonmag_only = spin_mode and not args.include_magnetic
    seed_spin = "zero" if args.zero_mag else "noise" if args.noise_mag else "none"

    print(f"Variant={VARIANT} (ispin1={bool(args.ispin1)}, seed_spin={seed_spin}, "
          f"magmom={args.magmom})")

    if args.total_only and not args.relaxed_magmoms:
        print("❌ --total_only requires --relaxed_magmoms (its spin init).")
        raise SystemExit(1)

    relax_map = None
    if args.relaxed_magmoms:
        if spin_mode and not args.total_only:
            print("❌ --relaxed_magmoms is mutually exclusive with the "
                  "spin-init modes (--ispin1/--zero_mag/--noise_mag/--magmom).")
            raise SystemExit(1)
        from neural_paw_dft.vasp_runner.relaxmag import load_relaxed_magmoms
        relax_map = load_relaxed_magmoms(args.relaxed_magmoms, args.pbe_only)
        print(f"Relaxed magmoms: {len(relax_map)} usable mp-ids from "
              f"{args.relaxed_magmoms} (others keep MP's own MAGMOM)")

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

    # runs.db is the SOLE skip-truth: hard-overwrite the local performed flag
    # from the manifest's terminal set. A dropped runs.db write (write_run
    # swallows "database is locked" under NFS contention) now causes at worst a
    # cheap idempotent re-run of one structure, never a permanent skip — which
    # is the failure mode the old additive merge produced. terminal_set
    # excludes in_progress so walltime-killed rows stay retryable.
    conn = None
    try:
        conn = runs_db.get_connection(args.RUNS_DB)
        done = runs_db.terminal_set(conn, VARIANT, df_slice["MP_ID"].astype(str))
        df.loc[df_slice.index, performed_col] = df_slice["MP_ID"].astype(str).isin(done)
        df_slice = df.iloc[start:end]
    except Exception as e:
        print(f"⚠️ Could not read runs manifest {args.RUNS_DB}: {e!r}. "
              f"Falling back to slice CSV performed flags.")

    # DB magnetic verdict per mp-id (scf_stats criterion over true_init), only
    # needed for the spin-init modes: they are different physical calculations
    # for a magnetic structure, so those rows are skipped as in
    # the total-pseudo-grid runner (not included).
    mag_map = {}
    if nonmag_only and conn is not None:
        try:
            mag_map = scf_stats.magnetic_map(conn, REPO_ROOT, MAG_VARIANT,
                                             scf_stats.DEFAULT_MAG_THRESHOLD)
        except Exception as e:
            print(f"⚠️ Could not build magnetic map from {args.RUNS_DB}: {e!r}.")
        print(f"Magnetic map: {len(mag_map)} classified "
              f"({sum(mag_map.values())} magnetic).")

    pending = df_slice[df_slice[performed_col] == False]
    if pending.empty:
        print(f"No pending rows in slice [{start}:{end}).")
        raise SystemExit(2)

    # Scan pending rows for the first runnable structure.
    #   - missing reference CHGCAR   -> TERMINAL: missing_chgcar
    #   - magnetic (DB criterion)    -> TERMINAL: skipped_magnetic  (spin modes only)
    #   - unknown magnetic status    -> TERMINAL: skipped_unknown_mag (spin modes only)
    idx = chgcar_path = mpid = None
    for cand in pending.index:
        cand_chgcar = df.at[cand, "CHGCAR_PATH"]
        cand_mpid = derive_mpid(df.loc[cand], cand_chgcar)

        if not os.path.isfile(cand_chgcar):
            print(f"⚠️ CHGCAR file not found: {cand_chgcar}. Marking as performed "
                  f"and skipping.")
            df.at[cand, performed_col] = True
            df.to_csv(args.CSV_PATH, index=False)
            runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                              chgcar_path=cand_chgcar, status="missing_chgcar",
                              source_csv=args.CSV_PATH, source_row=int(cand))
            continue

        if nonmag_only:
            if cand_mpid not in mag_map:
                print(f"⚠️ No {MAG_VARIANT} classification for {cand_mpid}; cannot "
                      f"confirm non-magnetic. Marking performed (skipped_unknown_mag).")
                df.at[cand, performed_col] = True
                df.to_csv(args.CSV_PATH, index=False)
                runs_db.write_run(conn, mp_id=cand_mpid, variant=VARIANT,
                                  chgcar_path=cand_chgcar, status="skipped_unknown_mag",
                                  source_csv=args.CSV_PATH, source_row=int(cand))
                continue

            if mag_map[cand_mpid]:
                print(f"↷ {cand_mpid} is magnetic (DB criterion); skipping.")
                df.at[cand, performed_col] = True
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
    # and must stay pending so the next sbatch iteration re-picks it. We flip
    # performed_<variant>=True only after a terminal result is written below.
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

    # The spin-init modes all seed from the converged total pseudo grid + total
    # augmentation and differ only in the spin channel: none (--ispin1/--magmom),
    # zeros (--zero_mag) or near-zero noise (--noise_mag). Same seed construction
    # as the total-pseudo-grid runner (not included), with the converged pseudo grid in place of the
    # ML-predicted one.
    seed_chgcar = chgcar_path
    seed_dir = None
    if spin_mode:
        seed_dir = os.path.join(run_base, "_seed")
        os.makedirs(seed_dir, exist_ok=True)
        converged_chg = os.path.join(seed_dir, "CHGCAR_converged")
        _materialize_chgcar(chgcar_path, converged_chg)
        seed_chgcar = os.path.join(seed_dir, "CHGCAR_seed")
        print(f"Building {VARIANT} seed (spin={seed_spin}) from the converged "
              f"CHGCAR …")
        try:
            build_total_only_chgcar(converged_chg, seed_chgcar, spin=seed_spin)
        except Exception as e:
            print(f"⚠️ Seed build failed for {mpid}: {e!r}. "
                  f"Marked performed; skipping structure.")
            runs_db.write_run(conn, mp_id=mpid, variant=VARIANT,
                              chgcar_path=chgcar_path, status="failed_seed_build",
                              source_csv=args.CSV_PATH, source_row=int(idx))
            df.at[idx, performed_col] = True
            df.to_csv(args.CSV_PATH, index=False)
            shutil.rmtree(seed_dir, ignore_errors=True)
            raise SystemExit(3)
        os.remove(converged_chg)

    relax_magmom = relax_map.get(mpid) if relax_map is not None else None
    magmom_source = ""
    if relax_map is not None:
        magmom_source = "relaxed" if relax_magmom is not None else "default_fallback"
        if relax_magmom is None:
            print(f"ℹ️ No usable relaxed magmoms for {mpid}; keeping MP's own "
                  f"MAGMOM (default_fallback).")

    wandb.init(
        project="VASP_SCF_TrueInit_MP",
        name=f"{VARIANT}_{base}",
        config={
            "csv_index": int(idx),
            "chgcar_path": chgcar_path,
            "mpid": mpid,
            "workdir": workdir,
            "grid_dims": grid_dims,
            "magmom_source": magmom_source,
            "slice_start": int(start),
            "slice_end": int(end),
        },
    )

    try:
        energy, total_mag, magnetization, spin, dav, rmm, total, others, wall = run_vasp_mp(
            mpid=mpid,
            workdir=workdir,
            icharg=1,
            chgcar_path=seed_chgcar,
            vasp_cmd=tuple(args.vasp_cmd),
            ispin1=args.ispin1,
            magmom=args.magmom,
            magmom_pattern=args.magmom_pattern,
            incar_overrides=incar_overrides,
            relax_magmom=relax_magmom,
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
        if seed_dir:
            shutil.rmtree(seed_dir, ignore_errors=True)
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
        f"magmom_source_{VARIANT}": magmom_source,
    })
    wandb.finish()

    header = [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        f"status_{VARIANT}", f"energy_{VARIANT}", f"total_mag_{VARIANT}",
        f"spin_{VARIANT}",
        f"scf_steps_dav_{VARIANT}", f"scf_steps_rmm_{VARIANT}",
        f"scf_steps_other_{VARIANT}", f"scf_steps_total_{VARIANT}",
        f"time_{VARIANT}", f"magmom_source_{VARIANT}",
    ]
    row_out = [
        chgcar_path, mpid, int(idx), int(start), int(end),
        status,
        "" if energy is None else energy,
        "" if total_mag is None else total_mag,
        "" if spin is None else spin,
        dav, rmm, other_total, total, wall, magmom_source,
    ]
    # All three writes land on the NFS-backed share; a hung lock here once held a
    # node for 19h at 0% CPU after the SCF had already succeeded. If the watchdog
    # fires the manifest row stays 'in_progress', which terminal_set treats as
    # retryable, so the next invocation re-runs this structure cleanly.
    with watchdog(600, f"post-run bookkeeping for {mpid}"):
        results.upsert_result(args.RESULTS_CSV, header, row_out)

        # Terminal result recorded (ok or failed): now it is safe to mark the row
        # performed so it is not re-run. A 'failed' row is retried only after
        # reconcile_csvs.py --clear-status failed flips it back.
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
                f"magmom_source_{VARIANT}": magmom_source,
            },
            source_csv=args.CSV_PATH, source_row=int(idx),
        )

    prune_workdir_keep_outputs(workdir)
    if seed_dir:
        shutil.rmtree(seed_dir, ignore_errors=True)

    print("Done.")
    print(f"Energy: {energy}")
    print(f"Total mag: {total_mag}")
    print(f"DAV: {dav}, RMM: {rmm}, OTHER: {other_total}, TOTAL: {total}")
    print(f"Time: {wall:.2f}s")

    if energy is None:
        raise SystemExit(3)


if __name__ == "__main__":
    run_guarded(main)
