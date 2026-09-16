#!/usr/bin/env python3
"""Submit all slices of the ML-augmentation SCF experiment.

Per-row VASP seed = converged-CHGCAR grid + ML-predicted augmentation
occupancies (from `--aug-pred-dir`). Same shape as
submit_conv_grid_sad_aug.py, but passes `--aug_pred_dir` instead of
`--sad_cache` and runs `run_ml_seed_mp.py`.

The variant tag is auto-derived from the prediction directory (e.g. a
`.../10k/...` path -> ml_aug_10k), so results/outdirs for the 1k/10k/50k models
never collide. Run this once per model directory.

Usage (from the project root on the cluster):
    python submissions/submit_ml_aug.py --aug-pred-dir /path/to/10k/.../aug_pred_dir_test_total
    python submissions/submit_ml_aug.py --aug-pred-dir <dir> --dry-run
    python submissions/submit_ml_aug.py --aug-pred-dir <dir> --num-slices 20
    python submissions/submit_ml_aug.py --aug-pred-dir <dir> --partition <name>
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=24:00:00
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --output={log_path}
#SBATCH --error={err_path}

# No `-e`: we deliberately inspect `$?` after each python call.
set -uo pipefail

PROJECT_DIR="{project_dir}"
CSV="{csv}"
OUTDIR="{outdir}"
RESULTS_CSV="{results_csv}"
AUG_PRED_DIR="{aug_pred_dir}"
MAG_AUG_PRED_DIR="{mag_aug_pred_dir}"
SPIN_PRED_DIR="{spin_pred_dir}"
RUNS_DB="{runs_db}"
VARIANT={variant}
NUM_RUNS={num_runs}
START_ROW=0
END_ROW={end_row}
RUNNER="{runner}"

: "${{PMG_VASP_PSP_DIR:?set PMG_VASP_PSP_DIR to your POTCAR directory}}"
: "${{MP_API_KEY:?set MP_API_KEY to your Materials Project API key}}"
: "${{WANDB_API_KEY:=}}"
: "${{WANDB_MODE:=offline}}"
export PMG_VASP_PSP_DIR
export MP_API_KEY
export MAPI_KEY="$MP_API_KEY"
export WANDB_API_KEY
export WANDB_MODE

# lmod's init dereferences optional vars (e.g. LD_PRELOAD); under `set -u`
# an unset one aborts the load and leaves vasp_std off PATH. Relax nounset here.
set +u
# Site-specific: load VASP (vasp_std on PATH) and the Python used for .venv.
# module load VASP Python
set -u

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "FATAL: PROJECT_DIR does not exist: $PROJECT_DIR" >&2
  exit 10
fi
cd "$PROJECT_DIR" || {{ echo "FATAL: cd to $PROJECT_DIR failed" >&2; exit 10; }}

if [[ ! -f .venv/bin/activate ]]; then
  echo "FATAL: venv not found at $PROJECT_DIR/.venv/bin/activate" >&2
  exit 11
fi
source .venv/bin/activate

if [[ ! -f "$RUNNER" ]]; then
  echo "FATAL: missing runner: $PROJECT_DIR/$RUNNER" >&2
  exit 12
fi
if [[ ! -f "$CSV" ]]; then
  echo "FATAL: task CSV not found: $PROJECT_DIR/$CSV" >&2
  exit 13
fi
if [[ ! -d "$PMG_VASP_PSP_DIR" ]]; then
  echo "FATAL: PMG_VASP_PSP_DIR does not exist: $PMG_VASP_PSP_DIR" >&2
  exit 14
fi
if [[ -z "${{MP_API_KEY:-}}" ]]; then
  echo "FATAL: MP_API_KEY is unset or empty" >&2
  exit 15
fi
if [[ -n "$AUG_PRED_DIR" && ! -d "$AUG_PRED_DIR" ]]; then
  echo "FATAL: AUG_PRED_DIR does not exist: $AUG_PRED_DIR" >&2
  exit 16
fi
if [[ -n "$MAG_AUG_PRED_DIR" && ! -d "$MAG_AUG_PRED_DIR" ]]; then
  echo "FATAL: MAG_AUG_PRED_DIR does not exist: $MAG_AUG_PRED_DIR" >&2
  exit 16
fi
if [[ -n "$SPIN_PRED_DIR" && ! -d "$SPIN_PRED_DIR" ]]; then
  echo "FATAL: SPIN_PRED_DIR does not exist: $SPIN_PRED_DIR" >&2
  exit 16
fi

runs_db_parent="$(dirname "$RUNS_DB")"
mkdir -p "$runs_db_parent" || {{ echo "FATAL: mkdir $runs_db_parent failed" >&2; exit 16; }}
results_parent="$(dirname "$RESULTS_CSV")"
mkdir -p "$results_parent" || {{ echo "FATAL: mkdir $results_parent failed" >&2; exit 17; }}

export PYTHONNOUSERSITE=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "Submission config: CSV=$CSV OUTDIR=$OUTDIR RESULTS_CSV=$RESULTS_CSV \
AUG_PRED_DIR=$AUG_PRED_DIR MAG_AUG_PRED_DIR=$MAG_AUG_PRED_DIR SPIN_PRED_DIR=$SPIN_PRED_DIR RUNS_DB=$RUNS_DB START_ROW=$START_ROW END_ROW=$END_ROW NUM_RUNS=$NUM_RUNS"

for run in $(seq 1 "$NUM_RUNS"); do
  echo "Starting {tag} run #$run at $(date)"

  python "$RUNNER" \\
    --csv "$CSV" \\
    --outdir "$OUTDIR" \\
    --results_csv "$RESULTS_CSV" \\
    {aug_pred_dir_arg}{mag_aug_pred_dir_arg}{spin_pred_dir_arg}--variant "$VARIANT" \\
    {mode_arg}\\
    --runs_db "$RUNS_DB" \\
    --start_row "$START_ROW" \\
    --end_row "$END_ROW" \\
    --vasp_cmd srun vasp_std

  status=$?

  if [ $status -eq 0 ]; then
    echo "Finished {tag} run #$run (OK) at $(date)"
  elif [ $status -eq 2 ]; then
    echo "Slice exhausted at run #$run; stopping at $(date)"
    break
  elif [ $status -eq 3 ]; then
    echo "{tag} run #$run FAILED for this structure (VASP/OUTCAR), skipping and continuing at $(date)"
    continue
  else
    echo "{tag} run #$run FAILED with unexpected status $status at $(date)" >&2
    exit $status
  fi
done

echo "All {tag} runs for this slice completed (or slice exhausted) at $(date)"
"""


def derive_tag(aug_pred_dir: str) -> str:
    """Auto-derive `ml_aug_<size>` from the prediction directory path.

    Mirrors `run_ml_seed_mp.py::derive_variant`: scans path components for
    one matching `^\\d+k$` or `full`, else falls back to a sanitized basename.
    """
    parts = [p for p in os.path.normpath(aug_pred_dir).split(os.sep) if p]
    for p in parts:
        if re.fullmatch(r"\d+k|full", p):
            return f"ml_aug_{p}"
    base = re.sub(r"[^0-9A-Za-z]+", "_", os.path.basename(os.path.normpath(aug_pred_dir)))
    return f"ml_aug_{base}".rstrip("_")


def split_csv(src: Path, dst: Path, start: int, end: int):
    if dst.exists():
        return
    with src.open() as f:
        header = next(f)
        rows = list(f)
    with dst.open("w") as f:
        f.write(header)
        f.writelines(rows[start:end])


def clear_variant_rows(runs_db_path: Path, variant: str, dry_run: bool):
    """Scoped clear: DELETE only this variant's rows from runs.db, after a
    timestamped backup. Other variants are untouched. On --dry-run, report the
    count that WOULD be cleared and delete nothing.
    """
    if not runs_db_path.is_file():
        print(f"  [rerun] runs.db not found at {runs_db_path}; nothing to clear.")
        return
    # Import lazily from the repo root (cwd is project_dir here) so the common
    # non-rerun path never needs vasp_runner on the path.
    sys.path.insert(0, os.getcwd())
    from neural_paw_dft.vasp_runner import runs_db
    conn = runs_db.get_connection(runs_db_path)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM runs WHERE variant=?", (variant,)
        ).fetchone()[0]
        if dry_run:
            print(f"  [rerun][DRY] would clear {n} '{variant}' rows from "
                  f"{runs_db_path} (no backup, no delete).")
            return
        backup = runs_db_path.with_name(f"{runs_db_path.name}.bak-{int(time.time())}")
        shutil.copy(runs_db_path, backup)
        conn.execute("DELETE FROM runs WHERE variant=?", (variant,))
        conn.commit()
        print(f"  [rerun] cleared {n} '{variant}' rows from {runs_db_path} "
              f"(backup: {backup.name}). Other variants untouched.")
    finally:
        conn.close()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--aug-pred-dir", default=None,
                    help="ML augmentation prediction directory for ONE model "
                         "(e.g. .../10k/.../aug_pred_dir_test_total).")
    ap.add_argument("--mag-aug-pred-dir", default=None,
                    help="Directory of ML DIFF-augmentation predictions "
                         "({mpid}_mag_aug.npz, the AugNet spin export), passed "
                         "through as --mag_aug_pred_dir. --aug-pred-dir may then "
                         "be omitted (total augmentation stays converged); "
                         "give --tag in that case.")
    ap.add_argument("--spin-pred-dir", default=None,
                    help="Directory of ML spin-density grids (<mpid>_spin.npy.lz4), "
                         "passed through as --spin_pred_dir: replaces the "
                         "converged diff GRID.")
    ap.add_argument("--mag-only", action="store_true",
                    help="Passed through as --mag_only: run ONLY the magnetic "
                         "structures (DB criterion).")
    ap.add_argument("--suffix", default=None,
                    help="Optional moniker appended to the variant tag for a "
                         "VERSIONED rerun (e.g. --suffix v2 -> ml_aug_1k_v2). "
                         "Fresh runs.db keys / result CSVs / outdirs; old run "
                         "preserved. Combine with --rerun to scoped-clear the "
                         "suffixed tag before resubmitting.")
    ap.add_argument("--rerun", action="store_true",
                    help="IN-PLACE rerun under the SAME tag: scoped-clear this "
                         "variant's rows from runs.db (after a timestamped "
                         "backup) so finished structures are redone. Leaves all "
                         "other variants untouched. May be combined with "
                         "--suffix to rerun a suffixed tag.")
    ap.add_argument("--src-csv", default="exps/grid_conv_aug_sad.csv",
                    help="Source task CSV (CHGCAR_PATH,MP_ID) of non-magnetic "
                         "test-set rows (default: exps/grid_conv_aug_sad.csv).")
    ap.add_argument("--num-slices", type=int, default=10,
                    help="Slices for the experiment (default: 10 → 10 sbatch jobs total).")
    ap.add_argument("--partition", default="compute",
                    help="SLURM partition (default: compute).")
    ap.add_argument("--project-dir", default=None,
                    help="Project root (default: current working directory).")
    ap.add_argument("--runs-db", default="runs/runs.db",
                    help="SQLite cross-experiment manifest (shared safely across all jobs).")
    ap.add_argument("--tag", default=None,
                    help="Override the auto-derived variant tag base (use when "
                         "the pred-dir path has no plain `<N>k`/`full` component).")
    ap.add_argument("--grid-pred-dir", default=None,
                    help="Directory of ML-predicted total-density CHGCARs. "
                         "When given the total pseudo grid comes from that "
                         "model too, so grid AND augmentation are both ML.")
    ap.add_argument("--nonmag-only", action="store_true",
                    help="Skip magnetic structures (DB criterion), matching "
                         "submit_total_ml_pg.py's behaviour.")
    ap.add_argument("--total-only", action="store_true",
                    help="Converged TOTAL valence density + ML total "
                         "augmentation only; no diff grid or diff aug.")
    ap.add_argument("--magmom", type=float, default=None,
                    help="Uniform per-ion INCAR MAGMOM (e.g. --magmom 0.001). "
                         "Intended for use with --total-only.")
    ap.add_argument("--ispin1", action="store_true",
                    help="Force ISPIN=1 in the runner (total-only seed, no "
                         "MAGMOM). Pass a --tag that records it, e.g. "
                         "ml_aug_full_c3n4k_ispin1.")
    ap.add_argument("--relaxed-magmoms", default=None,
                    help="Passed through as --relaxed_magmoms: replace INCAR "
                         "MAGMOM with the harvested MP relax-task moments "
                         "(fallback to MP's own MAGMOM + magmom_source flag). "
                         "Mutually exclusive with --magmom/--ispin1.")
    ap.add_argument("--pbe-only", dest="pbe_only",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="With --relaxed-magmoms: only GGA/GGA+U relax "
                         "moments (default).")
    ap.add_argument("--renorm", action="store_true",
                    help="Scale the ML-predicted total grid to the converged "
                         "reference's integrated charge (NELECT). Only meaningful "
                         "with --grid-pred-dir. ChargE3Net predictions are not "
                         "normalized in advance; ElectraFi's are.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write slice CSVs + SBATCH files but skip `sbatch` submission.")
    args = ap.parse_args()

    project_dir = Path(args.project_dir or os.getcwd()).resolve()
    os.chdir(project_dir)

    if args.aug_pred_dir is None and args.mag_aug_pred_dir is None:
        print("FATAL: need --aug-pred-dir and/or --mag-aug-pred-dir.", file=sys.stderr)
        sys.exit(2)
    if args.mag_only and args.nonmag_only:
        print("FATAL: --mag-only and --nonmag-only are mutually exclusive.",
              file=sys.stderr)
        sys.exit(2)
    aug_pred_dir = Path(args.aug_pred_dir) if args.aug_pred_dir else ""
    mag_aug_pred_dir = Path(args.mag_aug_pred_dir) if args.mag_aug_pred_dir else ""
    spin_pred_dir = Path(args.spin_pred_dir) if args.spin_pred_dir else ""
    tag = args.tag or derive_tag(str(aug_pred_dir or mag_aug_pred_dir))
    if args.suffix:
        clean_suffix = re.sub(r"[^0-9A-Za-z]+", "_", args.suffix).strip("_")
        if not clean_suffix:
            print(f"FATAL: --suffix {args.suffix!r} is empty after sanitizing.",
                  file=sys.stderr)
            sys.exit(2)
        tag = f"{tag}_{clean_suffix}"
    results_prefix = f"results/{tag}/MP_{tag.upper()}"   # e.g. results/ml_aug_10k/MP_ML_AUG_10K
    outdir_prefix = f"{tag.upper()}_RUNS_MP"       # e.g. ML_AUG_10K_RUNS_MP
    runner = "run_ml_seed_mp.py"
    print(f"=== Experiment {tag}: aug_pred_dir={aug_pred_dir or '(converged)'} "
          f"mag_aug_pred_dir={mag_aug_pred_dir or '(converged)'} "
          f"spin_pred_dir={spin_pred_dir or '(converged)'} ===")

    if args.rerun:
        clear_variant_rows(project_dir / args.runs_db, tag, args.dry_run)

    slices_dir = project_dir / "exps" / "slices"
    jobs_dir = project_dir / "submissions" / "jobs"
    logs_dir = project_dir / "logs"
    for d in (slices_dir, jobs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    src = project_dir / args.src_csv
    if not src.is_file():
        print(f"FATAL: source CSV missing: {src}", file=sys.stderr)
        sys.exit(2)

    with src.open() as f:
        next(f)  # skip header
        total = sum(1 for _ in f)

    slice_size = (total + args.num_slices - 1) // args.num_slices
    print(f"{total} rows → {args.num_slices} slices of up to {slice_size}")

    for i in range(args.num_slices):
        start = i * slice_size
        end = min(start + slice_size, total)
        if start >= total:
            break

        slice_id = f"{i:02d}_{start}-{end}"
        slice_n = end - start

        slice_csv_abs = slices_dir / f"{tag}_{slice_id}.csv"
        split_csv(src, slice_csv_abs, start, end)
        slice_csv_rel = slice_csv_abs.relative_to(project_dir)

        slice_results = f"{results_prefix}_{slice_id}.csv"
        slice_outdir = f"{outdir_prefix}_{slice_id}"
        job_name = f"{tag}_{slice_id}"
        log_path = logs_dir / f"{tag}_{slice_id}.log"
        err_path = logs_dir / f"{tag}_{slice_id}.err"
        sbatch_path = jobs_dir / f"{tag}_{slice_id}.sh"

        script = SBATCH_TEMPLATE.format(
            partition=args.partition,
            job_name=job_name,
            log_path=log_path,
            err_path=err_path,
            project_dir=project_dir,
            csv=slice_csv_rel,
            outdir=slice_outdir,
            results_csv=slice_results,
            aug_pred_dir=aug_pred_dir,
            mag_aug_pred_dir=mag_aug_pred_dir,
            spin_pred_dir=spin_pred_dir,
            aug_pred_dir_arg=('--aug_pred_dir "$AUG_PRED_DIR" \\\n    '
                              if aug_pred_dir else ""),
            mag_aug_pred_dir_arg=('--mag_aug_pred_dir "$MAG_AUG_PRED_DIR" \\\n    '
                                  if mag_aug_pred_dir else ""),
            spin_pred_dir_arg=('--spin_pred_dir "$SPIN_PRED_DIR" \\\n    '
                               if spin_pred_dir else ""),
            variant=tag,
            runs_db=args.runs_db,
            num_runs=slice_n,
            end_row=slice_n,
            runner=runner,
            tag=tag,
            mode_arg=(f"--grid_pred_dir {args.grid_pred_dir} "
                      if args.grid_pred_dir else "")
                     + ("--nonmag_only " if args.nonmag_only else "")
                     + ("--mag_only " if args.mag_only else "")
                     + ("--total_only " if args.total_only else "")
                     + ("--ispin1 " if args.ispin1 else "")
                     + (f"--magmom {args.magmom} " if args.magmom is not None else "")
                     + ((f"--relaxed_magmoms {args.relaxed_magmoms} "
                         + ("--pbe-only " if args.pbe_only else "--no-pbe-only "))
                        if args.relaxed_magmoms else "")
                     + ("--renorm " if args.renorm else ""),
        )
        sbatch_path.write_text(script)
        sbatch_path.chmod(0o755)

        if args.dry_run:
            print(f"  [DRY] wrote {sbatch_path.relative_to(project_dir)} "
                  f"({slice_n} rows)")
            continue

        result = subprocess.run(
            ["sbatch", str(sbatch_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  FAIL submitting {sbatch_path.name}:\n"
                  f"    stdout: {result.stdout.strip()}\n"
                  f"    stderr: {result.stderr.strip()}",
                  file=sys.stderr)
        else:
            print(f"  submitted {sbatch_path.name}: "
                  f"{result.stdout.strip()}")

    print("Done.")


if __name__ == "__main__":
    main()
