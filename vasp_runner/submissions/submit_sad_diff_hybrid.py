#!/usr/bin/env python3
"""Submit all slices of the sad_diff_hybrid experiment.

Hybrid CHGCAR = converged total + SAD diff (built per-row by the runner).
Mirrors submit_all_decomp.py's per-slice SBATCH template (passes --sad_cache),
but for a single experiment.

Usage (from the project root on the cluster):
    python submissions/submit_sad_diff_hybrid.py
    python submissions/submit_sad_diff_hybrid.py --dry-run
    python submissions/submit_sad_diff_hybrid.py --num-slices 20
    python submissions/submit_sad_diff_hybrid.py --partition <name>
"""
import argparse
import re
import os
import subprocess
import sys
from pathlib import Path


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time=40:00:00
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
SAD_CACHE="{sad_cache}"
RUNS_DB="{runs_db}"
NUM_RUNS={num_runs}
START_ROW=0
END_ROW={end_row}
RUNNER="{runner}"
VARIANT={variant}

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

mkdir -p "$SAD_CACHE" || {{ echo "FATAL: mkdir $SAD_CACHE failed" >&2; exit 16; }}
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
SAD_CACHE=$SAD_CACHE RUNS_DB=$RUNS_DB START_ROW=$START_ROW END_ROW=$END_ROW NUM_RUNS=$NUM_RUNS"

for run in $(seq 1 "$NUM_RUNS"); do
  echo "Starting {tag} run #$run at $(date)"

  python "$RUNNER" \\
    --csv "$CSV" \\
    --outdir "$OUTDIR" \\
    --results_csv "$RESULTS_CSV" \\
    --sad_cache "$SAD_CACHE" \\
    --variant "$VARIANT" \\
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


EXPERIMENTS = [
    {
        "tag": "sad_diff_hybrid",
        "src_csv": "exps/default_tasks.csv",
        "runner": "run_sad_diff_hybrid_mp.py",
        "results_prefix": "MP_SAD_DIFF_HYBRID_SCF",
        "outdir_prefix": "SAD_DIFF_HYBRID_SCF_RUNS_MP",
    },
]


def split_csv(src: Path, dst: Path, start: int, end: int):
    if dst.exists():
        return
    with src.open() as f:
        header = next(f)
        rows = list(f)
    with dst.open("w") as f:
        f.write(header)
        f.writelines(rows[start:end])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--num-slices", type=int, default=10,
                    help="Slices for the experiment (default: 10 → 10 sbatch jobs total).")
    ap.add_argument("--partition", default="compute",
                    help="SLURM partition (default: compute).")
    ap.add_argument("--project-dir", default=None,
                    help="Project root (default: current working directory).")
    ap.add_argument("--sad-cache", default="SAD_VASP_RUNS_MP",
                    help="SAD CHGCAR cache dir (shared safely across all jobs).")
    ap.add_argument("--runs-db", default="runs/runs.db",
                    help="SQLite cross-experiment manifest (shared safely across all jobs).")
    ap.add_argument("--suffix", default=None,
                    help="Optional moniker appended to the variant tag for a "
                         "VERSIONED rerun (e.g. --suffix spinfix). Fresh runs.db "
                         "keys / result CSVs / outdirs; old run preserved.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write slice CSVs + SBATCH files but skip `sbatch` submission.")
    args = ap.parse_args()

    project_dir = Path(args.project_dir or os.getcwd()).resolve()
    os.chdir(project_dir)

    slices_dir = project_dir / "exps" / "slices"
    jobs_dir = project_dir / "submissions" / "jobs"
    logs_dir = project_dir / "logs"
    for d in (slices_dir, jobs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    experiments = EXPERIMENTS
    if args.suffix:
        clean_suffix = re.sub(r"[^0-9A-Za-z]+", "_", args.suffix).strip("_")
        if not clean_suffix:
            print(f"FATAL: --suffix {args.suffix!r} is empty after sanitizing.",
                  file=sys.stderr)
            sys.exit(2)
        experiments = [
            {**e, "tag": f"{e['tag']}_{clean_suffix}",
             "results_prefix": f"results/{e['tag']}_{clean_suffix}/"
                              f"{e['tag'].upper()}_{clean_suffix.upper()}",
             "outdir_prefix": f"{e['tag'].upper()}_{clean_suffix.upper()}_RUNS_MP"}
            for e in EXPERIMENTS
        ]

    for exp in experiments:
        tag = exp["tag"]
        src = project_dir / exp["src_csv"]
        if not src.is_file():
            print(f"FATAL: source CSV missing: {src}", file=sys.stderr)
            sys.exit(2)

        with src.open() as f:
            next(f)  # skip header
            total = sum(1 for _ in f)

        slice_size = (total + args.num_slices - 1) // args.num_slices
        print(f"=== Experiment {tag}: {total} rows → "
              f"{args.num_slices} slices of up to {slice_size} ===")

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

            slice_results = f"{exp['results_prefix']}_{slice_id}.csv"
            slice_outdir = f"{exp['outdir_prefix']}_{slice_id}"
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
                sad_cache=args.sad_cache,
                runs_db=args.runs_db,
                num_runs=slice_n,
                end_row=slice_n,
                runner=exp["runner"],
                variant=tag,
                tag=tag,
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
