#!/usr/bin/env python3
"""Submit all slices of the true_init (ICHARG=1 + converged CHGCAR) experiment.

Same shape as submit_default.py: splits the input CSV into NUM_SLICES sub-CSVs
under exps/slices/ and writes one self-contained SBATCH script per slice into
submissions/jobs/, then submits via `sbatch`. Re-running is idempotent.

Pass `--ispin1` for the strictly non-magnetic Oracle: the converged reference
CHGCAR is stripped to a total-only seed (no diff grid, no diff augmentation) and
ISPIN=1 is forced in the INCAR, with magnetic structures skipped via the
database criterion. That is the exact control for `total_ml_pg_ispin1` — same
seed construction with the converged pseudo grid in place of the ML-predicted
one. It runs under the distinct `true_init_ispin1` tag — its own runs.db
variant, result CSVs, slice CSVs and outdirs — so the ISPIN=2 Oracle is untouched.

`--zero-mag`, `--noise-mag` and `--magmom0p1` are the ISPIN=2 spin-difference
stand-in experiments: converged total pseudo grid + converged total augmentation
in every case, with only the spin channel varying (explicit zeros / near-zero
noise / no diff channel plus MAGMOM=0.1 per ion). Like `--ispin1` they skip
magnetic structures, so all of them cover the Oracle's non-magnetic population
and pair 1:1 with it. Tags: `true_init_zeromag`, `true_init_noisemag`,
`true_init_magmom0p1`.

Usage (from the project root on the cluster):
    python submissions/submit_true_init.py
    python submissions/submit_true_init.py --ispin1
    python submissions/submit_true_init.py --zero-mag
    python submissions/submit_true_init.py --noise-mag
    python submissions/submit_true_init.py --magmom0p1
    python submissions/submit_true_init.py --dry-run
    python submissions/submit_true_init.py --num-slices 20
    python submissions/submit_true_init.py --partition <name>
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# Mirrors submit_default.py's SBATCH template (no --sad_cache; the runner
# reads the converged CHGCAR directly from CHGCAR_PATH).
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
RUNS_DB=$RUNS_DB START_ROW=$START_ROW END_ROW=$END_ROW NUM_RUNS=$NUM_RUNS"

for run in $(seq 1 "$NUM_RUNS"); do
  echo "Starting {tag} run #$run at $(date)"

  python "$RUNNER" \\
    --csv "$CSV" \\
    --outdir "$OUTDIR" \\
    --results_csv "$RESULTS_CSV" \\
    --runs_db "$RUNS_DB" \\
    --variant "$VARIANT" \\
    --start_row "$START_ROW" \\
    --end_row "$END_ROW" \\
    {mode_arg}--vasp_cmd srun vasp_std

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
        "tag": "true_init",
        "src_csv": "exps/default_tasks.csv",
        "runner": "run_true_init_mp.py",
        "results_prefix": "MP_TRUE_INIT_SCF",
        "outdir_prefix": "TRUE_INIT_SCF_RUNS_MP",
    },
]

# --ispin1 counterpart: total-only seed from the converged CHGCAR, ISPIN=1
# forced, magnetic structures skipped. Recorded under its own `true_init_ispin1`
# variant tag so it never collides with the ISPIN=2 Oracle in runs.db / results
# / slice CSVs.
EXPERIMENTS_ISPIN1 = [
    {
        "tag": "true_init_ispin1",
        "src_csv": "exps/default_tasks.csv",
        "runner": "run_true_init_mp.py",
        "results_prefix": "results/true_init_ispin1_scf/MP_TRUE_INIT_ISPIN1_SCF",
        "outdir_prefix": "TRUE_INIT_ISPIN1_SCF_RUNS_MP",
    },
]


def spin_experiment(tag: str):
    """One ISPIN=2 spin-difference stand-in experiment, isolated under `tag`."""
    up = tag.upper()
    return [
        {
            "tag": tag,
            "src_csv": "exps/default_tasks.csv",
            "runner": "run_true_init_mp.py",
            "results_prefix": f"results/{tag}_scf/MP_{up}_SCF",
            "outdir_prefix": f"{up}_SCF_RUNS_MP",
        },
    ]


def suffixed_experiment(exp: dict, suffix: str) -> dict:
    """Re-tag `exp` for a VERSIONED rerun, so it gets its own runs.db variant,
    result CSVs, slice CSVs and outdirs and never collides with the original.
    Uses the same naming scheme as `spin_experiment`."""
    tag = f"{exp['tag']}_{suffix}"
    up = tag.upper()
    return {
        **exp,
        "tag": tag,
        "results_prefix": f"results/{tag}_scf/MP_{up}_SCF",
        "outdir_prefix": f"{up}_SCF_RUNS_MP",
    }


def split_csv(src: Path, dst: Path, start: int, end: int):
    """Write data rows [start, end) from `src` (plus its header) to `dst`.
    No-op if `dst` already exists."""
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
    ap.add_argument("--runs-db", default="runs/runs.db",
                    help="SQLite cross-experiment manifest (shared safely across all jobs).")
    ap.add_argument("--ispin1", action="store_true",
                    help="Run the strictly non-magnetic Oracle: total-only seed "
                         "stripped from the converged CHGCAR, ISPIN=1 forced in "
                         "the INCAR (MAGMOM/NUPDOWN dropped), magnetic structures "
                         "skipped via the database criterion. Recorded under the "
                         "distinct `true_init_ispin1` tag, so it does not collide "
                         "with the ISPIN=2 Oracle.")
    ap.add_argument("--src-csv", default=None,
                    help="Override the experiment's source task CSV "
                         "(CHGCAR_PATH,MP_ID), e.g. a magnetic-only subset.")
    ap.add_argument("--include-magnetic", action="store_true",
                    help="With a spin-init mode: run magnetic structures too "
                         "instead of skipping them (same tag; rows already "
                         "terminal in runs.db stay skipped).")
    ap.add_argument("--zero-mag", action="store_true",
                    help="ISPIN=2, converged total grid + total augmentation with "
                         "an explicit ZERO diff grid and zero diff augmentation. "
                         "Magnetic structures skipped. Tag `true_init_zeromag`.")
    ap.add_argument("--noise-mag", action="store_true",
                    help="Like --zero-mag but with a near-zero symmetry-broken diff "
                         "channel instead of strict zeros. Tag `true_init_noisemag`.")
    ap.add_argument("--magmom0p1", action="store_true",
                    help="ISPIN=2, total-only converged seed (no diff channel) with "
                         "MAGMOM=0.1 on every ion, replacing MP's own MAGMOM. "
                         "Magnetic structures skipped. Tag `true_init_magmom0p1`.")
    ap.add_argument("--magmom", type=float, default=None,
                    help="Like --magmom0p1 but for an arbitrary value, e.g. "
                         "--magmom 0.001 -> tag `true_init_magmom0p001`.")
    ap.add_argument("--total-only", action="store_true",
                    help="ISPIN=2, total-only converged seed (no diff channel) "
                         "with the per-site MAGMOM from --relaxed-magmoms as "
                         "the spin init (required). Base tag `true_init_total`; "
                         "pair with --suffix, e.g. --suffix chgnetmag.")
    ap.add_argument("--suffix", default=None,
                    help="Optional moniker appended to the variant tag for a "
                         "VERSIONED rerun (e.g. --suffix spinfix -> "
                         "true_init_zeromag_spinfix). Fresh runs.db keys / "
                         "result CSVs / outdirs; the old run is preserved for "
                         "a paired before/after comparison.")
    ap.add_argument("--relaxed-magmoms", default=None,
                    help="Passed through as --relaxed_magmoms: replace INCAR "
                         "MAGMOM with the harvested MP relax-task moments "
                         "(fallback to MP's own MAGMOM + magmom_source flag). "
                         "Mutually exclusive with the spin-init modes; use "
                         "with --suffix, e.g. --suffix relaxmag.")
    ap.add_argument("--pbe-only", dest="pbe_only",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="With --relaxed-magmoms: only GGA/GGA+U relax "
                         "moments (default).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write slice CSVs + SBATCH files but skip `sbatch` submission.")
    args = ap.parse_args()

    modes = (args.ispin1, args.zero_mag, args.noise_mag, args.magmom0p1,
             args.magmom is not None, args.total_only)
    if sum(bool(m) for m in modes) > 1:
        print("FATAL: --ispin1, --zero-mag, --noise-mag, --magmom0p1, "
              "--magmom and --total-only are mutually exclusive.", file=sys.stderr)
        sys.exit(2)
    if args.total_only and not args.relaxed_magmoms:
        print("FATAL: --total-only requires --relaxed-magmoms (its spin init).",
              file=sys.stderr)
        sys.exit(2)

    if args.ispin1:
        experiments, mode_arg = EXPERIMENTS_ISPIN1, "--ispin1 "
    elif args.zero_mag:
        experiments, mode_arg = spin_experiment("true_init_zeromag"), "--zero_mag "
    elif args.noise_mag:
        experiments, mode_arg = spin_experiment("true_init_noisemag"), "--noise_mag "
    elif args.magmom0p1:
        experiments, mode_arg = spin_experiment("true_init_magmom0p1"), "--magmom 0.1 "
    elif args.magmom is not None:
        # Tag convention matches the existing runs: 0.001 -> true_init_magmom0p001.
        # Fixed-point, never exponent notation: repr(1e-05) would give `1e-05`.
        decimal = f"{args.magmom:.12f}".rstrip("0").rstrip(".")
        tag = f"true_init_magmom{decimal.replace('.', 'p')}"
        experiments, mode_arg = spin_experiment(tag), f"--magmom {decimal} "
    elif args.total_only:
        experiments, mode_arg = spin_experiment("true_init_total"), "--total_only "
    else:
        experiments, mode_arg = EXPERIMENTS, ""
    if args.relaxed_magmoms:
        if any(modes) and not args.total_only:
            print("FATAL: --relaxed-magmoms is mutually exclusive with the "
                  "spin-init modes.", file=sys.stderr)
            sys.exit(2)
        mode_arg += (f"--relaxed_magmoms {args.relaxed_magmoms} "
                     + ("--pbe-only " if args.pbe_only else "--no-pbe-only "))
    if args.include_magnetic:
        mode_arg += "--include-magnetic "

    if args.suffix:
        clean_suffix = re.sub(r"[^0-9A-Za-z]+", "_", args.suffix).strip("_")
        if not clean_suffix:
            print(f"FATAL: --suffix {args.suffix!r} is empty after sanitizing.",
                  file=sys.stderr)
            sys.exit(2)
        experiments = [suffixed_experiment(e, clean_suffix) for e in experiments]

    project_dir = Path(args.project_dir or os.getcwd()).resolve()
    os.chdir(project_dir)

    slices_dir = project_dir / "exps" / "slices"
    jobs_dir = project_dir / "submissions" / "jobs"
    logs_dir = project_dir / "logs"
    for d in (slices_dir, jobs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    for exp in experiments:
        tag = exp["tag"]
        src = project_dir / (args.src_csv or exp["src_csv"])
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
                runs_db=args.runs_db,
                variant=tag,
                num_runs=slice_n,
                end_row=slice_n,
                runner=exp["runner"],
                tag=tag,
                mode_arg=mode_arg,
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
