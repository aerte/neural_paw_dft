#!/usr/bin/env python3
"""Submit the GNOME_2 SCF experiments (new dataset category, own runs.db).

All four experiments run on the SAME population — the non-magnetic, Yb-free
1254 built by `exps/make_gnome2_tasks.py` — and take their POSCAR/INCAR/KPOINTS/
POTCAR verbatim from the reference run directories, so step counts are
directly comparable across variants.

  default      ICHARG=2, no seed.                          -> gnome_default
  true_init    ICHARG=1, converged reference CHGCAR.       -> gnome_true_init
  electrafi    ML total grid (ELECTRAFI-ref) + AugNet aug  -> gnome_mlaug_electrafi_magmom0p001
               + uniform MAGMOM, no diff channel.
  charge3net   Same, with the Charge3Net total grid.       -> gnome_mlaug_charge3net_magmom0p001

Usage (from the project root on the cluster):
    python submissions/submit_gnome2.py --exp default --num-slices 20 --dry-run
    python submissions/submit_gnome2.py --exp default --num-slices 20
    python submissions/submit_gnome2.py --exp true_init --num-slices 20
    python submissions/submit_gnome2.py --exp electrafi --num-slices 20
    python submissions/submit_gnome2.py --exp charge3net --num-slices 20
    python submissions/submit_gnome2.py --exp all --num-slices 20
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# Site-specific defaults; override on the command line or via the environment.
REF_RUNS_ROOT = os.environ.get("GNOME_REF_RUNS_ROOT", "/path/to/GNOME_REFERENCE_RUNS")
AUG_PRED_DIR = os.environ.get("GNOME_AUG_PRED_DIR", "/path/to/augnet_gnome/aug_pred_dir_test_total")
ELECTRAFI_DIR = os.environ.get("GNOME_ELECTRAFI_DIR", "/path/to/electrafi_ref/gnome_ecd")
CHARGE3NET_DIR = os.environ.get("GNOME_CHARGE3NET_DIR", "/path/to/charge3net/gnome")

BASELINE_RUNNER = "run_baseline_gnome.py"
ML_SEED_RUNNER = "run_ml_seed_gnome.py"


def magmom_tag(value: float) -> str:
    """0.001 -> '0p001' (the tag convention the MP MAGMOM variants use)."""
    return ("%g" % value).replace(".", "p").replace("-", "m")


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 24
#SBATCH --time={walltime}
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
REF_RUNS_ROOT="{ref_runs_root}"
RUNS_DB="{runs_db}"
VARIANT={variant}
NUM_RUNS={num_runs}
START_ROW=0
END_ROW={end_row}
RUNNER="{runner}"

# POTCARs are copied verbatim out of the reference runs, so no MP API key and no
# pseudopotential lookup is needed here — unlike every MP submit script.
: "${{WANDB_API_KEY:=}}"
: "${{WANDB_MODE:=offline}}"
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
if [[ ! -d "$REF_RUNS_ROOT" ]]; then
  echo "FATAL: REF_RUNS_ROOT does not exist: $REF_RUNS_ROOT" >&2
  exit 14
fi
{pred_dir_checks}
runs_db_parent="$(dirname "$RUNS_DB")"
mkdir -p "$runs_db_parent" || {{ echo "FATAL: mkdir $runs_db_parent failed" >&2; exit 18; }}
results_parent="$(dirname "$RESULTS_CSV")"
mkdir -p "$results_parent" || {{ echo "FATAL: mkdir $results_parent failed" >&2; exit 19; }}

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
    --ref_runs_root "$REF_RUNS_ROOT" \\
    --variant "$VARIANT" \\
    --runs_db "$RUNS_DB" \\
    --start_row "$START_ROW" \\
    --end_row "$END_ROW" \\
    {mode_args}--vasp_cmd srun vasp_std

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

PRED_DIR_CHECK = """\
if [[ ! -d "{path}" ]]; then
  echo "FATAL: {label} does not exist: {path}" >&2
  exit {code}
fi
"""


def experiments(magmom: float, renorm: bool = False, aug_pred_dir: str = AUG_PRED_DIR,
                electrafi_dir: str = ELECTRAFI_DIR, relaxed_magmoms: str = None,
                conv_spin: bool = False, spin_pred_dir: str = None,
                mag_aug_pred_dir: str = None, magmom_tag: str = "chgnetmag"):
    """Definition of the four GNOME_2 experiments.

    `renorm` adds `--renorm` to the two ML-seed experiments, rescaling the
    predicted total grid to the reference's integrated charge (NELECT). It is
    meaningless for the baselines, which carry no predicted grid.

    `aug_pred_dir` selects which AugNet model supplies the augmentation and
    `electrafi_dir` which ELECTRAFI model supplies the total grid; pair a
    non-default one with `--suffix` so it lands under its own variant tag.

    `relaxed_magmoms` (a per-site prediction CSV, e.g. CHGNet's) replaces the
    uniform MAGMOM on the two ML-seed experiments — the uniform value stays
    only as the fallback for gids missing from the CSV — and swaps the
    `magmom{mtag}` piece of their tags for `chgnetmag`. On the `default`
    baseline it swaps the reference INCAR's MAGMOM instead (reference MAGMOM
    as fallback) under the tag `gnome_default_chgnetmag`; `true_init` ignores
    it (a converged two-channel seed overrides the MAGMOM anyway).
    """
    if conv_spin and (spin_pred_dir or mag_aug_pred_dir):
        # The diff channel is (partly) ML too, so "convspin" would misname it.
        mtag = "allml"
    elif conv_spin:
        mtag = "convspin"
    else:
        mtag = magmom_tag if relaxed_magmoms else f"magmom{magmom_tag(magmom)}"
    rn = "--renorm " if renorm else ""
    rm = f'--relaxed_magmoms "{relaxed_magmoms}" ' if relaxed_magmoms else ""
    cs = "--conv_spin " if conv_spin else ""
    cs += f'--spin_pred_dir "{spin_pred_dir}" ' if spin_pred_dir else ""
    cs += f'--mag_aug_pred_dir "{mag_aug_pred_dir}" ' if mag_aug_pred_dir else ""
    return {
        "default": {
            # With --relaxed-magmoms the SAD baseline runs with the predicted
            # per-site MAGMOM (reference MAGMOM kept as fallback) under its
            # own tag, leaving gnome_default untouched.
            "tag": f"gnome_default_{mtag}" if relaxed_magmoms else "gnome_default",
            "runner": BASELINE_RUNNER,
            "mode_args": "--mode default " + rm,
            "pred_dirs": {},
        },
        "true_init": {
            "tag": "gnome_true_init",
            "runner": BASELINE_RUNNER,
            "mode_args": "--mode true_init ",
            "pred_dirs": {},
        },
        "true_init_total": {
            # Converged total grid + total augmentation, no diff channel; the
            # spin init comes from the predicted per-site MAGMOM. Requires
            # --relaxed-magmoms; not part of --exp all.
            "tag": f"gnome_true_init_total_{mtag}",
            "runner": BASELINE_RUNNER,
            "mode_args": "--mode true_init_total " + rm,
            "pred_dirs": {},
        },
        "electrafi": {
            "tag": f"gnome_mlaug_electrafi_{mtag}",
            "runner": ML_SEED_RUNNER,
            "mode_args": (f'--pred_dir "{electrafi_dir}" '
                          f'--aug_pred_dir "{aug_pred_dir}" '
                          f"--magmom {magmom} " + cs + rn + rm),
            "pred_dirs": {"PRED_DIR": electrafi_dir, "AUG_PRED_DIR": aug_pred_dir},
        },
        # Partial seeds against a converged remainder (require --conv-spin):
        # only the total augmentation is ML, or only the total grid is.
        "augnet_convrest": {
            "tag": "gnome_augnet_convrest",
            "runner": ML_SEED_RUNNER,
            "mode_args": (f'--aug_pred_dir "{aug_pred_dir}" ' + cs + rn),
            "pred_dirs": {"AUG_PRED_DIR": aug_pred_dir},
        },
        "electrafi_convrest": {
            "tag": "gnome_electrafi_convrest",
            "runner": ML_SEED_RUNNER,
            "mode_args": (f'--pred_dir "{electrafi_dir}" ' + cs + rn),
            "pred_dirs": {"PRED_DIR": electrafi_dir},
        },
        "charge3net": {
            "tag": f"gnome_mlaug_charge3net_{mtag}",
            "runner": ML_SEED_RUNNER,
            "mode_args": (f'--pred_dir "{CHARGE3NET_DIR}" '
                          f'--aug_pred_dir "{aug_pred_dir}" '
                          f"--magmom {magmom} " + cs + rn + rm),
            "pred_dirs": {"PRED_DIR": CHARGE3NET_DIR, "AUG_PRED_DIR": aug_pred_dir},
        },
    }


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
    """Scoped clear: DELETE only this variant's rows, after a timestamped backup."""
    if not runs_db_path.is_file():
        print(f"  [rerun] runs.db not found at {runs_db_path}; nothing to clear.")
        return
    sys.path.insert(0, os.getcwd())
    from neural_init.vasp_runner import runs_db
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


def submit_experiment(exp, args, project_dir: Path, total: int, src: Path):
    tag = exp["tag"]
    if args.suffix:
        clean = re.sub(r"[^0-9A-Za-z]+", "_", args.suffix).strip("_")
        if not clean:
            print(f"FATAL: --suffix {args.suffix!r} is empty after sanitizing.",
                  file=sys.stderr)
            sys.exit(2)
        tag = f"{tag}_{clean}"

    print(f"=== Experiment {tag} (runner={exp['runner']}) ===")

    if args.rerun:
        clear_variant_rows(project_dir / args.runs_db, tag, args.dry_run)

    checks = "".join(
        PRED_DIR_CHECK.format(label=label, path=path, code=15 + i)
        for i, (label, path) in enumerate(exp["pred_dirs"].items())
    )

    slices_dir = project_dir / "exps" / "slices"
    jobs_dir = project_dir / "submissions" / "jobs"
    logs_dir = project_dir / "logs"
    for d in (slices_dir, jobs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    # tag already carries the `gnome_` prefix, so no dataset prefix is added here.
    results_prefix = f"results/{tag}/{tag.upper()}"
    outdir_prefix = f"{tag.upper()}_RUNS"

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

        log_path = logs_dir / f"{tag}_{slice_id}.log"
        err_path = logs_dir / f"{tag}_{slice_id}.err"
        sbatch_path = jobs_dir / f"{tag}_{slice_id}.sh"

        script = SBATCH_TEMPLATE.format(
            walltime=args.walltime,
            partition=args.partition,
            job_name=f"{tag}_{slice_id}",
            log_path=log_path,
            err_path=err_path,
            project_dir=project_dir,
            csv=slice_csv_rel,
            outdir=f"{outdir_prefix}_{slice_id}",
            results_csv=f"{results_prefix}_{slice_id}.csv",
            ref_runs_root=REF_RUNS_ROOT,
            runs_db=args.runs_db,
            variant=tag,
            num_runs=slice_n,
            end_row=slice_n,
            runner=exp["runner"],
            tag=tag,
            pred_dir_checks=checks,
            mode_args=exp["mode_args"],
        )
        sbatch_path.write_text(script)
        sbatch_path.chmod(0o755)

        if args.dry_run:
            print(f"  [DRY] wrote {sbatch_path.relative_to(project_dir)} "
                  f"({slice_n} rows)")
            continue

        result = subprocess.run(["sbatch", str(sbatch_path)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAIL submitting {sbatch_path.name}:\n"
                  f"    stdout: {result.stdout.strip()}\n"
                  f"    stderr: {result.stderr.strip()}", file=sys.stderr)
        else:
            print(f"  submitted {sbatch_path.name}: {result.stdout.strip()}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True,
                    choices=("default", "true_init", "true_init_total",
                             "electrafi", "charge3net",
                             "augnet_convrest", "electrafi_convrest", "all"),
                    help="Which experiment to submit ('all' submits the "
                         "original four; true_init_total must be named "
                         "explicitly and requires --relaxed-magmoms).")
    ap.add_argument("--magmom", type=float, default=0.001,
                    help="Uniform MAGMOM for the two ML-seed experiments "
                         "(default: 0.001; also fixes their variant tag).")
    ap.add_argument("--spin-pred-dir", default=None,
                    help="Directory of ML spin-density grids (<gid>_spin.npy.lz4), "
                    "passed through as --spin_pred_dir (ML diff GRID). Requires "
                    "--conv-spin; retags the ML-seed experiments `_allml`.")
    ap.add_argument("--mag-aug-pred-dir", default=None,
                    help="Directory of {gid}_mag_aug.npz (AugNet spin export), "
                    "passed through as --mag_aug_pred_dir (ML diff AUG). Requires "
                    "--conv-spin; retags the ML-seed experiments `_allml`.")
    ap.add_argument("--aug-pred-dir", default=AUG_PRED_DIR,
                    help="AugNet augmentation prediction directory for the two "
                         "ML-seed experiments (default: the full-data model). "
                         "Pair a per-split model with --suffix 1k/10k/50k so it "
                         "lands under its own variant tag.")
    ap.add_argument("--electrafi-dir", default=ELECTRAFI_DIR,
                    help="ELECTRAFI total-grid prediction directory for --exp "
                         "electrafi (default: the ElectraFi-ref export). "
                         "Pair a per-split model with --suffix so it lands under "
                         "its own variant tag.")
    ap.add_argument("--conv-spin", action="store_true",
                    help="ML-seed experiments only: keep the CONVERGED spin "
                         "channel (diff grid + diff augmentation verbatim from "
                         "the reference), so the seed is ML in the total grid "
                         "and total augmentation only. Tags become "
                         "gnome_mlaug_<model>_convspin. Mutually exclusive "
                         "with --relaxed-magmoms.")
    ap.add_argument("--relaxed-magmoms", default=None,
                    help="Per-site MAGMOM prediction CSV (e.g. "
                         "chgnet_preds/chgnet_pred_gnome.csv) for the two "
                         "ML-seed experiments; replaces the uniform --magmom "
                         "(kept only as fallback) and retags them "
                         "gnome_mlaug_*_chgnetmag.")
    ap.add_argument("--magmom-tag", default="chgnetmag",
                    help="Tag piece naming the --relaxed-magmoms source "
                         "(default chgnetmag; e.g. oraclemag for "
                         "exps/oracle_magmoms_gnome.csv).")
    ap.add_argument("--src-csv", default="exps/gnome2_nonmag.csv",
                    help="Task CSV (CHGCAR_PATH,GNOME_ID) built by "
                         "exps/make_gnome2_tasks.py.")
    ap.add_argument("--num-slices", type=int, default=20,
                    help="Slices per experiment (default: 20 → 20 sbatch jobs).")
    ap.add_argument("--partition", default="compute",
                    help="SLURM partition (default: compute).")
    ap.add_argument("--walltime", default="24:00:00",
                    help="SBATCH --time (default: 24:00:00). Do NOT use the "
                         "partition maximum of 50:00:00: such jobs are never "
                         "backfilled and can sit pending for days.")
    ap.add_argument("--project-dir", default=None,
                    help="Project root (default: current working directory).")
    ap.add_argument("--runs-db", default="runs/runs_gnome.db",
                    help="SQLite manifest for the GNOME category "
                         "(separate from the MP runs/runs.db).")
    ap.add_argument("--suffix", default=None,
                    help="Moniker appended to every variant tag for a VERSIONED "
                         "rerun (fresh runs.db keys / result CSVs / outdirs).")
    ap.add_argument("--rerun", action="store_true",
                    help="IN-PLACE rerun under the SAME tag: scoped-clear this "
                         "variant's rows from runs.db after a timestamped backup.")
    ap.add_argument("--renorm", action="store_true",
                    help="Rescale the predicted total grid so it integrates to "
                         "NELECT before seeding (ML-seed experiments only). "
                         "Charge3Net predictions are not normalized in advance; "
                         "ElectraFi's are. Pair with --suffix renorm so the "
                         "result lands under its own variant tag.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Write slice CSVs + SBATCH files but skip submission.")
    args = ap.parse_args()

    project_dir = Path(args.project_dir or os.getcwd()).resolve()
    os.chdir(project_dir)

    src = project_dir / args.src_csv
    if not src.is_file():
        print(f"FATAL: source CSV missing: {src}\n"
              f"       Build it with: python exps/make_gnome2_tasks.py",
              file=sys.stderr)
        sys.exit(2)

    with src.open() as f:
        next(f)  # skip header
        total = sum(1 for _ in f)

    if args.relaxed_magmoms and not (project_dir / args.relaxed_magmoms).is_file():
        print(f"FATAL: --relaxed-magmoms CSV missing: "
              f"{project_dir / args.relaxed_magmoms}", file=sys.stderr)
        sys.exit(2)

    if args.exp == "true_init_total" and not args.relaxed_magmoms:
        print("FATAL: --exp true_init_total requires --relaxed-magmoms "
              "(its spin init).", file=sys.stderr)
        sys.exit(2)

    if args.conv_spin and args.relaxed_magmoms:
        print("FATAL: --conv-spin and --relaxed-magmoms are mutually exclusive "
              "(a converged spin channel makes MAGMOM irrelevant).",
              file=sys.stderr)
        sys.exit(2)
    CONVREST = ("augnet_convrest", "electrafi_convrest")
    if args.conv_spin and args.exp not in ("electrafi", "charge3net") + CONVREST:
        print("FATAL: --conv-spin applies to the ML-seed experiments only "
              "(--exp electrafi / charge3net / *_convrest).", file=sys.stderr)
        sys.exit(2)
    if args.exp in CONVREST and not args.conv_spin:
        print(f"FATAL: --exp {args.exp} is a partial seed against a converged "
              f"remainder; it requires --conv-spin.", file=sys.stderr)
        sys.exit(2)

    if (args.spin_pred_dir or args.mag_aug_pred_dir) and not args.conv_spin:
        print("FATAL: --spin-pred-dir/--mag-aug-pred-dir require --conv-spin.",
              file=sys.stderr)
        sys.exit(2)
    defs = experiments(args.magmom, args.renorm, args.aug_pred_dir,
                       args.electrafi_dir, args.relaxed_magmoms, args.conv_spin,
                       args.spin_pred_dir, args.mag_aug_pred_dir,
                       args.magmom_tag)
    # `all` stays the original four: true_init_total and the *_convrest partial
    # seeds must be named explicitly (they need --relaxed-magmoms / --conv-spin).
    names = ([n for n in defs if n not in ("true_init_total",) + CONVREST]
             if args.exp == "all" else [args.exp])
    for name in names:
        submit_experiment(defs[name], args, project_dir, total, src)

    print("Done.")


if __name__ == "__main__":
    main()
