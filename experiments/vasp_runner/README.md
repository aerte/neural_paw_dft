> **Moved.** The library is now `neural_paw_dft.vasp_runner` (install with `pip install -e ".[train,mp]"`
> from the repository root); the runner, submission and report scripts stay in this directory and run from here.

# clean_vasp_runner

Code used to measure how many SCF steps VASP needs when its starting charge
density is seeded from different sources: the built-in superposition of atomic
densities (SAD), the converged density (Oracle), and ML-predicted pieces of the
density (pseudo grid, PAW augmentation occupancies, spin channel, per-site
MAGMOM). The same protocol runs on Materials Project (MP) structures and on
GNoME structures.

Every run of a structure under one *variant* (seed recipe) becomes one row in
a SQLite manifest (`runs/runs.db` for MP, `runs/runs_gnome.db` for GNoME).
The report scripts aggregate those rows into the result tables.

## Layout

| Path | Role |
|---|---|
| `vasp_runner/` | Shared library: CHGCAR channel surgery (`chgcar.py`), SAD extraction, INCAR patching, OSZICAR parsing, per-site MAGMOM injection (`relaxmag.py`), ML spin-grid loader (`spin_npy.py`), the SQLite manifest (`runs_db.py`), and the two input sources (`sources/mp.py` rebuilds MP inputs from the MP API, `sources/gnome2.py` copies a reference run directory). |
| `run_default_mp.py` | MP baseline: ICHARG=2 (SAD), or SAD + injected per-site MAGMOM. |
| `run_true_init_mp.py` | MP Oracle: ICHARG=1 with the converged CHGCAR, plus the spin-init variants (`--total_only`, `--magmom`, ISPIN=1). |
| `run_conv_grid_sad_aug_mp.py` | Converged pseudo grid + SAD augmentation. |
| `run_sad_grid_conv_aug_mp.py` | SAD pseudo grid + converged augmentation. |
| `run_sad_diff_hybrid_mp.py` | Converged total channel + SAD spin (diff) channel. |
| `run_sad_total_hybrid_mp.py` | SAD total channel + converged spin channel. |
| `run_ml_seed_mp.py` | All ML-seeded MP variants: ML total grid and/or ML augmentation and/or ML spin grid and/or ML spin augmentation, with the remaining channels converged, plus the total-only + MAGMOM recipes. |
| `run_baseline_gnome.py` | GNoME baseline / Oracle / Oracle-total + MAGMOM. |
| `run_ml_seed_gnome.py` | GNoME ML-seeded variants (mirror of `run_ml_seed_mp.py`). |
| `submissions/submit_*.py` | One submitter per runner. Splits the task CSV into slices, writes one self-contained sbatch script per slice, submits it. |
| `scf_stats.py` | Magnetic / non-magnetic classification of MP ids from the Oracle rows in `runs.db` (used by the runners for `--nonmag-only` / `--mag-only` gates) and per-variant statistics. |
| `exps/report_iclr.py`, `exps/report_primary_iclr.py` | Build the result tables from `runs.db` / `runs_gnome.db` snapshots. |
| `make_tasks_mp.py`, `exps/make_gnome2_tasks.py` | Build the task CSVs. |
| `exps/harvest_*.py` | Build the per-site MAGMOM CSVs (MP-relaxed moments, converged "oracle" moments). |

## Requirements

* Python >= 3.13, VASP 5.4.4 (`vasp_std` on `PATH` inside the job), SLURM.
* Python dependencies are in `pyproject.toml` (`uv sync` or `pip install -e .`).
* Environment variables, read by every generated sbatch script:

| Variable | Meaning |
|---|---|
| `PMG_VASP_PSP_DIR` | pymatgen POTCAR directory (MP runners rebuild POTCARs from the MP task document). Required. |
| `MP_API_KEY` | Materials Project API key. Required by the MP runners and the MP harvest scripts (those also read it from a `.env` file in the project root). |
| `WANDB_API_KEY`, `WANDB_MODE` | Runners log to Weights & Biases. Defaults to `WANDB_MODE=offline`, so no key is needed. |
| `GNOME_REF_RUNS_ROOT` | Root of the GNoME reference run directories (`<gid>.chgcar/run_default/{INCAR,POSCAR,KPOINTS,POTCAR,CHGCAR,OUTCAR}`). |
| `GNOME_AUG_PRED_DIR`, `GNOME_ELECTRAFI_DIR`, `GNOME_CHARGE3NET_DIR` | Default ML prediction directories for `submit_gnome2.py` (all overridable on the command line). |

Each sbatch template has a placeholder comment where the site's `module load`
lines for VASP and Python go. Edit it once in each `submissions/submit_*.py`.
`--partition` defaults to `compute`; pass your partition name.

## Input files

None of the data is shipped. The scripts expect:

* **Task CSVs** with a header `CHGCAR_PATH,MP_ID` (MP) or `CHGCAR_PATH,GNOME_ID`
  (GNoME). `CHGCAR_PATH` is the converged reference CHGCAR (`.lz4` accepted
  for MP). Build with `make_tasks_mp.py` / `exps/make_gnome2_tasks.py`.
  The report scripts read `exps/gnome2_nonmag.csv` and `exps/gnome2_full.csv`
  to split GNoME into populations.
* **Per-site MAGMOM CSVs** (`--relaxed-magmoms`): one row per id, a `n_sites`
  column and a space-separated moment column named `magmom_chgnet`,
  `magmom_oracle` or `magmom_relaxed`, in POSCAR site order.
  `exps/harvest_relaxed_magmoms.py` and `exps/harvest_oracle_magmoms_mpapi.py`
  produce the MP ones from the MP API; `exps/harvest_oracle_magmoms.py`
  collects converged moments from a finished variant's `*_magnetization.jsonl`
  sidecars. CHGNet moments come from running CHGNet on the POSCARs (not included).
* **ML predictions**
  * total pseudo grid: one CHGCAR-format file per id in `--grid-pred-dir`
    (`--pred_dir` / `--electrafi-dir` for GNoME); add `--renorm` for models
    whose grid does not integrate to NELECT.
  * augmentation occupancies: `{id}_total_aug.npz` (AugNet export with
    `aug_sanvito_padded`, `schema_mask`, `mask`, `atomic_numbers`) in
    `--aug-pred-dir`; spin augmentation as `{id}_mag_aug.npz` in
    `--mag-aug-pred-dir`.
  * spin grid: `{id}_spin.npy.lz4`, float32 `(NGX,NGY,NGZ)` in e/Å³, in
    `--spin-pred-dir` (see `vasp_runner/spin_npy.py`).

## Running

All commands run from the project root on the cluster. Every submitter
accepts `--dry-run` (writes the slice CSVs and sbatch scripts, submits
nothing), `--num-slices N`, `--partition`, `--runs-db` and `--suffix` (versioned
rerun under `<tag>_<suffix>`).
Resubmitting is idempotent: ids already `ok` in the manifest are skipped, so a
walltime-killed slice is recovered by resubmitting the same command.

### MP baselines

```bash
# Default (SAD)                    -> variant "default"
python submissions/submit_default.py --src-csv tasks_mp.csv --num-slices 20
# SAD + per-site MAGMOM            -> "default_<suffix>"
python submissions/submit_default.py --relaxed-magmoms magmoms.csv --suffix chgnetmag --src-csv tasks_mp.csv

# Oracle (converged CHGCAR)        -> "true_init"
python submissions/submit_true_init.py --src-csv tasks_mp.csv --num-slices 20
# Oracle total + per-site MAGMOM   -> "true_init_total_<suffix>"
python submissions/submit_true_init.py --total-only --relaxed-magmoms magmoms.csv \
    --suffix chgnetmag --include-magnetic --src-csv tasks_mp.csv
# Magnetic classification helper (non-magnetic ids get a run, magnetic ids a
# skipped_magnetic row); the report scripts read this variant for populations.
python submissions/submit_true_init.py --magmom0p1 --src-csv tasks_mp.csv

# Channel hybrids (converged vs SAD by grid / augmentation / spin channel).
# These four read their task CSV from the `src_csv` entry in their EXPERIMENTS
# list (exps/default_tasks.csv, exps/grid_conv_aug_sad.csv, exps/grid_sad_aug_conv.csv); edit it or place
# your CSV at that path.
python submissions/submit_conv_grid_sad_aug.py   # conv grid + SAD aug   -> conv_grid_sad_aug
python submissions/submit_sad_grid_conv_aug.py   # SAD grid + conv aug   -> sad_grid_conv_aug
python submissions/submit_sad_diff_hybrid.py     # conv total + SAD diff -> sad_diff_hybrid
python submissions/submit_sad_total_hybrid.py    # SAD total + conv diff -> sad_total_hybrid
```

The SAD-based recipes extract the SAD density with a one-step VASP run and
cache it under `--sad-cache` (default `SAD_VASP_RUNS_MP/`).

### MP ML seeds (`submit_ml_aug.py` -> `run_ml_seed_mp.py`)

Any channel not given stays converged. `--tag` names the variant; the runner
prefixes `ml_aug_` unless the tag already starts with it.

```bash
AUG=/path/to/augnet_mp/aug_pred_dir_test_total
GRID=/path/to/electrafi_ref/pred_dens            # or a charge3net dir, with --renorm
SPIN=/path/to/spin_eval_grids/mp
AUGMAG=/path/to/augnet_mp_spin/aug_pred_dir_test_mag

# ML augmentation only (conv grid, conv spin)
python submissions/submit_ml_aug.py --aug-pred-dir $AUG --tag augnet_full --src-csv tasks_mp.csv --num-slices 20
# ML grid + ML augmentation, converged spin channel
python submissions/submit_ml_aug.py --aug-pred-dir $AUG --grid-pred-dir $GRID \
    --tag augnet_full_electrafi_ref_convspin --src-csv tasks_mp.csv --num-slices 20
# ML spin grid + ML spin augmentation, converged total
python submissions/submit_ml_aug.py --spin-pred-dir $SPIN --mag-aug-pred-dir $AUGMAG \
    --tag ml_aug_spinboth_convrest --src-csv tasks_mp.csv --num-slices 20
# Everything ML
python submissions/submit_ml_aug.py --aug-pred-dir $AUG --grid-pred-dir $GRID \
    --spin-pred-dir $SPIN --mag-aug-pred-dir $AUGMAG \
    --tag ml_aug_allml_electrafi_ref --src-csv tasks_mp.csv --num-slices 20
# ML grid + ML augmentation, no spin channel, spin from per-site MAGMOM
python submissions/submit_ml_aug.py --aug-pred-dir $AUG --grid-pred-dir $GRID \
    --total-only --relaxed-magmoms magmoms.csv \
    --tag augnet_full_electrafi_ref_chgnetmag --src-csv tasks_mp.csv --num-slices 20
```

`--nonmag-only` / `--mag-only` restrict a submission to one population
(classified from the Oracle rows already in `runs.db`).

Note: the GNoME "ML spin grid + ML spin augmentation, converged total" row
(`gnome_spinboth_convrest`) was submitted with a `--exp spinboth_convrest` option
that is not in this copy of `submit_gnome2.py`; the runner itself
(`run_ml_seed_gnome.py`) accepts `--spin_pred_dir`/`--mag_aug_pred_dir`
without a total prediction only if one of `--pred_dir`/`--aug_pred_dir` is
also given, so that recipe needs the submitter guard relaxed before use.

### GNoME (`submit_gnome2.py` -> `run_baseline_gnome.py` / `run_ml_seed_gnome.py`)

```bash
export GNOME_REF_RUNS_ROOT=/path/to/GNOME_REFERENCE_RUNS
AUGGN=/path/to/augnet_gnome/aug_pred_dir_test_total
E=/path/to/electrafi_ref/gnome_ecd
SPINGN=/path/to/spin_eval_grids/gnome
MAGGN=/path/to/augnet_gnome_spin/aug_pred_dir_test_mag

python submissions/submit_gnome2.py --exp default   --src-csv exps/gnome2_full.csv   # gnome_default
python submissions/submit_gnome2.py --exp true_init --src-csv exps/gnome2_full.csv   # gnome_true_init
python submissions/submit_gnome2.py --exp default --relaxed-magmoms magmoms_gnome.csv \
    --magmom-tag chgnetmag --src-csv exps/gnome2_full.csv                            # gnome_default_chgnetmag
python submissions/submit_gnome2.py --exp true_init_total --relaxed-magmoms magmoms_gnome.csv \
    --magmom-tag chgnetmag --src-csv exps/gnome2_full.csv                            # gnome_true_init_total_chgnetmag

# ML grid + ML augmentation, converged spin          -> gnome_mlaug_electrafi_convspin_<suffix>
python submissions/submit_gnome2.py --exp electrafi --conv-spin --aug-pred-dir $AUGGN \
    --electrafi-dir $E --suffix augnetfull --src-csv exps/gnome2_full.csv
# same with a charge3net grid (needs --renorm)       -> gnome_mlaug_charge3net_convspin_<suffix>
python submissions/submit_gnome2.py --exp charge3net --conv-spin --renorm --aug-pred-dir $AUGGN \
    --suffix augnetfull --src-csv exps/gnome2_full.csv
# ML augmentation only / ML grid only, converged rest
python submissions/submit_gnome2.py --exp augnet_convrest    --conv-spin --aug-pred-dir $AUGGN --suffix full --src-csv exps/gnome2_nonmag.csv
python submissions/submit_gnome2.py --exp electrafi_convrest --conv-spin --electrafi-dir $E    --suffix full --src-csv exps/gnome2_nonmag.csv
# everything ML                                       -> gnome_mlaug_electrafi_allml
python submissions/submit_gnome2.py --exp electrafi --conv-spin --aug-pred-dir $AUGGN \
    --electrafi-dir $E --spin-pred-dir $SPINGN --mag-aug-pred-dir $MAGGN --src-csv exps/gnome2_full.csv
# ML grid + ML augmentation, no spin channel, per-site MAGMOM
python submissions/submit_gnome2.py --exp electrafi --aug-pred-dir $AUGGN --electrafi-dir $E \
    --relaxed-magmoms magmoms_gnome.csv --magmom-tag chgnetmag --suffix augnetfull --src-csv exps/gnome2_full.csv
```

### Reports

Take a consistent snapshot of the manifests (they are written by running
jobs) and run the report scripts on the snapshots:

```bash
sqlite3 runs/runs.db       ".backup snap_mp.db"
sqlite3 runs/runs_gnome.db ".backup snap_gnome.db"
python exps/report_primary_iclr.py --db snap_mp.db --gnome_db snap_gnome.db --out primary_results.txt
python exps/report_iclr.py         --db snap_mp.db --gnome_db snap_gnome.db --out results_iclr.txt
python scf_stats.py --runs_db runs/runs.db --variants default true_init
```

The variant tags listed in `MP_ROWS`, `GNOME_ROWS` and `BASELINE_ROWS` at the
top of `exps/report_primary_iclr.py` are the ones the tables report; edit
those lists to match the tags you submitted.

## Outputs

* `runs/runs.db`, `runs/runs_gnome.db`: one row per `(id, variant)` with
  status, energy, total magnetisation, wall time and SCF step count.
* `results/<variant>/…csv` and `…_magnetization.jsonl`: per-slice result rows
  and the OUTCAR per-site moments of every finished run.
* `<VARIANT>_RUNS_MP/`, `<GNOME_VARIANT>_RUNS/`: VASP work directories, pruned
  to the small output files after each run.
* `logs/`, `submissions/jobs/`, `exps/slices/`: sbatch logs, generated job
  scripts and slice CSVs.
