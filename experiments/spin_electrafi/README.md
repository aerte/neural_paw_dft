> **Moved.** The code now lives in the installable package `neural_paw_dft.spin_electrafi`; install with
> `pip install -e ".[train]"` from the repository root (no separate `setup.py`; fairchem-core is no longer
> required). `train.py` / `eval_spin_constrained.py` are `neural_paw_dft/spin_electrafi/*.py`
> (`ndi-electrafi-train`), configs are packaged, checkpoints live in `trained_models/spin_electrafi/`.

# ELECTRAFI spin-density model

Joint prediction of the total charge density and the spin (charge-difference)
density of periodic crystals from structure alone. The backbone is EScAIP; the
readout places Gaussians per valence electron and evaluates them on the VASP grid.

This tree contains only the code used to train and evaluate the two spin-model
arms reported in the paper:

| arm | config | checkpoint |
|---|---|---|
| constrained (net moment rescaled to an external / oracle value) | `paper_runs/constrained_full_spin.yaml` | `trained_models/spin_density/constrained_spin_electrafi.ckpt` (5 epochs, step 1145500) |
| unconstrained (`spin_renorm: False`, no net moment at inference) | `paper_runs/unconstrained_full_spin.yaml` | `trained_models/spin_density/unconstrained_spin_electrafi.ckpt` (pre-divergence archive, step 200000) |

The two configs differ only in `spin_renorm` (and `model_name`/`run_name`). Both
are the full resolved configs the runs executed with, so a training run can be
reproduced with `--config` directly. The checkpoints are Lightning checkpoints
(`state_dict` plus optimizer state); they load strictly into `model.ELECTRAFI`.

`trained_models/total_density/` holds the charge-only (no spin channel) ELECTRAFI
models as flat state dicts (`torch.load` gives the state dict directly):
`ELECTRAFI_BEST.model_state_dict` and `ELECTRA_2026-01-07_logical-pond-118.model_state_dict`.

## Layout

```
train.py                    training / eval entry point (Lightning)
eval_spin_constrained.py    test-set evaluation of the spin channel with csv / oracle / no constraint
model/ELECTRAFI.py          the Lightning module: Gaussian readout, spin channel, losses, CHGCAR output
model/loss.py               DensityLoss (NMAE / MAE) and SpinDifferenceLoss
model/model_utils.py        KeOps Gaussian evaluation, plane-wave grid
model/data_utils.py         CHGCAR datasets (charge + spin grids)
model/escaip/               EScAIP backbone
tools/density_conversions.py  CHGCAR <-> tensors
tools/valence_picker.py     valence slot assignment
tools/atom_tools_mpfull.py  per-element valence tables (PAW setups)
tools/visualization.py      CHGCAR delta writer, Gaussian plots
utils/                      optimizers (Muon mix), model IO, path/file helpers, VASP loader
data_splits/                train/val/test id lists (mpfull2025 and subsets, cubic, ECD)
data_prep/                  Materials Project download script and id lists
eval_runs/                  SLURM eval wrappers and the CHGNet net-moment constraint CSVs
runs/                       SLURM training scripts for the two arms
paper_runs/                 the two resolved training configs
hpc_conf.yaml               base config the run scripts derive from
local_conf.yaml             small CPU config for the smoke test
```

## Installation

Python 3.11, CUDA 12.1 (plus cuDNN and cuTENSOR). With those on the path:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip uv
uv pip install -e .
```

`setup.py` lists the dependencies (torch, lightning, fairchem-core 1.10.0,
torch_geometric, torch_scatter, torch_cluster, pykeops, ase, lz4, dill, plotly).
KeOps must see CUDA; the run scripts assert this before starting because a
CPU-only KeOps silently segfaults minutes into a run.

## Data

Training expects the MP_FULL_2025 CHGCARs as `<dens_path>/<mp-id>.chgcar.lz4`
(lz4-compressed VASP CHGCAR with the magnetization block). Set `dens_path` in
the config (the shipped configs use the placeholder `/path/to/MP_FULL_2025`).
The train/val/test ids are in `data_splits/datasplits_mpfull2025.json`. The
out-of-distribution GNoME set is a directory of `.lz4` CHGCARs passed as
`--ood_path` to the eval script.

## Smoke test (CPU, ~minutes)

```
python test_spin_pipeline.py
```

Runs one train/val/test step on two lz4-compressed CHGCARs placed in `data/`
(`mp-8648.chgcar.lz4`, spin-polarised, and `mp-8648_nospin.chgcar.lz4`, the same
structure without the magnetization block; any pair with the same names works)
and checks both branches of the spin-loss gating.

## Training

Both arms were trained on one 140 GB GPU (240 GB host RAM) with SLURM; edit the
`#SBATCH` partition and module lines for your cluster first:

```
sbatch runs/full_spin.sh            # constrained arm
sbatch runs/full_spin_norenorm.sh   # unconstrained arm
```

Each script derives its config from `hpc_conf.yaml` into `runs/configs/` and
calls `python train.py --config <yaml>`; the resolved configs are identical to
the ones in `paper_runs/`, so

```
python train.py --config paper_runs/constrained_full_spin.yaml
```

is equivalent. Checkpoints go to `trained_models/checkpoints/<model_name>/`
(`last.ckpt`, plus a permanent `archive-*.ckpt` every 40k steps); `train.py`
resumes from the newest checkpoint in that directory automatically, and the
scripts requeue themselves on SIGUSR1 before the wall-clock limit. Set
`wandb: False` in the config to log without Weights & Biases.

## Inference / evaluation

`eval_spin_constrained.py` scores the spin and charge prediction per structure
and optionally dumps the predicted spin grids:

```
# constrained arm, net moment from CHGNet (unsigned site-moment sum per structure)
python eval_spin_constrained.py \
    --config paper_runs/constrained_full_spin.yaml \
    --ckpt trained_models/spin_density/constrained_spin_electrafi.ckpt \
    --constraint csv --constraint_csv eval_runs/chgnet_magmom_constraints_mp.csv --id_col MP_ID \
    --out_csv spin_eval_mp.csv [--save_grids spin_grids/mp]

# unconstrained arm: no constraint input at all
python eval_spin_constrained.py \
    --config paper_runs/unconstrained_full_spin.yaml \
    --ckpt trained_models/spin_density/unconstrained_spin_electrafi.ckpt \
    --constraint none --out_csv spin_eval_norenorm_mp.csv

# GNoME (out of distribution): add
    --ood_path /path/to/gnome_ecd --ood_name gnome_ecd
```

`--constraint oracle` uses the net moment of the reference grid (the training
signal) as an upper bound. Never run `csv`/`oracle` against the unconstrained
checkpoint: it was trained without the rescale. The SLURM wrappers
`eval_runs/eval_spin_chgnet_3090.sh <mp|gnome> <csv|oracle|none>` and
`eval_runs/eval_spin_norenorm200k_3090.sh <mp|gnome>` run exactly these
evaluations on a 24 GB GPU.

Output CSV columns per structure: `m_constraint`, `m_oracle_net`, `m_abs_sum`,
`m_pred_net`, `spin_nmae` (unitless, only defined where the reference
int|m| dv >= `spin_mag_min`), `spin_abs_err_e` (muB), `charge_err_pct`. The
dataset substitutes the next file when a CHGCAR fails to load, so deduplicate
by id before aggregating.

Predicted grids are written as `<id>_spin.npy.lz4`:

```python
import io, lz4.frame, numpy as np
m = np.load(io.BytesIO(lz4.frame.decompress(open(p, "rb").read())))   # (nx, ny, nz)
```
