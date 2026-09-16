> **Moved.** The code now lives in the installable package `neural_paw_dft.augnet`; install with
> `pip install -e ".[train]"` from the repository root (see the root README). The commands below run from
> this directory and refer to packaged data as `neural_paw_dft/augnet/{configs,stats,data_splits}`;
> checkpoints live in `trained_models/augnet/`.

# AugNet — PAW augmentation-charge prediction with a MACE backbone

AugNet predicts VASP PAW augmentation occupancies (the `augmentation occupancies`
blocks of a CHGCAR) directly from structure. A MACE backbone (upstream
`mace-torch`) feeds an element-agnostic, equivariant readout head that emits every
atom's occupancy matrix in projector space. The default model predicts the delta
from the free-atom occupancies (total channel); a second config trains the spin
(magnetization) channel.

## Layout

```
train.py / eval.py        CLI entry points (eval.py = train.py --eval-only)
src/
  train_augnet.py         Lightning module, dataset, CLI
  augnet_model.py         Schemas, irreps, masks
  paw_head_shared.py      Shared element-agnostic readout head
  paw_basis_transform.py  Sanvito (CHGCAR) <-> e3nn basis transforms
  paw_moments.py          Multipole-moment reconstruction
  paw_stats.py            Target shift/scale stats (CLI: python -m src.paw_stats)
  run_paw_chgcar.py       CHGCAR parsing and training-example assembly
  mp_potcar_map.py        Element -> POTCAR variant and l-channel map
  postprocess_sanvito_metrics.py  Offline metrics from exported predictions
configs/                  The configs used for the reported models
data_splits/              Train/val/test split JSONs
stats/                    Free-atom reference targets and neighbour-count tables
trained_models/           1k / 10k / 50k / full total-channel and spin_full checkpoints
reference_calcs/          Free-atom PAW references (inputs to the stats files)
scripts/                  Split, stats, and checkpoint tooling
cjlm/                     CJLM MoS2 benchmark (see cjlm/README.md)
```

## Installation

Python >= 3.10. Install torch for your CUDA build first, then the rest:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
```

`cuequivariance-torch` and `openequivariance` provide the fused GPU kernels the
configs enable (`enable_cueq`, `enable_oeq`). For CPU-only use set
`enable_oeq: false` and `accelerator: cpu`; cuEquivariance falls back to slow
CPU kernels on its own.

The trainer always creates a Weights & Biases logger. Export
`WANDB_MODE=disabled` (or set `wandb.mode: disabled` in the config) to run
without an account.

## Data

Training data is one CHGCAR per structure, named `<material_id>.chgcar` or
`<material_id>.chgcar.lz4`, in a flat directory. Point `project.data_dir` (and
`project.hpc_data_dir`) in the config at it, or override per run. Split files in
`data_splits/` list the material ids per split.

Per-split inputs a run needs:

- `stats/neighbor_stats_<split>.json`: measured average neighbour count,
  consumed when `model.avg_num_neighbors: auto`. Regenerate with
  `python scripts/neighbor_stats.py --data-dir <dir> --split-file <split.json> --out <file>`.
- `stats/paw_ref_freeatom.pt`: the free-atom reference target transform, built
  from `reference_calcs/` by `python scripts/make_free_atom_ref_stats.py`.
  Both are shipped for the four MP splits.

## Training

```bash
python train.py --config configs/config.yaml \
    --split-name full \
    --split-file data_splits/datasplits_mpfull2025_full.json \
    --neighbor-stats-file stats/neighbor_stats_mpfull2025_full.json \
    --out-dir augnet_runs/full
```

Use `1k`, `10k`, or `50k` in place of `full` for the smaller splits. The spin
channel model uses `configs/config_spin.yaml` with the same flags. `--resume auto`
continues from the newest checkpoint in the output directory. Each run writes
checkpoints, `paw_stats.pt` (the target transform it trained under),
`test_metrics_summary.json`, and one `<id>_<channel>_aug.npz` prediction export
per test structure.

## Evaluation

Score a checkpoint on any split. The `--stats-file` must be the transform the
weights were trained under (the run's own `paw_stats.pt`, or the shipped
`stats/paw_ref_freeatom.pt`) and `--neighbor-stats-file` must be the training
split's table.

```bash
python eval.py --config configs/config.yaml \
    --split-name full \
    --split-file data_splits/datasplits_mpfull2025_full.json \
    --stats-file stats/paw_ref_freeatom.pt \
    --neighbor-stats-file stats/neighbor_stats_mpfull2025_full.json \
    --init-from trained_models/full.ckpt \
    --out-dir augnet_runs/eval_full
```

The spin-channel model is evaluated the same way with `configs/config_spin.yaml`
and `--init-from trained_models/spin_full.ckpt`.

GNoME zero-shot: put the GNoME CHGCARs in a flat directory (one
`<id>.chgcar` per structure), write a split file listing every id under `test`,
and run the same command with `configs/config_gnome_eval.yaml`,
`--split-name gnome`, and that split file.

Sanvito-basis metrics (per-structure and pooled MAE / RMSE / maxAE, with the
magnetic split) are recomputed offline from the exported predictions:

```bash
python -m src.postprocess_sanvito_metrics --export-dir augnet_runs/eval_full/<run_name> --out-dir results
```

## CJLM MoS2 fine-tuning ablation

Prepare the data with the scripts in `cjlm/` (see `cjlm/README.md`). The ablation
arms all use `stats/paw_ref_freeatom_mosv.pt` (Mo_sv reference) and start from
`trained_models/full.ckpt`:

```bash
# fine-tune everything (backbone at 0.1x lr)
python train.py --config configs/config_cjlm_ablate10k_ft.yaml \
    --stats-file stats/paw_ref_freeatom_mosv.pt --init-from trained_models/full.ckpt \
    --run-name ft_full --out-dir augnet_runs/cjlm

# head only, pretrained head
python train.py --config configs/config_cjlm_ablate10k_ft.yaml \
    --stats-file stats/paw_ref_freeatom_mosv.pt --init-from trained_models/full.ckpt \
    --freeze-backbone --run-name ft_head --out-dir augnet_runs/cjlm

# head only, re-initialised head
python scripts/make_freshhead_ckpt.py trained_models/full.ckpt
python train.py --config configs/config_cjlm_ablate10k_ft.yaml \
    --stats-file stats/paw_ref_freeatom_mosv.pt --init-from trained_models/full_freshhead.ckpt \
    --freeze-backbone --run-name ft_freshhead --out-dir augnet_runs/cjlm

# from scratch
python train.py --config configs/config_cjlm_ablate10k_scratch.yaml \
    --stats-file stats/paw_ref_freeatom_mosv.pt --run-name scratch --out-dir augnet_runs/cjlm
```

Each arm writes a `periodic_e*_s<step>.ckpt` every 1000 steps. Evaluate one with
`eval.py --config configs/config_cjlm_ablate10k_ft.yaml --stats-file
stats/paw_ref_freeatom_mosv.pt --init-from <ckpt> --run-name eval_s<step>
--out-dir augnet_runs/cjlm/<arm>`, then score the whole tree under the paper's
L <= 2 filter with `python cjlm/score_ablation_l2.py augnet_runs/cjlm`.
