# Neural Electronic Initialization

Code for "Complete Neural Electronic Initialization Accelerates Materials DFT", packaged as one
installable distribution, `neural_init`, with an end-to-end inference pipeline:

structure (or CHGCAR) → ELECTRAFI total + spin density grids, AugNet PAW augmentation
occupancies, CHGNet site moments → a VASP-ready directory (`CHGCAR`, `INCAR` with
`ICHARG=1`, `POSCAR`, `POTCAR`, `KPOINTS`).

## Layout

```
neural_init/            the installable package
  spin_electrafi/           ELECTRAFI adapted for spin-difference densities (EScAIP backbone)
  augnet/                   AugNet: augmentation occupancies from a MACE backbone
  vasp_runner/              CHGCAR channel surgery, INCAR/OSZICAR helpers, VASP experiment plumbing
  pipeline/                 the inference workflow, YAML config and the `ndi` CLI
  models.py                 registry of weight files (names -> paths under the weights directory)
experiments/                training runs, SLURM submitters and report scripts for the paper (not installed)
trained_models/             model weights (not tracked; see below)
tests/
```

Each subpackage keeps its original README under `experiments/<name>/README.md`; the training and
evaluation commands there now run from `experiments/<name>/` against the installed package
(`python experiments/augnet/train.py ...`, `ndi-augnet-train ...`, `ndi-electrafi-train ...`).

## Installation

Python >= 3.10. One torch install serves all three models and CHGNet; pick the wheel for your
CUDA first (or skip this line to get PyPI's default build):

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.4.1" --index-url https://download.pytorch.org/whl/cu124   # optional, match your CUDA
pip install -e ".[cueq-cuda]"      # GPU: fused cuEquivariance kernels for AugNet
pip install -e .                   # CPU / macOS
pip check
```

Extras: `train` (wandb, plotly, ...), `oeq` (OpenEquivariance, needs torch>=2.7 and nvcc),
`mp` (Materials Project API for the experiment scripts), `dev` (pytest, ruff, build).

Notes on the dependency set:

- `fairchem-core` and `torch_scatter`/`torch_cluster` are no longer needed. The few fairchem
  helpers the EScAIP backbone used (periodic radius graph, distance smearing) are vendored under
  `neural_init/spin_electrafi/model/escaip/utils/` (MIT, see `LICENSE.fairchem`); numerics are unchanged.
- The EScAIP backbone was trained with e3nn >= 0.5 spherical harmonics up to l = 12 (component
  normalization). e3nn 0.4.4, which MACE pins, stops at l = 11 and normalizes differently, so the
  e3nn 0.5.1 `_spherical_harmonics` function is vendored verbatim (`LICENSE.e3nn`). AugNet itself was
  trained with e3nn 0.4.4 and uses it unchanged.
- `pykeops` compiles its kernels at first use and needs a C++ compiler (and CUDA for GPU runs) at runtime.
- `cuequivariance` / `cuequivariance-torch` are required because the AugNet checkpoints are stored in
  cuEquivariance module layout; they run on CPU without the CUDA kernel package.

### Weights

Weights are not in the repository. Put them under `trained_models/` (the default when installed
editable), or point `weights_dir` in the YAML or `NDI_WEIGHTS_DIR` at a directory laid out as

```
trained_models/augnet/{full,50k,10k,1k,spin_full}.ckpt
trained_models/spin_electrafi/spin_density/{constrained,unconstrained}_spin_electrafi.ckpt
trained_models/spin_electrafi/total_density/ELECTRAFI_BEST.model_state_dict
```

The registry names are in `neural_init/models.py` (`electrafi_spin_constrained`,
`augnet_total_full`, ...); any config entry also accepts a plain path.

## Usage

```bash
ndi config-template > ndi.yaml           # edit: potcar_dir, grid_dims or vasp_cmd, device, ...
ndi build POSCAR --config ndi.yaml --out fe2o3_seed/
ndi build CHGCAR.lz4 --config ndi.yaml   # structure and NGX/NGY/NGZ taken from the CHGCAR
ndi predict POSCAR --grid 60 60 60 --out preds/   # grids (.npy) and augmentation (.npz) only
```

`ndi build` writes `INCAR`/`POSCAR`/`POTCAR`/`KPOINTS` with pymatgen's `MPStaticSet`
(`ICHARG=1`, `ISTART=0`, `LCHARG=.TRUE.`, `MAGMOM` from CHGNet), then the `CHGCAR`, then
`ndi_prediction.json` (NELECT, integrals, moments, weights used). It never submits or runs the
calculation. The FFT grid comes from, in order: `grid_dims` / `--grid`, a CHGCAR input,
`NGXF/NGYF/NGZF` in `incar_overrides`, or, if `vasp.vasp_cmd` is set, a one-step VASP dry run.

Python:

```python
from neural_init.pipeline import Pipeline, load_config

pipe = Pipeline(load_config("ndi.yaml"))
pipe.build("POSCAR", "fe2o3_seed")            # full directory
pred = pipe.predict("POSCAR", grid_dims=(60, 60, 60))   # arrays only: pred.rho_total, pred.aug_total, ...
```

CHGNet moments are unsigned; the spin constraint passed to the constrained ELECTRAFI arm is their
sum (the paper's convention). Set `chgnet.enabled: false` or `--no-chgnet` to skip both.

## Example notebook

`examples/demo.ipynb` runs the whole thing on bcc Fe on CPU: CHGNet moments, ELECTRAFI grids,
AugNet occupancies, and a `CHGCAR` written to `examples/demo_out/`.

## Tests

```bash
pytest tests/                 # writer / input / config tests need no weights; checkpoint tests skip without them
```
