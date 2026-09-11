"""YAML-backed pipeline configuration."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ElectrafiConfig:
    checkpoint: str = "electrafi_spin_constrained"  # registry name (neural_init.models) or path
    train_config: str | None = None  # training yaml; None -> chosen from the checkpoint name/kind
    escaip_config: str | None = None  # None -> packaged model/escaip/escaip_config.yaml
    spin: bool = True  # False -> charge-only model, no magnetization grid
    spin_renorm: bool | None = None  # None -> value from train_config
    max_neighbors: int = 300  # eval-script CUDA setting (spin_electrafi/eval_spin_constrained.py)
    use_compile: bool = False


@dataclass
class AugnetConfig:
    total_checkpoint: str = "augnet_total_full"
    mag_checkpoint: str | None = "augnet_spin_full"  # None -> no magnetization augmentation
    stats: str | None = None  # None -> packaged stats/paw_ref_freeatom.pt
    enable_cueq: bool = True  # the shipped checkpoints are stored in cuEquivariance layout


@dataclass
class ChgnetConfig:
    enabled: bool = True  # False -> no MAGMOM override and no spin net-moment constraint
    model: str | None = None  # CHGNet.load(model_name=...) ; None -> default


@dataclass
class VaspConfig:
    potcar_dir: str | None = None  # exported as PMG_VASP_PSP_DIR; None -> pymatgen's own setting
    potcar_functional: str = "PBE_54"
    kpoints_reciprocal_density: int = 100
    incar_overrides: dict = field(default_factory=dict)  # applied last
    vasp_cmd: list[str] | None = None  # e.g. ["mpirun", "-np", "8", "vasp_std"]; enables the NGX dry run


@dataclass
class PipelineConfig:
    device: str = "auto"  # auto | cpu | cuda | cuda:N
    matmul_precision: str = "high"  # torch.set_float32_matmul_precision (training used "high")
    grid_dims: tuple[int, int, int] | None = None  # None -> from CHGCAR input -> dry run -> error
    out_dir: str = "ndi_out"
    weights_dir: str | None = None  # None -> $NDI_WEIGHTS_DIR or <repo>/trained_models
    electrafi: ElectrafiConfig = field(default_factory=ElectrafiConfig)
    augnet: AugnetConfig = field(default_factory=AugnetConfig)
    chgnet: ChgnetConfig = field(default_factory=ChgnetConfig)
    vasp: VaspConfig = field(default_factory=VaspConfig)


def _from_dict(cls, d: dict[str, Any], path: str = ""):
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"unknown config key(s) {sorted(unknown)} under '{path or 'root'}'")
    kwargs = {}
    for name, value in d.items():
        f = known[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _from_dict(type(default), value, f"{path}.{name}".strip("."))
        elif name == "grid_dims" and value is not None:
            kwargs[name] = tuple(int(x) for x in value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None) -> PipelineConfig:
    """Load a YAML config; ``None`` gives all defaults. Unknown keys are an error."""
    if path is None:
        return PipelineConfig()
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return _from_dict(PipelineConfig, data)


def config_template() -> str:
    return """\
# neural_init pipeline configuration (all keys optional; shown with their defaults)
device: auto                 # auto | cpu | cuda | cuda:0
matmul_precision: high       # torch float32 matmul precision; "high" (TF32 on Ampere+) matches training
grid_dims: null              # [NGX, NGY, NGZ]; null -> taken from a CHGCAR input, else VASP dry run, else error
out_dir: ndi_out
weights_dir: null            # null -> $NDI_WEIGHTS_DIR or <repo>/trained_models

electrafi:
  checkpoint: electrafi_spin_constrained   # registry name or path (.ckpt or .model_state_dict)
  train_config: null         # null -> constrained/unconstrained spin yaml or hpc_conf.yaml by checkpoint kind
  escaip_config: null        # null -> packaged model/escaip/escaip_config.yaml
  spin: true                 # false -> charge-only model, no magnetization grid
  spin_renorm: null          # null -> as in train_config (true for the constrained arm)
  max_neighbors: 300
  use_compile: false

augnet:
  total_checkpoint: augnet_total_full
  mag_checkpoint: augnet_spin_full         # null -> no magnetization augmentation block
  stats: null                # null -> packaged stats/paw_ref_freeatom.pt
  enable_cueq: true          # required by the shipped checkpoints (cuEquivariance weight layout)

chgnet:
  enabled: true              # false -> no MAGMOM override, no spin net-moment constraint
  model: null

vasp:
  potcar_dir: null           # e.g. /opt/vasp/potpaw_PBE.54 ; exported as PMG_VASP_PSP_DIR
  potcar_functional: PBE_54
  kpoints_reciprocal_density: 100
  incar_overrides: {}        # e.g. {ENCUT: 520, NCORE: 4}; applied last
  vasp_cmd: null             # e.g. [mpirun, -np, "8", vasp_std]; enables the one-step dry run for grid dims
"""
