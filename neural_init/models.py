"""Registry of shipped model weights.

Weights are not part of the wheel. They live under a weights directory, resolved as
``weights_dir`` (config) > ``$NDI_WEIGHTS_DIR`` > ``<repo>/trained_models`` (editable install).
"""
import os
from pathlib import Path

# name -> path relative to the weights directory
REGISTRY = {
    "augnet_total_full": "augnet/full.ckpt",
    "augnet_total_50k": "augnet/50k.ckpt",
    "augnet_total_10k": "augnet/10k.ckpt",
    "augnet_total_1k": "augnet/1k.ckpt",
    "augnet_spin_full": "augnet/spin_full.ckpt",
    "electrafi_total": "spin_electrafi/total_density/ELECTRAFI_BEST.model_state_dict",
    "electrafi_spin_constrained": "spin_electrafi/spin_density/constrained_spin_electrafi.ckpt",
    "electrafi_spin_unconstrained": "spin_electrafi/spin_density/unconstrained_spin_electrafi.ckpt",
}

# ELECTRAFI checkpoints carry no hyper-parameters; each needs the training config it was built with.
ELECTRAFI_TRAIN_CONFIG = {
    "electrafi_total": "hpc_conf.yaml",
    "electrafi_spin_constrained": "configs/constrained_full_spin.yaml",
    "electrafi_spin_unconstrained": "configs/unconstrained_full_spin.yaml",
}


def weights_dir(override=None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get("NDI_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[1] / "trained_models"


def resolve_weights(name_or_path: str, weights_root=None) -> Path:
    """An existing path wins; otherwise ``name_or_path`` must be a registry name."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    if name_or_path not in REGISTRY:
        raise KeyError(f"{name_or_path!r} is neither an existing file nor one of {sorted(REGISTRY)}")
    out = weights_dir(weights_root) / REGISTRY[name_or_path]
    if not out.is_file():
        raise FileNotFoundError(
            f"weights for {name_or_path!r} expected at {out}; set weights_dir in the config or NDI_WEIGHTS_DIR"
        )
    return out
