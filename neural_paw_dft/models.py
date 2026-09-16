"""Registry of shipped model weights.

Weights are not part of the wheel. They live under a weights directory, resolved as
``weights_dir`` (config) > ``$NDI_WEIGHTS_DIR`` > ``<repo>/trained_models`` (editable install)
> ``~/.cache/neural_paw_dft``. A registry file that is missing there is downloaded from the
Hugging Face repo ``HF_REPO`` into that directory.
"""
import os
from pathlib import Path

HF_REPO = "faerte/neural_paw_dft"

# name -> file name in the weights directory and in HF_REPO (flat layout)
REGISTRY = {
    "augnet_total_full": "augnet_total_full.safetensors",
    "augnet_total_50k": "augnet_total_50k.safetensors",
    "augnet_total_10k": "augnet_total_10k.safetensors",
    "augnet_total_1k": "augnet_total_1k.safetensors",
    "augnet_spin_full": "augnet_spin_full.safetensors",
    "electrafi_total": "electrafi_total.safetensors",
    "electrafi_total_v2": "electrafi_total_v2.safetensors",
    "electrafi_spin_constrained": "electrafi_spin_constrained.safetensors",
    "electrafi_spin_unconstrained": "electrafi_spin_unconstrained.safetensors",
}

# ELECTRAFI checkpoints carry no hyper-parameters; each needs the training config it was built with.
ELECTRAFI_TRAIN_CONFIG = {
    "electrafi_total": "hpc_conf.yaml",
    "electrafi_total_v2": "hpc_conf.yaml",
    "electrafi_spin_constrained": "configs/constrained_full_spin.yaml",
    "electrafi_spin_unconstrained": "configs/unconstrained_full_spin.yaml",
}


def weights_dir(override=None) -> Path:
    if override:
        return Path(override).expanduser()
    env = os.environ.get("NDI_WEIGHTS_DIR")
    if env:
        return Path(env).expanduser()
    repo_dir = Path(__file__).resolve().parents[1] / "trained_models"
    if repo_dir.is_dir():  # editable install from a checkout
        return repo_dir
    return Path.home() / ".cache" / "neural_paw_dft"


def augnet_sidecar(weights_path: Path) -> Path:
    """The ``<stem>.config.json`` next to an AugNet ``.safetensors`` file."""
    return weights_path.with_name(weights_path.name.removesuffix(".safetensors") + ".config.json")


def _download(filename: str, root: Path) -> Path:
    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(HF_REPO, filename, local_dir=str(root)))


def resolve_weights(name_or_path: str, weights_root=None) -> Path:
    """An existing path wins; otherwise ``name_or_path`` must be a registry name.

    Registry files missing from the weights directory are downloaded from ``HF_REPO``
    (AugNet files together with their ``.config.json`` sidecar).
    """
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p
    if name_or_path not in REGISTRY:
        raise KeyError(f"{name_or_path!r} is neither an existing file nor one of {sorted(REGISTRY)}")
    root = weights_dir(weights_root)
    out = root / REGISTRY[name_or_path]
    if not out.is_file():
        out = _download(REGISTRY[name_or_path], root)
    if name_or_path.startswith("augnet_") and not augnet_sidecar(out).is_file():
        _download(augnet_sidecar(out).name, root)
    return out
