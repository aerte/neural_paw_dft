"""ELECTRAFI inference: structure -> total (+ spin) density grid in e/Å^3."""
from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from neural_paw_dft._resources import resource_path
from neural_paw_dft.models import ELECTRAFI_TRAIN_CONFIG, resolve_weights
from neural_paw_dft.spin_electrafi.model.ELECTRAFI import ELECTRAFI
from neural_paw_dft.spin_electrafi.model.model_utils import PWGrid

from .config import ElectrafiConfig

# output-path keys ELECTRAFI.__init__ reads; all pointed at a scratch dir for inference
_PATH_KEYS = (
    "pred_dens_path_val", "pred_dens_path_test", "density_delta_path_val", "density_delta_path_test",
    "gaus_pos_path_val", "gaus_pos_path_test", "gaus_pos_outlier_path",
)


def _train_config_path(cfg: ElectrafiConfig, ckpt: Path) -> Path:
    if cfg.train_config:
        return Path(cfg.train_config)
    if cfg.checkpoint in ELECTRAFI_TRAIN_CONFIG:
        return resource_path("spin_electrafi", ELECTRAFI_TRAIN_CONFIG[cfg.checkpoint])
    raise ValueError(
        f"cannot determine the training config for {ckpt.name}: it carries the spin arm and "
        f"spin_renorm setting, and the wrong one silently changes the predicted density. "
        f"Use one of {sorted(ELECTRAFI_TRAIN_CONFIG)} as electrafi.checkpoint, or set "
        f"electrafi.train_config explicitly."
    )


def build_inference_config(cfg: ElectrafiConfig, ckpt: Path, has_spin_head: bool, work_dir: Path, n_atoms: int,
                           num_layers: int | None = None) -> dict:
    """The training config, trimmed so ELECTRAFI.__init__ has no training side effects."""
    with open(_train_config_path(cfg, ckpt)) as f:
        config = yaml.safe_load(f)
    esc = cfg.escaip_config or resource_path("spin_electrafi", "model/escaip/escaip_config.yaml")
    with open(esc) as f:
        config["escaip_config"] = yaml.safe_load(f)

    # Mirror eval_spin_constrained.py: these knobs are not part of the weights.
    backbone = config["escaip_config"]["model"]["backbone"]
    backbone["max_neighbors"] = int(cfg.max_neighbors)
    backbone["use_compile"] = bool(cfg.use_compile)
    # use_padding pads every batch to this many nodes; oversized structures would be truncated.
    backbone["max_num_nodes_per_batch"] = max(int(backbone.get("max_num_nodes_per_batch", 154)), int(n_atoms))
    if num_layers is not None:  # the charge-only models were trained with 3 layers, the spin models with 2
        backbone["num_layers"] = int(num_layers)

    # The spin head must match the checkpoint, not the yaml, for a strict load.
    if has_spin_head:
        config["spin_type"] = "total_diff"
    else:
        config.pop("spin_type", None)
    if cfg.spin_renorm is not None:
        config["spin_renorm"] = bool(cfg.spin_renorm)

    config.update(
        wandb=False,
        inference_only=True,
        write_stage_timing_csv=False,
        rotate_train=False, rotate_val=False, rotate_test=False,
        ood_names=[], ood_paths=[],
        debug=False,
        dens_path=str(work_dir),
        pw_grid=[32, 32, 32],  # placeholder; replaced per structure in predict_density
    )
    config.setdefault("loss_type", "normal")
    for k in _PATH_KEYS:
        config[k] = str(work_dir)
    return config


def load_electrafi(cfg: ElectrafiConfig, device: torch.device, n_atoms: int = 154, weights_dir=None) -> ELECTRAFI:
    ckpt = resolve_weights(cfg.checkpoint, weights_dir)
    if ckpt.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(ckpt)
    else:
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = blob.get("state_dict", blob)  # Lightning .ckpt or flat .model_state_dict
    has_spin_head = any(k.startswith("spin_w_net") for k in state)
    blocks = [int(m.group(1)) for k in state for m in [re.match(r"backbone\.transformer_blocks\.(\d+)\.", k)] if m]
    num_layers = max(blocks) + 1 if blocks else None
    if cfg.spin and not has_spin_head:
        raise ValueError(f"{ckpt.name} is a charge-only model; set electrafi.spin: false or pick a spin checkpoint")

    work_dir = Path(tempfile.mkdtemp(prefix="ndi_electrafi_"))
    config = build_inference_config(cfg, ckpt, has_spin_head, work_dir, n_atoms, num_layers)
    model = ELECTRAFI(train_files=[], test_files=[], validation_files=[], model_handler=None, config=config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    model._ndi_max_nodes = config["escaip_config"]["model"]["backbone"]["max_num_nodes_per_batch"]
    model._ndi_has_spin = has_spin_head
    return model


@torch.inference_mode()
def predict_density(
    model: ELECTRAFI,
    structure: Structure,
    grid_dims: tuple[int, int, int],
    n_elec: float,
    m_total: float | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Total and (if the model has a spin head) spin density on ``grid_dims``, in e/Å^3.

    ``n_elec`` fixes the integral of the total density exactly; ``m_total`` is the net
    moment in electrons used for the constrained spin arm (ignored when |m| < spin_mag_min
    or the model was trained with spin_renorm: false).
    """
    atoms = AseAtomsAdaptor.get_atoms(structure)  # fresh copy: the model mutates atoms.pbc
    grid_dims = tuple(int(x) for x in grid_dims)
    if tuple(model.pw.real_shape) != grid_dims:
        model.pw = PWGrid(grid_dims, device=model.device, dtype=model.precision)
    out = model(atoms, n_elec=float(n_elec), m_total=m_total)
    rho = out["rho"].detach().float().cpu().numpy()
    rho_spin = out["rho_spin"]
    if rho_spin is not None:
        rho_spin = rho_spin.detach().float().cpu().numpy()
    return rho, rho_spin
