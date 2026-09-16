"""CHGNet per-site magnetic moments (unsigned, in mu_B), used for INCAR MAGMOM and the
ELECTRAFI net-moment constraint (sum of unsigned site moments, the paper's convention)."""
from __future__ import annotations

import numpy as np
import torch
from pymatgen.core import Structure


def load_chgnet(model_name: str | None, device: torch.device):
    from chgnet.model import CHGNet

    kwargs = {"use_device": device.type}
    if model_name:
        kwargs["model_name"] = model_name
    return CHGNet.load(**kwargs)


def predict_site_moments(chgnet, structure: Structure) -> np.ndarray:
    pred = chgnet.predict_structure(structure)
    m = np.asarray(pred["m"], dtype=float).reshape(-1)
    if m.size != len(structure):
        raise RuntimeError(f"CHGNet returned {m.size} moments for {len(structure)} sites")
    return m
