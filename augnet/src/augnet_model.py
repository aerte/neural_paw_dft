# augnet_model.py
from __future__ import annotations

import os

# e3nn (<=0.4.x) torch.load()s cached constants at import; torch>=2.6 rejects them
# under the default weights_only=True. Must be set before e3nn is imported.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


import torch
from e3nn import o3

from src.mp_potcar_map import MP_POTCAR_BY_Z

# ----------------------------
# PAW schema definitions
# ----------------------------

SCHEMA_L_CHANNELS: dict[int, list[int]] = {
    15: [0, 0, 1],
    33: [0, 0, 1, 1],
    78: [0, 0, 1, 1, 2],
    138: [0, 0, 1, 1, 2, 2],
    390: [0, 0, 1, 1, 2, 2, 3, 3],
}

SCHEMA_IRREPS: dict[int, str] = {
    15: "4x0e + 2x1o + 1x2e",
    33: "6x0e + 4x1o + 3x2e",
    78: "7x0e + 6x1o + 6x2e + 2x3o + 1x4e",
    138: "9x0e + 8x1o + 10x2e + 4x3o + 3x4e",
    390: "12x0e + 12x1o + 17x2e + 12x3o + 10x4e + 4x5o + 3x6e",
}

# Derived from the MP POTCAR map so every element with a projector schema is
# representable whether or not it appears in the training split.
Z_TO_SCHEMA: dict[int, int] = {z: e["n"] for z, e in MP_POTCAR_BY_Z.items()}


def expected_dim(schema: int) -> int:
    return o3.Irreps(SCHEMA_IRREPS[schema]).dim


for _schema in SCHEMA_IRREPS:
    assert expected_dim(_schema) == _schema, (_schema, expected_dim(_schema))


# ----------------------------
# Block layouts
# ----------------------------


def build_sanvito_blocks(l_channels: list[int]):
    """Blocks in CHGCAR/Sanvito flat order: pair-major -> L -> M=-L..L."""
    blocks = []
    pos = 0

    for i, l1 in enumerate(l_channels):
        for j, l2 in enumerate(l_channels):
            if i <= j:
                for L in range(abs(l1 - l2), l1 + l2 + 1, 2):
                    n = 2 * L + 1
                    blocks.append(
                        {
                            "pair": (i, j),
                            "l1": l1,
                            "l2": l2,
                            "L": L,
                            "M": list(range(-L, L + 1)),
                            "sanvito_slice": slice(pos, pos + n),
                        }
                    )
                    pos += n

    return blocks


def build_e3nn_slices(blocks):
    """e3nn grouped order: all L=0 blocks, then L=1, ...; Sanvito order within each L."""
    pos = 0
    e3nn_slices = {}

    for L in sorted({b["L"] for b in blocks}):
        for b in [bb for bb in blocks if bb["L"] == L]:
            n = 2 * L + 1
            key = (b["pair"], b["L"])
            e3nn_slices[key] = slice(pos, pos + n)
            pos += n

    return e3nn_slices


def irrep_copy_slices(irreps: o3.Irreps) -> list[tuple[int, slice, o3.Irrep]]:
    """One (copy_index, coefficient_slice, irrep) per irrep copy."""
    out = []
    pos = 0
    copy_idx = 0

    for mul, ir in irreps:
        dim = ir.dim
        for _ in range(mul):
            sl = slice(pos, pos + dim)
            out.append((copy_idx, sl, ir))
            pos += dim
            copy_idx += 1

    assert pos == irreps.dim
    return out


SCHEMA_COPY_SLICES = {
    schema: irrep_copy_slices(o3.Irreps(irreps))
    for schema, irreps in SCHEMA_IRREPS.items()
}

SCHEMA_NUM_COPIES = {
    schema: len(SCHEMA_COPY_SLICES[schema]) for schema in SCHEMA_IRREPS
}

MAX_SCHEMA_COPIES = max(SCHEMA_NUM_COPIES.values())

MAX_SCHEMA_DIM = max(SCHEMA_IRREPS)

# Angular momentum L of every coefficient, in e3nn grouped order, per schema.
SCHEMA_COMP_L: dict[int, torch.Tensor] = {}
for _schema in SCHEMA_IRREPS:
    _comp_l = torch.zeros(_schema, dtype=torch.long)
    for _copy_idx, _sl, _ir in SCHEMA_COPY_SLICES[_schema]:
        _comp_l[_sl] = _ir.l
    SCHEMA_COMP_L[_schema] = _comp_l


def lmaxmix_component_mask(
    atomic_numbers: torch.Tensor,
    lmaxmix: int | torch.Tensor | None,
    max_dim: int = MAX_SCHEMA_DIM,
) -> torch.Tensor:
    """
    Per-atom keep-mask over padded e3nn-order coefficients: True where L <= LMAXMIX.

    VASP stores a literal 0.0 for L > LMAXMIX.
    lmaxmix: int (whole batch), per-atom LongTensor (negative = unknown, keep all),
    or None (keep all). Padding past each atom's schema is always False.
    """
    device = atomic_numbers.device
    n_atoms = int(atomic_numbers.shape[0])
    mask = torch.zeros((n_atoms, max_dim), dtype=torch.bool, device=device)

    if lmaxmix is None:
        cut = None
    elif torch.is_tensor(lmaxmix):
        cut = lmaxmix.to(device=device, dtype=torch.long).reshape(-1)
        if cut.shape[0] != n_atoms:
            raise ValueError(f"lmaxmix has {cut.shape[0]} entries, expected {n_atoms}")
    else:
        cut = torch.full((n_atoms,), int(lmaxmix), dtype=torch.long, device=device)

    for z in atomic_numbers.unique().tolist():
        schema = Z_TO_SCHEMA[int(z)]
        rows = atomic_numbers == z
        if cut is None:
            mask[rows, :schema] = True
            continue
        comp_l = SCHEMA_COMP_L[schema].to(device)
        keep = comp_l.unsqueeze(0) <= cut[rows].unsqueeze(1)
        keep |= (cut[rows] < 0).unsqueeze(1)
        mask[rows, :schema] = keep

    return mask
