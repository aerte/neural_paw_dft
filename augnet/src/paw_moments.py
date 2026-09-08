# paw_moments.py
"""
Spectral moments Tr(D^k) of the on-site PAW density matrix, reconstructed from the
augmentation-occupancy coefficients.

Moments rather than eigenvalues: on-site matrices are highly degenerate, so eigh
gradients (1/(lambda_i - lambda_j)) blow up; Tr(D^k) is a smooth polynomial
invariant carrying the same spectral information.

Reconstruction: D[i:m1, j:m2] = sum_{L,M} alpha(l1,l2,L) * w3j(l1,l2,L)[m1,m2,M] * c[L][M],
with alpha the least-squares pseudo-inverse of the projection D -> c. VASP stores
only the same-parity part of off-diagonal blocks, so for k >= 2 these are moments of
the recoverable D-hat (a lower bound); Tr(D) is exact. Each Q_L is orthogonal, so the
moments are basis-independent and computed directly on e3nn-order coefficients.
"""
from __future__ import annotations

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # e3nn constants under torch>=2.6

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import torch
from e3nn import o3

from src.augnet_model import (
    SCHEMA_L_CHANNELS,
    build_sanvito_blocks,
    build_e3nn_slices,
    Z_TO_SCHEMA,
)


@lru_cache(maxsize=None)
def build_reconstruction(schema: int) -> dict:
    """Per-schema reconstruction plan: N_i and one placement per stored (pair, L) block
    with row/col slices, W = w3j * alpha, alpha, the coefficient slice and offdiag."""
    l_channels = SCHEMA_L_CHANNELS[schema]

    offsets: List[int] = []
    pos = 0
    for l in l_channels:
        offsets.append(pos)
        pos += 2 * l + 1
    N_i = pos

    blocks = build_sanvito_blocks(l_channels)
    e3_slices = build_e3nn_slices(blocks)

    placements = []
    for b in blocks:
        i, j = b["pair"]
        l1, l2, L = b["l1"], b["l2"], b["L"]

        w3j = o3.wigner_3j(l1, l2, L).to(torch.float64)     # [2l1+1, 2l2+1, 2L+1]
        denom = (w3j ** 2).sum(dim=(0, 1))                  # [2L+1], M-independent
        alpha = torch.where(denom > 0, 1.0 / denom, torch.zeros_like(denom))
        W = w3j * alpha.view(1, 1, -1)

        placements.append(
            {
                "row": slice(offsets[i], offsets[i] + 2 * l1 + 1),
                "col": slice(offsets[j], offsets[j] + 2 * l2 + 1),
                "W": W,
                "alpha": alpha,
                "coeff": e3_slices[(b["pair"], L)],
                "offdiag": (i != j),
            }
        )

    return {"N_i": N_i, "placements": placements, "schema": schema}


def reconstruct_D(
    coeff_padded: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """coeff_padded [n_atoms, 390] physical e3nn-order -> {schema: (idx, D [n, N_i, N_i])}."""
    device = coeff_padded.device
    dtype = coeff_padded.dtype

    schemas = [Z_TO_SCHEMA[int(z)] for z in atomic_numbers.tolist()]
    schemas_t = torch.tensor(schemas, device=device)

    out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    for schema in sorted(set(schemas)):
        idx = torch.where(schemas_t == schema)[0]
        rec = build_reconstruction(schema)
        N = rec["N_i"]
        n = idx.numel()

        sub = coeff_padded.index_select(0, idx)             # [n, 390]
        D = coeff_padded.new_zeros((n, N, N))

        for p in rec["placements"]:
            W = p["W"].to(device=device, dtype=dtype)
            c = sub[:, p["coeff"]]                          # [n, 2L+1]
            block = torch.einsum("abm,nm->nab", W, c)       # [n, 2l1+1, 2l2+1]

            # Out-of-place assembly keeps autograd intact.
            contrib = coeff_padded.new_zeros((n, N, N))
            contrib[:, p["row"], p["col"]] = block
            if p["offdiag"]:
                contrib[:, p["col"], p["row"]] = block.transpose(1, 2)
            D = D + contrib

        out[schema] = (idx, D)

    return out


def spectral_moments(D: torch.Tensor, orders: Sequence[int]) -> torch.Tensor:
    """D [n, N, N] -> [n, len(orders)] with Tr(D^k)."""
    max_k = max(orders)
    powers = {1: D}
    Dk = D
    for k in range(2, max_k + 1):
        Dk = torch.matmul(Dk, D)
        powers[k] = Dk

    cols = [torch.diagonal(powers[k], dim1=1, dim2=2).sum(dim=1) for k in orders]
    return torch.stack(cols, dim=1)


def moments_from_coeffs(
    coeff_padded: torch.Tensor,
    atomic_numbers: torch.Tensor,
    orders: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (moments [n_atoms, len(orders)], valid [n_atoms])."""
    orders = list(orders)
    n_atoms = coeff_padded.shape[0]

    moments = coeff_padded.new_zeros((n_atoms, len(orders)))
    valid = torch.zeros(n_atoms, dtype=torch.bool, device=coeff_padded.device)

    for schema, (idx, D) in reconstruct_D(coeff_padded, atomic_numbers).items():
        m = spectral_moments(D, orders)                     # [n_schema, len(orders)]
        moments = moments.index_copy(0, idx, m)
        valid = valid.index_fill(0, idx, True)

    return moments, valid


def moment_regularization_loss(
    pred_coeff_padded: torch.Tensor,
    target_coeff_padded: torch.Tensor,
    atomic_numbers: torch.Tensor,
    lambda_tr1: float = 0.05,
    lambda_tr2: float = 0.02,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Relative MSE of Tr(D) and Tr(D^2) between prediction and target."""
    zero = pred_coeff_padded.new_tensor(0.0)

    if lambda_tr1 == 0.0 and lambda_tr2 == 0.0:
        return zero, {
            "moment_tr1_rel_mse": zero,
            "moment_tr2_rel_mse": zero,
        }

    pred_m, pred_valid = moments_from_coeffs(pred_coeff_padded, atomic_numbers, [1, 2])
    target_m, target_valid = moments_from_coeffs(target_coeff_padded, atomic_numbers, [1, 2])
    valid = pred_valid & target_valid

    if not bool(valid.any().detach().cpu()):
        return zero, {
            "moment_tr1_rel_mse": zero,
            "moment_tr2_rel_mse": zero,
        }

    rel = (pred_m[valid] - target_m[valid]) / (target_m[valid].abs() + eps)
    rel_sq = rel.pow(2)

    tr1_rel_mse = rel_sq[:, 0].mean()
    tr2_rel_mse = rel_sq[:, 1].mean()
    loss = lambda_tr1 * tr1_rel_mse + lambda_tr2 * tr2_rel_mse

    return loss, {
        "moment_tr1_rel_mse": tr1_rel_mse,
        "moment_tr2_rel_mse": tr2_rel_mse,
    }


def tr_d2_closed_form(
    coeff_padded: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """Verification helper: Tr(D^2) = sum_blocks w_ij * sum_M alpha[M] c[M]^2
    (w_ij = 1 diagonal, 2 off-diagonal), without assembling D."""
    n_atoms = coeff_padded.shape[0]
    out = coeff_padded.new_zeros((n_atoms,))
    schemas = [Z_TO_SCHEMA[int(z)] for z in atomic_numbers.tolist()]
    schemas_t = torch.tensor(schemas, device=coeff_padded.device)

    for schema in sorted(set(schemas)):
        idx = torch.where(schemas_t == schema)[0]
        rec = build_reconstruction(schema)
        sub = coeff_padded.index_select(0, idx)
        acc = sub.new_zeros((idx.numel(),))
        for p in rec["placements"]:
            alpha = p["alpha"].to(device=sub.device, dtype=sub.dtype)
            c = sub[:, p["coeff"]]                          # [n, 2L+1]
            w = 2.0 if p["offdiag"] else 1.0
            acc = acc + w * (alpha.view(1, -1) * c * c).sum(dim=1)
        out = out.index_copy(0, idx, acc)

    return out
