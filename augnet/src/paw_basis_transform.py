from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import torch
from e3nn import o3

from src.augnet_model import (
    SCHEMA_L_CHANNELS,
    build_sanvito_blocks,
    build_e3nn_slices,
    Z_TO_SCHEMA,
)
from src.mp_potcar_map import MP_POTCAR_BY_Z


# Per-element override of the physical projector order (config data.potcar_l_channels).
_L_CHANNEL_OVERRIDES: Dict[int, Tuple[int, ...]] = {}

_SYMBOL_TO_Z = {e["element"]: z for z, e in MP_POTCAR_BY_Z.items()}


def set_l_channel_overrides(overrides: Dict) -> None:
    """Override the physical CHGCAR projector order per element, e.g.
    {"Mo": [0, 0, 1, 1, 2, 2]} for Mo_sv data where MP uses Mo_pv. Each sequence must
    be a permutation of that element's mp_potcar_map l_channels. {} restores defaults."""
    resolved: Dict[int, Tuple[int, ...]] = {}

    for key, l_channels in (overrides or {}).items():
        if isinstance(key, str):
            if key not in _SYMBOL_TO_Z:
                raise ValueError(f"Unknown element symbol in l_channel override: {key}")
            z = _SYMBOL_TO_Z[key]
        else:
            z = int(key)

        entry = MP_POTCAR_BY_Z.get(z)
        if entry is None:
            raise ValueError(f"No mp_potcar_map entry for Z={z}; cannot validate override.")

        l_new = tuple(int(x) for x in l_channels)
        if sorted(l_new) != sorted(entry["l_channels"]):
            raise ValueError(
                f"l_channel override for Z={z} ({entry['element']}) must be a permutation "
                f"of {tuple(entry['l_channels'])}, got {l_new}"
            )

        resolved[z] = l_new

    global _L_CHANNEL_OVERRIDES
    if resolved == _L_CHANNEL_OVERRIDES:
        return

    _L_CHANNEL_OVERRIDES = resolved
    physical_to_canonical_blocks.cache_clear()


@lru_cache(maxsize=None)
def physical_to_canonical_blocks(z: int):
    """Map from the element's physical CHGCAR partial-wave layout to the canonical
    per-schema e3nn layout (SCHEMA_L_CHANNELS = sorted projectors; e.g. Ti_pv stores
    (1,1,2,2,0,0)). Returns [(sanvito_slice, e3nn_slice, L)] per stored block. Only
    same-parity blocks are stored, so this is a pure permutation with no sign change."""
    schema = Z_TO_SCHEMA[int(z)]
    l_canon = list(SCHEMA_L_CHANNELS[schema])

    override = _L_CHANNEL_OVERRIDES.get(int(z))
    if override is not None:
        l_phys = list(override)
    else:
        entry = MP_POTCAR_BY_Z.get(int(z))
        l_phys = list(entry["l_channels"]) if entry is not None else list(l_canon)

    order = sorted(range(len(l_phys)), key=lambda i: l_phys[i])
    phys_to_canon = [0] * len(l_phys)
    for canon_pos, phys_idx in enumerate(order):
        phys_to_canon[phys_idx] = canon_pos

    phys_blocks = build_sanvito_blocks(l_phys)
    canon_e3 = build_e3nn_slices(build_sanvito_blocks(l_canon))

    out = []
    for b in phys_blocks:
        i, j = b["pair"]
        ci, cj = phys_to_canon[i], phys_to_canon[j]
        if ci > cj:
            ci, cj = cj, ci
        out.append((b["sanvito_slice"], canon_e3[((ci, cj), b["L"])], b["L"]))

    return out


def sanvito_schema_and_lmaxmix_masks(
    atomic_numbers: torch.Tensor,
    lmaxmix: "int | None",
    max_dim: int = 390,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Masks over a padded Sanvito-order vector: schema_mask (slots the element stores)
    and lmaxmix_mask (slots with L <= lmaxmix; equals schema_mask when lmaxmix is None).
    Indexed in the element's physical CHGCAR order, unlike lmaxmix_component_mask."""
    device = atomic_numbers.device
    n_atoms = int(atomic_numbers.shape[0])
    schema_mask = torch.zeros((n_atoms, max_dim), dtype=torch.bool, device=device)
    lmaxmix_mask = torch.zeros((n_atoms, max_dim), dtype=torch.bool, device=device)

    for z in atomic_numbers.unique().tolist():
        z = int(z)
        schema = Z_TO_SCHEMA[z]
        rows = atomic_numbers == z
        schema_mask[rows, :schema] = True

        if lmaxmix is None:
            lmaxmix_mask[rows, :schema] = True
            continue

        keep = torch.zeros(max_dim, dtype=torch.bool, device=device)
        for sanvito_slice, _e3nn_slice, L in physical_to_canonical_blocks(z):
            if L <= int(lmaxmix):
                keep[sanvito_slice] = True
        lmaxmix_mask[rows] = keep

    return schema_mask, lmaxmix_mask


def fibonacci_sphere(n: int, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """Deterministic near-uniform points on S^2, [n, 3]."""
    i = torch.arange(n, dtype=dtype, device=device)
    phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0, dtype=dtype, device=device)))

    z = 1.0 - 2.0 * (i + 0.5) / n
    r = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
    theta = phi * i

    x = r * torch.cos(theta)
    y = r * torch.sin(theta)

    return torch.stack([x, y, z], dim=-1)


def sanvito_real_sh(L: int, xyz: torch.Tensor) -> torch.Tensor:
    """Sanvito real spherical harmonics, M=-L..L, for unit xyz [N, 3] -> [N, 2L+1].
    Explicit Cartesian forms for L <= 4; e3nn with (y, z, x) coordinates above."""
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    pi = torch.pi

    if L == 0:
        sfact = 0.5 * (1.0 / pi) ** 0.5
        return torch.stack([sfact * torch.ones_like(x)], dim=-1)

    if L == 1:
        pfact = (3.0 / (4.0 * pi)) ** 0.5
        return torch.stack(
            [
                pfact * y,
                pfact * z,
                pfact * x,
            ],
            dim=-1,
        )

    if L == 2:
        dfact = 0.5 * (15.0 / pi) ** 0.5
        dfact0 = 0.25 * (5.0 / pi) ** 0.5
        return torch.stack(
            [
                dfact * x * y,
                dfact * y * z,
                dfact0 * (3.0 * z**2 - 1.0),
                dfact * x * z,
                dfact * (x**2 - y**2) / 2.0,
            ],
            dim=-1,
        )

    if L == 3:
        ffact3 = 0.25 * (35.0 / (2.0 * pi)) ** 0.5
        ffact2 = 0.50 * (105.0 / pi) ** 0.5
        ffact1 = 0.25 * (21.0 / (2.0 * pi)) ** 0.5
        ffact0 = 0.25 * (7.0 / pi) ** 0.5
        return torch.stack(
            [
                ffact3 * y * (3.0 * x**2 - y**2),
                ffact2 * y * x * z,
                ffact1 * y * (5.0 * z**2 - 1.0),
                ffact0 * (5.0 * z**3 - 3.0 * z),
                ffact1 * x * (5.0 * z**2 - 1.0),
                ffact2 * (x**2 - y**2) * z / 2.0,
                ffact3 * x * (x**2 - 3.0 * y**2),
            ],
            dim=-1,
        )

    if L == 4:
        gfact4 = 0.75 * (35.0 / pi) ** 0.5
        gfact3 = 0.75 * (35.0 / (2.0 * pi)) ** 0.5
        gfact2 = 0.75 * (5.0 / pi) ** 0.5
        gfact1 = 0.75 * (5.0 / (2.0 * pi)) ** 0.5
        gfact0 = 0.1875 * (1.0 / pi) ** 0.5
        return torch.stack(
            [
                gfact4 * x * y * (x**2 - y**2),
                gfact3 * y * (3.0 * x**2 - y**2) * z,
                gfact2 * x * y * (7.0 * z**2 - 1.0),
                gfact1 * y * (7.0 * z**3 - 3.0 * z),
                gfact0 * (35.0 * z**4 - 30.0 * z**2 + 3.0),
                gfact1 * x * (7.0 * z**3 - 3.0 * z),
                gfact2 * (x**2 - y**2) * (7.0 * z**2 - 1.0) / 2.0,
                gfact3 * x * (x**2 - 3.0 * y**2) * z,
                gfact4
                * (x**2 * (x**2 - 3.0 * y**2) - y**2 * (3.0 * x**2 - y**2))
                / 4.0,
            ],
            dim=-1,
        )

    xyz_yzx = xyz[:, [1, 2, 0]]
    return o3.spherical_harmonics(
        L,
        xyz_yzx,
        normalize=True,
        normalization="integral",
    ).to(dtype=xyz.dtype, device=xyz.device)


@lru_cache(None)
def basis_matrix_sanvito_to_e3nn(
    L: int,
    n_points: int = 2000,
) -> torch.Tensor:
    """Orthogonal Q_L [2L+1, 2L+1] with E(x) = S(x) @ Q_L.T, so c_E = Q_L @ c_S."""
    xyz = fibonacci_sphere(n_points, dtype=torch.float64)

    S = sanvito_real_sh(L, xyz)  # [N, d]
    E = o3.spherical_harmonics(
        L,
        xyz,
        normalize=True,
        normalization="integral",
    ).to(dtype=torch.float64)  # [N, d]

    A = torch.linalg.lstsq(S, E).solution  # [d, d]
    Q = A.T.contiguous()

    # SVD projection to the nearest orthogonal matrix.
    U, _, Vh = torch.linalg.svd(Q)
    Q = U @ Vh

    return Q


def print_basis_diagnostics(max_L: int = 6):
    for L in range(max_L + 1):
        Q = basis_matrix_sanvito_to_e3nn(L)
        d = 2 * L + 1
        I = torch.eye(d, dtype=Q.dtype)
        orth_err = (Q.T @ Q - I).abs().max().item()

        xyz = fibonacci_sphere(200, dtype=torch.float64)
        S = sanvito_real_sh(L, xyz)
        E = o3.spherical_harmonics(
            L,
            xyz,
            normalize=True,
            normalization="integral",
        ).to(dtype=torch.float64)

        E_pred = S @ Q.T
        fit_err = (E_pred - E).abs().max().item()

        print(f"L={L}: Q shape={tuple(Q.shape)}, orth_err={orth_err:.3e}, basis_fit_err={fit_err:.3e}")
        if L == 1:
            print("Q_L1 =")
            print(Q)


def sanvito_coeff_block_to_e3nn(coeff_san: torch.Tensor, L: int) -> torch.Tensor:
    Q = basis_matrix_sanvito_to_e3nn(L).to(device=coeff_san.device, dtype=coeff_san.dtype)
    return coeff_san @ Q.T


def e3nn_coeff_block_to_sanvito(coeff_e3: torch.Tensor, L: int) -> torch.Tensor:
    Q = basis_matrix_sanvito_to_e3nn(L).to(device=coeff_e3.device, dtype=coeff_e3.dtype)
    return coeff_e3 @ Q


def sanvito_to_e3nn_with_basis_padded(
    y_sanvito: torch.Tensor,
    mask_sanvito: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sanvito/CHGCAR order + basis -> e3nn grouped order + basis."""
    y_e3nn = torch.zeros_like(y_sanvito)
    mask_e3nn = torch.zeros_like(mask_sanvito)

    for i, z in enumerate(atomic_numbers.tolist()):
        for san, e3, L in physical_to_canonical_blocks(int(z)):
            coeff_san = y_sanvito[i, san]
            coeff_e3 = sanvito_coeff_block_to_e3nn(coeff_san, L)

            y_e3nn[i, e3] = coeff_e3
            mask_e3nn[i, e3] = True

    return y_e3nn, mask_e3nn


def e3nn_to_sanvito_with_basis_padded(
    y_e3nn: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> torch.Tensor:
    """e3nn grouped order + basis -> Sanvito/CHGCAR order + basis."""
    y_sanvito = torch.zeros_like(y_e3nn)

    for i, z in enumerate(atomic_numbers.tolist()):
        for san, e3, L in physical_to_canonical_blocks(int(z)):
            coeff_e3 = y_e3nn[i, e3]
            coeff_san = e3nn_coeff_block_to_sanvito(coeff_e3, L)

            y_sanvito[i, san] = coeff_san

    return y_sanvito


def check_basis_roundtrip_for_schema(schema: int, ntrials: int = 10):
    blocks = build_sanvito_blocks(SCHEMA_L_CHANNELS[schema])
    e3_slices = build_e3nn_slices(blocks)

    for _ in range(ntrials):
        y_san = torch.randn(schema, dtype=torch.float64)
        y_e3 = torch.zeros_like(y_san)

        for b in blocks:
            L = b["L"]
            san = b["sanvito_slice"]
            e3 = e3_slices[(b["pair"], L)]
            y_e3[e3] = sanvito_coeff_block_to_e3nn(y_san[san], L)

        y_san_back = torch.zeros_like(y_san)
        for b in blocks:
            L = b["L"]
            san = b["sanvito_slice"]
            e3 = e3_slices[(b["pair"], L)]
            y_san_back[san] = e3nn_coeff_block_to_sanvito(y_e3[e3], L)

        err = (y_san_back - y_san).abs().max().item()
        assert err < 1e-10, (schema, err)

    print(f"[PASS] schema {schema}: basis-aware Sanvito <-> e3nn coefficient roundtrip")


if __name__ == "__main__":
    print_basis_diagnostics(max_L=6)
    for schema in [15, 33, 78, 138, 390]:
        check_basis_roundtrip_for_schema(schema)
