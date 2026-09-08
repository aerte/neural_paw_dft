# paw_head_shared.py
"""
Element-agnostic PAW readout: one parameter set shared by every schema and element.

D_ij^LM = sum_n f_n <p_i|psi_n><psi_n|p_j> is a symmetric outer product in projector
space, so the head predicts projector-space coefficients with one shared o3.Linear
(h -> c[slot, k, m], k = 1..rank) and builds each (i, j, L) block by parameter-free
Clebsch-Gordan coupling: block_M = sum_k w_k * CG(c_i^k, c_j^k)_M.

Every schema's partial-wave list is a prefix of [0,0,1,1,2,2,3,3], so the full
390-coefficient block set is computed once per atom and each atom's schema is
gathered out of it. The parity selection rule (l_i + l_j + L even) reproduces each
schema's declared irreps exactly (asserted at import).

sum_k c^k (x) c^k is PSD, but targets are standardized (shifted), so a signed
combination over k is needed; rank >= 64 spans every schema's symmetric block space.
Blocks with L <= linear_max_l are read off h by a direct linear map instead.
"""

from __future__ import annotations

import os

os.environ.setdefault(
    "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1"
)  # e3nn constants under torch>=2.6

import math

import torch
from e3nn import o3
from torch import nn

from src.augnet_model import (
    MAX_SCHEMA_COPIES,
    MAX_SCHEMA_DIM,
    SCHEMA_IRREPS,
    SCHEMA_L_CHANNELS,
    SCHEMA_NUM_COPIES,
    Z_TO_SCHEMA,
    build_e3nn_slices,
    build_sanvito_blocks,
)

# ----------------------------
# Maximal partial-wave basis
# ----------------------------

MAX_PW_L: list[int] = list(SCHEMA_L_CHANNELS[MAX_SCHEMA_DIM])

SCHEMA_NUM_PW: dict[int, int] = {}
for _schema, _l_channels in SCHEMA_L_CHANNELS.items():
    _n = len(_l_channels)
    assert list(_l_channels) == MAX_PW_L[:_n], (
        f"schema {_schema} partial waves {list(_l_channels)} are not a prefix of "
        f"the maximal basis {MAX_PW_L}; the shared head assumes a nested basis"
    )
    SCHEMA_NUM_PW[_schema] = _n


def _blocks_and_slices(l_channels: list[int]):
    """(pair, L) -> slice in the e3nn grouped layout for this partial-wave list."""
    blocks = build_sanvito_blocks(list(l_channels))
    return build_e3nn_slices(blocks)


MAX_E3NN_SLICES: dict[tuple[tuple[int, int], int], slice] = _blocks_and_slices(MAX_PW_L)

for _schema in SCHEMA_IRREPS:
    _slices = _blocks_and_slices(SCHEMA_L_CHANNELS[_schema])
    _mults: dict[int, int] = {}
    for _pair, _L in _slices:
        _mults[_L] = _mults.get(_L, 0) + 1
    _declared = {ir.l: mul for mul, ir in o3.Irreps(SCHEMA_IRREPS[_schema])}
    assert _mults == _declared, (
        f"schema {_schema}: CG coupling of partial waves gives {_mults}, "
        f"but SCHEMA_IRREPS declares {_declared}"
    )


class SharedPAWHead(nn.Module):
    """
    Args:
        hidden_irreps: backbone node-feature irreps; must contain l = 0..3.
        rank: number of outer-product terms k (even).
        linear_max_l: blocks with L <= this come from a direct linear map on h;
            -1 = none. Cannot exceed the largest l in hidden_irreps.
        block_mixing: learn a per-block combination over k after the coupling;
            False uses a fixed +-1 signature.

    forward(h, atomic_numbers) returns
        y_pad           [n_atoms, 390]   standardized coefficients, e3nn order
        mask            [n_atoms, 390]   True over each atom's schema
        ragged          dict schema -> {"atom_indices", "y", "gate_logits"}
        gate_logits_pad [n_atoms, MAX_SCHEMA_COPIES]  zeros
        gate_mask       [n_atoms, MAX_SCHEMA_COPIES]
    """

    def __init__(
        self,
        hidden_irreps: str,
        rank: int = 64,
        block_mixing: bool = True,
        linear_max_l: int = 0,
    ):
        super().__init__()

        if rank % 2 != 0:
            raise ValueError(f"rank must be even (signed signature), got {rank}")

        self.block_mixing = bool(block_mixing)
        self.linear_max_l = int(linear_max_l)

        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.rank = int(rank)
        self.max_dim = MAX_SCHEMA_DIM
        self.max_copies = MAX_SCHEMA_COPIES

        hidden_ls = {ir.l for _mul, ir in self.hidden_irreps}
        missing = sorted(set(MAX_PW_L) - hidden_ls)
        if missing:
            raise ValueError(
                f"hidden_irreps {self.hidden_irreps} lacks l={missing}, which the "
                f"maximal partial-wave basis {MAX_PW_L} needs. Raise max_ell."
            )

        max_hidden_l = max(hidden_ls)
        if self.linear_max_l > max_hidden_l:
            raise ValueError(
                f"linear_max_l={self.linear_max_l} exceeds the largest l in "
                f"hidden_irreps ({max_hidden_l}); those blocks would be "
                f"identically zero. Raise max_ell or lower linear_max_l."
            )

        self.slot_irreps = o3.Irreps([(self.rank, (l, (-1) ** l)) for l in MAX_PW_L])
        self.to_coeffs = o3.Linear(self.hidden_irreps, self.slot_irreps)

        signature = torch.ones(self.rank)
        signature[self.rank // 2 :] = -1.0
        self.register_buffer("signature", signature, persistent=False)

        # Plan order: all linear blocks, then all coupled blocks, grouped by
        # (l1, l2, L). Must match what couple() concatenates -- it is NOT the
        # MAX_E3NN_SLICES order, and a mismatch silently permutes same-L copies.
        groups: dict[tuple[int, int, int], dict[str, list]] = {}
        for (pair, L), sl in MAX_E3NN_SLICES.items():
            i, j = pair
            g = groups.setdefault(
                (MAX_PW_L[i], MAX_PW_L[j], L), {"i": [], "j": [], "blocks": []}
            )
            g["i"].append(i)
            g["j"].append(j)
            g["blocks"].append((pair, L))

        lin_blocks, quad_blocks = [], []  # (slot_i, slot_j, L, (pair, L))
        for (l1, l2, L), g in groups.items():
            dst = lin_blocks if L <= self.linear_max_l else quad_blocks
            for i, j, pair_L in zip(g["i"], g["j"], g["blocks"]):
                dst.append((i, j, L, pair_L))

        plan_cols: list[int] = []
        for _, _, _, pair_L in lin_blocks + quad_blocks:
            sl = MAX_E3NN_SLICES[pair_L]
            plan_cols.extend(range(sl.start, sl.stop))
        assert len(plan_cols) == self.max_dim
        assert sorted(plan_cols) == list(range(self.max_dim))

        max_to_plan = [0] * self.max_dim
        for plan_col, max_col in enumerate(plan_cols):
            max_to_plan[max_col] = plan_col

        self.n_linear_blocks = len(lin_blocks)
        self.n_coupled_blocks = len(quad_blocks)
        self._plan_blocks = lin_blocks + quad_blocks

        # .simplify() is load-bearing: one instruction per block makes e3nn emit
        # ~50 tiny GPU kernels (~20x slower). Blocks are L-major so it merges
        # without reordering.
        if lin_blocks:
            self.linear_head = o3.Linear(
                self.hidden_irreps,
                o3.Irreps(
                    [(1, (L, (-1) ** L)) for _, _, L, _ in lin_blocks]
                ).simplify(),
            )
        else:
            self.linear_head = None

        # Grouped einsums, not one o3.TensorProduct with per-block instructions:
        # that was ~18x slower on GPU (many tiny kernels).
        quad_groups = []  # (l1, l2, L, slot_i, slot_j, weight_slice)
        pos = 0
        for (l1, l2, L), g in groups.items():
            if L <= self.linear_max_l:
                continue
            n = len(g["i"])
            quad_groups.append(
                (l1, l2, L, list(g["i"]), list(g["j"]), slice(pos, pos + n))
            )
            pos += n
        assert pos == self.n_coupled_blocks
        self._quad_groups = quad_groups

        for gi, (l1, l2, L, _, _, _) in enumerate(quad_groups):
            # sqrt(2L+1) makes one coupling term unit-variance for unit-variance inputs.
            self.register_buffer(
                f"_w_{gi}",
                o3.wigner_3j(l1, l2, L) * math.sqrt(2 * L + 1),
                persistent=False,
            )

        slot_slices, pos = [], 0
        for l in MAX_PW_L:
            width = self.rank * (2 * l + 1)
            slot_slices.append((pos, pos + width))
            pos += width
        assert pos == self.slot_irreps.dim
        self._slot_slices = slot_slices

        if self.block_mixing:
            w0 = torch.randn(self.n_coupled_blocks, self.rank) / math.sqrt(self.rank)
            self.coupling_weight = nn.Parameter(w0)
        else:
            w0 = self.signature.unsqueeze(0).repeat(self.n_coupled_blocks, 1)
            self.register_buffer(
                "coupling_weight", w0 / math.sqrt(self.rank), persistent=False
            )

        # Per-schema gather: plan-order column for each of the schema's coefficients.
        schemas = sorted(SCHEMA_IRREPS)
        self._schemas = schemas
        gather = torch.zeros((len(schemas), self.max_dim), dtype=torch.long)
        keep = torch.zeros((len(schemas), self.max_dim), dtype=torch.bool)
        copies = torch.zeros((len(schemas), self.max_copies), dtype=torch.bool)

        for row, schema in enumerate(schemas):
            for (pair, L), sl in _blocks_and_slices(SCHEMA_L_CHANNELS[schema]).items():
                src = MAX_E3NN_SLICES[(pair, L)]
                for off in range(sl.stop - sl.start):
                    gather[row, sl.start + off] = max_to_plan[src.start + off]
            keep[row, :schema] = True
            copies[row, : SCHEMA_NUM_COPIES[schema]] = True

        self.register_buffer("_gather", gather, persistent=False)
        self.register_buffer("_keep", keep, persistent=False)
        self.register_buffer("_copies", copies, persistent=False)
        self.register_buffer(
            "_row_to_schema", torch.tensor(schemas, dtype=torch.long), persistent=False
        )

        max_z = max(Z_TO_SCHEMA)
        z_row = torch.zeros(max_z + 1, dtype=torch.long)
        for z, schema in Z_TO_SCHEMA.items():
            z_row[z] = schemas.index(schema)
        self.register_buffer("_z_to_row", z_row, persistent=False)

    def couple(
        self, coeffs: torch.Tensor, l0: torch.Tensor | None = None
    ) -> torch.Tensor:
        """All 390 block coefficients in plan order [linear blocks | coupled blocks].
        coeffs: [n_atoms, slot_irreps.dim] from to_coeffs; l0: linear_head output."""
        if self.linear_head is not None and l0 is None:
            raise ValueError(
                "a linear readout is configured, so couple() needs its output"
            )

        n_atoms = coeffs.shape[0]
        c = [
            coeffs[:, a:b].reshape(n_atoms, self.rank, 2 * l + 1)
            for l, (a, b) in zip(MAX_PW_L, self._slot_slices)
        ]

        pieces = [l0] if l0 is not None else []
        for gi, (_, _, _, slot_i, slot_j, wsl) in enumerate(self._quad_groups):
            w = getattr(self, f"_w_{gi}")
            mix = self.coupling_weight[wsl].unsqueeze(0).unsqueeze(-1)
            a = torch.stack([c[i] for i in slot_i], dim=1) * mix
            b = torch.stack([c[j] for j in slot_j], dim=1)
            out = torch.einsum("npka,npkb,abm->npm", a, b, w)
            pieces.append(out.reshape(n_atoms, -1))

        return torch.cat(pieces, dim=1)

    def forward(self, h: torch.Tensor, atomic_numbers: torch.Tensor):
        n_atoms = h.shape[0]

        coeffs = self.to_coeffs(h)
        l0 = self.linear_head(h) if self.linear_head is not None else None
        y_plan = self.couple(coeffs, l0)

        rows = self._z_to_row[atomic_numbers]
        mask = self._keep[rows]
        y_pad = y_plan.gather(1, self._gather[rows]) * mask

        gate_logits_pad = h.new_zeros((n_atoms, self.max_copies))
        gate_mask = self._copies[rows]

        ragged = {}
        schemas_per_atom = self._row_to_schema[rows]
        for schema in self._schemas:
            idx = torch.where(schemas_per_atom == schema)[0]
            if idx.numel() == 0:
                continue
            ragged[schema] = {
                "atom_indices": idx,
                "y": y_pad[idx, :schema],
                "gate_logits": gate_logits_pad[idx, : SCHEMA_NUM_COPIES[schema]],
            }

        return y_pad, mask, ragged, gate_logits_pad, gate_mask


if __name__ == "__main__":
    torch.manual_seed(0)

    hidden = "128x0e + 128x1o + 128x2e + 128x3o"
    head = SharedPAWHead(hidden)

    atomic_numbers = torch.tensor([1, 8, 35, 42, 57])  # schemas 15, 33, 78, 138, 390
    h = torch.randn(len(atomic_numbers), o3.Irreps(hidden).dim)

    y, mask, ragged, gl, gm = head(h, atomic_numbers)
    print("y", tuple(y.shape), "filled per atom", mask.sum(dim=1).tolist())
    print("params", sum(p.numel() for p in head.parameters()))
