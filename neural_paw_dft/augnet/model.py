"""AugNet model definition and the helpers needed to rebuild it from a config.

Kept separate from the trainer so inference (``neural_paw_dft.pipeline.augnet``) does not import
the training code.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from e3nn import o3
from mace.data import AtomicData
from mace.modules import interaction_classes
from mace.modules.models import MACE
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig, OEQConfig, get_layout
from mace.tools.utils import AtomicNumberTable

from .paw_head_shared import SharedPAWHead


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = dict_to_namespace(v)
        elif isinstance(v, list):
            out[k] = [
                dict_to_namespace(x) if isinstance(x, dict) else x
                for x in v
            ]
        else:
            out[k] = v
    return SimpleNamespace(**out)


def hidden_irreps_from_width(width: int, max_ell: int) -> str:
    terms = []
    for ell in range(max_ell + 1):
        parity = "e" if ell % 2 == 0 else "o"
        terms.append(f"{int(width)}x{ell}{parity}")
    return " + ".join(terms)


def atomic_data_from_item(item: Dict, z_table: AtomicNumberTable, r_max: float):
    from mace.data import Configuration

    atomic_numbers_np = item["atomic_numbers"].cpu().numpy().astype(np.int64)
    positions_np = item["positions"].cpu().numpy().astype(np.float64)
    cell_np = item["cell"].cpu().numpy().astype(np.float64)
    pbc_tuple = tuple(bool(x) for x in item["pbc"].cpu().numpy().tolist())

    n_atoms = atomic_numbers_np.shape[0]

    config = Configuration(
        atomic_numbers=atomic_numbers_np,
        positions=positions_np,
        properties={
            "energy": np.array(0.0, dtype=np.float64),
            "forces": np.zeros((n_atoms, 3), dtype=np.float64),
        },
        property_weights={
            "energy": 0.0,
            "forces": 0.0,
        },
        cell=cell_np,
        pbc=pbc_tuple,
        weight=1.0,
        config_type="Default",
        head="Default",
    )

    return AtomicData.from_config(
        config,
        z_table=z_table,
        cutoff=r_max,
        heads=["Default"],
    )


def build_cueq_config(cfg: SimpleNamespace) -> Optional[CuEquivarianceConfig]:
    if not getattr(cfg.model, "enable_cueq", False):
        return None

    layout = getattr(cfg.model, "cueq_layout", "mul_ir")

    # Hybrid with OEQ: oeq owns the channelwise conv, so cueq must not claim it
    # (optimize_all would route the conv to cueq's non-fused TP and leave oeq dead).
    hybrid_with_oeq = bool(getattr(cfg.model, "enable_oeq", False))
    config = CuEquivarianceConfig(
        enabled=True,
        layout=layout,
        group="O3_e3nn",
        optimize_all=not hybrid_with_oeq,
        optimize_linear=hybrid_with_oeq,
        optimize_symmetric=hybrid_with_oeq,
        optimize_fctp=hybrid_with_oeq,
        optimize_channelwise=False,
        conv_fusion=False,
    )
    if not config.enabled:
        print("WARNING: enable_cueq set but cuequivariance unavailable; using e3nn.")
        return None
    return config


def build_oeq_config(cfg: SimpleNamespace) -> Optional[OEQConfig]:
    if not getattr(cfg.model, "enable_oeq", False):
        return None

    config = OEQConfig(
        enabled=True,
        optimize_all=True,
        conv_fusion="atomic",
    )
    if not config.enabled:
        print("WARNING: enable_oeq set but openequivariance unavailable; using e3nn.")
        return None
    return config


class MACEBackbone(nn.Module):
    def __init__(
        self,
        atomic_numbers: List[int],
        hidden_irreps: str,
        r_max: float,
        max_ell: int,
        num_interactions: int,
        correlation: int,
        avg_num_neighbors: float,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        use_reduced_cg: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        super().__init__()

        if use_reduced_cg and not CUET_AVAILABLE:
            raise RuntimeError(
                "use_reduced_cg=True needs cuequivariance: it selects the reduced "
                "Clebsch-Gordan basis, which changes the symmetric-contraction weight "
                "shapes. Install cuequivariance, or load weights exported with "
                "use_reduced_cg=False."
            )

        self.atomic_numbers = sorted(set(int(z) for z in atomic_numbers))
        self.z_table = AtomicNumberTable(self.atomic_numbers)
        self.r_max = r_max
        self.hidden_irreps = o3.Irreps(hidden_irreps)
        self.node_feats_irreps = (self.hidden_irreps * num_interactions).simplify()

        self.model = MACE(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
            max_ell=max_ell,
            interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
            num_interactions=num_interactions,
            num_elements=len(self.atomic_numbers),
            hidden_irreps=self.hidden_irreps,
            MLP_irreps=o3.Irreps("64x0e"),
            atomic_energies=np.zeros(len(self.atomic_numbers), dtype=np.float64),
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=self.atomic_numbers,
            correlation=correlation,
            gate=torch.nn.functional.silu,
            radial_MLP=[64, 64, 64],
            keep_last_layer_irreps=True,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

        # The head is pure e3nn (mul_ir); transpose back when cueq runs in ir_mul.
        self._to_mul_ir = None
        if get_layout(cueq_config) == "ir_mul":
            import cuequivariance as cue
            import cuequivariance_torch as cuet

            self._to_mul_ir = cuet.TransposeIrrepsLayout(
                cue.Irreps(cueq_config.group, str(self.node_feats_irreps)),
                source=cue.ir_mul,
                target=cue.mul_ir,
                use_fallback=True,
            )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        out = self.model(
            batch,
            training=self.training,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
            compute_edge_forces=False,
            compute_atomic_stresses=False,
        )

        if "node_feats" not in out:
            raise RuntimeError("MACE did not return node_feats.")

        node_feats = out["node_feats"]
        if self._to_mul_ir is not None:
            node_feats = self._to_mul_ir(node_feats)

        return {"node_feats": node_feats}


class PAWAugModel(nn.Module):
    def __init__(
        self,
        hidden_irreps: str,
        atomic_numbers: List[int],
        target_channel: str,
        r_max: float,
        max_ell: int,
        num_interactions: int,
        correlation: int,
        avg_num_neighbors: float,
        head_rank: int = 64,
        head_block_mixing: bool = True,
        head_linear_max_l: int = 0,
        use_reduced_cg: bool = True,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        super().__init__()

        if target_channel not in {"total", "mag"}:
            raise ValueError(f"target_channel must be total or mag, got {target_channel}")

        self.target_channel = target_channel

        self.backbone = MACEBackbone(
            atomic_numbers=atomic_numbers,
            hidden_irreps=hidden_irreps,
            r_max=r_max,
            max_ell=max_ell,
            num_interactions=num_interactions,
            correlation=correlation,
            avg_num_neighbors=avg_num_neighbors,
            use_reduced_cg=use_reduced_cg,
            cueq_config=cueq_config,
            oeq_config=oeq_config,
        )

        self.paw_head = SharedPAWHead(
            hidden_irreps=str(self.backbone.node_feats_irreps),
            rank=head_rank,
            block_mixing=head_block_mixing,
            linear_max_l=head_linear_max_l,
        )

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        h = self.backbone(batch)["node_feats"]
        y, schema_mask, ragged, _, _ = self.paw_head(
            h=h,
            atomic_numbers=batch["atomic_numbers"],
        )
        return {
            "paw_padded": y,
            "paw_schema_mask": schema_mask,
            "paw_ragged": ragged,
        }
