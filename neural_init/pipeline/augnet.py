"""AugNet inference: structure -> PAW augmentation occupancies in CHGCAR (Sanvito) order."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from neural_init._resources import resource_path
from neural_init.augnet.augnet_model import Z_TO_SCHEMA, lmaxmix_component_mask
from neural_init.augnet.paw_basis_transform import (
    e3nn_to_sanvito_with_basis_padded,
    sanvito_schema_and_lmaxmix_masks,
)
from neural_init.augnet.paw_stats import PAWStats
from neural_init.augnet.train_augnet import (
    PAWAugModel,
    _atomic_data_from_item,
    build_cueq_config,
    build_oeq_config,
    dict_to_namespace,
    hidden_irreps_from_width,
)
from neural_init.models import resolve_weights


class AugNetPredictor:
    """One loaded checkpoint (total or mag channel) plus its target transform."""

    def __init__(self, ckpt_path: Path, stats_path: Path, device: torch.device, enable_cueq: bool = True):
        from mace.tools.utils import AtomicNumberTable

        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hp = blob["hyper_parameters"]
        cfg = dict_to_namespace(hp["config"])
        self.channel: str = hp["target_channel"]
        cfg.model.enable_cueq = bool(enable_cueq)
        cfg.model.enable_oeq = False  # CUDA-JIT kernel; the conv tensor product carries no weights
        self.r_max = float(cfg.model.r_max)

        avg = cfg.model.avg_num_neighbors  # resolved float stored by train_augnet.main()
        if isinstance(avg, str):
            raise ValueError(f"{ckpt_path.name}: avg_num_neighbors={avg!r} was not resolved at training time")

        atomic_numbers = sorted(Z_TO_SCHEMA)
        self.z_table = AtomicNumberTable(atomic_numbers)
        self.model = PAWAugModel(
            hidden_irreps=hidden_irreps_from_width(cfg.model.hidden_width, cfg.model.max_ell),
            atomic_numbers=atomic_numbers,
            target_channel=self.channel,
            r_max=self.r_max,
            max_ell=int(cfg.model.max_ell),
            num_interactions=int(cfg.model.num_interactions),
            correlation=int(cfg.model.correlation),
            avg_num_neighbors=float(avg),
            head_rank=int(getattr(cfg.model, "head_rank", 64)),
            head_block_mixing=bool(getattr(cfg.model, "head_block_mixing", True)),
            head_linear_max_l=int(getattr(cfg.model, "head_linear_max_l", 0)),
            cueq_config=build_cueq_config(cfg),
            oeq_config=build_oeq_config(cfg),
        )
        # LightningPAWModule wrapped the model as `self.model`
        state = {k[len("model."):]: v for k, v in blob["state_dict"].items() if k.startswith("model.")}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        params = {n for n, _ in self.model.named_parameters()}
        missing_params = [k for k in missing if k in params]
        if unexpected or missing_params:  # same rule as train_augnet.load_init_weights: buffers may differ
            raise RuntimeError(
                f"{ckpt_path.name}: state_dict mismatch (unexpected={unexpected[:5]}, missing params={missing_params[:5]})"
            )
        self.model.to(device).eval()
        self.device = device
        self.stats = PAWStats.load(str(stats_path))

    def _batch(self, structure: Structure):
        from mace.tools.torch_geometric import Batch

        z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
        bad = sorted({int(x) for x in z.tolist() if int(x) not in Z_TO_SCHEMA})
        if bad:
            raise KeyError(f"AugNet has no PAW schema for Z={bad} (89 MP POTCAR elements supported)")
        item = {
            "atomic_numbers": z,
            "positions": torch.tensor(structure.cart_coords, dtype=torch.float64),
            "cell": torch.tensor(structure.lattice.matrix, dtype=torch.float64),
            "pbc": torch.tensor([True, True, True]),
        }
        batch = Batch.from_data_list([_atomic_data_from_item(item, self.z_table, self.r_max)])
        n = int(z.shape[0])
        if "batch" not in batch:
            batch["batch"] = torch.zeros(n, dtype=torch.long)
        if "ptr" not in batch:
            batch["ptr"] = torch.tensor([0, n], dtype=torch.long)
        batch["head"] = torch.zeros(1, dtype=torch.long)
        batch["node_heads"] = torch.zeros(n, dtype=torch.long)
        batch["atomic_numbers"] = z
        return batch.to(self.device)

    @torch.no_grad()
    def predict(self, structure: Structure, lmaxmix: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (aug_sanvito_padded [n_atoms, 390], schema_mask [n_atoms, 390]).

        Same order of operations as train_augnet.export_test_augmentation_predictions:
        de-standardize (adds the free-atom occupancies) in e3nn order, zero L>LMAXMIX
        components as VASP writes them, then rotate/permute into CHGCAR order.
        """
        batch = self._batch(structure)
        out = self.model(batch)
        z = batch["atomic_numbers"]
        y = self.stats.de_standardize(out["paw_padded"], z, self.channel)
        y = y * out["paw_schema_mask"].to(y.dtype)
        if lmaxmix is not None:
            y = y * lmaxmix_component_mask(z, int(lmaxmix)).to(y.dtype)
        y_san = e3nn_to_sanvito_with_basis_padded(y, z)
        schema_mask, _ = sanvito_schema_and_lmaxmix_masks(z, None)
        return y_san.detach().cpu().numpy(), schema_mask.detach().cpu().numpy()


def load_augnet(checkpoint: str, stats: str | None, device: torch.device, enable_cueq: bool = True,
                weights_dir=None) -> AugNetPredictor:
    ckpt = resolve_weights(checkpoint, weights_dir)
    stats_path = Path(stats) if stats else resource_path("augnet", "stats/paw_ref_freeatom.pt")
    return AugNetPredictor(ckpt, stats_path, device, enable_cueq)


def aug_blocks(aug_padded: np.ndarray, atomic_numbers) -> dict[int, np.ndarray]:
    """pymatgen ``data_aug`` layout: 1-based ion index -> the element's schema-length block."""
    return {i + 1: np.asarray(aug_padded[i, : Z_TO_SCHEMA[int(z)]], dtype=float) for i, z in enumerate(atomic_numbers)}
