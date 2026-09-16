# train_augnet.py
from __future__ import annotations

import os
# e3nn (<=0.4.x) torch.load()s cached constants at import; torch>=2.6 rejects them
# under the default weights_only=True. Must be set before e3nn is imported.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse
import csv
import json
import math
import random
import signal
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from e3nn import o3
from torch.utils.data import Dataset, DataLoader

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        Callback,
        ModelCheckpoint,
        LearningRateMonitor,
        ModelSummary,
    )
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    pl = None
    Callback = object
    ModelCheckpoint = None
    LearningRateMonitor = None
    ModelSummary = None
    WandbLogger = None

from mace.data import AtomicData
from mace.tools.utils import AtomicNumberTable
from mace.modules.models import MACE
from mace.modules import interaction_classes
from mace.modules.wrapper_ops import CUET_AVAILABLE, CuEquivarianceConfig, OEQConfig, get_layout

from neural_paw_dft._resources import resolve_data_path
from .run_paw_chgcar import (
    build_training_example_from_chgcar_dual,
    predicted_aug_blocks_text,
    resolve_lmaxmix,
)
from .paw_basis_transform import (
    e3nn_to_sanvito_with_basis_padded,
    sanvito_schema_and_lmaxmix_masks,
    set_l_channel_overrides,
)
from .augnet_model import Z_TO_SCHEMA, lmaxmix_component_mask
from .paw_head_shared import SharedPAWHead
from .paw_moments import moment_regularization_loss, moments_from_coeffs
from .paw_stats import PAWStats, compute_paw_stats, standardize, de_standardize


# ----------------------------
# Config utilities
# ----------------------------

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


def namespace_to_dict(ns: Any) -> Any:
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    if isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    if isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    return ns


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        else:
            out[key] = v
    return out


def load_config(config_path: str | Path) -> SimpleNamespace:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    if cfg_dict is None:
        raise ValueError(f"Config file is empty: {config_path}")

    cfg = dict_to_namespace(cfg_dict)
    cfg.config_path = str(config_path)
    return cfg


def str_to_bool_or_auto(x) -> str:
    if isinstance(x, bool):
        return "true" if x else "false"
    x = str(x).lower().strip()
    if x not in {"auto", "true", "false"}:
        raise ValueError(f"Expected auto/true/false, got {x}")
    return x


def should_use_hpc_paths(cfg: SimpleNamespace) -> bool:
    mode = str_to_bool_or_auto(getattr(cfg.runtime, "use_hpc_paths", "auto"))
    if mode == "true":
        return True
    if mode == "false":
        return False
    return torch.cuda.is_available()


def apply_runtime_paths(cfg: SimpleNamespace) -> SimpleNamespace:
    use_hpc = should_use_hpc_paths(cfg)
    cfg.runtime.is_hpc = use_hpc

    if use_hpc:
        cfg.project.effective_data_dir = cfg.project.hpc_data_dir
        cfg.project.effective_out_dir = cfg.project.hpc_out_dir
    else:
        cfg.project.effective_data_dir = cfg.project.data_dir
        cfg.project.effective_out_dir = cfg.project.out_dir

    return cfg


def make_project_name_from_config(cfg: SimpleNamespace) -> str:
    base = getattr(cfg.project, "base_name", "augnet")
    split_name = getattr(cfg.project, "split_name", None)

    if split_name is None or str(split_name).strip() == "":
        return base

    return f"{base}-{split_name}"


def make_run_name_from_config(cfg: SimpleNamespace) -> str:
    """Run name from the knobs that vary between runs; the full config is logged to wandb."""
    channel = cfg.data.augmentation_channel
    if channel == "mag" and str(getattr(cfg.data, "mag_scale_source", "mag")) == "total":
        channel = "magtscale"

    mix = "mix" if bool(getattr(cfg.model, "head_block_mixing", True)) else "nomix"
    lml = int(getattr(cfg.model, "head_linear_max_l", 0))
    l0 = f"lin{lml}" if lml >= 0 else "quad"
    head_tag = f"shared{int(getattr(cfg.model, 'head_rank', 64))}{mix}{l0}"

    name = (
        f"{channel}_{head_tag}"
        f"_w{cfg.model.hidden_width}"
        f"_l{cfg.model.max_ell}"
        f"_r{cfg.model.r_max:g}"
        f"_c{cfg.model.correlation}"
    )

    split_name = str(getattr(cfg.project, "split_name", "") or "").strip()
    if split_name:
        name = f"{name}_{split_name}"

    return name.replace(".", "p")


def apply_auto_names(cfg: SimpleNamespace) -> SimpleNamespace:
    project_name = make_project_name_from_config(cfg)
    run_name = make_run_name_from_config(cfg)

    if not hasattr(cfg.project, "name") or str(cfg.project.name).lower() == "auto":
        cfg.project.name = project_name

    if str(getattr(cfg.project, "run_name", "auto")).lower() == "auto":
        cfg.project.run_name = run_name

    if getattr(cfg.wandb, "project", "auto") is None or str(cfg.wandb.project).lower() == "auto":
        cfg.wandb.project = project_name

    if getattr(cfg.wandb, "name", "auto") is None or str(cfg.wandb.name).lower() == "auto":
        cfg.wandb.name = run_name

    return cfg


def hidden_irreps_from_width(width: int, max_ell: int) -> str:
    terms = []
    for ell in range(max_ell + 1):
        parity = "e" if ell % 2 == 0 else "o"
        terms.append(f"{int(width)}x{ell}{parity}")
    return " + ".join(terms)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------
# Files / splits
# ----------------------------

def material_id_from_path(path: str | Path) -> str:
    name = Path(path).name
    for suffix in [
        ".chgcar.lz4",
        ".CHGCAR.lz4",
        ".chgcar",
        ".CHGCAR",
        ".vasp",
        ".lz4",
    ]:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def find_chgcars(data_dir: str | Path) -> List[Path]:
    data_dir = Path(data_dir)

    patterns = [
        "*.chgcar",
        "*.CHGCAR",
        "*.chgcar.lz4",
        "*.CHGCAR.lz4",
        "*CHGCAR*",
        "*.vasp",
    ]

    files = []
    for pat in patterns:
        files.extend(data_dir.glob(pat))
        files.extend(data_dir.rglob(pat))

    files = sorted(set(p for p in files if p.is_file()))

    if not files:
        raise FileNotFoundError(f"No CHGCAR-like files found in {data_dir}")

    return files


def load_split_json(split_file: str | Path) -> Dict[str, List[str]]:
    with open(split_file, "r") as f:
        split = json.load(f)

    for key in ["train", "validation", "test"]:
        if key not in split:
            raise ValueError(f"Split file {split_file} missing key '{key}'")

    return split


def build_material_file_index(data_dir: str | Path) -> Dict[str, Path]:
    files = find_chgcars(data_dir)

    file_index = {}
    duplicates = {}

    for p in files:
        mid = material_id_from_path(p)

        if mid in file_index:
            duplicates.setdefault(mid, []).append(p)
            continue

        file_index[mid] = p

    if duplicates:
        print(f"Warning: found {len(duplicates)} duplicate material IDs. Keeping first occurrence.")
        for mid, paths in list(duplicates.items())[:10]:
            print(f"  duplicate {mid}: kept {file_index[mid]}, ignored {paths[:3]}")

    print(f"Indexed {len(file_index)} CHGCAR-like files in {data_dir}")

    return file_index


def files_from_split_ids(
    file_index: Dict[str, Path],
    split_ids: List[str],
    split_name: str,
    strict: bool = False,
):
    files = []
    missing = []

    for x in split_ids:
        mid = material_id_from_path(x)

        if mid in file_index:
            files.append(file_index[mid])
        else:
            missing.append(mid)

    if missing:
        msg = (
            f"{split_name}: skipping {len(missing)} missing IDs. "
            f"First missing: {missing[:10]}"
        )

        if strict:
            raise FileNotFoundError(msg)

        print("Warning:", msg)

    return files, missing


def get_train_val_test_files(cfg: SimpleNamespace):
    """HPC: split JSON (unresolvable IDs skipped). Local CPU: on-the-fly element-covered split."""
    use_json_split = (
        bool(cfg.runtime.is_hpc)
        and bool(getattr(cfg.data, "use_split_file_on_hpc", True))
    )

    if use_json_split:
        split_file = resolve_data_path(cfg.data.split_file, "augnet")
        split = load_split_json(split_file)

        strict = bool(getattr(cfg.data, "strict_split_ids", False))
        file_index = build_material_file_index(cfg.project.effective_data_dir)

        train_files, train_missing = files_from_split_ids(
            file_index,
            split["train"],
            "train",
            strict=strict,
        )
        val_files, val_missing = files_from_split_ids(
            file_index,
            split["validation"],
            "validation",
            strict=strict,
        )
        test_files, test_missing = files_from_split_ids(
            file_index,
            split["test"],
            "test",
            strict=strict,
        )

        missing = {
            "train": train_missing,
            "validation": val_missing,
            "test": test_missing,
        }

        print(f"Using split file: {split_file}")
        print(
            f"  train={len(train_files)} / {len(split['train'])} "
            f"validation={len(val_files)} / {len(split['validation'])} "
            f"test={len(test_files)} / {len(split['test'])}"
        )

        if len(train_files) == 0:
            raise RuntimeError("No train files resolved from split.")
        if len(val_files) == 0:
            raise RuntimeError("No validation files resolved from split.")
        if len(test_files) == 0:
            print("Warning: no test files resolved from split; test/export will be skipped.")

        return train_files, val_files, test_files, True, missing

    files = find_chgcars(cfg.project.effective_data_dir)
    print("Using local on-the-fly element-covered split.")
    return files, None, None, False, {"train": [], "validation": [], "test": []}


def make_element_covered_split(dataset: Dataset, val_fraction: float, seed: int = 42):
    rng = random.Random(seed)
    all_indices = []

    elem_sets = {}
    global_counts = {}

    for idx in range(len(dataset)):
        item = dataset[idx]
        if item is None:
            continue
        elems = set(int(z) for z in item["atomic_numbers"].tolist())
        all_indices.append(idx)
        elem_sets[idx] = elems

        for z in elems:
            global_counts[z] = global_counts.get(z, 0) + 1

    n = len(all_indices)
    target_n_val = max(1, int(round(n * val_fraction)))

    candidate_val = []
    for idx in all_indices:
        if all(global_counts[z] >= 2 for z in elem_sets[idx]):
            candidate_val.append(idx)

    rng.shuffle(candidate_val)

    val_indices = []
    train_indices = set(all_indices)
    train_elem_counts = dict(global_counts)

    for idx in candidate_val:
        if len(val_indices) >= target_n_val:
            break

        elems = elem_sets[idx]
        can_move = all(train_elem_counts[z] - 1 >= 1 for z in elems)

        if can_move:
            val_indices.append(idx)
            train_indices.remove(idx)
            for z in elems:
                train_elem_counts[z] -= 1

    if len(val_indices) == 0:
        raise RuntimeError(
            "Could not create element-covered split. Use split JSON or more data."
        )

    train_indices = sorted(train_indices)
    val_indices = sorted(val_indices)

    print("Element-covered split:")
    print(f"  train={len(train_indices)} val={len(val_indices)}")

    return train_indices, val_indices


# ----------------------------
# Dataset / collate
# ----------------------------

def resolve_avg_num_neighbors(cfg: SimpleNamespace) -> float:
    """A float in cfg.model.avg_num_neighbors is used as is; "auto" reads the measured
    table (neighbor_stats.py) and requires an exact r_max hit, never interpolating."""
    raw = getattr(cfg.model, "avg_num_neighbors", None)

    if raw is not None and not (isinstance(raw, str) and raw.lower() == "auto"):
        return float(raw)

    path = getattr(cfg.model, "neighbor_stats_file", None)
    if not path:
        raise ValueError(
            "model.avg_num_neighbors: auto needs model.neighbor_stats_file (or "
            "--neighbor-stats-file) pointing at a neighbor_stats_<split>.json. "
            "Generate it with scripts/neighbor_stats.py."
        )
    path = resolve_data_path(path, "augnet")

    blob = json.loads(Path(path).read_text())
    table = {float(k): v for k, v in blob["avg_num_neighbors"].items() if v is not None}

    r_max = float(cfg.model.r_max)
    if r_max not in table:
        raise KeyError(
            f"r_max={r_max:g} is not in {path} (measured: "
            f"{sorted(table)}). Re-run scripts/neighbor_stats.py with "
            f"R_MAXES='... {r_max:g} ...' -- refusing to interpolate, since a wrong "
            "avg_num_neighbors silently mis-scales message passing."
        )
    return float(table[r_max])


def resolve_lmaxmix_mode(mode: str | None) -> str:
    mode = str(mode or "off").strip().lower()
    if mode in {"off", "none", "null", ""}:
        return "off"
    if mode == "auto":
        return "auto"
    raise ValueError(f"data.lmaxmix_mode must be auto or off; got {mode!r}")


class PAWCHGCARDataset(Dataset):
    def __init__(
        self,
        files: List[Path],
        target_order: str = "e3nn",
        target_channel: str = "total",
        stats: "PAWStats | None" = None,
        l_channel_overrides: Dict | None = None,
        lmaxmix_mode: str = "off",
    ):
        self.files = list(files)
        self.target_order = target_order
        self.target_channel = target_channel
        # "auto": mask each structure's L>LMAXMIX coefficients (VASP writes them as 0.0).
        self.lmaxmix_mode = resolve_lmaxmix_mode(lmaxmix_mode)
        # Kept on the dataset so spawned dataloader workers re-apply them.
        self.l_channel_overrides = dict(l_channel_overrides or {})
        self.stats = stats
        self._skipped: set[str] = set()

        if target_channel not in {"total", "mag"}:
            raise ValueError(f"target_channel must be total or mag, got {target_channel}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        path = str(self.files[idx])

        if self.l_channel_overrides:
            set_l_channel_overrides(self.l_channel_overrides)

        # An unreadable CHGCAR must not kill the run: None is dropped by collate.
        try:
            batch_stub, targets = build_training_example_from_chgcar_dual(
                path,
                target_order=self.target_order,
                with_lmaxmix=(self.lmaxmix_mode == "auto"),
            )
        except Exception as e:
            if path not in self._skipped:
                self._skipped.add(path)
                print(
                    f"  [skip] {path}: {type(e).__name__}: {e}",
                    file=sys.stderr,
                    flush=True,
                )
            return None

        total_target = targets["total_target"].float()
        total_mask = targets["total_mask"].bool()
        mag_target = targets["mag_target"].float()
        mag_mask = targets["mag_mask"].bool()

        if self.target_channel == "total":
            target = total_target
            mask = total_mask
        else:
            target = mag_target
            mask = mag_mask

        atomic_numbers = batch_stub.atomic_numbers.long()

        lmaxmix = resolve_lmaxmix(targets.get("lmaxmix"))
        if lmaxmix is not None:
            mask = mask & lmaxmix_component_mask(atomic_numbers, lmaxmix)

        target_phys = target
        if self.stats is not None:
            target = standardize(target, atomic_numbers, self.target_channel, self.stats)

        # sum_atoms |Tr(D_mag)| in electrons, from the physical mag target regardless of
        # the trained channel; Tr(D) reads only L=0 blocks, which LMAXMIX never truncates.
        if self.target_order == "e3nn" and bool(targets["has_mag"]):
            mag_tr1, _ = moments_from_coeffs(mag_target, atomic_numbers, [1])
            mag_moment = float(mag_tr1[:, 0].abs().sum())
        else:
            mag_moment = 0.0

        return {
            "path": path,
            "material_id": material_id_from_path(path),

            "atomic_numbers": atomic_numbers,
            "positions": batch_stub.positions.float(),
            "cell": batch_stub.cell.float(),
            "pbc": batch_stub.pbc.bool(),

            "target": target,
            "target_phys": target_phys,
            "mask": mask,
            "has_mag": torch.tensor(bool(targets["has_mag"]), dtype=torch.bool),
            "mag_moment": torch.tensor(mag_moment, dtype=torch.float32),
            "lmaxmix": -1 if lmaxmix is None else int(lmaxmix),  # -1 == not detected
        }


def _atomic_data_from_item(item: Dict, z_table: AtomicNumberTable, r_max: float):
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


def collate_mace_graphs(items: List[Dict], z_table: AtomicNumberTable, r_max: float):
    items = [it for it in items if it is not None]
    if len(items) == 0:
        return None

    from mace.tools.torch_geometric import Batch

    data_list = [_atomic_data_from_item(item, z_table, r_max) for item in items]
    batch = Batch.from_data_list(data_list)

    n_total = sum(int(it["atomic_numbers"].shape[0]) for it in items)
    n_structures = len(items)

    if "batch" not in batch:
        batch["batch"] = torch.zeros(n_total, dtype=torch.long)
    if "ptr" not in batch:
        batch["ptr"] = torch.tensor([0, n_total], dtype=torch.long)

    batch["head"] = torch.zeros(n_structures, dtype=torch.long)
    batch["node_heads"] = torch.zeros(n_total, dtype=torch.long)

    # Node order == items order, so per-atom fields concatenate aligned.
    batch["atomic_numbers"] = torch.cat([it["atomic_numbers"] for it in items], dim=0)
    batch["target"] = torch.cat([it["target"] for it in items], dim=0)
    batch["target_phys"] = torch.cat([it["target_phys"] for it in items], dim=0)
    batch["mask"] = torch.cat([it["mask"] for it in items], dim=0)

    batch["paths"] = [it["path"] for it in items]
    batch["material_ids"] = [it["material_id"] for it in items]
    batch["lmaxmix"] = [it["lmaxmix"] for it in items]
    batch["has_mag"] = torch.stack([it["has_mag"].reshape(()) for it in items])
    batch["mag_moment"] = torch.stack([it["mag_moment"].reshape(()) for it in items])

    return batch


# ----------------------------
# Model
# ----------------------------

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


# ----------------------------
# Loss / metrics
# ----------------------------

def masked_mse_loss(pred, target, mask):
    if not mask.any():
        return pred.new_tensor(0.0)
    return (((pred - target) ** 2) * mask.float()).sum() / mask.float().sum()


def masked_mae(pred, target, mask):
    if not mask.any():
        return pred.new_tensor(0.0)
    return ((pred - target).abs() * mask.float()).sum() / mask.float().sum()


def masked_huber_loss(pred, target, mask, delta=1.0):
    if not mask.any():
        return pred.new_tensor(0.0)

    err = pred - target
    abs_err = err.abs()
    quad = torch.minimum(abs_err, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    lin = abs_err - quad
    elem = 0.5 * quad * quad + delta * lin

    return (elem * mask.float()).sum() / mask.float().sum()


def masked_error_stats(pred, target, mask):
    valid_err = (pred - target)[mask]

    if valid_err.numel() == 0:
        z = pred.new_tensor(0.0)
        return {"mae": z, "rmse": z, "maxae": z}

    abs_err = valid_err.abs()

    return {
        "mae": abs_err.mean(),
        "rmse": torch.sqrt((valid_err ** 2).mean()),
        "maxae": abs_err.max(),
    }


def relative_mae_by_atom(pred, target, mask, eps: float = 1e-8):
    abs_err = (pred - target).abs() * mask.float()
    abs_tgt = target.abs() * mask.float()

    mae = abs_err.sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)
    scale = abs_tgt.sum(dim=1) / mask.float().sum(dim=1).clamp_min(1.0)

    return (mae / (scale + eps)).mean()


def sanvito_component_error_stats_from_e3nn(pred, target, mask, atomic_numbers):
    pred_sanvito = e3nn_to_sanvito_with_basis_padded(pred, atomic_numbers)
    target_sanvito = e3nn_to_sanvito_with_basis_padded(target, atomic_numbers)
    return masked_error_stats(pred_sanvito, target_sanvito, mask)


def spin_error_stats(pred_phys, target_phys, mask, batch, mag_min: float):
    """Per-structure (num, den, magnetic): num = sum|pred - m|, den = sum|m| over the
    valid coefficients, so num/den == 1 is the trivial m == 0 predictor. Normalizing by
    sum|m| rather than the net moment keeps antiferromagnets well-defined.
    magnetic = sum_atoms |Tr(D_mag)| >= mag_min electrons."""
    absdiff = ((pred_phys - target_phys).abs() * mask.to(pred_phys.dtype)).sum(dim=1)
    absref = (target_phys.abs() * mask.to(target_phys.dtype)).sum(dim=1)

    node_batch = batch["batch"]
    n_structures = int(node_batch.max().item()) + 1 if node_batch.numel() else 0

    num = absdiff.new_zeros(n_structures).index_add_(0, node_batch, absdiff)
    den = absref.new_zeros(n_structures).index_add_(0, node_batch, absref)

    magnetic = batch["mag_moment"].to(num.device).reshape(-1)[:n_structures] >= mag_min

    return num, den, magnetic


def single_channel_loss_and_metrics(model, batch, cfg, paw_stats=None, channel="total"):
    out = model(batch)

    pred = out["paw_padded"]
    target = batch["target"]  # standardized when paw_stats is set
    target_phys = batch["target_phys"]
    mask = batch["mask"]

    if cfg.loss.name == "mse":
        value_loss = masked_mse_loss(pred, target, mask)
    elif cfg.loss.name == "mae":
        value_loss = masked_mae(pred, target, mask)
    elif cfg.loss.name == "huber":
        value_loss = masked_huber_loss(pred, target, mask, delta=cfg.loss.delta)
    else:
        raise ValueError(f"Unknown loss name: {cfg.loss.name}")

    pred_phys = (
        de_standardize(pred, batch["atomic_numbers"], channel, paw_stats)
        if paw_stats is not None else pred
    )
    # Moments and Sanvito stats read the whole padded vector, so LMAXMIX-masked
    # components must be zeroed like the target rather than left free.
    pred_phys = pred_phys * mask.to(pred_phys.dtype)

    moment_enabled = (
        channel == "total"
        and bool(getattr(cfg.loss, "moment_reg_enabled", False))
    )
    if moment_enabled:
        moment_loss, moment_stats = moment_regularization_loss(
            pred_phys,
            target_phys,
            batch["atomic_numbers"],
            lambda_tr1=float(getattr(cfg.loss, "moment_lambda_tr1", 0.05)),
            lambda_tr2=float(getattr(cfg.loss, "moment_lambda_tr2", 0.02)),
            eps=float(getattr(cfg.loss, "moment_eps", 1e-6)),
        )
    else:
        moment_loss = pred.new_tensor(0.0)
        moment_stats = {
            "moment_tr1_rel_mse": pred.new_tensor(0.0),
            "moment_tr2_rel_mse": pred.new_tensor(0.0),
        }

    loss = value_loss + moment_loss

    e3nn_stats = masked_error_stats(pred, target, mask)
    sanvito_stats = sanvito_component_error_stats_from_e3nn(
        pred_phys,
        target_phys,
        mask,
        batch["atomic_numbers"],
    )

    if channel == "mag":
        spin_num, spin_den, spin_magnetic = spin_error_stats(
            pred_phys,
            target_phys,
            mask,
            batch,
            mag_min=float(getattr(cfg.loss, "spin_mag_min", 0.1)),
        )
        n_mag = int(spin_magnetic.sum())
        spin_nmae = (
            (spin_num[spin_magnetic].sum() / spin_den[spin_magnetic].sum().clamp_min(1e-12))
            if n_mag > 0 else pred.new_tensor(0.0)
        )
        spin_abs_err = spin_num.mean() if spin_num.numel() else pred.new_tensor(0.0)
    else:
        spin_nmae = pred.new_tensor(0.0)
        spin_abs_err = pred.new_tensor(0.0)
        n_mag = 0

    metrics = {
        "loss": loss,
        "loss_value": value_loss.detach(),
        "spin_nmae": spin_nmae.detach(),
        "spin_abs_err": spin_abs_err.detach(),
        "spin_n_magnetic": n_mag,
        "loss_moment": moment_loss.detach(),
        "moment_tr1_rel_mse": moment_stats["moment_tr1_rel_mse"].detach(),
        "moment_tr2_rel_mse": moment_stats["moment_tr2_rel_mse"].detach(),
        "mae": e3nn_stats["mae"],
        "rmse": e3nn_stats["rmse"],
        "maxae": e3nn_stats["maxae"],
        "sanvito_mae": sanvito_stats["mae"],
        "sanvito_rmse": sanvito_stats["rmse"],
        "sanvito_maxae": sanvito_stats["maxae"],
        "rel_mae": relative_mae_by_atom(pred, target, mask),
    }

    return loss, metrics, out


# ----------------------------
# Lightning
# ----------------------------

class LightningPAWModule(pl.LightningModule if pl is not None else object):
    def __init__(
        self,
        model: nn.Module,
        cfg: SimpleNamespace,
        target_channel: str,
        out_dir: str | Path,
        paw_stats: "PAWStats | None" = None,
    ):
        super().__init__()

        self.model = model
        self.cfg = cfg
        self.target_channel = target_channel
        self.out_dir = Path(out_dir)
        self.paw_stats = paw_stats

        self.save_hyperparameters(
            {
                "config": namespace_to_dict(cfg),
                "target_channel": target_channel,
            },
            ignore=["model"],
        )

        self.test_acc = None
        self.test_per_structure = []
        self.test_csv_path = None
        self.test_speed_warmup_batches = int(
            getattr(cfg.lightning, "inference_speed_warmup_batches", 2)
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        if batch is None:
            return None
        return batch.to(device)

    def forward(self, batch):
        return self.model(batch)

    def configure_optimizers(self):
        # insert(0) keeps pg0=backbone/pg1=head so existing wandb lr series names hold.
        groups = [
            {
                "params": self.model.paw_head.parameters(),
                "lr": self.cfg.optimizer.lr,
            },
        ]
        if not bool(getattr(self.cfg.optimizer, "freeze_backbone", False)):
            groups.insert(0, {
                "params": self.model.backbone.parameters(),
                "lr": self.cfg.optimizer.lr * self.cfg.optimizer.backbone_lr_multiplier,
            })

        return torch.optim.AdamW(
            groups,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

    def _shared_step(self, batch, stage: str):
        if batch is None:
            return None

        loss, metrics, _ = single_channel_loss_and_metrics(
            model=self.model,
            batch=batch,
            cfg=self.cfg,
            paw_stats=self.paw_stats,
            channel=self.target_channel,
        )

        log_data = {
            f"{stage}/loss": metrics["loss"],
            f"{stage}/loss_value": metrics["loss_value"],
            f"{stage}/loss_moment": metrics["loss_moment"],
            f"{stage}/moment_tr1_rel_mse": metrics["moment_tr1_rel_mse"],
            f"{stage}/moment_tr2_rel_mse": metrics["moment_tr2_rel_mse"],

            f"{stage}/{self.target_channel}_mae": metrics["mae"],
            f"{stage}/{self.target_channel}_rmse": metrics["rmse"],
            f"{stage}/{self.target_channel}_maxae": metrics["maxae"],
            f"{stage}/{self.target_channel}_rel_mae": metrics["rel_mae"],

            f"{stage}/sanvito_{self.target_channel}_mae": metrics["sanvito_mae"],
            f"{stage}/sanvito_{self.target_channel}_rmse": metrics["sanvito_rmse"],
            f"{stage}/sanvito_{self.target_channel}_maxae": metrics["sanvito_maxae"],
        }

        # Only log spin_nmae on batches that hold a magnetic structure; a 0.0
        # placeholder would pull the epoch mean toward "perfect".
        if self.target_channel == "mag" and metrics["spin_n_magnetic"] > 0:
            log_data[f"{stage}/spin_nmae"] = metrics["spin_nmae"]
            log_data[f"{stage}/spin_abs_err"] = metrics["spin_abs_err"]

        self.log_dict(
            log_data,
            on_step=(stage == "train"),
            on_epoch=True,
            prog_bar=(stage != "train"),
            logger=True,
            batch_size=1,
            sync_dist=False,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_test_epoch_start(self):
        self.test_per_structure = []
        self.test_acc = {
            "n_batches": 0,
            "n_structures": 0,
            "n_atoms": 0,
            "n_coeff": 0,

            "sum_loss": 0.0,
            "sum_value_loss": 0.0,
            "sum_moment_loss": 0.0,

            "sum_abs_err": 0.0,
            "sum_sq_err": 0.0,
            "max_abs_err": 0.0,

            "sum_sanvito_abs_err": 0.0,
            "sum_sanvito_sq_err": 0.0,
            "max_sanvito_abs_err": 0.0,

            "spin_num_magnetic": 0.0,
            "spin_den_magnetic": 0.0,
            "n_magnetic": 0,

            "inference_time_s": 0.0,
            "timed_batches": 0,
            "timed_structures": 0,
            "timed_atoms": 0,
        }

    def _sync_if_cuda(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def test_step(self, batch, batch_idx):
        if batch is None:
            return None

        self.model.eval()

        # Model forward only; CUDA events measure GPU execution directly.
        if self.device.type == "cuda":
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            self._sync_if_cuda()
            start_evt.record()
            out = self.model(batch)
            end_evt.record()
            self._sync_if_cuda()
            forward_ms = start_evt.elapsed_time(end_evt)
        else:
            t0 = time.perf_counter()
            out = self.model(batch)
            forward_ms = 1000.0 * (time.perf_counter() - t0)

        pred = out["paw_padded"]
        target = batch["target"]
        mask = batch["mask"]

        loss, metrics, _ = single_channel_loss_and_metrics(
            model=self.model,
            batch=batch,
            cfg=self.cfg,
            paw_stats=self.paw_stats,
            channel=self.target_channel,
        )

        if self.paw_stats is not None:
            pred_phys = de_standardize(pred, batch["atomic_numbers"], self.target_channel, self.paw_stats)
            target_phys = batch["target_phys"]
        else:
            pred_phys, target_phys = pred, target

        pred_phys = pred_phys * mask.to(pred_phys.dtype)

        pred_sanvito = e3nn_to_sanvito_with_basis_padded(
            pred_phys,
            batch["atomic_numbers"],
        )
        target_sanvito = e3nn_to_sanvito_with_basis_padded(
            target_phys,
            batch["atomic_numbers"],
        )

        n_atoms = int(batch["atomic_numbers"].shape[0])
        n_structures = len(batch["paths"])

        # `mask` is e3nn-ordered and does not line up with the Sanvito tensors;
        # rebuild it in Sanvito order, per structure since LMAXMIX is.
        node_batch = batch["batch"]
        mask_sanvito = torch.zeros_like(mask)
        for s in range(n_structures):
            rows = node_batch == s
            lmaxmix_s = int(batch["lmaxmix"][s])
            schema_mask_s, lmaxmix_mask_s = sanvito_schema_and_lmaxmix_masks(
                batch["atomic_numbers"][rows],
                None if lmaxmix_s < 0 else lmaxmix_s,
            )
            mask_sanvito[rows] = schema_mask_s & lmaxmix_mask_s

        err = (pred_phys - target_phys)[mask]
        sanvito_err = (pred_sanvito - target_sanvito)[mask_sanvito]

        n_coeff = int(err.numel())

        if n_coeff > 0:
            abs_err = err.abs()
            sanvito_abs_err = sanvito_err.abs()

            self.test_acc["sum_abs_err"] += float(abs_err.sum().detach().cpu())
            self.test_acc["sum_sq_err"] += float((err ** 2).sum().detach().cpu())
            self.test_acc["max_abs_err"] = max(
                self.test_acc["max_abs_err"],
                float(abs_err.max().detach().cpu()),
            )

            self.test_acc["sum_sanvito_abs_err"] += float(sanvito_abs_err.sum().detach().cpu())
            self.test_acc["sum_sanvito_sq_err"] += float((sanvito_err ** 2).sum().detach().cpu())
            self.test_acc["max_sanvito_abs_err"] = max(
                self.test_acc["max_sanvito_abs_err"],
                float(sanvito_abs_err.max().detach().cpu()),
            )

            self.test_acc["n_coeff"] += n_coeff

        # Per-structure / per-atom errors in physical e3nn space.
        maskf = mask.float()
        diff = pred_phys - target_phys
        per_atom_count = maskf.sum(dim=1).clamp_min(1.0)
        atomic_mae = (diff.abs() * maskf).sum(dim=1) / per_atom_count
        atomic_mse = (diff ** 2 * maskf).sum(dim=1) / per_atom_count

        total_count = maskf.sum().clamp_min(1.0)
        total_mae = (diff.abs() * maskf).sum() / total_count
        total_mse = (diff ** 2 * maskf).sum() / total_count

        # Tr(D), Tr(D^2) relative squared error, same definition as the moment regularizer.
        pred_m, _ = moments_from_coeffs(pred_phys, batch["atomic_numbers"], [1, 2])
        target_m, _ = moments_from_coeffs(target_phys, batch["atomic_numbers"], [1, 2])
        rel_sq = ((pred_m - target_m) / (target_m.abs() + 1e-6)) ** 2  # [n_atoms, 2]

        def _round_list(t):
            return [round(float(v), 8) for v in t.detach().cpu().tolist()]

        spin_num, spin_den, spin_magnetic = spin_error_stats(
            pred_phys, target_phys, mask, batch,
            mag_min=float(getattr(self.cfg.loss, "spin_mag_min", 0.1)),
        )
        struct_num = float(spin_num.sum())
        struct_den = float(spin_den.sum())
        is_magnetic = bool(spin_magnetic.any())

        if self.target_channel == "mag" and is_magnetic:
            self.test_acc["spin_num_magnetic"] += struct_num
            self.test_acc["spin_den_magnetic"] += struct_den
            self.test_acc["n_magnetic"] += 1

        self.test_per_structure.append({
            "material_id": batch["material_ids"][0],
            "n_atoms": n_atoms,
            "n_coeff": n_coeff,
            "lmaxmix": batch["lmaxmix"][0],
            "atomic_numbers": batch["atomic_numbers"].detach().cpu().tolist(),
            "total_mae": round(float(total_mae), 8),
            "atomic_mae": _round_list(atomic_mae),
            "total_mse": round(float(total_mse), 8),
            "atomic_mse": _round_list(atomic_mse),
            "total_tr1_rel_mse": round(float(rel_sq[:, 0].mean()), 8),
            "atomic_tr1_rel_mse": _round_list(rel_sq[:, 0]),
            "total_tr2_rel_mse": round(float(rel_sq[:, 1].mean()), 8),
            "atomic_tr2_rel_mse": _round_list(rel_sq[:, 1]),
            "mag_moment": round(float(batch["mag_moment"].sum()), 8),
            "is_magnetic": int(is_magnetic),
            "spin_nmae": round(struct_num / max(struct_den, 1e-12), 8),
            "forward_ms": round(forward_ms, 4),
        })

        if batch_idx >= self.test_speed_warmup_batches:
            self.test_acc["inference_time_s"] += forward_ms / 1000.0
            self.test_acc["timed_batches"] += 1
            self.test_acc["timed_structures"] += n_structures
            self.test_acc["timed_atoms"] += n_atoms

        self.test_acc["n_batches"] += 1
        self.test_acc["n_structures"] += n_structures
        self.test_acc["n_atoms"] += n_atoms
        self.test_acc["sum_loss"] += float(metrics["loss"].detach().cpu())
        self.test_acc["sum_value_loss"] += float(metrics["loss_value"].detach().cpu())
        self.test_acc["sum_moment_loss"] += float(metrics["loss_moment"].detach().cpu())

    def on_test_epoch_end(self):
        acc = self.test_acc
        eps = 1e-12

        n_batches = max(acc["n_batches"], 1)
        n_coeff = max(acc["n_coeff"], 1)
        inference_time_s = max(acc["inference_time_s"], eps)

        summary = {
            "target_channel": self.target_channel,

            "test_n_batches": acc["n_batches"],
            "test_n_structures": acc["n_structures"],
            "test_n_atoms": acc["n_atoms"],
            "test_n_coeff": acc["n_coeff"],

            "test_loss": acc["sum_loss"] / n_batches,
            "test_loss_value": acc["sum_value_loss"] / n_batches,
            "test_loss_moment": acc["sum_moment_loss"] / n_batches,

            "test_component_mae": acc["sum_abs_err"] / n_coeff,
            "test_component_rmse": math.sqrt(acc["sum_sq_err"] / n_coeff),
            "test_component_maxae": acc["max_abs_err"],

            "test_sanvito_component_mae": acc["sum_sanvito_abs_err"] / n_coeff,
            "test_sanvito_component_rmse": math.sqrt(acc["sum_sanvito_sq_err"] / n_coeff),
            "test_sanvito_component_maxae": acc["max_sanvito_abs_err"],

            # Pooled over the magnetic test structures only; 1.0 == trivial m == 0 predictor.
            "test_spin_nmae": (
                acc["spin_num_magnetic"] / max(acc["spin_den_magnetic"], eps)
                if acc["n_magnetic"] > 0 else float("nan")
            ),
            "test_n_magnetic": acc["n_magnetic"],

            "test_inference_time_s": acc["inference_time_s"],
            "test_timed_batches": acc["timed_batches"],
            "test_timed_structures": acc["timed_structures"],
            "test_timed_atoms": acc["timed_atoms"],
            "test_structures_per_second": acc["timed_structures"] / inference_time_s,
            "test_atoms_per_second": acc["timed_atoms"] / inference_time_s,
            "test_ms_per_structure": 1000.0 * inference_time_s / max(acc["timed_structures"], 1),
            "test_ms_per_atom": 1000.0 * inference_time_s / max(acc["timed_atoms"], 1),
        }

        self.log_dict(
            {f"test/{k}": v for k, v in summary.items() if isinstance(v, (int, float))},
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )

        if self.trainer.is_global_zero:
            self.out_dir.mkdir(parents=True, exist_ok=True)

            with open(self.out_dir / "test_metrics_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            with open(self.out_dir / "test_metrics_summary.txt", "w") as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")

            print(f"Wrote test summary to {self.out_dir}")

            csv_path = self.out_dir / "test_per_structure_errors.csv"
            fieldnames = [
                "material_id", "n_atoms", "n_coeff", "lmaxmix", "atomic_numbers",
                "total_mae", "atomic_mae",
                "total_mse", "atomic_mse",
                "total_tr1_rel_mse", "atomic_tr1_rel_mse",
                "total_tr2_rel_mse", "atomic_tr2_rel_mse",
                "mag_moment", "is_magnetic", "spin_nmae",
                "forward_ms",
            ]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.test_per_structure:
                    writer.writerow({
                        k: json.dumps(v) if isinstance(v, list) else v
                        for k, v in row.items()
                    })

            self.test_csv_path = str(csv_path)
            print(f"Wrote per-structure error CSV to {csv_path}")


# ----------------------------
# Export
# ----------------------------

@torch.no_grad()
def export_test_augmentation_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: str | Path,
    target_channel: str,
    paw_stats: "PAWStats | None" = None,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    n_written = 0

    for batch in loader:
        if batch is None:
            continue

        batch = batch.to(device)

        out = model(batch)
        pred_e3nn = out["paw_padded"]
        if paw_stats is not None:
            pred_e3nn = de_standardize(pred_e3nn, batch["atomic_numbers"], target_channel, paw_stats)
        # Masked (L>LMAXMIX) coefficients go out as literal 0.0, as VASP writes them.
        pred_e3nn = pred_e3nn * batch["mask"].to(pred_e3nn.dtype)
        pred_sanvito = e3nn_to_sanvito_with_basis_padded(
            pred_e3nn,
            batch["atomic_numbers"],
        )

        material_id = batch["material_ids"][0]
        atomic_numbers = batch["atomic_numbers"].detach().cpu()
        y = pred_sanvito.detach().cpu()

        # Masks in y's own (Sanvito) order; batch["mask"] is e3nn-ordered.
        lmaxmix = int(batch["lmaxmix"][0])
        lmaxmix = None if lmaxmix < 0 else lmaxmix
        schema_mask, lmaxmix_mask = sanvito_schema_and_lmaxmix_masks(
            atomic_numbers, lmaxmix,
        )

        npz_path = out_dir / f"{material_id}_{target_channel}_aug.npz"
        txt_path = out_dir / f"{material_id}_{target_channel}_aug.txt"

        np.savez_compressed(
            npz_path,
            material_id=material_id,
            source_path=batch["paths"][0],
            channel=target_channel,
            atomic_numbers=atomic_numbers.numpy(),
            aug_sanvito_padded=y.numpy(),
            schema_mask=schema_mask.numpy(),
            lmaxmix_mask=lmaxmix_mask.numpy(),
            lmaxmix=-1 if lmaxmix is None else lmaxmix,
            mask=(schema_mask & lmaxmix_mask).numpy(),  # kept for existing readers
        )

        txt = predicted_aug_blocks_text(
            y_total_sanvito=y,
            atomic_numbers=atomic_numbers,
            y_mag_sanvito=None,
        )
        txt_path.write_text(txt)

        n_written += 1

    print(f"Wrote {n_written} test augmentation files to {out_dir}")


# ----------------------------
# Trainer setup
# ----------------------------

# Tells the SLURM resubmit wrapper "wall clock cut this short, state is on disk, resubmit".
EXIT_RESUBMIT = 42


class PreemptionCheckpoint(Callback):
    """Checkpoint and stop when SLURM sends the wall-clock warning (SIGUSR1, forwarded
    by the SLURM resubmit wrapper). The handler only sets a flag; the save happens from a hook.
    limit_val_batches is zeroed on the way out, since Lightning would otherwise run a
    full validation pass before stopping and eat the grace period."""

    def __init__(self, out_dir: Path, sig: int = signal.SIGUSR1):
        self.out_dir = Path(out_dir)
        self.preempted = False
        self._signalled = False
        signal.signal(sig, self._on_signal)

    def _on_signal(self, signum, _frame):
        self._signalled = True
        print(f"Signal {signum} received: checkpointing and stopping at the next hook.",
              flush=True)

    def _request_stop(self, trainer) -> None:
        if not self._signalled or self.preempted:
            return
        self.preempted = True
        path = self.out_dir / f"preempt_e{trainer.current_epoch:03d}_s{trainer.global_step:07d}.ckpt"
        trainer.save_checkpoint(path)
        print(f"Saved preemption checkpoint to {path}", flush=True)
        trainer.limit_val_batches = 0
        trainer.should_stop = True

    def on_train_batch_end(self, trainer, *_args, **_kwargs):
        self._request_stop(trainer)

    def on_validation_end(self, trainer, *_args, **_kwargs):
        self._request_stop(trainer)


def make_lightning_logger(cfg: SimpleNamespace, out_dir: Path):
    if not getattr(cfg.wandb, "enabled", False):
        return False

    tags = list(getattr(cfg.wandb, "tags", None) or [])
    split_name = str(getattr(cfg.project, "split_name", "") or "").strip()
    if split_name and split_name not in tags:
        tags.append(split_name)

    # Pin the wandb run id per out_dir so a chained resubmission logs as one run.
    run_id_file = out_dir / "wandb_run_id.txt"
    if run_id_file.exists():
        run_id = run_id_file.read_text().strip()
    else:
        # uuid, not random: `random` is seeded, so every run would claim the same id.
        run_id = uuid.uuid4().hex[:8]
        run_id_file.write_text(run_id)

    return WandbLogger(
        project=cfg.wandb.project,
        entity=getattr(cfg.wandb, "entity", None),
        name=cfg.wandb.name,
        save_dir=str(out_dir),
        tags=tags,
        log_model=False,
        config=flatten_dict(namespace_to_dict(cfg)),
        id=run_id,
        resume="allow",
    )


def make_lightning_trainer(cfg: SimpleNamespace, out_dir: Path):
    """Returns (trainer, preemption_callback)."""
    # Nothing in out_dir is ever deleted: _remove_checkpoint is Lightning's only
    # pruning path, and filenames carry epoch+step so saves never overwrite.
    class KeepAllModelCheckpoint(ModelCheckpoint):
        def _remove_checkpoint(self, trainer, filepath):
            return

    preemption = PreemptionCheckpoint(out_dir)

    callbacks = [
        ModelSummary(max_depth=int(getattr(cfg.lightning, "model_summary_depth", 2))),
        LearningRateMonitor(logging_interval="step"),
        preemption,
        # Must stay the first ModelCheckpoint: trainer.checkpoint_callback resolves to it.
        # enable_version_counter=False keeps last.ckpt's name stable across a resubmit.
        KeepAllModelCheckpoint(
            dirpath=str(out_dir),
            filename="best_e{epoch:03d}_s{step:07d}",
            auto_insert_metric_name=False,
            monitor="val/loss",
            mode="min",
            save_top_k=int(getattr(cfg.lightning, "save_top_k", 1)),
            save_last=True,
            enable_version_counter=False,
        ),
    ]

    if bool(getattr(cfg.lightning, "save_every_epoch", False)):
        callbacks.append(
            KeepAllModelCheckpoint(
                dirpath=str(out_dir),
                filename="epoch_{epoch:03d}",
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=-1,
            )
        )

    # Mid-epoch safety net for wall-clock kills. Step-triggered because Lightning's
    # manual-optimization branch ignores train_time_interval; monitor=None +
    # save_top_k=1 writes unconditionally (save_top_k=0 would be a no-op). 0 disables.
    ckpt_steps = int(getattr(cfg.lightning, "checkpoint_every_n_steps", 30000) or 0)
    if ckpt_steps > 0:
        callbacks.append(
            KeepAllModelCheckpoint(
                dirpath=str(out_dir),
                filename="periodic_e{epoch:03d}_s{step:07d}",
                auto_insert_metric_name=False,
                monitor=None,
                save_top_k=1,
                save_last=False,
                every_n_train_steps=ckpt_steps,
            )
        )

    trainer = pl.Trainer(
        accelerator=getattr(cfg.lightning, "accelerator", "auto"),
        devices=getattr(cfg.lightning, "devices", "auto"),
        precision=getattr(cfg.lightning, "precision", 32),
        max_epochs=int(cfg.epochs),
        logger=make_lightning_logger(cfg, out_dir),
        callbacks=callbacks,
        log_every_n_steps=int(getattr(cfg.lightning, "log_every_n_steps", 1)),
        default_root_dir=str(out_dir),
        enable_checkpointing=True,
        enable_model_summary=True,
    )
    return trainer, preemption


def resolve_resume_ckpt(resume: Optional[str], out_dir: Path) -> Optional[str]:
    """'auto' takes the newest *.ckpt by mtime (periodic_/epoch_/last do not sort by
    name in time order) and starts fresh when there is none; an explicit path must exist."""
    if not resume:
        return None

    if resume != "auto":
        ckpt = Path(resume).expanduser()
        if not ckpt.is_file():
            raise FileNotFoundError(f"--resume checkpoint does not exist: {ckpt}")
        print(f"Resuming from {ckpt}")
        return str(ckpt)

    found = sorted(out_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
    if not found:
        print(f"--resume auto: no checkpoint in {out_dir}, starting fresh")
        return None

    print(f"--resume auto: resuming from {found[-1]}")
    return str(found[-1])


# Architecture keys that must agree between an --init-from checkpoint and this run.
# r_max and avg_num_neighbors change no tensor shape, so a mismatch would be silent.
INIT_FROM_ARCH_KEYS = (
    "hidden_width", "max_ell", "r_max", "num_interactions", "correlation",
    "avg_num_neighbors", "head_rank", "head_block_mixing", "head_linear_max_l",
)
INIT_FROM_ARCH_DEFAULTS = {"head_rank": 64, "head_block_mixing": True, "head_linear_max_l": 0}


def load_init_weights(lit_model, ckpt_path: str, cfg: SimpleNamespace,
                      allow_channel_change: bool = False) -> None:
    """Load parameters only (no optimizer/epoch/loop state), in place on the live module
    so OpenEquivariance's lazily-created conv constants stay on the GPU.

    Strict for parameters. Missing non-parameter buffers are tolerated: an e3nn build
    registers Wigner-3j tables etc. that an OEQ-built checkpoint never stored; they are
    constants rebuilt identically at construction."""
    ckpt = Path(ckpt_path).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(f"--init-from checkpoint does not exist: {ckpt}")

    state = torch.load(ckpt, map_location="cpu")

    hparams = state.get("hyper_parameters", {}) or {}
    old_model = ((hparams.get("config") or {}).get("model") or {})
    def _eff(src, key):
        got = src.get(key, None) if isinstance(src, dict) else getattr(src, key, None)
        return INIT_FROM_ARCH_DEFAULTS.get(key, None) if got is None else got

    diffs = [
        f"{k}: ckpt={_eff(old_model, k)!r} run={_eff(cfg.model, k)!r}"
        for k in INIT_FROM_ARCH_KEYS
        if k in old_model and _eff(old_model, k) != _eff(cfg.model, k)
    ]
    old_channel = hparams.get("target_channel")
    if old_channel is not None and old_channel != cfg.data.augmentation_channel:
        if allow_channel_change:
            print(f"--init-from: channel change {old_channel!r} -> "
                  f"{cfg.data.augmentation_channel!r} (allowed explicitly). The head "
                  f"weights were trained on a different target; this is a transfer "
                  f"initialization, not a resume.")
        else:
            diffs.append(
                f"target_channel: ckpt={old_channel!r} run={cfg.data.augmentation_channel!r}"
            )
    if diffs:
        raise ValueError(
            f"--init-from architecture mismatch between {ckpt} and this config:\n  "
            + "\n  ".join(diffs)
            + "\nSame weights under a different architecture are a different model."
        )

    missing, unexpected = lit_model.load_state_dict(state["state_dict"], strict=False)

    param_names = {n for n, _ in lit_model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    missing_buffers = [k for k in missing if k not in param_names]

    if missing_params or unexpected:
        raise RuntimeError(
            f"--init-from state_dict does not match the model built from this config:\n"
            f"  missing parameters ({len(missing_params)}): {missing_params[:10]}\n"
            f"  unexpected keys ({len(unexpected)}): {list(unexpected)[:10]}"
        )

    print(f"--init-from: loaded {len(state['state_dict'])} weight tensors from {ckpt} "
          f"(weights only; training starts at step 0)")
    if missing_buffers:
        print(f"--init-from: {len(missing_buffers)} non-parameter buffers not in the "
              f"checkpoint, kept as constructed (openequivariance/e3nn build "
              f"difference), e.g. {missing_buffers[:3]}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split-name", type=str, default=None,
                        help="Override project.split_name (wandb/project naming).")
    parser.add_argument("--split-file", type=str, default=None,
                        help="Override data.split_file (datasplits_*.json).")
    parser.add_argument("--stats-file", type=str, default=None,
                        help="Override data.stats_file: load precomputed PAW stats "
                             "instead of recomputing from the train split.")
    parser.add_argument("--neighbor-stats-file", type=str, default=None,
                        help="Override model.neighbor_stats_file: the measured "
                             "avg_num_neighbors table (neighbor_stats.py). Only read "
                             "when model.avg_num_neighbors is 'auto'.")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Override the effective output dir.")
    parser.add_argument("--resume", type=str, default=None, metavar="CKPT",
                        help="Resume training from a checkpoint path, or 'auto' for the "
                             "most recently modified *.ckpt in the output dir (starts "
                             "fresh when there is none).")
    parser.add_argument("--head-rank", type=int, default=None,
                        help="Override model.head_rank: number of outer-product terms in "
                             "the shared head. Must be even; 64 spans every schema's "
                             "symmetric block space.")
    mix_group = parser.add_mutually_exclusive_group()
    mix_group.add_argument("--head-block-mixing", dest="head_block_mixing",
                           action="store_true", default=None,
                           help="Learn a per-block combination over the rank index "
                                "after the CG coupling. Default on.")
    mix_group.add_argument("--no-head-block-mixing", dest="head_block_mixing",
                           action="store_false", default=None,
                           help="Fixed +-1 signature, no post-coupling mix.")
    parser.add_argument("--head-linear-max-l", type=int, default=None, metavar="L",
                        help="Predict every block with L <= this from a direct linear "
                             "map on h instead of the quadratic coupling. 0 = L=0 only "
                             "(default); -1 = none; cannot exceed max_ell.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override project.run_name (and the wandb run name).")
    parser.add_argument("--channel", type=str, default=None, choices=["total", "mag"],
                        help="Override data.augmentation_channel: 'mag' trains the "
                             "spin-difference augmentation occupancies.")
    parser.add_argument("--mag-scale-source", type=str, default=None,
                        choices=["mag", "total"],
                        help="Override data.mag_scale_source: which channel's per-Z "
                             "sigma standardizes the mag target (mag channel only).")
    parser.add_argument("--init-from", type=str, default=None, metavar="CKPT",
                        help="Initialize weights from a checkpoint WITHOUT resuming "
                             "optimizer/epoch state (unlike --resume). The architecture "
                             "keys recorded in the checkpoint must match this config.")
    parser.add_argument("--init-from-allow-channel-change", action="store_true",
                        help="Let --init-from load a checkpoint trained on the other "
                             "augmentation channel (total -> mag transfer).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training: run the test split and write the usual "
                             "test_metrics_* files. Pair with --init-from and a fresh "
                             "--out-dir.")
    parser.add_argument("--freeze-backbone", dest="freeze_backbone",
                        action="store_true", default=None,
                        help="Override optimizer.freeze_backbone: train the PAW head "
                             "only, holding the MACE backbone fixed.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs (Trainer max_epochs). With --resume this "
                             "extends a finished run: a 5-epoch checkpoint resumed with "
                             "--epochs 10 trains 5 more epochs.")
    args = parser.parse_args()

    if args.eval_only and args.resume:
        raise ValueError(
            "--eval-only and --resume are mutually exclusive: --eval-only does not train. "
            "Use --init-from to choose the weights to evaluate."
        )

    if pl is None:
        raise ImportError("Install Lightning with: pip install lightning")

    cfg = load_config(args.config)
    if args.split_name is not None:
        cfg.project.split_name = args.split_name
    if args.split_file is not None:
        cfg.data.split_file = args.split_file
    if args.stats_file is not None:
        cfg.data.stats_file = args.stats_file
    if args.neighbor_stats_file is not None:
        cfg.model.neighbor_stats_file = args.neighbor_stats_file
    if args.run_name is not None:
        cfg.project.run_name = args.run_name
        cfg.wandb.name = args.run_name
    if args.channel is not None:
        cfg.data.augmentation_channel = args.channel
    if args.mag_scale_source is not None:
        cfg.data.mag_scale_source = args.mag_scale_source
    if args.head_rank is not None:
        cfg.model.head_rank = args.head_rank
    if args.head_block_mixing is not None:
        cfg.model.head_block_mixing = args.head_block_mixing
    if args.head_linear_max_l is not None:
        cfg.model.head_linear_max_l = args.head_linear_max_l
    if args.freeze_backbone is not None:
        cfg.optimizer.freeze_backbone = args.freeze_backbone
    if args.epochs is not None:
        cfg.epochs = args.epochs
    cfg = apply_runtime_paths(cfg)
    if args.out_dir is not None:
        cfg.project.effective_out_dir = args.out_dir
    cfg = apply_auto_names(cfg)

    seed_everything(cfg.seed)
    pl.seed_everything(cfg.seed, workers=True)

    target_channel = cfg.data.augmentation_channel
    if target_channel not in {"total", "mag"}:
        raise ValueError(f"augmentation_channel must be total or mag, got {target_channel}")

    # Non-MP POTCARs store the augmentation blocks in a different projector order.
    l_channel_overrides = namespace_to_dict(getattr(cfg.data, "potcar_l_channels", None) or {})
    if l_channel_overrides:
        set_l_channel_overrides(l_channel_overrides)
        print(f"POTCAR projector-order overrides: {l_channel_overrides}")

    lmaxmix_mode = resolve_lmaxmix_mode(getattr(cfg.data, "lmaxmix_mode", "off"))

    out_dir = Path(cfg.project.effective_out_dir) / cfg.project.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(namespace_to_dict(cfg), f, sort_keys=False)

    print(f"Project:        {cfg.project.name}")
    print(f"Run name:       {cfg.project.run_name}")
    print(f"Target channel: {target_channel}")
    print(f"LMAXMIX mask:   {lmaxmix_mode}")
    print(f"HPC paths:      {cfg.runtime.is_hpc}")
    print(f"Data dir:       {cfg.project.effective_data_dir}")
    print(f"Out dir:        {out_dir}")

    train_files, val_files, test_files, using_json_split, split_missing = get_train_val_test_files(cfg)

    if using_json_split and any(len(v) > 0 for v in split_missing.values()):
        missing_path = out_dir / "missing_split_ids.json"
        with open(missing_path, "w") as f:
            json.dump(split_missing, f, indent=2)
        print(f"Wrote missing split ID log to {missing_path}")

    stats_min_atoms = int(getattr(cfg.data, "stats_min_atoms", 50))

    # Standardization stats: precomputed file, or computed from this run's train split.
    stats_file = getattr(cfg.data, "stats_file", None)
    use_stats_file = (
        stats_file is not None
        and str(stats_file).strip().lower() not in {"", "none", "null", "auto"}
    )
    if use_stats_file:
        stats_file = resolve_data_path(stats_file, "augnet")
        paw_stats = PAWStats.load(str(stats_file))
        print(f"Loaded precomputed PAW standardization stats from {stats_file}")
        if lmaxmix_mode == "auto" and paw_stats.meta.get("lmaxmix_mode", "off") != "auto":
            print(f"  WARNING: {stats_file} predates the LMAXMIX mask "
                  f"(meta.lmaxmix_mode={paw_stats.meta.get('lmaxmix_mode', 'off')!r}). "
                  f"Its L>=3 scale for d/f elements is under-estimated by sqrt(1-p_Z); "
                  f"re-run python -m src.paw_stats to regenerate it.",
                  file=sys.stderr)

    dataset_kwargs = dict(
        target_channel=target_channel,
        l_channel_overrides=l_channel_overrides,
        lmaxmix_mode=lmaxmix_mode,
    )

    if using_json_split:
        if not use_stats_file:
            paw_stats = compute_paw_stats([str(f) for f in train_files], min_atoms=stats_min_atoms,
                                          l_channel_overrides=l_channel_overrides,
                                          lmaxmix_mode=lmaxmix_mode)
        train_set = PAWCHGCARDataset(train_files, stats=paw_stats, **dataset_kwargs)
        val_set = PAWCHGCARDataset(val_files, stats=paw_stats, **dataset_kwargs)
        test_set = PAWCHGCARDataset(test_files, stats=paw_stats, **dataset_kwargs)
    else:
        dataset = PAWCHGCARDataset(train_files, **dataset_kwargs)
        train_idx, val_idx = make_element_covered_split(dataset, cfg.val_fraction, cfg.seed)
        if not use_stats_file:
            train_subset_files = [dataset.files[i] for i in train_idx]
            paw_stats = compute_paw_stats([str(f) for f in train_subset_files], min_atoms=stats_min_atoms,
                                          l_channel_overrides=l_channel_overrides,
                                          lmaxmix_mode=lmaxmix_mode)
        dataset.stats = paw_stats
        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        test_set = None

    # mag_scale_source "total": standardize the mag channel with the TOTAL channel's
    # per-Z scale, so non-magnetic SCF residue is not amplified to O(1). The shift stays
    # the mag channel's own. The mutated table is what gets saved, so de-standardization
    # at test/export time stays consistent.
    mag_scale_source = str(getattr(cfg.data, "mag_scale_source", "mag") or "mag").lower()
    if mag_scale_source not in {"mag", "total"}:
        raise ValueError(f"data.mag_scale_source must be mag or total, got {mag_scale_source!r}")
    if target_channel == "mag" and mag_scale_source == "total":
        paw_stats.scale["mag"] = paw_stats.scale["total"].clone()
        paw_stats.meta["mag_scale_source"] = "total"
        print("mag_scale_source: total -- the mag channel is standardized with the "
              "TOTAL channel's per-Z scale (shift stays the mag channel's own).")

    if lmaxmix_mode == "auto" and paw_stats.meta.get("lmaxmix_hist"):
        print(f"LMAXMIX (train files): {paw_stats.meta['lmaxmix_hist']} "
              f"('none' = no truncation visible, nothing masked)")

    paw_stats.save(out_dir / "paw_stats.pt")
    print(f"Saved PAW standardization stats to {out_dir / 'paw_stats.pt'}")

    atomic_numbers = sorted(Z_TO_SCHEMA.keys())

    hidden_irreps = hidden_irreps_from_width(
        cfg.model.hidden_width,
        cfg.model.max_ell,
    )

    print(f"Hidden irreps: {hidden_irreps}")

    cueq_config = build_cueq_config(cfg)
    if cueq_config is not None:
        print(f"cuequivariance enabled (layout={cueq_config.layout_str})")

    oeq_config = build_oeq_config(cfg)
    if oeq_config is not None:
        print("openequivariance enabled (fused conv tensor product)")

    avg_num_neighbors = resolve_avg_num_neighbors(cfg)
    print(f"avg_num_neighbors: {avg_num_neighbors:.4g} (r_max={cfg.model.r_max:g}, "
          f"config={getattr(cfg.model, 'avg_num_neighbors', None)!r})")
    # So the saved/logged config carries the resolved scalar, not "auto".
    cfg.model.avg_num_neighbors = avg_num_neighbors

    model = PAWAugModel(
        hidden_irreps=hidden_irreps,
        atomic_numbers=atomic_numbers,
        target_channel=target_channel,
        r_max=cfg.model.r_max,
        max_ell=cfg.model.max_ell,
        num_interactions=cfg.model.num_interactions,
        correlation=cfg.model.correlation,
        avg_num_neighbors=avg_num_neighbors,
        head_rank=int(getattr(cfg.model, "head_rank", 64)),
        head_block_mixing=bool(getattr(cfg.model, "head_block_mixing", True)),
        head_linear_max_l=int(getattr(cfg.model, "head_linear_max_l", 0)),
        use_reduced_cg=bool(getattr(cfg.model, "use_reduced_cg", True)),
        cueq_config=cueq_config,
        oeq_config=oeq_config,
    )

    # Frozen here rather than in configure_optimizers so ModelSummary reports the
    # real trainable count.
    if bool(getattr(cfg.optimizer, "freeze_backbone", False)):
        model.backbone.requires_grad_(False)
        print("freeze_backbone: training the PAW head only.")

    def make_loader(ds, shuffle: bool, batch_size: int):
        nw = int(getattr(cfg, "num_workers", 0))
        extra = {"persistent_workers": True, "prefetch_factor": 4} if nw > 0 else {}
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda items: collate_mace_graphs(
                items,
                z_table=model.backbone.z_table,
                r_max=model.backbone.r_max,
            ),
            num_workers=nw,
            **extra,
        )

    # Val/test stay at batch_size=1 so the per-structure test CSV has one structure per batch.
    train_loader = make_loader(train_set, shuffle=True, batch_size=cfg.batch_size)
    val_loader = make_loader(val_set, shuffle=False, batch_size=1)
    test_loader = (
        make_loader(test_set, shuffle=False, batch_size=1) if test_set is not None else None
    )

    lit_model = LightningPAWModule(
        model=model,
        cfg=cfg,
        target_channel=target_channel,
        out_dir=out_dir,
        paw_stats=paw_stats,
    )

    if args.init_from is not None:
        load_init_weights(lit_model, args.init_from, cfg,
                          allow_channel_change=args.init_from_allow_channel_change)

    trainer, preemption = make_lightning_trainer(cfg, out_dir)

    if args.eval_only and test_loader is None:
        raise ValueError(
            "--eval-only needs a test split: set data.split_file (--split-file). "
            "Without one the run has no test set to evaluate."
        )

    if args.eval_only:
        print("--eval-only: skipping training, going straight to test.")
    else:
        trainer.fit(
            lit_model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=resolve_resume_ckpt(args.resume, out_dir),
        )

        if preemption.preempted:
            print(f"Training preempted; exiting {EXIT_RESUBMIT} to request resubmission.",
                  flush=True)
            sys.exit(EXIT_RESUBMIT)

        # Load best weights in place and test with ckpt_path=None: a Lightning
        # checkpoint reload leaves OpenEquivariance's lazily-created conv constants
        # on CPU ("Tensor 'L1_in' is not on the GPU").
        if test_loader is not None:
            best_ckpt = trainer.checkpoint_callback.best_model_path
            if best_ckpt:
                ckpt = torch.load(best_ckpt, map_location=lit_model.device)
                lit_model.load_state_dict(ckpt["state_dict"])

    if test_loader is not None:
        trainer.test(
            lit_model,
            dataloaders=test_loader,
            ckpt_path=None,
        )

        if bool(getattr(cfg.data, "export_test_predictions", False)):
            # trainer.test() teardown moves the module back to CPU; the export runs
            # outside Lightning and OEQ's conv kernel is GPU-only.
            export_device = trainer.strategy.root_device
            lit_model.to(export_device)
            if getattr(cfg.data, "aug_pred_dir_test", "auto") == "auto":
                aug_pred_dir_test = out_dir / f"aug_pred_dir_test_{target_channel}"
            else:
                aug_pred_dir_test = Path(cfg.data.aug_pred_dir_test)

            export_test_augmentation_predictions(
                model=lit_model.model,
                loader=test_loader,
                device=export_device,
                out_dir=aug_pred_dir_test,
                target_channel=target_channel,
                paw_stats=lit_model.paw_stats,
            )

    print("Done")

    if getattr(lit_model, "test_csv_path", None):
        print(f"Per-structure test error CSV: {lit_model.test_csv_path}")


if __name__ == "__main__":
    main()
