"""End-to-end: structure or CHGCAR -> ML-seeded VASP input directory."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Structure

from neural_init.augnet.paw_basis_transform import sanvito_schema_and_lmaxmix_masks

from . import assemble, inputs
from .augnet import AugNetPredictor, aug_blocks, load_augnet
from .chgnet import load_chgnet, predict_site_moments
from .config import PipelineConfig
from .electrafi import load_electrafi, predict_density


@dataclass
class Prediction:
    structure: Structure
    grid_dims: tuple[int, int, int]
    nelect: float
    rho_total: np.ndarray  # e/Å^3, shape grid_dims
    rho_spin: np.ndarray | None  # e/Å^3 (spin-up minus spin-down)
    aug_total: dict[int, np.ndarray]  # 1-based ion index -> CHGCAR-order block
    aug_diff: dict[int, np.ndarray] | None
    site_moments: np.ndarray | None  # CHGNet, unsigned mu_B
    m_total: float | None  # net-moment constraint passed to ELECTRAFI
    lmaxmix: int | None
    meta: dict = field(default_factory=dict)


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


class Pipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        self.device = resolve_device(self.cfg.device)
        self._electrafi = None
        self._augnet: dict[str, AugNetPredictor] = {}
        self._chgnet = None

    # ---------- lazy model loading ----------
    def electrafi(self, n_atoms: int):
        if self._electrafi is None or self._electrafi._ndi_max_nodes < n_atoms:
            self._electrafi = load_electrafi(self.cfg.electrafi, self.device, n_atoms, self.cfg.weights_dir)
            # EScAIPBackbone.__init__ forces "high"; the user's choice wins.
            torch.set_float32_matmul_precision(self.cfg.matmul_precision)
        return self._electrafi

    def augnet(self, channel: str) -> AugNetPredictor:
        if channel not in self._augnet:
            ckpt = self.cfg.augnet.total_checkpoint if channel == "total" else self.cfg.augnet.mag_checkpoint
            pred = load_augnet(ckpt, self.cfg.augnet.stats, self.device, self.cfg.augnet.enable_cueq, self.cfg.weights_dir)
            if pred.channel != channel:
                raise ValueError(f"{ckpt} predicts the {pred.channel!r} channel, expected {channel!r}")
            self._augnet[channel] = pred
        return self._augnet[channel]

    def chgnet(self):
        if self._chgnet is None:
            self._chgnet = load_chgnet(self.cfg.chgnet.model, self.device)
        return self._chgnet

    # ---------- steps ----------
    def site_moments(self, structure: Structure) -> np.ndarray | None:
        if not (self.cfg.chgnet.enabled and self.cfg.electrafi.spin):
            return None
        return predict_site_moments(self.chgnet(), structure)

    def predict(
        self,
        inp: inputs.StructureInput | Structure | str | os.PathLike,
        grid_dims: tuple[int, int, int] | None = None,
        nelect: float | None = None,
        lmaxmix: int | None = None,
        site_moments: np.ndarray | None = None,
    ) -> Prediction:
        """Run the three models. Grid dims: argument > config > CHGCAR input (no VASP here)."""
        if isinstance(inp, Structure):
            inp = inputs.StructureInput(inp, None, "<Structure>")
        elif not isinstance(inp, inputs.StructureInput):
            inp = inputs.load_input(inp)
        structure = inp.structure
        grid_dims = grid_dims or self.cfg.grid_dims or inp.grid_dims
        if grid_dims is None:
            raise ValueError("grid dims unknown: pass grid_dims, set grid_dims in the config, or give a CHGCAR input")
        nelect = float(nelect) if nelect is not None else inputs.nelect_for(structure)

        if site_moments is None:
            site_moments = self.site_moments(structure)
        m_total = float(np.sum(site_moments)) if site_moments is not None else None

        model = self.electrafi(len(structure))
        rho_total, rho_spin = predict_density(model, structure, grid_dims, nelect, m_total)
        if not self.cfg.electrafi.spin:
            rho_spin = None

        zs = [site.specie.Z for site in structure]
        aug_pad, _ = self.augnet("total").predict(structure, lmaxmix)
        aug_total = aug_blocks(aug_pad, zs)
        aug_diff = None
        if rho_spin is not None and self.cfg.augnet.mag_checkpoint:
            mag_pad, _ = self.augnet("mag").predict(structure, lmaxmix)
            aug_diff = aug_blocks(mag_pad, zs)

        return Prediction(
            structure=structure, grid_dims=tuple(grid_dims), nelect=nelect,
            rho_total=rho_total, rho_spin=rho_spin, aug_total=aug_total, aug_diff=aug_diff,
            site_moments=site_moments, m_total=m_total, lmaxmix=lmaxmix,
            meta={
                "input": inp.path,
                "electrafi": self.cfg.electrafi.checkpoint,
                "augnet_total": self.cfg.augnet.total_checkpoint,
                "augnet_mag": self.cfg.augnet.mag_checkpoint if aug_diff is not None else None,
                "chgnet": self.cfg.chgnet.enabled and site_moments is not None,
                "device": str(self.device),
            },
        )

    def build(self, input_path: str | os.PathLike, out_dir: str | os.PathLike | None = None) -> Path:
        """Write CHGCAR, INCAR, POSCAR, POTCAR(.spec), KPOINTS and ndi_prediction.json to ``out_dir``."""
        out_dir = Path(out_dir or self.cfg.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        inp = inputs.load_input(input_path)
        structure = inp.structure
        spin = self.cfg.electrafi.spin

        # INCAR first: it decides LMAXMIX (for the augmentation blocks) and carries MAGMOM.
        site_moments = self.site_moments(structure)
        incar = assemble.write_vasp_inputs(structure, out_dir, self.cfg.vasp, site_moments, spin_polarized=spin)
        potcar = out_dir / "POTCAR"
        nelect = inputs.nelect_for(structure, potcar if potcar.is_file() else None)
        lmaxmix = int(incar.get("LMAXMIX", 2))

        grid_dims = self.cfg.grid_dims or inp.grid_dims or assemble.grid_dims_from_incar(structure, incar)
        if grid_dims is None:
            if not self.cfg.vasp.vasp_cmd:
                raise ValueError(
                    "grid dims unknown: set grid_dims, give a CHGCAR input, set NGXF/NGYF/NGZF in "
                    "vasp.incar_overrides, or configure vasp.vasp_cmd for a one-step dry run"
                )
            grid_dims = assemble.dry_run_grid_dims(structure, self.cfg.vasp, out_dir / "dryrun")

        pred = self.predict(inp, grid_dims, nelect, lmaxmix, site_moments)
        assemble.build_chgcar(
            structure, pred.rho_total, pred.rho_spin, pred.aug_total, pred.aug_diff,
            pred.site_moments, out_dir / "CHGCAR", nelect=nelect,
        )
        self._write_meta(pred, out_dir)
        return out_dir

    # ---------- outputs ----------
    def save_prediction(self, pred: Prediction, out_dir: str | os.PathLike, name: str = "structure") -> Path:
        """Grids as .npy and augmentation as .npz in the AugNet export layout
        (readable by vasp_runner.chgcar._aug_dict_from_npz)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "rho_total.npy", pred.rho_total)
        if pred.rho_spin is not None:
            np.save(out_dir / "rho_spin.npy", pred.rho_spin)
        z = torch.tensor([s.specie.Z for s in pred.structure])
        for channel, blocks in (("total", pred.aug_total), ("mag", pred.aug_diff)):
            if blocks is None:
                continue
            padded = np.zeros((len(z), 390))
            for i, blk in blocks.items():
                padded[i - 1, : blk.size] = blk
            schema_mask, lmaxmix_mask = sanvito_schema_and_lmaxmix_masks(z, pred.lmaxmix)
            np.savez_compressed(
                out_dir / f"{name}_{channel}_aug.npz",
                material_id=name, channel=channel, atomic_numbers=z.numpy(),
                aug_sanvito_padded=padded, schema_mask=schema_mask.numpy(), lmaxmix_mask=lmaxmix_mask.numpy(),
                lmaxmix=-1 if pred.lmaxmix is None else pred.lmaxmix,
                mask=(schema_mask & lmaxmix_mask).numpy(),
            )
        if pred.site_moments is not None:
            np.savetxt(out_dir / "magmom_chgnet.txt", pred.site_moments[None, :], fmt="%.6f")
        self._write_meta(pred, out_dir)
        return out_dir

    @staticmethod
    def _write_meta(pred: Prediction, out_dir: Path) -> None:
        meta = {
            **pred.meta,
            "n_sites": len(pred.structure),
            "grid_dims": list(pred.grid_dims),
            "nelect": pred.nelect,
            "total_integral": float(pred.rho_total.sum() * pred.structure.volume / pred.rho_total.size),
            "spin_integral": None if pred.rho_spin is None else float(pred.rho_spin.sum() * pred.structure.volume / pred.rho_spin.size),
            "m_total_constraint": pred.m_total,
            "site_moments_chgnet": None if pred.site_moments is None else [float(x) for x in pred.site_moments],
            "lmaxmix": pred.lmaxmix,
        }
        (out_dir / "ndi_prediction.json").write_text(json.dumps(meta, indent=2))
