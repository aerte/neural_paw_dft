from __future__ import annotations

import csv

import hashlib
import logging
import math
import os
import time

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ase import Atoms
from ase.data import chemical_symbols
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data

from model.data_utils import DensityDataset, DensityStream, _dl_worker_init, collate_fn
from model.escaip.EScAIP import EScAIPBackbone
from model.loss import DensityLoss, SpinDifferenceLoss
from model.model_utils import PWGrid, struc_from_gaussians_keops
from tools.density_conversions import cd_to_chgcar_mat
from tools.valence_picker import pick_valence_slots, required_valence_slots
from tools.visualization import (
    create_chg_delta,
    plot_atoms_with_displacements,
    plot_atoms_with_gaussian_shapes,
)
from utils.model_handling import ModelIO
from utils.train_helper_funcs import set_optimizer, set_optimizers_mixed

logger = logging.getLogger(__file__)



class ELECTRAFI(L.LightningModule):
    __version__ = 1

    def __init__(
        self,
        train_files: list[str] = None,
        test_files: list[str] = None,
        validation_files: list[str] = None,
        model_handler: ModelIO = None,
        config: dict = None,
        **kwargs,
    ):
        super().__init__()

        self.loss = DensityLoss(config=config, device=self.device)

        escaip_cfg = config["escaip_config"]
        self.gaus_per_electrons = config["gaus_per_electrons"]
        self.master_units = config["master_units"]
        self.cutoff = config["cutoff"]
        self.negative_contributions = config["negative_contributions"]
        self.config = config

        self.train_files = train_files
        self.validation_files = validation_files
        self.test_files = test_files
        self.dens_path = config["dens_path"]
        self.pred_dens_path_val = config["pred_dens_path_val"]
        self.pred_dens_path_test = config["pred_dens_path_test"]
        self.density_delta_path_val = config["density_delta_path_val"]
        self.density_delta_path_test = config["density_delta_path_test"]
        self.gaus_pos_path_val = config["gaus_pos_path_val"]
        self.gaus_pos_path_test = config["gaus_pos_path_test"]
        self.ood_paths = {
            name: path for name, path in zip(config["ood_names"], config["ood_paths"])
        }
        self.model_handler = model_handler

        # Spin channel: "total_diff" is the only mode.
        self.spin_type = config.get("spin_type", None)
        self.spin_enabled = self.spin_type == "total_diff"

        self.precision = torch.float32

        from tools.atom_tools_mpfull import VALENCE_OPTIONS, valence_dict

        self.VALENCE_OPTIONS = VALENCE_OPTIONS
        self.valence_dict = valence_dict

        # Backbone
        backbone_cfg = escaip_cfg["model"]["backbone"]
        needed_slots = required_valence_slots(self.VALENCE_OPTIONS)
        backbone_cfg["valence_slots"] = needed_slots
        backbone_cfg["use_pbc"] = self.config["pbc"] or self.config["wrap"]
        backbone_cfg["atom_embedding_size"] = config["master_units"]
        backbone_cfg["max_radius"] = self.cutoff
        self.backbone = EScAIPBackbone(**backbone_cfg)

        self.wsm_network = nn.Sequential(
            nn.Linear(self.gaus_per_electrons, self.gaus_per_electrons * 3),
            nn.Mish(),
            nn.Linear(self.gaus_per_electrons * 3, 2 * self.gaus_per_electrons),
        )

        self.tens_scale_net = nn.Sequential(
            nn.Linear(self.gaus_per_electrons, self.gaus_per_electrons),
            nn.Mish(),
            nn.Linear(self.gaus_per_electrons, self.gaus_per_electrons),
        )

        if self.spin_enabled:
            # Spin channel: second set of signed weights over the charge Gaussians (same shape as wsm_network).
            self.spin_w_net = nn.Sequential(
                nn.Linear(self.gaus_per_electrons, self.gaus_per_electrons * 3),
                nn.Mish(),
                nn.Linear(self.gaus_per_electrons * 3, 2 * self.gaus_per_electrons),
            )

            self.spin_loss = SpinDifferenceLoss(config=config, device=self.device)
            self.spin_loss_weight = float(config.get("spin_loss_weight", 0.2))
            # Below this int|m|dv the target is treated as non-magnetic and trained against zero.
            self.spin_mag_min = float(config.get("spin_mag_min", 0.1))  # electrons
            self.spin_warmup_steps = int(config.get("spin_warmup_steps", 500))

            # cd_diff is not resampled under cop/com rotation centres; only "origin" is supported.
            for split in ("train", "val", "test"):
                if config.get(f"rotate_{split}", False):
                    center = config.get(
                        "rotate_train_center"
                        if split == "train"
                        else "rotate_eval_center",
                        "origin",
                    )
                    if center != "origin":
                        raise ValueError(
                            f"spin_type={self.spin_type} requires rotate_{split}_center"
                            f"='origin' (got '{center}'): cd_diff is not resampled."
                        )

        # PW
        self.pw = PWGrid(
            tuple(self.config["pw_grid"]), device=self.device, dtype=self.precision
        )
        self.pw_norm = self.config.get("pw_norm", "backward")
        self.fft_set_dc = bool(self.config.get("fft_set_dc_to_electrons", False))

        self._err_sum = {"train": 0.0, "val": 0.0, "test": 0.0}
        self._err_count = {"train": 0, "val": 0, "test": 0}

        # EMA baseline for the loss-spike skip in training_step (not checkpointed on purpose).
        self._loss_ema = 0.0
        self._loss_ema_n = 0

        self.val_csv_path = os.path.join(
            self.gaus_pos_path_val, "val_inference_times.csv"
        )
        self.test_csv_path = os.path.join(
            self.gaus_pos_path_test, "test_inference_times.csv"
        )

        if not os.path.exists(self.val_csv_path):
            with open(self.val_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "n_atoms", "inference_seconds"])
        if not os.path.exists(self.test_csv_path):
            with open(self.test_csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "name",
                        "file",
                        "epoch",
                        "n_atoms",
                        "inference_seconds",
                        "density_error",
                    ]
                )

        # Inference timing CSV
        self.write_stage_timing_csv = bool(
            self.config.get("write_stage_timing_csv", False)
        )
        self.stage_timing_splits = set(
            self.config.get("stage_timing_splits", ["val", "test"])
        )

        self.val_stage_timing_csv_path = os.path.join(
            self.gaus_pos_path_val, "val_stage_timings.csv"
        )
        self.test_stage_timing_csv_path = os.path.join(
            self.gaus_pos_path_test, "test_stage_timings.csv"
        )

        if self.write_stage_timing_csv:
            if "val" in self.stage_timing_splits and not os.path.exists(
                self.val_stage_timing_csv_path
            ):
                with open(self.val_stage_timing_csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "epoch",
                            "file",
                            "n_atoms",
                            "n_electrons",
                            "backbone_seconds",
                            "gaussian_construction_seconds",
                            "grid_construction_seconds",
                        ]
                    )

            if "test" in self.stage_timing_splits and not os.path.exists(
                self.test_stage_timing_csv_path
            ):
                with open(self.test_stage_timing_csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(
                        [
                            "name",
                            "file",
                            "epoch",
                            "n_atoms",
                            "n_electrons",
                            "backbone_seconds",
                            "gaussian_construction_seconds",
                            "grid_construction_seconds",
                        ]
                    )

    def _append_val_inference_csv(self, epoch: int, n_atoms: int, seconds: float):
        # Assumes self.val_csv_path has been set up in __init__
        with open(self.val_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(epoch), int(n_atoms), float(seconds)])

    def _append_test_inference_csv(
        self,
        name: str,
        file: str,
        epoch: int,
        n_atoms: int,
        seconds: float,
        error: float,
    ):
        # Assumes self.val_csv_path has been set up in __init__
        with open(self.test_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    str(name),
                    str(file),
                    int(epoch),
                    int(n_atoms),
                    float(seconds),
                    float(error),
                ]
            )

    def _append_val_stage_timing_csv(
        self,
        epoch: int,
        file: str,
        n_atoms: int,
        n_electrons: float,
        backbone_seconds: float,
        gaussian_construction_seconds: float,
        grid_construction_seconds: float,
    ):
        with open(self.val_stage_timing_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    int(epoch),
                    str(file),
                    int(n_atoms),
                    float(n_electrons),
                    float(backbone_seconds),
                    float(gaussian_construction_seconds),
                    float(grid_construction_seconds),
                ]
            )

    def _append_test_stage_timing_csv(
        self,
        name: str,
        file: str,
        epoch: int,
        n_atoms: int,
        n_electrons: float,
        backbone_seconds: float,
        gaussian_construction_seconds: float,
        grid_construction_seconds: float,
    ):
        with open(self.test_stage_timing_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    str(name),
                    str(file),
                    int(epoch),
                    int(n_atoms),
                    float(n_electrons),
                    float(backbone_seconds),
                    float(gaussian_construction_seconds),
                    float(grid_construction_seconds),
                ]
            )

    def _now_sync(self) -> float:
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.time()

    def _rotation_matrix_from_axis_angle(
        self, axis: np.ndarray, angle: float
    ) -> np.ndarray:
        """
        Rodrigues' rotation formula.
        axis: shape (3,), assumed nonzero
        angle: radians
        returns: (3,3) rotation matrix
        """
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)

        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c

        R = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=np.float64,
        )
        return R

    def _random_rotation_matrix(
        self, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """
        Uniform random 3D rotation via random unit quaternion.
        """
        if rng is None:
            rng = np.random.default_rng()

        u1, u2, u3 = rng.random(3)

        q1 = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q2 = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q3 = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q4 = np.sqrt(u1) * np.cos(2 * np.pi * u3)

        # quaternion = (x, y, z, w)
        x, y, z, w = q1, q2, q3, q4

        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return R

    def _deterministic_rng_from_key(self, key: str) -> np.random.Generator:
        """
        Deterministic RNG from a string key, useful for reproducible val/test rotations.
        """
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], byteorder="little", signed=False)
        return np.random.default_rng(seed)

    def _rotate_atoms(
        self,
        atoms: Atoms,
        R: np.ndarray,
        wrap: bool = True,
        center_mode: str = "origin",
    ):
        atoms_rot = atoms.copy()

        pos = np.asarray(atoms_rot.positions, dtype=np.float64)
        cell_old = np.asarray(atoms_rot.cell.array, dtype=np.float64)

        if center_mode == "com":
            center = np.asarray(atoms_rot.get_center_of_mass(), dtype=np.float64)
        elif center_mode == "cop":
            center = pos.mean(axis=0)
        else:
            center = np.zeros(3, dtype=np.float64)

        pos_rot = (pos - center) @ R.T + center
        cell_new = cell_old @ R.T

        atoms_rot.positions = pos_rot
        atoms_rot.set_cell(cell_new, scale_atoms=False)

        if wrap and np.any(atoms_rot.pbc):
            atoms_rot.wrap()

        meta = {
            "R": R,
            "center_cart": center,
            "cell_old": cell_old,
            "cell_new": cell_new,
            "center_mode": center_mode,
        }
        return atoms_rot, meta

    def _maybe_rotate_structure(
        self, struc: Atoms, split: str, filename: str | None = None
    ):
        do_rotate = bool(self.config.get(f"rotate_{split}", False))
        if not do_rotate:
            return struc, None

        if split == "train":
            rng = np.random.default_rng()
            center_mode = self.config.get("rotate_train_center", "origin")
        else:
            deterministic = bool(self.config.get("rotate_eval_deterministic", True))
            center_mode = self.config.get("rotate_eval_center", "origin")
            if deterministic:
                key = f"{split}::{filename if filename is not None else 'none'}"
                rng = self._deterministic_rng_from_key(key)
            else:
                rng = np.random.default_rng()

        R = self._random_rotation_matrix(rng)
        struc_rot, meta = self._rotate_atoms(
            struc,
            R=R,
            wrap=bool(self.config.get("wrap", True) or self.config.get("pbc", True)),
            center_mode=center_mode,
        )
        return struc_rot, meta

    def _renormalize_density_sum(
        self, rho_new: torch.Tensor, rho_old: torch.Tensor
    ) -> torch.Tensor:
        s_old = rho_old.sum().clamp_min(1e-12)
        s_new = rho_new.sum().clamp_min(1e-12)
        return rho_new * (s_old / s_new)

    def _maybe_transform_density_target(
        self, density: torch.Tensor, rot_meta: dict | None
    ) -> torch.Tensor:
        """
        For origin rotation, the density tensor can stay unchanged.
        For COP/COM, resample target density onto the rotated cell grid.
        """
        if rot_meta is None:
            return density

        center_mode = rot_meta["center_mode"]
        if center_mode == "origin":
            return density

        rho_old = density.to(self.device)
        cell_old = torch.as_tensor(
            rot_meta["cell_old"], device=self.device, dtype=self.precision
        )
        cell_new = torch.as_tensor(
            rot_meta["cell_new"], device=self.device, dtype=self.precision
        )
        R = torch.as_tensor(rot_meta["R"], device=self.device, dtype=self.precision)
        center_cart = torch.as_tensor(
            rot_meta["center_cart"], device=self.device, dtype=self.precision
        )

        rho_new = rigidly_resample_density_to_new_cell(
            rho_old=rho_old,
            cell_old=cell_old,
            cell_new=cell_new,
            R=R,
            center_cart=center_cart,
        )
        rho_new = self._renormalize_density_sum(rho_new, rho_old)
        return rho_new

    # ---------- forward ----------
    def forward(
        self,
        atoms: Atoms,
        n_elec: int | None = None,
        sampled_points: torch.Tensor | None = None,
        m_total: float | None = None,
        **kwargs,
    ):
        if self.pw.Gx.device != self.device or self.pw.Gx.dtype != self.precision:
            self.pw.to(dtype=self.precision, device=self.device)

        cell0 = torch.tensor(atoms.cell[:], dtype=self.precision, device=self.device)
        self.pw.set_cell(cell0)

        V = float(atoms.get_volume())
        nx, ny, nz = self.pw.real_shape
        Nreal = nx * ny * nz
        dv = V / Nreal

        # spin_renorm: False disables the net-moment rescale of the spin weights.
        if m_total is not None and not self.config.get("spin_renorm", True):
            m_total = None
        enc = self._encode_gaussians(atoms, n_elec=n_elec, dv=dv, m_total=m_total)
        mu, Sigma, weights = enc["mu"], enc["Sigma"], enc["weights"]
        scal_mults = enc["scal_mults"]
        n_multiples = enc["n_multiples"]
        raw_irrep_out = enc["raw"]
        cell = enc["cell"]
        displacements = enc["displacements"]
        scalars = raw_irrep_out["rawirrep_scalars"]

        # grid construction timing
        t0 = self._now_sync()
        rho_out, rho_full = self._gaussians_to_rho(mu, Sigma, weights, sampled_points)
        t1 = self._now_sync()
        grid_construction_seconds = t1 - t0

        # Spin (charge-difference) density: same Gaussians, second set of weights.
        rho_spin = None
        if self.spin_enabled:
            rho_spin, _ = self._gaussians_to_rho(
                mu, Sigma, enc["weights_spin"], sampled_points
            )

        return {
            "rho": rho_out,
            "rho_full": rho_full,
            "rho_spin": rho_spin,
            "cell": cell,
            "weights": weights,
            "pos_disp": displacements.view(-1, 3),
            "cov": Sigma,
            "scal_mults": scal_mults,
            "n_multiples": n_multiples,
            "mu": mu,
            "timings": {
                **enc.get("timings", {}),
                "grid_construction_seconds": grid_construction_seconds,
            },
        }

    def _gaussians_to_rho(
        self,
        mu: torch.Tensor,
        Sigma: torch.Tensor,
        weights: torch.Tensor,
        sampled_points: torch.Tensor | None,
    ):
        """Build a real-space density from Gaussians via the shared FFT path.

        Returns (rho_out, rho_full) where rho_full is None when points are sampled.
        """
        cG = struc_from_gaussians_keops(self.pw, mu, Sigma, weights)
        rho_full = torch.fft.irfftn(cG, s=self.pw.real_shape, norm=self.pw_norm).real
        if sampled_points is None:
            return rho_full, rho_full
        return rho_full.flatten().index_select(0, sampled_points), None

    def _encode_gaussians(
        self,
        atoms: Atoms,
        n_elec: float | None = None,
        dv: float | None = None,
        m_total: float | None = None,
    ):
        atoms.pbc = self.config["pbc"] or self.config["wrap"]

        per_atom_slots = None
        if n_elec is not None:
            per_atom_slots, chosen_assignment, exact, delta = pick_valence_slots(
                atoms,
                Ne_target=float(n_elec),
                tol=float(self.config.get("electron_sum_tol", 1e-6)),
                device=self.device,
                VALENCE_OPTIONS=self.VALENCE_OPTIONS,
                valence_dict=self.valence_dict,
            )

        # Build pyg batch with chosen slots (or slot 0 if unknown)
        batch = self.atoms_to_pyg(
            atoms, device=self.device, slots_per_atom=per_atom_slots
        )

        # backbone pass timing
        t0 = self._now_sync()
        raw = self.backbone(batch)
        t1 = self._now_sync()
        backbone_seconds = t1 - t0

        # Gaussian construction timing
        t2 = self._now_sync()

        if raw["data"].node_padding_mask is not None:
            m = raw["data"].node_padding_mask
            for key in list(raw.keys()):
                if key.startswith("rawirrep_"):
                    raw[key] = raw[key][m]

        scalars = raw["rawirrep_scalars"]
        vectors = raw["rawirrep_vectors"]
        tensors = raw["rawirrep_tensors"]

        species_ids = batch.species_ids.to(self.device)

        valence = self._valence_from_slots(atoms, species_ids)
        n_multiples = (valence * self.gaus_per_electrons).to(self.device)
        n_valence_elec = torch.sum(valence)

        cell = torch.tensor(atoms.cell[:], dtype=self.precision, device=self.device)

        at_pos = torch.as_tensor(
            atoms.positions, device=self.device, dtype=self.precision
        )
        pos_base = (
            at_pos.view(n_multiples.shape[0], -1)
            .repeat_interleave(n_multiples, 0)
            .view(-1, 3)
        )

        # Charge channel (electron-normalized when n_elec is given)
        mu, Sigma, weights, disp, scalars_sliced = self._build_gaussians(
            scalars,
            vectors,
            tensors,
            self.wsm_network,
            self.tens_scale_net,
            pos_base,
            n_multiples,
            n_valence_elec,
            n_elec=n_elec,
            dv=dv,
        )

        enc = {
            "mu": mu,
            "Sigma": Sigma,
            "weights": weights,
            "scal_mults": weights,
            "n_multiples": n_multiples,
            "raw": raw,
            "cell": cell,
            "displacements": disp,
        }

        # Spin channel: signed weights over the same (mu, Sigma), normalised to m_total like the charge channel to n_elec.
        if self.spin_enabled:
            enc["weights_spin"] = self._build_spin_weights(
                scalars_sliced, m_total=m_total, dv=dv
            )

        t3 = self._now_sync()
        gaussian_construction_seconds = t3 - t2

        enc["timings"] = {
            "backbone_seconds": backbone_seconds,
            "gaussian_construction_seconds": gaussian_construction_seconds,
        }
        return enc

    def _build_gaussians(
        self,
        scalars: torch.Tensor,
        vectors: torch.Tensor,
        tensors: torch.Tensor,
        wsm_net: nn.Module,
        tens_net: nn.Module,
        pos_base: torch.Tensor,
        n_multiples: torch.Tensor,
        n_valence_elec: torch.Tensor,
        n_elec: float | None,
        dv: float | None,
    ):
        """Turn raw irreps (scalars/vectors/tensors) into Gaussian (mu, Sigma, weights).

        Also returns the sliced scalars, which the spin head reuses to produce a
        second set of weights over the same Gaussians.
        """
        scalars_sliced = self.slice(scalars, n_multiples).view(
            n_valence_elec, self.gaus_per_electrons
        )
        vectors_sliced = self.slice(vectors, n_multiples).view(
            n_valence_elec, self.gaus_per_electrons, 3
        )
        tensors_sliced = self.slice(tensors, n_multiples).view(
            n_valence_elec, self.gaus_per_electrons, 3, 3
        )

        wsm = wsm_net(scalars_sliced)
        w = wsm[:, : self.gaus_per_electrons]
        w_logits = w.reshape(-1)
        aux = wsm[:, self.gaus_per_electrons : self.gaus_per_electrons * 2].reshape(-1)

        mode = self.config.get("signed_weights", "tanh_softplus")
        mag_max = self.config["weight_mag_max"]
        if mode == "tanh_softplus":
            mag = F.softplus(w_logits).clamp(max=mag_max)
            s = torch.tanh(aux)
            weights = s * mag
        elif mode == "softsign":
            mag = F.softplus(w_logits).clamp(max=mag_max)
            s = F.softsign(aux)
            weights = s * mag
        else:
            weights = F.softplus(w_logits).clamp(max=mag_max)
            weights = weights * torch.sign(aux)
        weights = weights.reshape(-1)

        disp = vectors_sliced.reshape(-1, 3)
        mu = pos_base + disp

        T = tensors_sliced.reshape(-1, 3, 3)
        Sigma = T @ T.mT
        sigma_norm = torch.linalg.norm(T, dim=(-2, -1)).clamp_min(1e-6)
        Sigma = Sigma / sigma_norm.unsqueeze(-1).unsqueeze(-1)
        s = tens_net(scalars_sliced).view(-1)
        s = torch.nn.functional.softplus(s)
        s = s.clamp(
            min=self.config["sigma_scale_min"], max=self.config["sigma_scale_max"]
        )
        Sigma = s.view(-1, 1, 1) * Sigma

        # Cap trace(Sigma) so that every eigenvalue is bounded (sigma_scale_max only bounds s).
        tr_max = float(self.config.get("sigma_trace_max", 0.0))
        if tr_max > 0.0:
            tr = Sigma.diagonal(dim1=-2, dim2=-1).sum(-1)
            Sigma = Sigma * (tr_max / tr.clamp_min(1e-12)).clamp(max=1.0).view(-1, 1, 1)

        # Rescale so the density integrates to the valence electron count. The signed
        # sum is floored relative to sum|w| so the factor is bounded by 1/wsum_rel_floor.
        if n_elec is not None:
            wsum = weights.sum()
            rel_floor = float(self.config.get("wsum_rel_floor", 1e-3))
            floor = rel_floor * weights.abs().sum().clamp_min(1e-12)
            sign = torch.where(wsum < 0, -torch.ones_like(wsum), torch.ones_like(wsum))
            safe_wsum = torch.where(wsum.abs() < floor, sign * floor, wsum)
            weights = weights * ((n_elec / dv) / safe_wsum)

        return mu, Sigma, weights, disp, scalars_sliced

    def _build_spin_weights(
        self,
        scalars_sliced: torch.Tensor,
        m_total: float | None,
        dv: float | None,
    ):
        """Signed Gaussian weights for the spin channel, over the charge channel's
        (mu, Sigma). Rescaled to integrate to the net magnetization m_total when that
        is meaningfully non-zero -- the direct analogue of the charge channel's n_elec
        normalization, and the only thing that pins the amplitude of m(r).
        """
        G = self.gaus_per_electrons
        wsm = self.spin_w_net(scalars_sliced)
        mag = F.softplus(wsm[:, :G].reshape(-1)).clamp(max=self.config["weight_mag_max"])
        weights = torch.tanh(wsm[:, G : 2 * G].reshape(-1)) * mag

        # Skip the rescale for near-zero net moment (antiferromagnets); the NMAE loss still carries scale.
        if m_total is not None and abs(m_total) >= self.spin_mag_min:
            wsum = weights.sum()
            safe_wsum = torch.where(
                wsum.abs() < 1e-6, torch.full_like(wsum, 1e-6), wsum
            )
            # Bound the rescale factor: signed weights cancel strongly here.
            scale = ((m_total / dv) / safe_wsum).clamp(-10.0, 10.0)
            weights = weights * scale

        return weights

    # ---------- training / eval ----------
    def configure_optimizers(self):
        if self.config.get("optimizer", "").lower() == "muon_mix":
            opts = set_optimizers_mixed(self, self.config)
            self.automatic_optimization = False
        else:
            # your legacy single-optimizer path
            opts = [set_optimizer(self, self.config)]

        anneal_milestones = list(self.config["lr_dec_every"] * np.arange(1, 500))
        scheds = []
        for opt in opts:
            scheds.append(
                torch.optim.lr_scheduler.MultiStepLR(
                    opt, milestones=anneal_milestones, gamma=self.config["lr_gamma"]
                )
            )
        # Lightning is fine with multiple optimizers+schedulers on the same loss
        return opts, scheds

    def _sample_cfg(self, split: str):
        # Prefer *_sample_frac; fallback to *_n_points for legacy
        key_frac = f"{split}_sample_frac"
        key_n = f"{split}_n_points"
        if key_frac in self.config and self.config[key_frac]:
            return self.config[key_frac]
        return self.config.get(key_n, 0)

    def _spin_stats(self, cd_diff, dv: float):
        """Net moment and total |m|, both in electrons.

        m_total normalizes the predicted amplitude (analogue of n_elec for charge);
        it is derived from the target grid exactly as Ne_true already is.
        """
        if not self.spin_enabled or cd_diff is None:
            return None, 0.0
        return float(cd_diff.sum()) * dv, float(cd_diff.abs().sum()) * dv

    def _apply_spin_loss(self, split, rho_spin, cd_diff, dv, sampled_points):
        """Weighted spin loss + metrics. Note spin_nmae == 1.0 is exactly the
        trivial m == 0 predictor, so < 1.0 is the bar this channel has to clear.
        """
        loss, nmae, abs_err = self.spin_loss.compute_spin_loss(
            rho_spin, cd_diff, dv, sampled_points
        )
        self._log_split_metric(split, "spin_loss", loss)
        self._log_split_metric(split, "spin_abs_err_e", abs_err)
        if nmae is not None:
            self._log_split_metric(split, "spin_nmae", nmae)

        w = self.spin_loss_weight
        if split == "train" and self.spin_warmup_steps > 0:
            # Ramp the spin term in so it does not dominate the charge gradient early.
            w *= min(1.0, (self.global_step + 1) / self.spin_warmup_steps)
        return w * loss

    def training_step(self, batch, batch_idx: int):
        density, struc, n_elec, grid_dict, filename, pos_grid, cd_diff = batch[0]
        struc, rot_meta = self._maybe_rotate_structure(
            struc, split="train", filename=str(filename)
        )
        density = self._maybe_transform_density_target(density, rot_meta)
        nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
        if (self.pw.nx, self.pw.ny, self.pw.nz) != (nx, ny, nz):
            self.pw = PWGrid((nx, ny, nz), device=self.device, dtype=self.precision)
        self.pw.set_cell(
            torch.tensor(struc.cell[:], dtype=self.precision, device=self.device)
        )

        sampled_points = self._maybe_sample_points(grid_dict, self._sample_cfg("train"))
        nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
        dv = float(struc.get_volume()) / (nx * ny * nz)
        Ne_true = float(density.sum().detach().cpu().item() * dv)
        m_spin, _ = self._spin_stats(cd_diff, dv)
        out = self(
            struc, n_elec=Ne_true, sampled_points=sampled_points, m_total=m_spin
        )
        if self.config["log_cov_stats"]:
            self._log_cov_stats(out["cov"], split="train", step_idx=self.global_step)
        self._check_electron_mismatch(
            split="train",
            filename=str(filename),
            struc=struc,
            ne_true_grid=Ne_true,
            n_multiples=out["n_multiples"],
        )

        rho = out["rho"]

        pred_cd = rho
        n_val_elec = Ne_true
        # self._append_val_inference_csv(epoch=self.current_epoch, n_atoms=len(struc), seconds=(end - start))
        loss, dens_error = self.loss.compute_total_loss(
            sys=struc,
            pred_cd=pred_cd,
            true_cd_tens=density,
            grid_dict=grid_dict,
            n_valence_electrons=n_val_elec,
            sampled_points=sampled_points,
            training=True,
            volume=None,
        )
        # Non-magnetic systems are kept and trained against m == 0.
        if self.spin_enabled and out["rho_spin"] is not None and cd_diff is not None:
            loss = loss + self._apply_spin_loss(
                "train", out["rho_spin"], cd_diff, dv, sampled_points
            )
        self._maybe_report_outlier(
            split="train",
            err_scalar=float(dens_error.detach().cpu().item()),
            filename=str(filename),
            formula=struc.get_chemical_formula(),
            struc=struc,
            grid_dict=grid_dict,
            forward_out=out,  # contains "cov" and "n_multiples"
            dv=dv,
            ne_true_grid=Ne_true,  # = sum(density)*dv
        )
        np_err = float(torch.round(1000 * dens_error.detach().cpu(), decimals=5) / 10.0)
        self.log(
            "Train Density Err %",
            np_err,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        self._log_split_metric("train", "num_nodes", len(struc))
        self._log_split_metric("train", "total_density_err_pct", np_err)
        self._log_split_metric("train", "total_density_nmae", dens_error)
        density_mae = self.loss.compute_mae_loss(pred_cd, density, sampled_points)
        self._log_split_metric("train", "total_density_mae", density_mae)
        # Covariance-trace telemetry (cheap: a sum, no eigendecomposition).
        with torch.no_grad():
            _tr = out["cov"].diagonal(dim1=-2, dim2=-1).sum(-1)
            self._log_split_metric("train", "cov_trace_p50", _tr.median())
            self._log_split_metric("train", "cov_trace_max", _tr.max())
            _tr_max = float(self.config.get("sigma_trace_max", 0.0))
            if _tr_max > 0.0:
                self._log_split_metric(
                    "train", "cov_sat_frac", (_tr > 0.9 * _tr_max).float().mean()
                )
        if self.config.get("optimizer", "").lower() == "muon_mix":
            # Grab the two optimizers in the same order you returned them
            opt_muon, opt_adamw = self.optimizers()

            # Zero-grad (Lightning won't do this for manual opt)
            opt_muon.zero_grad(set_to_none=True)
            opt_adamw.zero_grad(set_to_none=True)

            # Skip the step on a non-finite loss so one bad batch can't poison weights
            if not torch.isfinite(loss):
                self.log(
                    "train/nonfinite_skip",
                    1.0,
                    on_step=True,
                    prog_bar=False,
                    batch_size=1,
                )
                return None

            # Skip steps whose charge NMAE exceeds skip_loss_spike_factor x an EMA baseline
            # (the EMA is not updated on skipped steps).
            spike_factor = float(self.config.get("skip_loss_spike_factor", 50.0))
            _err = float(dens_error.detach())
            if spike_factor > 0.0 and math.isfinite(_err):
                if self._loss_ema_n >= self.config.get(
                    "skip_loss_spike_warmup", 200
                ) and _err > spike_factor * max(self._loss_ema, 1e-12):
                    self.log(
                        "train/loss_spike_skip",
                        1.0,
                        on_step=True,
                        prog_bar=False,
                        batch_size=1,
                    )
                    self._log_split_metric(
                        "train", "loss_spike_ratio", _err / max(self._loss_ema, 1e-12)
                    )
                    logger.warning(
                        f"[TRAIN] skipping step {self.global_step}: NMAE {_err:.6g} > "
                        f"{spike_factor}x EMA {self._loss_ema:.6g} | file='{filename}'"
                    )
                    return None
                _beta = 0.99
                self._loss_ema = (
                    _err
                    if self._loss_ema_n == 0
                    else _beta * self._loss_ema + (1.0 - _beta) * _err
                )
                self._loss_ema_n += 1

            # Backward
            self.manual_backward(loss)

            # Skip the step on any non-finite gradient (norm clipping would spread the NaN to every parameter).
            if self.config.get("skip_nonfinite_grads", True):
                bad = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for group in (opt_muon, opt_adamw)
                    for pg in group.param_groups
                    for p in pg["params"]
                )
                if bad:
                    self.log(
                        "train/nonfinite_grad_skip",
                        1.0,
                        on_step=True,
                        prog_bar=False,
                        batch_size=1,
                    )
                    opt_muon.zero_grad(set_to_none=True)
                    opt_adamw.zero_grad(set_to_none=True)
                    return None

            # Gradient clipping (manual opt: Trainer-level clip is ignored, must clip here)
            if self.config.get("clip_grad", False):
                cv = self.config["gradient_clip_value"]
                self.clip_gradients(
                    opt_muon, gradient_clip_val=cv, gradient_clip_algorithm="norm"
                )
                self.clip_gradients(
                    opt_adamw, gradient_clip_val=cv, gradient_clip_algorithm="norm"
                )

            # Step both
            opt_muon.step()
            opt_adamw.step()


            return {"loss": loss}
        return loss

    def on_train_epoch_end(self):
        # Manual optimization does not auto-step LR schedulers.
        if not self.automatic_optimization:
            scheds = self.lr_schedulers()
            if scheds is None:
                return
            if not isinstance(scheds, (list, tuple)):
                scheds = [scheds]
            for sch in scheds:
                sch.step()

    def validation_step(self, batch, batch_idx: int):
        density, struc, n_elec, grid_dict, filename, pos_grid, cd_diff = batch[0]
        struc, rot_meta = self._maybe_rotate_structure(
            struc, split="val", filename=str(filename)
        )
        density = self._maybe_transform_density_target(density, rot_meta)

        nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
        dv = float(struc.get_volume()) / (nx * ny * nz)
        Ne_true = float(density.sum().detach().cpu().item() * dv)
        m_spin, _ = self._spin_stats(cd_diff, dv)
        if (self.pw.nx, self.pw.ny, self.pw.nz) != (nx, ny, nz):
            self.pw = PWGrid((nx, ny, nz), device=self.device, dtype=self.precision)
        self.pw.set_cell(
            torch.tensor(struc.cell[:], dtype=self.precision, device=self.device)
        )
        sampled_points = None
        if torch.cuda.is_available():
            self.eval()
            torch.cuda.synchronize()
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                starter.record()
                if (self.pw.nx, self.pw.ny, self.pw.nz) != (nx, ny, nz):
                    self.pw = PWGrid(
                        (nx, ny, nz), device=self.device, dtype=self.precision
                    )
                self.pw.set_cell(
                    torch.tensor(
                        struc.cell[:], dtype=self.precision, device=self.device
                    )
                )
                out = self(
                    struc,
                    n_elec=Ne_true,
                    sampled_points=sampled_points,
                    m_total=m_spin,
                )
                if self.write_stage_timing_csv and "val" in self.stage_timing_splits:
                    timings = out.get("timings", {})
                    self._append_val_stage_timing_csv(
                        epoch=self.current_epoch,
                        file=str(filename),
                        n_atoms=len(struc),
                        n_electrons=Ne_true,
                        backbone_seconds=timings.get("backbone_seconds", float("nan")),
                        gaussian_construction_seconds=timings.get(
                            "gaussian_construction_seconds", float("nan")
                        ),
                        grid_construction_seconds=timings.get(
                            "grid_construction_seconds", float("nan")
                        ),
                    )
                ender.record()
            torch.cuda.synchronize()
            dt = (
                starter.elapsed_time(ender) / 1000.0
            )  # ender time is in ms, so we divide by 1000 to get seconds
        else:
            start = time.time()
            with torch.inference_mode():
                if (self.pw.nx, self.pw.ny, self.pw.nz) != (nx, ny, nz):
                    self.pw = PWGrid(
                        (nx, ny, nz), device=self.device, dtype=self.precision
                    )
                self.pw.set_cell(
                    torch.tensor(
                        struc.cell[:], dtype=self.precision, device=self.device
                    )
                )
                out = self(
                    struc,
                    n_elec=Ne_true,
                    sampled_points=sampled_points,
                    m_total=m_spin,
                )
            dt = time.time() - start
        self._log_cov_stats(out["cov"], split="val", step_idx=self.global_step)
        self._check_electron_mismatch(
            split="val",
            filename=str(filename),
            struc=struc,
            ne_true_grid=Ne_true,
            n_multiples=out["n_multiples"],
        )
        rho = out["rho"]

        pred_cd = rho
        n_val_elec = Ne_true

        self.log(
            "Val Inference time",
            dt,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=1,
        )
        loss, density_error = self.loss.compute_total_loss(
            sys=struc,
            pred_cd=pred_cd,
            true_cd_tens=density,
            grid_dict=grid_dict,
            n_valence_electrons=n_val_elec,
            sampled_points=sampled_points,
            training=False,
            volume=None,
        )
        if self.spin_enabled and out["rho_spin"] is not None and cd_diff is not None:
            loss = loss + self._apply_spin_loss(
                "val", out["rho_spin"], cd_diff, dv, sampled_points
            )
        self._maybe_report_outlier(
            split="val",
            err_scalar=float(density_error.detach().cpu().item()),
            filename=str(filename),
            formula=struc.get_chemical_formula(),
            struc=struc,
            grid_dict=grid_dict,
            forward_out=out
            if "out" in locals()
            else None,  # if you want the same stats
            dv=dv,
            ne_true_grid=Ne_true,
        )

        np_err = float(
            torch.round(1000 * density_error.detach().cpu(), decimals=5) / 10.0
        )
        self.log(
            "Validation Density Err %",
            np_err,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        self._log_split_metric("val", "inference_time", dt)
        self._log_split_metric("val", "num_nodes", len(struc))
        self._log_split_metric("val", "total_density_err_pct", np_err)
        self._log_split_metric("val", "total_density_nmae", density_error)
        density_mae = self.loss.compute_mae_loss(pred_cd, density, sampled_points)
        self._log_split_metric("val", "total_density_mae", density_mae)

        if batch_idx == 0 and self.config["save_model"]:
            self.model_handler.save(self)

        if (
            self.config["construct_val_cd"]
            and batch_idx % self.config["visualize_every"] == 0
        ):
            path = os.path.join(self.dens_path, filename)
            id = filename.split("/")[-1]
            id_no_ending = id.split(".")[0]
            filename = f"{self.pred_dens_path_val}{id_no_ending}_{struc.get_chemical_formula()}_{float(density_error)}_val_{batch_idx}.CHGCAR"
            name_iter_str = (
                f"{id_no_ending}_{struc.get_chemical_formula()}_val_{batch_idx}"
            )
            cd_to_chgcar_mat(
                original_file=path,
                atoms=struc,
                cd=pred_cd.detach().cpu().numpy(),
                filename=filename,
            )
            create_chg_delta(
                pred_dens_file=filename,
                true_dens_file=path,
                delta_folder=self.density_delta_path_val,
                name_iter_str=name_iter_str,
            )
            vis_base = f"{struc.get_chemical_formula()}_val_{batch_idx}"
            ax1, path1 = plot_atoms_with_displacements(
                struc,
                mu=out["mu"],
                n_multiples=out["n_multiples"],
                cell=out["cell"],
                save_dir=self.config["gaus_pos_path_val"],
                base_name=vis_base,
                structure_error=float(density_error),
                dpi=180,
            )

            ax2, path2 = plot_atoms_with_gaussian_shapes(
                struc,
                mu=out["mu"],
                weights=out["weights"],
                Sigma=out["cov"],
                mode="ellipsoids",
                iso_sigma=self.config.get("viz_iso_sigma", 0.01),
                mesh_nu=self.config.get("viz_mesh_nu", 20),
                mesh_nv=self.config.get("viz_mesh_nv", 12),
                alpha=self.config.get("viz_alpha", 0.5),
                max_shapes=self.config.get("viz_max_shapes", 400),
                save_dir=self.config["gaus_pos_path_val"],
                base_name=vis_base,
                structure_error=float(density_error),
                dpi=180,
            )
        return loss

    def test_step(self, batch, batch_idx: int):
        density, struc, n_elec, grid_dict, filename, ood_name, cd_diff = batch[0]
        struc, rot_meta = self._maybe_rotate_structure(
            struc, split="test", filename=str(filename)
        )
        density = self._maybe_transform_density_target(density, rot_meta)

        nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
        dv = float(struc.get_volume()) / (nx * ny * nz)
        Ne_true = float(density.sum().detach().cpu().item() * dv)
        m_spin, _ = self._spin_stats(cd_diff, dv)
        sampled_points = None
        data_name = str(ood_name) if ood_name is not None else "default"
        if torch.cuda.is_available():
            self.eval()
            torch.cuda.synchronize()
        if torch.cuda.is_available():
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                starter.record()
                if (self.pw.nx, self.pw.ny, self.pw.nz) != (nx, ny, nz):
                    self.pw = PWGrid(
                        (nx, ny, nz), device=self.device, dtype=self.precision
                    )
                self.pw.set_cell(
                    torch.tensor(
                        struc.cell[:], dtype=self.precision, device=self.device
                    )
                )
                out = self(
                    struc,
                    n_elec=Ne_true,
                    sampled_points=sampled_points,
                    m_total=m_spin,
                )
                if self.write_stage_timing_csv and "test" in self.stage_timing_splits:
                    timings = out.get("timings", {})
                    self._append_test_stage_timing_csv(
                        name=data_name,
                        file=str(filename),
                        epoch=self.current_epoch,
                        n_atoms=len(struc),
                        n_electrons=Ne_true,
                        backbone_seconds=timings.get("backbone_seconds", float("nan")),
                        gaussian_construction_seconds=timings.get(
                            "gaussian_construction_seconds", float("nan")
                        ),
                        grid_construction_seconds=timings.get(
                            "grid_construction_seconds", float("nan")
                        ),
                    )
                ender.record()
            torch.cuda.synchronize()
            dt = (
                starter.elapsed_time(ender) / 1000.0
            )  # ender time is in ms, so we divide by 1000 to get seconds
        else:
            start = time.time()
            with torch.inference_mode():
                if (self.pw.nx, self.pw.ny, self.pw.nz) != (nx, ny, nz):
                    self.pw = PWGrid(
                        (nx, ny, nz), device=self.device, dtype=self.precision
                    )
                self.pw.set_cell(
                    torch.tensor(
                        struc.cell[:], dtype=self.precision, device=self.device
                    )
                )
                out = self(
                    struc,
                    n_elec=Ne_true,
                    sampled_points=sampled_points,
                    m_total=m_spin,
                )
            dt = time.time() - start
        test_split = f"test/{data_name}" if ood_name is not None else "test"
        self._log_cov_stats(out["cov"], split=test_split, step_idx=self.global_step)
        rho = out["rho"]
        self._check_electron_mismatch(
            split="test",
            filename=str(filename),
            struc=struc,
            ne_true_grid=Ne_true,
            n_multiples=out["n_multiples"],
        )

        pred_cd = rho
        n_val_elec = Ne_true

        self.log(
            f"Test Inference time ({data_name})",
            dt,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=1,
        )
        loss, density_error = self.loss.compute_total_loss(
            sys=struc,
            pred_cd=pred_cd,
            true_cd_tens=density,
            grid_dict=grid_dict,
            n_valence_electrons=n_val_elec,
            sampled_points=sampled_points,
            training=False,
            volume=None,
        )
        if self.spin_enabled and out["rho_spin"] is not None and cd_diff is not None:
            loss = loss + self._apply_spin_loss(
                test_split, out["rho_spin"], cd_diff, dv, sampled_points
            )

        np_err = float(
            torch.round(1000 * density_error.detach().cpu(), decimals=5) / 10.0
        )
        self._append_test_inference_csv(
            name=data_name,
            file=filename,
            epoch=self.current_epoch,
            n_atoms=len(struc),
            seconds=dt,
            error=np_err,
        )
        self.log(
            f"Test Density Err % ({data_name})",
            np_err,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        self._log_split_metric(test_split, "inference_time", dt)
        self._log_split_metric(test_split, "num_nodes", len(struc))
        self._log_split_metric(test_split, "total_density_err_pct", np_err)
        self._log_split_metric(test_split, "total_density_nmae", density_error)
        density_mae = self.loss.compute_mae_loss(pred_cd, density, sampled_points)
        self._log_split_metric(test_split, "total_density_mae", density_mae)

        if ood_name is not None:
            save = True
            create_viz = False
        elif (
            self.config["construct_test_cd"]
            and batch_idx % self.config["visualize_every"] == 0
        ):
            save = True
            create_viz = True
        elif str(self.config["data_split"]).startswith("mpfull2025"):
            save = True
            create_viz = False
        else:
            save = False
            create_viz = False
        if save:
            if ood_name is not None:
                path = os.path.join(self.ood_paths[ood_name], filename)
            else:
                path = os.path.join(self.dens_path, filename)
            id = path.split("/")[-1]
            id_no_ending = id.split(".")[0]
            if ood_name is not None:
                pred_dens_path_test = os.path.join(
                    self.pred_dens_path_test, f"{ood_name}/"
                )
                density_delta_path_test = os.path.join(
                    self.density_delta_path_test, f"{ood_name}/"
                )
                gaus_pos_path_test = os.path.join(
                    self.gaus_pos_path_test, f"{ood_name}/"
                )
                os.makedirs(pred_dens_path_test, exist_ok=True)
                os.makedirs(density_delta_path_test, exist_ok=True)
                os.makedirs(gaus_pos_path_test, exist_ok=True)
            else:
                pred_dens_path_test = self.pred_dens_path_test
                density_delta_path_test = self.density_delta_path_test
                gaus_pos_path_test = self.gaus_pos_path_test
            filename = f"{pred_dens_path_test}{id_no_ending}_{struc.get_chemical_formula()}_{float(density_error)}_test_{batch_idx}.CHGCAR"
            name_iter_str = (
                f"{id_no_ending}_{struc.get_chemical_formula()}_test_{batch_idx}"
            )
            cd_to_chgcar_mat(
                original_file=path,
                atoms=struc,
                cd=pred_cd.detach().cpu().numpy(),
                filename=filename,
            )

            if create_viz:
                create_chg_delta(
                    pred_dens_file=filename,
                    true_dens_file=path,
                    delta_folder=density_delta_path_test,
                    name_iter_str=name_iter_str,
                )
                vis_base = f"{struc.get_chemical_formula()}_test_{batch_idx}"
                ax1, path1 = plot_atoms_with_displacements(
                    struc,
                    mu=out["mu"],
                    n_multiples=out["n_multiples"],
                    cell=out["cell"],
                    save_dir=gaus_pos_path_test,
                    base_name=vis_base,
                    structure_error=float(density_error),
                    dpi=180,
                )

                ax2, path2 = plot_atoms_with_gaussian_shapes(
                    struc,
                    mu=out["mu"],
                    weights=out["weights"],
                    Sigma=out["cov"],
                    mode="ellipsoids",
                    iso_sigma=self.config.get("viz_iso_sigma", 0.01),
                    mesh_nu=self.config.get("viz_mesh_nu", 20),
                    mesh_nv=self.config.get("viz_mesh_nv", 12),
                    alpha=self.config.get("viz_alpha", 0.5),
                    max_shapes=self.config.get("viz_max_shapes", 400),
                    save_dir=gaus_pos_path_test,
                    base_name=vis_base,
                    structure_error=float(density_error),
                    dpi=180,
                )
        return loss

    def _check_electron_mismatch(
        self,
        *,
        split: str,
        filename: str,
        struc: Atoms,
        ne_true_grid: float,
        n_multiples: torch.Tensor,
    ):
        ne_val = float(n_multiples.sum().item()) / float(self.gaus_per_electrons)
        delta = ne_val - float(ne_true_grid)
        tol = 1e-3  # grid sums are exact floats
        if abs(delta) > tol:
            msg = (
                f"[{split.upper()}] ELECTRON SUM MISMATCH | "
                f"valence_sum={ne_val:.6f}, grid_sum={float(ne_true_grid):.6f}, Δ={delta:.6f} | "
                f"file='{filename}' | formula={struc.get_chemical_formula()}"
            )
            print(msg)
            logger.warning(msg)

    def _maybe_report_outlier(
        self,
        split: str,
        err_scalar: float,
        filename: str,
        formula: str,
        *,
        struc: Atoms | None = None,
        grid_dict: dict | None = None,
        forward_out: dict | None = None,
        dv: float | None = None,
        ne_true_grid: float | None = None,
    ):
        """
        err_scalar: NMAE as scalar (e.g., 0.083 == 8.3%).
        Uses running mean BEFORE including current example.
        If it's an outlier (>2× current mean), prints detailed diagnostics:
          - natoms, cell volume
          - grid spacings (Å) along a,b,c and their minimum
          - σ percentiles (p05/p50/p95) from Gaussian covariances
          - compare σ_p05 to 0.7×min(grid_spacing)
          - total electrons from reference grid (sum(dens)*dv) if provided
          - total valence electrons = sum(n_multiples)/gaus_per_electrons
            and min/max per-atom n_multiples
        """
        sum_so_far = self._err_sum[split]
        n_so_far = self._err_count[split]

        def _format_floats(xs):
            return ", ".join(f"{float(x):.5g}" for x in xs)

        is_outlier = False
        running_mean = None
        if n_so_far > 0:
            running_mean = sum_so_far / n_so_far
            is_outlier = err_scalar > 2.0 * running_mean

        # Always update after check
        self._err_sum[split] += float(err_scalar)
        self._err_count[split] += 1

        if not is_outlier:
            return

        # --- Base header ---
        hdr = (
            f"[{split.upper()}] NMAE outlier: {err_scalar:.6f} "
            f"(> 2× {running_mean:.6f}) | file='{filename}' | formula={formula}"
        )
        print(hdr)
        logger.warning(hdr)

        # --- Safely gather optional diagnostics ---
        try:
            natoms = len(struc) if struc is not None else None
            vol = float(struc.get_volume()) if struc is not None else None

            # grid spacings along lattice vectors (Å)
            s_a = s_b = s_c = s_min = None
            if struc is not None and grid_dict is not None:
                nx, ny, nz = grid_dict["nx"], grid_dict["ny"], grid_dict["nz"]
                a_len = float(np.linalg.norm(np.asarray(struc.cell[0])))
                b_len = float(np.linalg.norm(np.asarray(struc.cell[1])))
                c_len = float(np.linalg.norm(np.asarray(struc.cell[2])))
                s_a, s_b, s_c = a_len / nx, b_len / ny, c_len / nz
                s_min = min(s_a, s_b, s_c)

            # sigma percentiles from covariances
            p05 = p50 = p95 = None
            alias_flag = None
            if forward_out is not None and ("cov" in forward_out):
                Sigma = forward_out["cov"]  # (M,3,3)
                stats, _, _ = self._cov_eig_stats(Sigma)  # uses sqrt(eigs) for σ
                p05 = float(stats["sigma_p5"])
                p50 = float(stats["sigma_p50"])
                p95 = float(stats["sigma_p95"])
                if s_min is not None:
                    alias_flag = p05 < 0.7 * s_min

            # electrons (reference vs valence)
            ne_grid = ne_true_grid  # already = sum(density)*dv if passed
            ne_val = None
            nm_sum = nm_min = nm_max = None
            if forward_out is not None and ("n_multiples" in forward_out):
                nmult = forward_out["n_multiples"].detach().to("cpu")
                nm_sum = int(nmult.sum().item())
                nm_min = int(nmult.min().item()) if nmult.numel() else None
                nm_max = int(nmult.max().item()) if nmult.numel() else None
                ne_val = (
                    nm_sum / float(self.gaus_per_electrons)
                    if self.gaus_per_electrons
                    else None
                )

            # --- Compose diagnostics text ---
            lines = []
            if natoms is not None and vol is not None:
                lines.append(f"  natoms={natoms}, volume={vol:.5g} Å³")
            if s_a is not None:
                lines.append(
                    f"  grid_spacing (Å): a={s_a:.5g}, b={s_b:.5g}, c={s_c:.5g} | min={s_min:.5g}"
                )
            if p05 is not None:
                lines.append(
                    f"  σ percentiles (Å): p05={p05:.5g}, p50={p50:.5g}, p95={p95:.5g}"
                )
            if (p05 is not None) and (s_min is not None):
                flag_txt = "RISK" if alias_flag else "OK"
                lines.append(
                    f"  σ_p05 {'<' if alias_flag else '>='} 0.7×min(Δx): "
                    f"{p05:.5g} {'<' if alias_flag else '>='} {0.7 * s_min:.5g}  → {flag_txt}"
                )
            if ne_grid is not None:
                lines.append(f"  electrons (reference grid, ∑ρ·dv) = {ne_grid:.6g}")
            if ne_val is not None:
                lines.append(
                    f"  electrons (valence sum) = {ne_val:.6g}  "
                    f"[n_multiples: sum={nm_sum}, per-atom min={nm_min}, max={nm_max}]"
                )

            if lines:
                diag = "\n".join(lines)
                print(diag)
                logger.warning(diag)

                # visualize & log this outlier
                try:
                    # only proceed if we have what we need for plotting
                    if (
                        forward_out is not None
                        and all(k in forward_out for k in ("mu", "weights", "cov"))
                        and (struc is not None)
                    ):
                        # Decide output dir: <gaus_pos_outlier_path>/<split>/
                        base_out_dir = self.config.get("gaus_pos_outlier_path", None)
                        if base_out_dir is not None:
                            out_dir = os.path.join(base_out_dir, split.lower())
                            self._ensure_dir(out_dir)

                            # basename: formula + short file id + error
                            try:
                                file_id = os.path.splitext(os.path.basename(filename))[
                                    0
                                ]
                            except Exception:
                                file_id = (
                                    str(filename).replace("/", "_").replace("\\", "_")
                                )

                            base_name = f"{file_id}_{formula}_err{err_scalar:.4f}"

                            iso_sigma = self.config.get("viz_iso_sigma", 0.01)
                            mesh_nu = self.config.get("viz_mesh_nu", 20)
                            mesh_nv = self.config.get("viz_mesh_nv", 12)
                            alpha = self.config.get("viz_alpha", 0.5)
                            max_shapes = self.config.get("viz_max_shapes", 400)

                            ax, plot_path = plot_atoms_with_gaussian_shapes(
                                struc,
                                mu=forward_out["mu"],
                                weights=forward_out["weights"],
                                Sigma=forward_out["cov"],
                                mode="ellipsoids",
                                iso_sigma=iso_sigma,
                                mesh_nu=mesh_nu,
                                mesh_nv=mesh_nv,
                                alpha=alpha,
                                max_shapes=max_shapes,
                                save_dir=out_dir,
                                base_name=base_name,
                                structure_error=float(err_scalar),
                                dpi=180,
                            )

                            # Append CSV record next to the plots
                            csv_path = os.path.join(out_dir, "outliers.csv")
                            self._append_outlier_csv(
                                csv_path,
                                [(str(filename), float(err_scalar), str(formula))],
                            )

                            logger.info(
                                f"[{split.upper()}] Outlier plot saved: {plot_path}"
                            )
                            logger.info(
                                f"[{split.upper()}] Outlier CSV updated: {csv_path}"
                            )
                        else:
                            logger.warning(
                                "gaus_pos_outlier_path not set; skipping outlier plot/CSV."
                            )
                    else:
                        logger.warning(
                            "Forward output missing keys or structure is None; skipping outlier plot/CSV."
                        )
                except Exception as viz_e:
                    logger.exception(
                        f"Failed to generate/save outlier visualization/CSV: {viz_e}"
                    )
        except Exception as e:
            logger.exception(f"Failed to print diagnostics for outlier: {e}")

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _append_outlier_csv(self, csv_path: str, rows: list[tuple[str, float, str]]):
        header = ["filename", "error", "formula"]
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if not file_exists:
                w.writerow(header)
            for r in rows:
                w.writerow(r)

    @torch.no_grad()
    def _cov_eig_stats(self, Sigma: torch.Tensor):
        """
        Sigma: (M, 3, 3) symmetric PD covariances (Å^2).
        Returns a dict of scalar stats + percentiles (on sqrt-eigs = length scales).
        """
        # Ensure symmetric (tiny safety) then eigs
        Ssym = 0.5 * (Sigma + Sigma.transpose(-1, -2))
        eigs = torch.linalg.eigvalsh(Ssym)  # (M, 3)
        eigs = eigs.clamp_min(1e-12)  # numerical floor

        # Also look at length scales (std devs)
        sigmas = torch.sqrt(eigs)  # (M, 3)

        # Flatten across all Gaussians & axes
        e = eigs.reshape(-1)  # Å^2
        s = sigmas.reshape(-1)  # Å

        # Basic stats
        stats = {
            "eig_min": e.min().item(),
            "eig_mean": e.mean().item(),
            "eig_max": e.max().item(),
            # anisotropy via log condition number (robust):
            "log_cond_eig": (eigs.max(dim=-1).values / eigs.min(dim=-1).values)
            .log()
            .mean()
            .item(),
            # geometric mean of eigs (volume scale):
            "eig_geom_mean": torch.exp(torch.log(e + 1e-24).mean()).item(),
            # same but in σ (length scale) space:
            "sigma_min": s.min().item(),
            "sigma_mean": s.mean().item(),
            "sigma_max": s.max().item(),
        }

        # Percentiles (useful to catch tails)
        q = torch.tensor([0.01, 0.05, 0.5, 0.95, 0.99], device=s.device)
        qs = torch.quantile(s, q)
        for p, val in zip([1, 5, 50, 95, 99], qs.tolist()):
            stats[f"sigma_p{p}"] = val

        return stats, eigs, sigmas

    def _log_split_metric(
        self, split: str, name: str, value, *, prog_bar: bool = False
    ):
        # Unified {split}/{name} scalar logging; accepts tensors or python scalars.
        if torch.is_tensor(value):
            value = value.detach().float().item()
        self.log(
            f"{split}/{name}",
            float(value),
            on_step=True,
            on_epoch=True,
            prog_bar=prog_bar,
            logger=True,
            batch_size=1,
        )

    def _log_cov_stats(
        self, Sigma: torch.Tensor, split: str, step_idx: int | None = None
    ):
        """
        Logs min/mean/max eigen-related stats to Lightning + W&B.
        `split` is "train"/"val"/"test".
        """
        stats, eigs, sigmas = self._cov_eig_stats(Sigma)

        # Lightning logs
        self.log(
            f"{split}/eig_min",
            stats["eig_min"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/eig_mean",
            stats["eig_mean"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/eig_max",
            stats["eig_max"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/log_cond_eig",
            stats["log_cond_eig"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_min",
            stats["sigma_min"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_mean",
            stats["sigma_mean"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_max",
            stats["sigma_max"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_p01",
            stats["sigma_p1"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_p05",
            stats["sigma_p5"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_p50",
            stats["sigma_p50"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_p95",
            stats["sigma_p95"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            f"{split}/sigma_p99",
            stats["sigma_p99"],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )

    def _set_dc(self, cG: torch.Tensor, target: torch.Tensor | float) -> torch.Tensor:
        """Return a copy of cG with DC component set to target; no in-place ops."""
        mask = torch.zeros_like(cG)
        # mask is a fresh leaf tensor with no grad; safe to write in-place
        mask[0, 0, 0] = 1.0
        target_t = torch.as_tensor(target, dtype=cG.dtype, device=cG.device)
        return cG * (1.0 - mask) + target_t * mask

    def _zero_dc(self, cG: torch.Tensor) -> torch.Tensor:
        """Return cG with DC zeroed; no in-place ops."""
        mask = torch.zeros_like(cG)
        mask[0, 0, 0] = 1.0
        return cG * (1.0 - mask)

    # in ELECTRAFI
    def _valence_from_slots(self, atoms, species_ids: torch.Tensor) -> torch.Tensor:
        MAX_Z = int(
            self.config.get(
                "max_num_elements",
                getattr(self.backbone.molecular_graph_cfg, "max_num_elements", 119),
            )
        )
        Z = torch.tensor(atoms.numbers, dtype=torch.long, device=self.device)
        slots = ((species_ids - Z) // MAX_Z).to(torch.long)  # integer slot id
        syms = [chemical_symbols[int(z)] for z in Z.tolist()]

        vals = []
        for s, sl in zip(syms, slots.tolist()):
            opts = self.VALENCE_OPTIONS.get(s, [self.valence_dict.get(s, 0)])
            if sl < 0 or sl >= len(opts):
                v = opts[0]  # safe fallback
            else:
                v = opts[sl]
            vals.append(int(v))
        return torch.tensor(vals, dtype=torch.long, device=self.device)

    # ---------- utils ----------
    def atoms_to_pyg(
        self,
        atoms,
        device=torch.device("cpu"),
        slots_per_atom: torch.Tensor | None = None,
    ):
        natoms = len(atoms)
        Z = torch.tensor(atoms.numbers, dtype=torch.long, device=device)  # (N,)
        MAX_Z = int(
            self.config.get(
                "max_num_elements",
                getattr(self.backbone.molecular_graph_cfg, "max_num_elements", 119),
            )
        )
        if slots_per_atom is None:
            slots = torch.zeros_like(Z)
        else:
            slots = slots_per_atom.to(device=device, dtype=torch.long)

        species_ids = Z + slots * MAX_Z  # (N,)

        data = Data(
            pos=torch.tensor(
                atoms.positions, dtype=torch.float32, device=device
            ).contiguous(),
            atomic_numbers=Z,
            species_ids=species_ids,  # NEW
            cell=torch.tensor(atoms.cell.array, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .contiguous(),
            pbc=torch.tensor(atoms.pbc, dtype=torch.bool, device=device).unsqueeze(0),
            batch=torch.zeros(natoms, dtype=torch.long, device=device),
            natoms=torch.tensor([natoms], dtype=torch.long, device=device),
            num_nodes=natoms,
            num_graphs=1,
        )
        return Batch.from_data_list([data])

    def slice(self, tensor, n_multiples):
        return torch.cat(
            [
                tensor[i, torch.arange(tensor.size(1))[:num]]
                for i, num in enumerate(n_multiples)
            ]
        )

    @torch.no_grad()
    def _sample_modes(self, frac: float) -> torch.Tensor:
        g = self.pw.Gnorm.reshape(-1)
        N = g.numel()
        M = max(1, int(frac * N))
        if self.band_sample == "stratified":
            # Split |G| into K quantile bands, take ~M/K from each
            K = 6
            qs = torch.quantile(g, torch.linspace(0, 1, K + 1, device=g.device))
            take = []
            per = max(1, M // K)
            for k in range(K):
                mask = (g >= qs[k]) & (g <= qs[k + 1])
                idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                if idx.numel() > per:
                    # prefer lower |G| early
                    choose = idx[torch.randperm(idx.numel(), device=g.device)[:per]]
                else:
                    choose = idx
                take.append(choose)
            sel = torch.cat(take)
            if sel.numel() > M:
                sel = sel[:M]
        else:
            sel = torch.randperm(N, device=g.device)[:M]
        return sel

    def _maybe_sample_points(self, grid_dict: dict, frac_or_n) -> torch.Tensor | None:
        """
        Accepts either a fraction in (0,1] or an integer count (>1).
        Returns None if falsy/zero.
        """
        if not frac_or_n:
            return None
        N = grid_dict["nx"] * grid_dict["ny"] * grid_dict["nz"]
        if isinstance(frac_or_n, float) and 0 < frac_or_n <= 1.0:
            k = max(1, int(round(frac_or_n * N)))
        else:
            k = min(int(frac_or_n), N)
        return torch.randint(high=N, size=(k,), device=self.device)

    # loaders
    def train_dataloader(self):
        ds = DensityStream(
            self.train_files,
            self.dens_path,
            self.config,
            None,
            shuffle_each_epoch=False,
        )
        return DataLoader(
            ds,
            batch_size=1,
            collate_fn=collate_fn,
            num_workers=max(4, os.cpu_count() // 2),
            persistent_workers=True,
            prefetch_factor=8,
            pin_memory=True,
            worker_init_fn=_dl_worker_init,
            multiprocessing_context="spawn",
        )

    def val_dataloader(self):
        return DataLoader(
            DensityDataset(self.validation_files, self.dens_path, self.config, None),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            DensityDataset(self.test_files, self.dens_path, self.config, None),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
        )

    def ood_dataloader(self, files, path, name):
        return DataLoader(
            DensityDataset(files, path, self.config, name),
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=0,
        )

    def on_fit_start(self):
        if self.config["debug"]:
            install_nan_hooks(self.backbone)  # catch inside the GNN
            install_nan_hooks(self.wsm_network)  # catch Mish MLP

    def on_after_backward(self):
        # Will raise immediately on the first offending parameter
        if self.config["debug"]:
            check_param_grads(self)


def install_nan_hooks(module):
    def _check(tag, t):
        if torch.is_tensor(t) and t.dtype.is_floating_point:
            if not torch.isfinite(t).all():
                raise RuntimeError(f"Non-finite detected in {tag}")

    def fwd_hook(m, inp, out):
        xs = inp if isinstance(inp, (tuple, list)) else (inp,)
        ys = out if isinstance(out, (tuple, list)) else (out,)
        for x in xs:
            _check(f"{m.__class__.__name__} input", x)
        for y in ys:
            _check(f"{m.__class__.__name__} output", y)

    def bwd_hook(m, grad_in, grad_out):
        for g in grad_in:
            if g is not None:
                _check(f"{m.__class__.__name__} grad_in", g)
        for g in grad_out:
            if g is not None:
                _check(f"{m.__class__.__name__} grad_out", g)

    for m in module.modules():
        m.register_forward_hook(fwd_hook)
        m.register_full_backward_hook(bwd_hook)


def check_param_grads(module):
    for name, p in module.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise RuntimeError(
                f"Non-finite grad in {name} (norm={p.grad.norm().item():.3e})"
            )


def _rotate_atoms(
    self, atoms: Atoms, R: np.ndarray, wrap: bool = True, center_mode: str = "origin"
):
    atoms_rot = atoms.copy()

    pos = np.asarray(atoms_rot.positions, dtype=np.float64)
    cell_old = np.asarray(atoms_rot.cell.array, dtype=np.float64)

    if center_mode == "com":
        center = np.asarray(atoms_rot.get_center_of_mass(), dtype=np.float64)
    elif center_mode == "cop":
        center = pos.mean(axis=0)
    else:
        center = np.zeros(3, dtype=np.float64)

    pos_rot = (pos - center) @ R.T + center
    cell_new = cell_old @ R.T

    atoms_rot.positions = pos_rot
    atoms_rot.set_cell(cell_new, scale_atoms=False)

    if wrap and np.any(atoms_rot.pbc):
        atoms_rot.wrap()

    meta = {
        "R": R,
        "center_cart": center,
        "cell_old": cell_old,
        "cell_new": cell_new,
        "center_mode": center_mode,
    }
    return atoms_rot, meta


import torch


def _frac_grid(nx: int, ny: int, nz: int, device, dtype):
    """
    Voxel centers in fractional coordinates, shape (nx, ny, nz, 3).
    Uses cell-attached grid. Choose endpoint convention consistently with your CHGCAR reading.
    """
    fx = torch.arange(nx, device=device, dtype=dtype) / nx
    fy = torch.arange(ny, device=device, dtype=dtype) / ny
    fz = torch.arange(nz, device=device, dtype=dtype) / nz
    X, Y, Z = torch.meshgrid(fx, fy, fz, indexing="ij")
    return torch.stack([X, Y, Z], dim=-1)  # (nx, ny, nz, 3)


def _cart_from_frac(f: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    # f: (..., 3), cell: (3, 3)
    return torch.matmul(f, cell)


def _frac_from_cart(x: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    # x: (..., 3), cell: (3, 3)
    return torch.linalg.solve(cell.transpose(-2, -1), x.transpose(-2, -1)).transpose(
        -2, -1
    )


def _wrap_frac01(f: torch.Tensor) -> torch.Tensor:
    return f - torch.floor(f)


def _periodic_trilinear_sample(
    rho: torch.Tensor, frac_coords: torch.Tensor
) -> torch.Tensor:
    """
    Periodic trilinear interpolation on a 3D density tensor.

    rho: (nx, ny, nz)
    frac_coords: (..., 3) in [0,1) fractional coordinates
    returns: (...) sampled values
    """
    device = rho.device
    dtype = rho.dtype
    nx, ny, nz = rho.shape

    f = _wrap_frac01(frac_coords)
    x = f[..., 0] * nx
    y = f[..., 1] * ny
    z = f[..., 2] * nz

    x0 = torch.floor(x).long() % nx
    y0 = torch.floor(y).long() % ny
    z0 = torch.floor(z).long() % nz

    x1 = (x0 + 1) % nx
    y1 = (y0 + 1) % ny
    z1 = (z0 + 1) % nz

    tx = (x - torch.floor(x)).to(dtype)
    ty = (y - torch.floor(y)).to(dtype)
    tz = (z - torch.floor(z)).to(dtype)

    c000 = rho[x0, y0, z0]
    c001 = rho[x0, y0, z1]
    c010 = rho[x0, y1, z0]
    c011 = rho[x0, y1, z1]
    c100 = rho[x1, y0, z0]
    c101 = rho[x1, y0, z1]
    c110 = rho[x1, y1, z0]
    c111 = rho[x1, y1, z1]

    c00 = c000 * (1 - tz) + c001 * tz
    c01 = c010 * (1 - tz) + c011 * tz
    c10 = c100 * (1 - tz) + c101 * tz
    c11 = c110 * (1 - tz) + c111 * tz

    c0 = c00 * (1 - ty) + c01 * ty
    c1 = c10 * (1 - ty) + c11 * ty

    c = c0 * (1 - tx) + c1 * tx
    return c


def rigidly_resample_density_to_new_cell(
    rho_old: torch.Tensor,
    cell_old: torch.Tensor,
    cell_new: torch.Tensor,
    R: torch.Tensor,
    center_cart: torch.Tensor,
) -> torch.Tensor:
    """
    Build rho_new on the new cell grid so that it is the old density transformed by:
        x' = (x - c) R^T + c

    We sample the old density at:
        x = c + (x' - c) R

    Inputs:
      rho_old: (nx, ny, nz)
      cell_old: (3, 3)
      cell_new: (3, 3)
      R: (3, 3)
      center_cart: (3,)
    Returns:
      rho_new: (nx, ny, nz)
    """
    device = rho_old.device
    dtype = rho_old.dtype
    nx, ny, nz = rho_old.shape

    frac_new = _frac_grid(nx, ny, nz, device=device, dtype=dtype)  # (nx,ny,nz,3)
    x_new = _cart_from_frac(frac_new, cell_new)  # Cartesian points in new grid

    c = center_cart.to(device=device, dtype=dtype).view(1, 1, 1, 3)
    R = R.to(device=device, dtype=dtype)

    # inverse rigid map: x_old = c + (x_new - c) @ R
    x_old = c + torch.matmul(x_new - c, R)

    frac_old = _frac_from_cart(x_old, cell_old)
    rho_new = _periodic_trilinear_sample(rho_old, frac_old)

    return rho_new
