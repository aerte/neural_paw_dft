#!/usr/bin/env python
from __future__ import division, print_function

import os
import tempfile

import lz4
import lz4.frame
import numpy as np
import torch
from ase import Atoms
from ase.calculators.vasp import VaspChargeDensity


def normalize_density(
    n_elec: int,
    density: torch.tensor,
    true_density: torch.tensor,
    grid_dict: dict,
    sys: Atoms,
    points: torch.tensor,
    volume: float = None,
):
    if points is not None:
        sum_target = true_density.flatten().index_select(0, points).sum()
        sum_pred = density.flatten().sum()
        factor = sum_target / sum_pred
        cd_sum_to_num_electrons = density * factor
    else:
        # n_elec = true_density.flatten()[points].sum() * dv
        sum_target = true_density.flatten().sum()
        sum_pred = density.flatten().sum()
        factor = sum_target / sum_pred
        cd_sum_to_num_electrons = density * factor
    # integrated_num_electrons = density.sum() * dv
    # factor = n_elec / integrated_num_electrons
    # cd_sum_to_num_electrons = density * factor
    return cd_sum_to_num_electrons, n_elec


def check_number_of_electrons(
    sys: Atoms, cd: np.array, gridrefinement: int = 2, grid_dict: dict = None
):
    # Get integrated values

    if grid_dict:
        dv = sys.get_volume() / (grid_dict["nx"] * grid_dict["ny"] * grid_dict["nz"])
        integrated_num_electrons = cd.sum() * dv
    else:
        dv = sys.get_volume() / (grid_dict["nx"] * grid_dict["ny"] * grid_dict["nz"])
        integrated_num_electrons = cd.sum() * dv / gridrefinement**3
    # print(f"Number of electrons in {sys.get_chemical_formula()} is {np.round(integrated_num_electrons, 2)}")
    return integrated_num_electrons


def chgcar_to_cd_old(chgcar_file):
    with lz4.frame.open(chgcar_file, mode="rb") as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    with os.fdopen(tmpfd, "wb") as tmpfile:
        tmpfile.write(filecontent)
    vcd = VaspChargeDensity(tmppath)
    os.remove(tmppath)
    charge_density = vcd.chg
    try:
        atoms = vcd.atoms[0]
    except:
        print(f"No atoms in chgcar file {chgcar_file}")
    cd = torch.FloatTensor(charge_density).squeeze(axis=0)
    return cd, atoms


def chgcar_to_cd(chgcar_file):
    # Load (lz4 -> temp -> ASE reader)
    with lz4.frame.open(chgcar_file, mode="rb") as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    with os.fdopen(tmpfd, "wb") as tmpfile:
        tmpfile.write(filecontent)

    vcd = VaspChargeDensity(tmppath)
    os.remove(tmppath)

    # --- normalize atoms ---
    try:
        atoms = vcd.atoms[-1]  # last ionic step if multiple
    except Exception:
        atoms = vcd.atoms[0]

    chg = vcd.chg  # could be list / ndarray (3D/4D/flat)
    chgdiff = getattr(vcd, "chgdiff", None)  # magnetization if present

    # Helper: ensure we end up with a (nx, ny, nz) total charge density
    def to_3d_total(x):
        """
        Return a (nx, ny, nz) *total* charge grid.

        Rules:
        - If x is 3D: return as-is (assumed total).
        - If x is 4D and first dim == 2: treat as two spin channels -> sum [0] + [1].
          Otherwise treat first dim as steps and take the last step.
        - If x is flat:
            * If size == N -> reshape (Fortran order).
            * If size == 2N -> interpret as two concatenated grids -> sum halves, reshape (Fortran).
        """
        x = np.asarray(x)

        if x.ndim == 3:
            return x

        if x.ndim == 4:
            if x.shape[0] == 2:
                # Two spin channels explicitly -> sum
                return x[0] + x[1]
            # Otherwise assume leading dim is steps; take last step
            return x[-1]

        if x.ndim == 1:
            # Prefer ngridpts if present
            ngridpts = getattr(vcd, "ngridpts", None)
            if ngridpts is None:
                raise ValueError(
                    "Flat charge array but missing ngridpts; cannot infer shape."
                )
            nx, ny, nz = map(int, ngridpts)
            N = nx * ny * nz
            flat = x.ravel()
            if flat.size == N:
                return flat.reshape(nx, ny, nz, order="F")
            if flat.size == 2 * N:
                # Two grids concatenated (e.g., up and down) -> sum
                return (flat[:N] + flat[N:]).reshape(nx, ny, nz, order="F")
            raise ValueError(
                f"Cannot infer 3D shape from flat array of size {flat.size} (N={N})."
            )

        raise ValueError(f"Unsupported charge array shape {x.shape}.")

    def conversion(input, vcd_input):
        # Prefer using ASE semantics where:
        #   vcd.chg      == total charge density
        #   vcd.chgdiff  == magnetization density (if spin-polarized)
        try:
            chg_out = to_3d_total(input[-1] if isinstance(input, list) else input)
            return chg_out
        except Exception:
            # Fallback: handle flat buffers robustly (sum two halves if present)
            ngridpts = getattr(vcd_input, "ngridpts", None)
            if ngridpts is None:
                raise
            nx, ny, nz = map(int, ngridpts)
            N = nx * ny * nz
            flat = np.asarray(input).ravel()
            if flat.size == 2 * N:
                chg_out = (flat[:N] + flat[N:]).reshape(nx, ny, nz, order="F")
            elif flat.size == N:
                chg_out = flat.reshape(nx, ny, nz, order="F")
            else:
                raise ValueError(
                    f"Unexpected flat buffer size {flat.size}; expected N={N} or 2N={2 * N}."
                )
            return chg_out

    chg_out = conversion(chg, vcd)
    cd = torch.as_tensor(chg_out, dtype=torch.float32)

    if chgdiff:
        chgdiff_out = conversion(chgdiff, vcd)
        cd_diff = torch.as_tensor(chgdiff_out, dtype=torch.float32)
    else:
        # No spin difference in the CHGCAR (non-spin-polarized): fall back to a zero
        # grid matching the total density. The spin loss is gated on the target's
        # absolute-sum downstream, so this placeholder is never actually trained on.
        cd_diff = torch.zeros_like(cd)

    return cd, atoms, cd_diff


def cd_to_chgcar_mat(original_file: str, atoms: Atoms, cd: np.array, filename: str):

    with lz4.frame.open(original_file, mode="rb") as fp:
        filecontent = fp.read()
    tmpfd, tmppath = tempfile.mkstemp(prefix="tmpchgcar")
    with os.fdopen(tmpfd, "wb") as tmpfile:
        tmpfile.write(filecontent)

    vcd = VaspChargeDensity(tmppath)
    os.remove(tmppath)

    # Overwrite with ONLY total charge density for the last step
    vcd.chg = [cd]
    if hasattr(vcd, "chgdiff"):
        vcd.chgdiff = []  # drop magnetization for a pure-total file

    vcd.atoms = [atoms]
    vcd.write(filename, format="chgcar")
    return cd, atoms


def get_density(path):
    density, struc, cd_diff = chgcar_to_cd(path)
    grid_dict = {"nx": density.shape[0], "ny": density.shape[1], "nz": density.shape[2]}
    n_elec = check_number_of_electrons(struc, density, grid_dict=grid_dict)
    return density, struc, n_elec, grid_dict, cd_diff
