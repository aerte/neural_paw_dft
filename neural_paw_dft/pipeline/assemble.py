"""Write the VASP input directory: CHGCAR from ML pieces, INCAR/POSCAR/POTCAR/KPOINTS via MPStaticSet."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Chgcar

from neural_paw_dft.augnet.augnet_model import Z_TO_SCHEMA
from neural_paw_dft.vasp_runner.chgcar import (
    dims_line,
    rewrite_aug_headers,
    splice_spin_moment_line,
    get_chgcar_grid_dims_textparse,
)
from neural_paw_dft.vasp_runner.relaxmag import apply_relaxed_magmom

from .config import VaspConfig

_PARALLEL_TAGS = ("NPAR", "NCORE", "KPAR", "NSIM")  # site-specific; never inherit them


def synth_moment_lines(site_moments: np.ndarray | None, n_ions: int) -> list[str]:
    """The per-ion moment block VASP needs between the total augmentation and the spin grid
    (ceil(n_ions/5) lines, 5 values each). VASP parses it list-directed and does not use the
    values as initial moments, so CHGNet moments or zeros are both fine."""
    vals = np.zeros(n_ions) if site_moments is None else np.asarray(site_moments, dtype=float).reshape(-1)
    if vals.size != n_ions:
        raise ValueError(f"{vals.size} site moments for {n_ions} ions")
    lines = []
    for i in range(0, n_ions, 5):
        lines.append("".join(f"{v:18.11E}" for v in vals[i : i + 5]))
    return lines


def build_chgcar(
    structure: Structure,
    rho_total: np.ndarray,
    rho_spin: np.ndarray | None,
    aug_total: dict[int, np.ndarray],
    aug_diff: dict[int, np.ndarray] | None,
    site_moments: np.ndarray | None,
    out_path: str | os.PathLike,
    nelect: float | None = None,
    renorm_tol: float = 1e-3,
) -> Path:
    """CHGCAR from scratch. Grids are densities in e/Å^3 (CHGCAR stores rho*V_cell);
    ``aug_*`` map 1-based ion index -> occupancy block in CHGCAR order."""
    out_path = Path(out_path)
    n = len(structure)
    zs = [site.specie.Z for site in structure]
    if sorted(aug_total) != list(range(1, n + 1)):
        raise ValueError("aug_total must have one block per ion, keyed 1..n_ions")
    for i, z in enumerate(zs, start=1):
        if aug_total[i].size != Z_TO_SCHEMA[z]:
            raise ValueError(f"ion {i} (Z={z}): augmentation block has {aug_total[i].size} values, POTCAR expects {Z_TO_SCHEMA[z]}")
    rho_total = np.asarray(rho_total, dtype=np.float64)
    if rho_spin is not None and np.shape(rho_spin) != rho_total.shape:
        raise ValueError(f"spin grid {np.shape(rho_spin)} != total grid {rho_total.shape}")

    volume = float(structure.volume)
    total = rho_total * volume
    if nelect is not None:
        integ = float(total.sum() / total.size)  # sum(rho*V)/N = integral of rho
        if integ > 0 and abs(integ - nelect) / nelect > renorm_tol:
            print(f"[ndi] rescaling total grid: integrates to {integ:.4f} e, NELECT={nelect:.4f}")
            total *= nelect / integ

    data = {"total": total}
    data_aug = {"total": {k: np.asarray(v, dtype=float) for k, v in aug_total.items()}}
    spin = rho_spin is not None
    if spin:
        data["diff"] = np.asarray(rho_spin, dtype=np.float64) * volume
        if aug_diff is None:
            aug_diff = {k: np.zeros_like(v) for k, v in data_aug["total"].items()}
        data_aug["diff"] = {k: np.asarray(v, dtype=float) for k, v in aug_diff.items()}

    chg = Chgcar(Poscar(structure), data, data_aug=data_aug)
    chg.is_spin_polarized = spin  # pymatgen fixes this at construction; make it explicit
    chg.write_file(str(out_path))
    rewrite_aug_headers(str(out_path))  # VASP's fixed-width header, or it silently ignores the file
    if spin:
        ok = splice_spin_moment_line(str(out_path), dims_line(chg), synth_moment_lines(site_moments, n))
        if not ok:
            raise RuntimeError(f"could not locate the spin grid header in {out_path}")
    return out_path


def _potcars_available(vasp: VaspConfig) -> bool:
    if vasp.potcar_dir:
        os.environ["PMG_VASP_PSP_DIR"] = str(Path(vasp.potcar_dir).expanduser())
        return True
    if os.environ.get("PMG_VASP_PSP_DIR"):
        return True
    from pymatgen.core import SETTINGS

    return bool(SETTINGS.get("PMG_VASP_PSP_DIR"))


def write_vasp_inputs(
    structure: Structure,
    out_dir: str | os.PathLike,
    vasp: VaspConfig,
    magmoms: np.ndarray | None,
    spin_polarized: bool,
) -> Incar:
    """INCAR/POSCAR/POTCAR/KPOINTS for an ICHARG=1 static run from pymatgen's MPStaticSet.

    Site order is kept (sort_structure=False) so POSCAR, MAGMOM and the CHGCAR agree.
    Without a POTCAR directory a POTCAR.spec is written instead of POTCAR.
    """
    from pymatgen.io.vasp.sets import MPStaticSet

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    user = {"ICHARG": 1, "ISTART": 0, "LCHARG": True, "LWAVE": False, "ISPIN": 2 if spin_polarized else 1}
    user.update(vasp.incar_overrides)
    vis = MPStaticSet(
        structure,
        user_incar_settings=user,
        user_potcar_functional=vasp.potcar_functional,
        reciprocal_density=vasp.kpoints_reciprocal_density,
        sort_structure=False,
    )
    vis.write_input(str(out_dir), potcar_spec=not _potcars_available(vasp))

    incar = Incar.from_file(str(out_dir / "INCAR"))
    for tag in _PARALLEL_TAGS:
        if tag not in vasp.incar_overrides:
            incar.pop(tag, None)
    if spin_polarized:
        if magmoms is not None and "MAGMOM" not in vasp.incar_overrides:
            apply_relaxed_magmom(incar, [float(m) for m in magmoms])
    else:
        incar.pop("MAGMOM", None)
        incar.pop("NUPDOWN", None)
    incar.update(vasp.incar_overrides)
    incar.write_file(str(out_dir / "INCAR"))
    return incar


def dry_run_grid_dims(structure: Structure, vasp: VaspConfig, work_dir: str | os.PathLike) -> tuple[int, int, int]:
    """One-step (NELM=1, Gamma-only) VASP run to read NGXF/NGYF/NGZF from the CHGCAR it writes.
    Mirrors vasp_runner.sad_extract; uses the same ENCUT/PREC as the final inputs."""
    if not vasp.vasp_cmd:
        raise ValueError("vasp.vasp_cmd is not configured; cannot dry-run for grid dims")
    work_dir = Path(work_dir)
    write_vasp_inputs(structure, work_dir, vasp, magmoms=None, spin_polarized=False)
    if not (work_dir / "POTCAR").is_file():
        raise FileNotFoundError("dry run needs real POTCARs; set vasp.potcar_dir or PMG_VASP_PSP_DIR")
    Kpoints.gamma_automatic((1, 1, 1)).write_file(str(work_dir / "KPOINTS"))
    incar = Incar.from_file(str(work_dir / "INCAR"))
    incar.update({"ICHARG": 2, "ISTART": 0, "NELM": 1, "LCHARG": True, "LWAVE": False, "ISMEAR": 0, "SIGMA": 0.05})
    incar.write_file(str(work_dir / "INCAR"))
    with open(work_dir / "vasp.out", "w") as log:
        subprocess.run(list(vasp.vasp_cmd), cwd=str(work_dir), check=False, stdout=log, stderr=subprocess.STDOUT)
    chgcar = work_dir / "CHGCAR"
    if not chgcar.is_file() or chgcar.stat().st_size == 0:
        raise RuntimeError(f"dry run produced no CHGCAR in {work_dir}; see vasp.out")
    return get_chgcar_grid_dims_textparse(str(chgcar))


def grid_dims_from_incar(incar: Incar) -> tuple[int, int, int] | None:
    """NGXF/NGYF/NGZF if the INCAR sets them explicitly."""
    if all(k in incar for k in ("NGXF", "NGYF", "NGZF")):
        return int(incar["NGXF"]), int(incar["NGYF"]), int(incar["NGZF"])
    return None

