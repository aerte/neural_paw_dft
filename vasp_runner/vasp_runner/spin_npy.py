"""Loader for the ELECTRAFI spin-density (``diff``) grid predictions.

Layout (verified 2026-08-28 against the converged references):

    <pred_dir>/<id>_spin.npy.lz4

  * ``<id>`` is the task id verbatim (``mp-1002574`` for MP, the 10-hex gid
    for GNoME); one file per structure, nothing else in the directory.
  * Each file is an lz4-frame-compressed ``.npy`` holding a float32 array of
    shape ``(NGX, NGY, NGZ)`` -- the SAME index order as pymatgen's
    ``Chgcar.data[...]`` (no transpose needed).
  * Values are the spin density ``rho_up - rho_down`` in e/A^3, i.e. PHYSICAL
    units, NOT the ``rho * V`` convention of the CHGCAR grid; multiply by the
    cell volume to obtain a CHGCAR ``diff`` grid (``grid.sum()/grid.size`` is
    then the total moment in mu_B).

MP:    <spin_eval_grids>/mp_csv/     (1943 mpids)
GNoME: <spin_eval_grids>/gnome_csv/  (1904 gids)
"""
import io
import os

import lz4.frame
import numpy as np

SUFFIX = "_spin.npy.lz4"


def index_spin_dir(pred_dir: str) -> dict:
    """Map ``<id>`` -> absolute path for every ``<id>_spin.npy.lz4`` in ``pred_dir``."""
    out = {}
    for name in os.listdir(pred_dir):
        if name.endswith(SUFFIX):
            out[name[: -len(SUFFIX)]] = os.path.join(pred_dir, name)
    return out


def load_spin_grid(path: str) -> np.ndarray:
    """Decompress and load one ``*_spin.npy.lz4`` -> float32 ``(NGX, NGY, NGZ)`` in e/A^3."""
    with open(path, "rb") as fh:
        raw = lz4.frame.decompress(fh.read())
    arr = np.load(io.BytesIO(raw))
    if arr.ndim != 3:
        raise RuntimeError(f"{path}: expected a 3D grid, got shape {arr.shape}")
    return arr


def spin_grid_as_chgcar_diff(spin_grid: np.ndarray, ref_chg) -> np.ndarray:
    """Convert an e/A^3 spin grid into a CHGCAR-convention ``diff`` grid for ``ref_chg``.

    Checks the FFT shape against the reference's ``total`` grid (the prediction
    must live on the reference's grid to be spliced into it) and scales by the
    reference cell volume so the result matches ``ref_chg.data['diff']``'s units.
    """
    ref_shape = ref_chg.data["total"].shape
    if spin_grid.shape != ref_shape:
        raise RuntimeError(
            f"Spin grid shape {spin_grid.shape} != reference grid {ref_shape}"
        )
    return spin_grid.astype(np.float64) * ref_chg.structure.volume
