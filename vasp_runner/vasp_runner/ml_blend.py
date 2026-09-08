"""Build an ML-initialized CHGCAR by replacing total density while preserving
the true magnetization (ISPIN=2). Lifted from the legacy charge3net runners (not included).
"""
import numpy as np


def _get_chgcar_dim(chg):
    """Robust (nx, ny, nz) for a Chgcar across pymatgen versions."""
    dim = getattr(chg, "dim", None)
    if dim is not None:
        dim = tuple(dim)
        if len(dim) == 3:
            return dim

    data = chg.data
    if isinstance(data, dict) and len(data) > 0:
        arr = np.asarray(next(iter(data.values())))
    else:
        arr = np.asarray(data)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D grid, got shape {arr.shape}")
    return tuple(arr.shape)


def _extract_total_and_diff(chg):
    """Return (total, diff_or_None) as 3D numpy arrays."""
    from pymatgen.electronic_structure.core import Spin

    data = chg.data
    if isinstance(data, dict):
        if (Spin.up in data) and (Spin.down in data):
            up = np.asarray(data[Spin.up])
            dn = np.asarray(data[Spin.down])
            return up + dn, up - dn
        if "total" in data:
            total = np.asarray(data["total"])
            diff = np.asarray(data["diff"]) if "diff" in data else None
            return total, diff
        arr = np.asarray(next(iter(data.values())))
        return arr, None
    return np.asarray(data), None


def _rescale_ml_to_true_charge(ml_grid, true_chg):
    """Rescale ml_grid so its integrated charge matches true_chg."""
    from pymatgen.electronic_structure.core import Spin

    data = true_chg.data
    if isinstance(data, dict):
        if (Spin.up in data) and (Spin.down in data):
            true_tot = np.asarray(data[Spin.up]) + np.asarray(data[Spin.down])
        elif "total" in data:
            true_tot = np.asarray(data["total"])
        else:
            true_tot = np.asarray(next(iter(data.values())))
    else:
        true_tot = np.asarray(data)

    dim = true_tot.shape
    assert ml_grid.shape == dim, f"ML grid shape {ml_grid.shape} != template shape {dim}"

    cell_vol = true_chg.structure.lattice.volume
    npoints = int(dim[0] * dim[1] * dim[2])
    dv = cell_vol / npoints

    q_true = true_tot.sum() * dv
    q_ml = ml_grid.sum() * dv
    if q_ml == 0:
        raise RuntimeError("ML grid has zero integrated charge, cannot rescale")
    scale = q_true / q_ml
    return ml_grid * scale


def build_ml_initialized_chgcar(template_true_chgcar: str,
                                ml_chgcar_path: str,
                                out_path: str):
    """Replace the *total* density of `template_true_chgcar` with the ML
    prediction (rescaled to match the true integrated charge) while
    preserving the true magnetization. Writes a CHGCAR to `out_path`.
    """
    from pymatgen.io.vasp.outputs import Chgcar
    from pymatgen.electronic_structure.core import Spin

    true_chg = Chgcar.from_file(template_true_chgcar)
    dim = _get_chgcar_dim(true_chg)

    ml_chg = Chgcar.from_file(ml_chgcar_path)
    ml_tot, _ml_diff = _extract_total_and_diff(ml_chg)
    if ml_tot.shape != dim:
        raise RuntimeError(f"ML grid shape {ml_tot.shape} != template shape {dim}")

    ml_grid = _rescale_ml_to_true_charge(ml_tot, true_chg)

    data_obj = true_chg.data
    if isinstance(data_obj, dict):
        if (Spin.up in data_obj) and (Spin.down in data_obj):
            n_up_true = np.asarray(data_obj[Spin.up])
            n_dn_true = np.asarray(data_obj[Spin.down])
            if n_up_true.shape != dim or n_dn_true.shape != dim:
                raise RuntimeError(
                    f"Spin channels have shapes {n_up_true.shape}, {n_dn_true.shape}, "
                    f"but ML grid shape is {dim}"
                )
            m_true = n_up_true - n_dn_true
            true_chg.data = {
                Spin.up: 0.5 * (ml_grid + m_true),
                Spin.down: 0.5 * (ml_grid - m_true),
            }
        elif "total" in data_obj:
            new_data = dict(data_obj)
            new_data["total"] = ml_grid
            true_chg.data = new_data
        else:
            new_data = {}
            for key, arr in data_obj.items():
                arr_np = np.asarray(arr)
                if arr_np.shape != dim:
                    raise RuntimeError(
                        f"Template CHGCAR channel {key} has shape {arr_np.shape}, "
                        f"but ML grid shape is {dim}"
                    )
                new_data[key] = ml_grid.copy()
            true_chg.data = new_data
    else:
        true_chg.data = ml_grid.copy()

    true_chg.write_file(out_path)


def nmae(pred, true) -> float:
    """Normalized Mean Absolute Error: sum|pred-true| / sum|true|."""
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.sum(np.abs(pred - true)) / (np.sum(np.abs(true))))


def compute_nmae_between_chgcars(true_chgcar_path: str, pred_chgcar_path: str):
    """Return (nmae_total, nmae_diff_or_None)."""
    from pymatgen.io.vasp.outputs import Chgcar

    chg_true = Chgcar.from_file(true_chgcar_path)
    chg_pred = Chgcar.from_file(pred_chgcar_path)

    tot_true, diff_true = _extract_total_and_diff(chg_true)
    tot_pred, diff_pred = _extract_total_and_diff(chg_pred)

    if tot_true.shape != tot_pred.shape:
        raise RuntimeError(
            f"Total grid shape mismatch: {tot_true.shape} vs {tot_pred.shape}"
        )

    n_total = nmae(tot_pred, tot_true)
    n_diff = None
    if (diff_true is not None) and (diff_pred is not None):
        if diff_true.shape != diff_pred.shape:
            raise RuntimeError(
                f"Diff grid shape mismatch: {diff_true.shape} vs {diff_pred.shape}"
            )
        n_diff = nmae(diff_pred, diff_true)
    return n_total, n_diff
