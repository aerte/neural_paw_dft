"""Whole pipeline on tiny cells (needs the weights; skipped otherwise). CPU, small grids."""
import json

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Chgcar

from neural_paw_dft.augnet.augnet_model import Z_TO_SCHEMA
from neural_paw_dft.pipeline import Pipeline, PipelineConfig
from neural_paw_dft.pipeline.config import ElectrafiConfig

from ..conftest import weights_available

NEEDED = ["electrafi_spin_constrained", "electrafi_total", "augnet_total_full", "augnet_spin_full"]
pytestmark = pytest.mark.skipif(not weights_available(*NEEDED), reason="model weights not available")

GRID = (20, 20, 20)


def _fe():
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def _nacl():
    return Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_build_spin_polarized_fe(tmp_path, monkeypatch):
    monkeypatch.delenv("PMG_VASP_PSP_DIR", raising=False)
    Poscar(_fe()).write_file(str(tmp_path / "POSCAR"))
    cfg = PipelineConfig(device="cpu", grid_dims=GRID, out_dir=str(tmp_path / "out"))
    out = Pipeline(cfg).build(tmp_path / "POSCAR")

    chg = Chgcar.from_file(str(out / "CHGCAR"))
    nelect = 2 * 14  # Fe_pv
    assert abs(chg.data["total"].sum() / chg.data["total"].size - nelect) < 1e-4 * nelect
    assert chg.is_spin_polarized and chg.data["diff"].shape == GRID
    assert chg.data_aug["total"][1].size == Z_TO_SCHEMA[26] == chg.data_aug["diff"][1].size
    assert np.all(np.isfinite(chg.data["diff"]))

    incar = Incar.from_file(str(out / "INCAR"))
    meta = json.loads((out / "ndi_prediction.json").read_text())
    assert incar["ICHARG"] == 1 and incar["ISPIN"] == 2 and incar["LMAXMIX"] == 4
    assert len(incar["MAGMOM"]) == 2
    np.testing.assert_allclose(incar["MAGMOM"], meta["site_moments_chgnet"], rtol=1e-6)
    assert meta["m_total_constraint"] == pytest.approx(sum(meta["site_moments_chgnet"]))
    assert meta["nelect"] == nelect and meta["grid_dims"] == list(GRID)
    # CHGNet gives bcc Fe a sizeable moment, so the constrained arm rescaled the spin grid to it
    spin_int = chg.data["diff"].sum() / chg.data["diff"].size
    assert spin_int == pytest.approx(meta["m_total_constraint"], rel=1e-3)
    for f in ("POSCAR", "KPOINTS", "POTCAR.spec"):
        assert (out / f).is_file()


def test_predict_charge_only_nacl(tmp_path):
    cfg = PipelineConfig(
        device="cpu",
        electrafi=ElectrafiConfig(checkpoint="electrafi_total", spin=False),
    )
    cfg.chgnet.enabled = False
    cfg.augnet.mag_checkpoint = None
    Poscar(_nacl()).write_file(str(tmp_path / "POSCAR"))
    pipe = Pipeline(cfg)
    pred = pipe.predict(tmp_path / "POSCAR", grid_dims=GRID)
    assert pred.rho_spin is None and pred.aug_diff is None and pred.site_moments is None
    assert pred.rho_total.shape == GRID
    assert pred.rho_total.sum() * pred.structure.volume / pred.rho_total.size == pytest.approx(pred.nelect, rel=1e-5)
    assert pred.nelect == 7 + 7  # Na_pv, Cl
    out = pipe.save_prediction(pred, tmp_path / "pred")
    npz = np.load(out / "structure_total_aug.npz")
    assert npz["aug_sanvito_padded"].shape == (2, 390)
    assert npz["schema_mask"][0].sum() == Z_TO_SCHEMA[11] and npz["schema_mask"][1].sum() == Z_TO_SCHEMA[17]
    # the npz is readable by the experiment code's loader
    from neural_paw_dft.vasp_runner.chgcar import aug_dict_from_npz

    ref = {i + 1: np.zeros(Z_TO_SCHEMA[s.specie.Z]) for i, s in enumerate(pred.structure)}
    blocks = aug_dict_from_npz(str(out / "structure_total_aug.npz"), pred.structure, ref)
    np.testing.assert_allclose(blocks[1], pred.aug_total[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_cpu_vs_gpu_density_agree(tmp_path):
    Poscar(_fe()).write_file(str(tmp_path / "POSCAR"))
    outs = []
    for dev in ("cpu", "cuda"):
        cfg = PipelineConfig(device=dev, grid_dims=GRID)
        cfg.chgnet.enabled = False
        outs.append(Pipeline(cfg).predict(tmp_path / "POSCAR"))
    a, b = outs[0].rho_total, outs[1].rho_total
    assert np.abs(a - b).sum() / a.sum() < 1e-3
