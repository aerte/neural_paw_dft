"""Strict loads of the shipped weights (skipped unless present or NDI_TEST_DOWNLOAD_WEIGHTS=1)."""
import subprocess
import sys

import pytest
import torch

from .conftest import weights_available as _have


@pytest.mark.parametrize("name", ["electrafi_spin_constrained", "electrafi_spin_unconstrained",
                                  "electrafi_total", "electrafi_total_v2"])
def test_electrafi_strict_load(name):
    if not _have(name):
        pytest.skip(f"{name} weights missing")
    from neural_paw_dft.pipeline.config import ElectrafiConfig
    from neural_paw_dft.pipeline.electrafi import load_electrafi

    cfg = ElectrafiConfig(checkpoint=name, spin=not name.startswith("electrafi_total"))
    model = load_electrafi(cfg, torch.device("cpu"), n_atoms=8)
    assert model._ndi_has_spin == (not name.startswith("electrafi_total"))


@pytest.mark.parametrize("name,channel", [("augnet_total_full", "total"), ("augnet_spin_full", "mag")])
def test_augnet_strict_load(name, channel):
    if not _have(name):
        pytest.skip(f"{name} weights missing")
    from neural_paw_dft.pipeline.augnet import load_augnet

    pred = load_augnet(name, None, torch.device("cpu"))
    assert pred.channel == channel


def test_augnet_rejects_unknown_element_and_cueq_flag():
    if not _have("augnet_total_full"):
        pytest.skip("augnet_total_full weights missing")
    from pymatgen.core import Lattice, Structure

    from neural_paw_dft.pipeline.augnet import load_augnet

    pred = load_augnet("augnet_total_full", None, torch.device("cpu"))
    with pytest.raises(KeyError, match="no PAW schema"):
        pred.predict(Structure(Lattice.cubic(4.0), ["Og"], [[0, 0, 0]]))
    with pytest.raises(ValueError, match="enable_cueq"):
        load_augnet("augnet_total_full", None, torch.device("cpu"), enable_cueq=True)


def test_electrafi_inference_without_lightning():
    """The `train` extra is optional: ELECTRAFI must load and predict with Lightning unimportable."""
    if not _have("electrafi_total"):
        pytest.skip("electrafi_total weights missing")
    code = """
import builtins, sys
_real = builtins.__import__
def _blocked(name, *a, **k):
    if name.split('.')[0] in {'lightning', 'pytorch_lightning'}:
        raise ImportError('blocked ' + name)
    return _real(name, *a, **k)
builtins.__import__ = _blocked
import torch
from pymatgen.core import Lattice, Structure
from neural_paw_dft.pipeline.config import ElectrafiConfig
from neural_paw_dft.pipeline.electrafi import load_electrafi, predict_density
m = load_electrafi(ElectrafiConfig(checkpoint='electrafi_total', spin=False), torch.device('cpu'), n_atoms=2)
rho, _ = predict_density(m, Structure(Lattice.cubic(5.43), ['Si', 'Si'], [[0, 0, 0], [.25, .25, .25]]), (12, 12, 12), n_elec=8.0)
assert abs(rho.sum() * 5.43**3 / rho.size - 8.0) < 1e-3
assert 'lightning' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_constrained_spin_model_requires_m_total():
    if not _have("electrafi_spin_constrained", "electrafi_spin_unconstrained"):
        pytest.skip("spin weights missing")
    from pymatgen.core import Lattice, Structure

    from neural_paw_dft.pipeline.config import ChgnetConfig, ElectrafiConfig, PipelineConfig
    from neural_paw_dft.pipeline.electrafi import load_electrafi, predict_density
    from neural_paw_dft.pipeline.pipeline import Pipeline

    fe = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    con = load_electrafi(ElectrafiConfig(checkpoint="electrafi_spin_constrained"), torch.device("cpu"), n_atoms=2)
    with pytest.raises(ValueError, match="m_total"):
        predict_density(con, fe, (12, 12, 12), n_elec=16.0)
    _, rho_spin = predict_density(con, fe, (12, 12, 12), n_elec=16.0, m_total=4.4)
    assert abs(rho_spin.sum() * fe.volume / rho_spin.size - 4.4) < 1e-3

    unc = load_electrafi(ElectrafiConfig(checkpoint="electrafi_spin_unconstrained"), torch.device("cpu"), n_atoms=2)
    predict_density(unc, fe, (12, 12, 12), n_elec=16.0)  # no constraint needed

    cfg = PipelineConfig(device="cpu", grid_dims=(12, 12, 12), chgnet=ChgnetConfig(enabled=False))
    with pytest.raises(ValueError, match="net-moment constraint"):
        Pipeline(cfg).predict(fe)
    cfg.electrafi.spin = False  # spin grid discarded: no constraint required
    assert Pipeline(cfg).predict(fe).rho_spin is None
