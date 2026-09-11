"""Strict loads of the shipped weights (skipped when trained_models/ is absent)."""
import pytest
import torch

from neural_init.models import REGISTRY, weights_dir

WEIGHTS = weights_dir()
pytestmark = pytest.mark.skipif(not WEIGHTS.is_dir(), reason="no trained_models/ directory")


def _have(name):
    return (WEIGHTS / REGISTRY[name]).is_file()


@pytest.mark.parametrize("name", ["electrafi_spin_constrained", "electrafi_spin_unconstrained", "electrafi_total"])
def test_electrafi_strict_load(name):
    if not _have(name):
        pytest.skip(f"{name} weights missing")
    from neural_init.pipeline.config import ElectrafiConfig
    from neural_init.pipeline.electrafi import load_electrafi

    cfg = ElectrafiConfig(checkpoint=name, spin=name != "electrafi_total")
    model = load_electrafi(cfg, torch.device("cpu"), n_atoms=8)
    assert model._ndi_has_spin == (name != "electrafi_total")


@pytest.mark.parametrize("name,channel", [("augnet_total_full", "total"), ("augnet_spin_full", "mag")])
def test_augnet_strict_load(name, channel):
    if not _have(name):
        pytest.skip(f"{name} weights missing")
    from neural_init.pipeline.augnet import load_augnet

    pred = load_augnet(name, None, torch.device("cpu"))
    assert pred.channel == channel
