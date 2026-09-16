"""ELECTRAFI training-config resolution and inference-config trimming (no weights needed)."""
from pathlib import Path

import pytest

from neural_paw_dft.models import ELECTRAFI_TRAIN_CONFIG
from neural_paw_dft.pipeline.config import ElectrafiConfig
from neural_paw_dft.pipeline.electrafi import _train_config_path, _work_dir, build_inference_config


@pytest.mark.parametrize("name", sorted(ELECTRAFI_TRAIN_CONFIG))
def test_registry_names_resolve_to_packaged_yaml(name):
    p = _train_config_path(ElectrafiConfig(checkpoint=name), Path("x.safetensors"))
    assert p.is_file() and p.name == Path(ELECTRAFI_TRAIN_CONFIG[name]).name


def test_bare_path_without_train_config_raises():
    with pytest.raises(ValueError, match="training config"):
        _train_config_path(ElectrafiConfig(checkpoint="/some/where.safetensors"), Path("where.safetensors"))


def test_explicit_train_config_wins(tmp_path):
    y = tmp_path / "conf.yaml"
    y.write_text("x: 1\n")
    assert _train_config_path(ElectrafiConfig(checkpoint="/a.ckpt", train_config=str(y)), Path("a.ckpt")) == y


def test_build_inference_config_trims_training_and_fits_structure(tmp_path):
    cfg = ElectrafiConfig(checkpoint="electrafi_total", spin=False)
    c = build_inference_config(cfg, Path("electrafi_total.safetensors"), has_spin_head=False, work_dir=tmp_path,
                               n_atoms=500, num_layers=3)
    assert c["wandb"] is False and c["inference_only"] is True and c["debug"] is False
    assert "spin_type" not in c
    bb = c["escaip_config"]["model"]["backbone"]
    assert bb["max_num_nodes_per_batch"] >= 500 and bb["num_layers"] == 3
    assert c["dens_path"] == str(tmp_path)

    c2 = build_inference_config(ElectrafiConfig(checkpoint="electrafi_spin_constrained"), Path("s.safetensors"),
                                has_spin_head=True, work_dir=tmp_path, n_atoms=2)
    assert c2["spin_type"] == "total_diff"


def test_work_dir_is_shared_per_process():
    a, b = _work_dir(), _work_dir()
    assert a == b and a.is_dir()
