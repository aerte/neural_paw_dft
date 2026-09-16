"""Weights-free tests of the weights registry (no download is attempted)."""
import pytest

from neural_paw_dft.models import REGISTRY, augnet_sidecar, resolve_weights, weights_dir


def test_registry_is_flat_safetensors():
    for name, fname in REGISTRY.items():
        assert fname == f"{name}.safetensors"


def test_sidecar_name(tmp_path):
    assert augnet_sidecar(tmp_path / "augnet_total_full.safetensors") == tmp_path / "augnet_total_full.config.json"


def test_existing_path_wins(tmp_path):
    f = tmp_path / "anything.ckpt"
    f.write_bytes(b"x")
    assert resolve_weights(str(f)) == f


def test_unknown_name_raises():
    with pytest.raises(KeyError):
        resolve_weights("not_a_model")


def test_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("NDI_WEIGHTS_DIR", str(tmp_path))
    assert weights_dir() == tmp_path
    assert weights_dir(tmp_path / "x") == tmp_path / "x"


def test_local_registry_file_is_used_without_download(tmp_path, monkeypatch):
    monkeypatch.setenv("NDI_WEIGHTS_DIR", str(tmp_path))
    (tmp_path / REGISTRY["electrafi_total"]).write_bytes(b"x")
    assert resolve_weights("electrafi_total") == tmp_path / "electrafi_total.safetensors"
