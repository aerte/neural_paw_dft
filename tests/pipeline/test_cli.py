"""`ndi` argument handling, without running any model."""
import argparse

import pytest

from neural_paw_dft import __version__
from neural_paw_dft.pipeline import cli
from neural_paw_dft.pipeline.config import PipelineConfig, config_template


def test_config_template_subcommand(capsys):
    assert cli.main(["config-template"]) == 0
    assert capsys.readouterr().out == config_template()


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_parse_incar_overrides():
    assert cli._parse_incar_overrides(["encut=520", "LREAL=Auto", "NGXF=48"]) == {"ENCUT": 520.0, "LREAL": "Auto", "NGXF": 48}
    assert cli._parse_incar_overrides(None) == {}
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        cli._parse_incar_overrides(["ENCUT"])


def _args(**kw):
    base = dict(device=None, grid=None, weights_dir=None, no_spin=False, no_chgnet=False, incar=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_apply_flags():
    cfg = cli._apply_flags(PipelineConfig(), _args(device="cpu", grid=[8, 9, 10], weights_dir="/w", no_chgnet=True, incar=["ENCUT=400"]))
    assert cfg.device == "cpu" and cfg.grid_dims == (8, 9, 10) and cfg.weights_dir == "/w"
    assert cfg.chgnet.enabled is False and cfg.vasp.incar_overrides["ENCUT"] == 400.0
    assert cfg.electrafi.spin is True and cfg.electrafi.checkpoint == "electrafi_spin_constrained"


def test_no_spin_swaps_registry_spin_checkpoint_only():
    cfg = cli._apply_flags(PipelineConfig(), _args(no_spin=True))
    assert cfg.electrafi.spin is False and cfg.electrafi.checkpoint == "electrafi_total"
    custom = PipelineConfig()
    custom.electrafi.checkpoint = "/my/own.ckpt"
    cfg = cli._apply_flags(custom, _args(no_spin=True))
    assert cfg.electrafi.spin is False and cfg.electrafi.checkpoint == "/my/own.ckpt"
