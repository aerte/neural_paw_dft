import numpy as np
import pytest
import yaml
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

from neural_paw_dft.pipeline import inputs
from neural_paw_dft.pipeline.config import PipelineConfig, config_template, load_config


def test_template_roundtrip(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(config_template())
    cfg = load_config(p)
    assert cfg == PipelineConfig()  # the template documents the defaults


def test_load_config_partial_and_unknown(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump({"grid_dims": [24, 24, 30], "electrafi": {"spin": False}, "vasp": {"incar_overrides": {"ENCUT": 520}}}))
    cfg = load_config(p)
    assert cfg.grid_dims == (24, 24, 30) and cfg.electrafi.spin is False and cfg.vasp.incar_overrides == {"ENCUT": 520}
    assert cfg.augnet.total_checkpoint == "augnet_total_full"
    p.write_text(yaml.safe_dump({"electrafi": {"spinn": True}}))
    with pytest.raises(KeyError, match="spinn"):
        load_config(p)


def test_load_input_poscar_and_nelect(tmp_path):
    s = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    Poscar(s).write_file(str(tmp_path / "POSCAR"))
    inp = inputs.load_input(tmp_path / "POSCAR")
    assert inp.grid_dims is None and len(inp.structure) == 2
    assert inputs.nelect_for(inp.structure) == 2 * 14  # Fe_pv in the MP set


def test_load_input_chgcar(tmp_path):
    from pymatgen.io.vasp.outputs import Chgcar

    s = Structure(Lattice.cubic(4.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    Chgcar(Poscar(s), {"total": np.ones((6, 8, 10))}).write_file(str(tmp_path / "CHGCAR"))
    inp = inputs.load_input(tmp_path / "CHGCAR")
    assert inp.grid_dims == (6, 8, 10) and [x.specie.symbol for x in inp.structure] == ["Na", "Cl"]
