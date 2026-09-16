"""Small pipeline helpers that need neither weights nor VASP."""
import lz4.frame
import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Chgcar

from neural_paw_dft.augnet.augnet_model import Z_TO_SCHEMA
from neural_paw_dft.pipeline import assemble, inputs
from neural_paw_dft.pipeline.augnet import aug_blocks
from neural_paw_dft.pipeline.config import VaspConfig
from neural_paw_dft.pipeline.pipeline import resolve_device
from neural_paw_dft.vasp_runner.chgcar import get_chgcar_dim


def _nacl():
    return Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_aug_blocks_cuts_each_element_to_its_schema():
    zs = [11, 17]
    padded = np.arange(2 * 390, dtype=float).reshape(2, 390)
    blocks = aug_blocks(padded, zs)
    assert set(blocks) == {1, 2}
    for i, z in enumerate(zs, start=1):
        assert blocks[i].shape == (Z_TO_SCHEMA[z],)
        np.testing.assert_array_equal(blocks[i], padded[i - 1, : Z_TO_SCHEMA[z]])


def test_grid_dims_from_incar():
    assert assemble.grid_dims_from_incar(Incar({"NGXF": 40, "NGYF": 42, "NGZF": 44})) == (40, 42, 44)
    assert assemble.grid_dims_from_incar(Incar({"NGXF": 40})) is None


def test_potcars_available(monkeypatch, tmp_path):
    monkeypatch.delenv("PMG_VASP_PSP_DIR", raising=False)
    monkeypatch.setattr("pymatgen.core.SETTINGS", {}, raising=False)
    assert assemble._potcars_available(VaspConfig()) is False
    assert assemble._potcars_available(VaspConfig(potcar_dir=str(tmp_path))) is True
    assert assemble._potcars_available(VaspConfig()) is True  # potcar_dir exported PMG_VASP_PSP_DIR


def test_dry_run_requires_vasp_cmd(tmp_path):
    with pytest.raises(ValueError, match="vasp_cmd"):
        assemble.dry_run_grid_dims(_nacl(), VaspConfig(), tmp_path)


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_load_input_lz4_chgcar(tmp_path):
    s = _nacl()
    grid = (6, 7, 8)
    chg = Chgcar(Poscar(s), {"total": np.ones(grid)})
    raw = tmp_path / "CHGCAR"
    chg.write_file(str(raw))
    packed = tmp_path / "CHGCAR.lz4"
    with open(raw, "rb") as f_in, lz4.frame.open(packed, "wb") as f_out:
        f_out.write(f_in.read())
    inp = inputs.load_input(packed)
    assert inp.grid_dims == grid and len(inp.structure) == 2 and inp.path == str(packed)
    assert get_chgcar_dim(Chgcar.from_file(str(raw))) == grid
    assert not list(tmp_path.glob("ndi_chgcar_*"))  # temp copy removed


def test_load_input_ase_fallback(tmp_path):
    from ase.build import bulk
    from ase.io import write

    p = tmp_path / "cu.xyz"
    write(str(p), bulk("Cu", "fcc", a=3.6, cubic=True))
    inp = inputs.load_input(p)
    assert len(inp.structure) == 4 and inp.grid_dims is None
