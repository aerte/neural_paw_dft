"""CHGCAR writer and VASP-input generation, no model weights needed."""
import re

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Chgcar

from neural_paw_dft.augnet.augnet_model import Z_TO_SCHEMA
from neural_paw_dft.augnet.run_paw_chgcar import predicted_aug_blocks_text
from neural_paw_dft.pipeline import assemble
from neural_paw_dft.pipeline.config import VaspConfig
from neural_paw_dft.vasp_runner.chgcar import read_spin_moment_line, dims_line

AUG_HDR = re.compile(r"^augmentation occupancies\s*(\d+)\s*(\d+)$")


def _structure(n_fe=6, n_o=6):
    """12 atoms so ion indices reach two digits (the header-width bug regime)."""
    rng = np.random.default_rng(0)
    lat = Lattice.cubic(8.0)
    species = ["Fe"] * n_fe + ["O"] * n_o
    return Structure(lat, species, rng.random((n_fe + n_o, 3)))


def _aug(structure, seed):
    rng = np.random.default_rng(seed)
    return {i + 1: rng.normal(size=Z_TO_SCHEMA[s.specie.Z]) for i, s in enumerate(structure)}


def test_build_chgcar_roundtrip(tmp_path):
    s = _structure()
    grid = (8, 8, 8)
    rng = np.random.default_rng(1)
    rho = rng.random(grid) + 0.1
    spin = rng.normal(size=grid) * 0.01
    nelect = 6 * 14 + 6 * 6
    aug_t, aug_d = _aug(s, 2), _aug(s, 3)
    moments = np.linspace(0.5, 3.0, len(s))
    out = assemble.build_chgcar(s, rho, spin, aug_t, aug_d, moments, tmp_path / "CHGCAR", nelect=nelect)

    chg = Chgcar.from_file(str(out))
    assert chg.is_spin_polarized
    # total grid was renormalised to NELECT, spin grid kept
    assert abs(chg.data["total"].sum() / chg.data["total"].size - nelect) < 1e-6 * nelect
    np.testing.assert_allclose(chg.data["diff"], spin * s.volume, rtol=1e-8, atol=1e-12)
    assert chg.structure.matches(s, ltol=1e-6, stol=1e-6)
    for i in range(1, len(s) + 1):
        assert chg.data_aug["total"][i].size == Z_TO_SCHEMA[s[i - 1].specie.Z]
        np.testing.assert_allclose(chg.data_aug["total"][i], aug_t[i], rtol=1e-9)
        np.testing.assert_allclose(chg.data_aug["diff"][i], aug_d[i], rtol=1e-9)

    # VASP's fixed-width header on every block, both channels
    headers = [ln.rstrip("\n") for ln in open(out) if ln.startswith("augmentation occupancies")]
    assert len(headers) == 2 * len(s)
    for ln in headers:
        m = AUG_HDR.match(ln)
        assert m, ln
        assert ln == f"augmentation occupancies{int(m[1]):4d}{int(m[2]):4d}"

    # per-ion moment block sits right before the spin grid and holds our values
    lines = read_spin_moment_line(str(out), dims_line(chg), len(s))
    assert lines is not None and len(lines) == -(-len(s) // 5)
    np.testing.assert_allclose([float(x) for x in " ".join(lines).split()], moments, rtol=1e-9)


def test_build_chgcar_total_only_no_moment_block(tmp_path):
    s = _structure(2, 2)
    rho = np.ones((4, 4, 4))
    out = assemble.build_chgcar(s, rho, None, _aug(s, 4), None, None, tmp_path / "CHGCAR")
    chg = Chgcar.from_file(str(out))
    assert not chg.is_spin_polarized and "diff" not in chg.data
    assert read_spin_moment_line(str(out), dims_line(chg), len(s)) is None


def test_build_chgcar_rejects_wrong_block_length(tmp_path):
    s = _structure(1, 1)
    aug = _aug(s, 5)
    aug[1] = aug[1][:-1]
    with pytest.raises(ValueError, match="augmentation block"):
        assemble.build_chgcar(s, np.ones((2, 2, 2)), None, aug, None, None, tmp_path / "CHGCAR")


def test_aug_values_match_augnet_text_formatter(tmp_path):
    """Same numbers as AugNet's own VASP-block writer (formatting differs, values must not)."""
    import torch

    s = _structure(1, 2)
    aug = _aug(s, 6)
    out = assemble.build_chgcar(s, np.ones((2, 2, 2)), None, aug, None, None, tmp_path / "CHGCAR")
    chg = Chgcar.from_file(str(out))
    z = torch.tensor([x.specie.Z for x in s])
    padded = torch.zeros(len(s), 390, dtype=torch.float64)
    for i, blk in aug.items():
        padded[i - 1, : blk.size] = torch.tensor(blk)
    txt = predicted_aug_blocks_text(y_total_sanvito=padded, atomic_numbers=z, y_mag_sanvito=None)
    vals = [float(t) for ln in txt.splitlines() if not ln.startswith("augmentation") for t in ln.split()]
    ours = np.concatenate([chg.data_aug["total"][i] for i in range(1, len(s) + 1)])
    np.testing.assert_allclose(vals, ours, rtol=1e-9)


def test_synth_moment_lines():
    lines = assemble.synth_moment_lines(np.arange(7.0), 7)
    assert len(lines) == 2 and len(lines[0].split()) == 5 and len(lines[1].split()) == 2
    assert assemble.synth_moment_lines(None, 3) == ["".join(f"{0.0:18.11E}" for _ in range(3))]


def test_write_vasp_inputs(tmp_path, monkeypatch):
    monkeypatch.delenv("PMG_VASP_PSP_DIR", raising=False)
    s = _structure(2, 3)
    vasp = VaspConfig(incar_overrides={"ENCUT": 450, "NCORE": 4})
    magmoms = np.array([2.5, 2.5, 0.1, 0.1, 0.1])
    incar = assemble.write_vasp_inputs(s, tmp_path, vasp, magmoms, spin_polarized=True)

    assert incar["ICHARG"] == 1 and incar["ISTART"] == 0 and incar["LCHARG"] is True and incar["ISPIN"] == 2
    assert incar["ENCUT"] == 450 and incar["NCORE"] == 4  # overrides win, incl. a parallel tag
    assert "NPAR" not in incar and "KPAR" not in incar
    np.testing.assert_allclose(incar["MAGMOM"], magmoms)
    assert incar["LMAXMIX"] == 4  # Fe is a d element
    assert (tmp_path / "KPOINTS").is_file() and (tmp_path / "POSCAR").is_file()
    assert (tmp_path / "POTCAR.spec").is_file()  # no POTCAR library available in the test env
    # site order preserved (MPStaticSet would otherwise sort by electronegativity)
    written = Poscar.from_file(str(tmp_path / "POSCAR")).structure
    assert [x.specie.symbol for x in written] == [x.specie.symbol for x in s]
    assert Incar.from_file(str(tmp_path / "INCAR"))["ISPIN"] == 2


def test_write_vasp_inputs_ispin1_drops_magmom(tmp_path, monkeypatch):
    monkeypatch.delenv("PMG_VASP_PSP_DIR", raising=False)
    s = _structure(1, 1)
    incar = assemble.write_vasp_inputs(s, tmp_path, VaspConfig(), None, spin_polarized=False)
    assert incar["ISPIN"] == 1 and "MAGMOM" not in incar
