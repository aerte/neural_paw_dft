"""Structure / CHGCAR input handling, NELECT and LMAXMIX lookups."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Incar

from neural_paw_dft.augnet.mp_potcar_map import MP_POTCAR_BY_Z


@dataclass
class StructureInput:
    structure: Structure  # site order preserved (it is the CHGCAR / POSCAR / MAGMOM order)
    grid_dims: tuple[int, int, int] | None  # known when the input was a CHGCAR
    path: str


def _is_chgcar(path: Path) -> bool:
    name = path.name.lower()
    return "chgcar" in name or name.endswith(".chgcar.lz4")


def load_input(path: str | os.PathLike) -> StructureInput:
    """CHGCAR(.lz4): structure + grid dims from the file. Otherwise any pymatgen/ASE-readable structure."""
    path = Path(path)
    if _is_chgcar(path):
        from pymatgen.io.vasp.outputs import Chgcar

        from neural_paw_dft.vasp_runner.ml_blend import _get_chgcar_dim

        src = path
        tmp = None
        if path.suffix == ".lz4":
            import lz4.frame

            fd, tmp = tempfile.mkstemp(prefix="ndi_chgcar_")
            os.close(fd)
            with lz4.frame.open(path, "rb") as f_in, open(tmp, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            src = Path(tmp)
        try:
            chg = Chgcar.from_file(str(src))
        finally:
            if tmp:
                os.remove(tmp)
        return StructureInput(chg.structure, tuple(int(x) for x in _get_chgcar_dim(chg)), str(path))

    try:
        structure = Structure.from_file(str(path))
    except Exception:  # not a pymatgen format: fall back to ASE
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor

        structure = AseAtomsAdaptor.get_structure(read(str(path)))
    return StructureInput(structure, None, str(path))


def nelect_for(structure: Structure, potcar_path: str | os.PathLike | None = None) -> float:
    """Total valence electron count.

    From the written POTCAR when available (exact for the run); otherwise from the
    Materials Project POTCAR table the models were trained against.
    """
    if potcar_path is not None and Path(potcar_path).is_file():
        from pymatgen.io.vasp.inputs import Potcar

        potcar = Potcar.from_file(str(potcar_path))
        zval = {p.element: p.nelectrons for p in potcar}
        return float(sum(zval[site.specie.symbol] for site in structure))
    missing = sorted({site.specie.symbol for site in structure if site.specie.Z not in MP_POTCAR_BY_Z})
    if missing:
        raise KeyError(f"no MP POTCAR entry for {missing}; the models do not cover these elements")
    return float(sum(MP_POTCAR_BY_Z[site.specie.Z]["zval"] for site in structure))


def read_lmaxmix(incar_path: str | os.PathLike) -> int:
    """LMAXMIX from an INCAR (VASP default 2)."""
    return int(Incar.from_file(str(incar_path)).get("LMAXMIX", 2))
