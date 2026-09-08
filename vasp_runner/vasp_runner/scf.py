"""Two VASP runners — one using MPStaticSet (GNOME) and one using exact MP
TaskDoc inputs. Each returns the same tuple so callers can be uniform.
"""
import os
import shutil
import subprocess
import time

import lz4.frame

from .incar import patch_lmaxmix_for_dftu
from .oszicar import count_scf_breakdown_from_oszicar


_KEEP_AFTER_RUN = frozenset({"OUTCAR", "OSZICAR"})


def prune_workdir_keep_outputs(workdir: str) -> None:
    """Delete every entry in `workdir` except OUTCAR and OSZICAR.

    Called once per row, after OSZICAR/OUTCAR have been parsed and the
    results CSV + runs.db manifest have been updated, to reclaim the disk
    space taken by CHGCAR/WAVECAR/vasprun.xml/etc. Swallows errors:
    cleanup must never turn into a job-killing exception.
    """
    if not os.path.isdir(workdir):
        return
    try:
        names = os.listdir(workdir)
    except OSError:
        return
    for name in names:
        if name in _KEEP_AFTER_RUN:
            continue
        path = os.path.join(workdir, name)
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                os.remove(path)
        except (FileNotFoundError, OSError):
            pass


def _copy_chgcar_into(workdir: str, chgcar_path: str):
    dest = os.path.join(workdir, "CHGCAR")
    try:
        if os.path.exists(dest) and os.path.samefile(chgcar_path, dest):
            return
    except FileNotFoundError:
        pass
    if chgcar_path.endswith(".lz4"):
        with lz4.frame.open(chgcar_path, "rb") as src, open(dest, "wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy(chgcar_path, dest)


def run_vasp_pymatgen(atoms,
                      workdir,
                      icharg,
                      chgcar_path=None,
                      vasp_cmd=("vasp_std",),
                      grid_dims=None):
    """GNOME-style runner: build INCAR/KPOINTS/POTCAR via MPStaticSet,
    optionally seed CHGCAR, run VASP, parse OSZICAR + OUTCAR.

    Returns:
        (energy, dav, rmm, total, other_counts, wall)
    """
    # Local imports so install_spglib_patch() takes effect before pymatgen
    # symmetry paths get pulled in.
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.sets import MPStaticSet
    from pymatgen.io.vasp.outputs import Outcar

    os.makedirs(workdir, exist_ok=True)

    structure = AseAtomsAdaptor.get_structure(atoms)
    user_incar = {
        "ICHARG": icharg,
        "ISTART": 0,
        "LCHARG": True,
        "LWAVE": False,
    }
    mp_set = MPStaticSet(structure, user_incar_settings=user_incar)
    mp_set.write_input(workdir)

    incar_path = os.path.join(workdir, "INCAR")
    patch_lmaxmix_for_dftu(incar_path)

    if chgcar_path is not None:
        _copy_chgcar_into(workdir, chgcar_path)

    start_time = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError as e:
        wall = time.time() - start_time
        print(f"❌ VASP run failed (return code {e.returncode}) in {workdir}")
        osz_path = os.path.join(workdir, "OSZICAR")
        dav, rmm, total, other = count_scf_breakdown_from_oszicar(osz_path)
        return None, dav, rmm, total, other, wall

    wall = time.time() - start_time

    osz_path = os.path.join(workdir, "OSZICAR")
    dav, rmm, total, other = count_scf_breakdown_from_oszicar(osz_path)

    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        print(f"❌ OUTCAR missing in {workdir}; treating as failed run.")
        return None, dav, rmm, total, other, wall
    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
    except Exception as e:
        print(f"❌ Failed to parse OUTCAR in {workdir}: {e!r}")
        energy = None
    return energy, dav, rmm, total, other, wall


def run_vasp_mp(mpid: str,
                workdir: str,
                icharg: int,
                chgcar_path,
                vasp_cmd,
                grid_dims=None):
    """MP-style runner: write the exact TaskDoc inputs, then VASP.

    Returns:
        (energy, dav, rmm, total, other_counts, wall)
    """
    from pymatgen.io.vasp.inputs import Incar
    from pymatgen.io.vasp.outputs import Outcar
    from .sources.mp import write_mp_inputs_for_mpid

    os.makedirs(workdir, exist_ok=True)
    write_mp_inputs_for_mpid(mpid, workdir)

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    for bad in ["NPAR", "NCORE", "KPAR", "NSIM"]:
        if bad in incar:
            incar.pop(bad)
    incar["ICHARG"] = icharg
    incar["ISTART"] = 0
    incar["LCHARG"] = True
    incar.write_file(incar_path)

    if chgcar_path is not None:
        _copy_chgcar_into(workdir, chgcar_path)

    start_time = time.time()
    try:
        subprocess.run(vasp_cmd, cwd=workdir, check=True)
    except subprocess.CalledProcessError:
        wall = time.time() - start_time
        osz_path = os.path.join(workdir, "OSZICAR")
        dav, rmm, total, other = count_scf_breakdown_from_oszicar(osz_path)
        return None, dav, rmm, total, other, wall

    wall = time.time() - start_time

    osz_path = os.path.join(workdir, "OSZICAR")
    dav, rmm, total, other = count_scf_breakdown_from_oszicar(osz_path)

    outcar_path = os.path.join(workdir, "OUTCAR")
    if not os.path.isfile(outcar_path):
        return None, dav, rmm, total, other, wall
    try:
        outcar = Outcar(outcar_path)
        energy = outcar.final_energy
    except Exception:
        energy = None
    return energy, dav, rmm, total, other, wall
