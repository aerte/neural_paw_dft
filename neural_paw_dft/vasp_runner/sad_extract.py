"""SAD (Superposition of Atomic Densities) CHGCAR extraction.

VASP doesn't expose its starting SAD density as a standalone file, so we
extract one by running a single-electronic-step SCF (``ICHARG=2``, ``NELM=1``,
``LCHARG=T``) and reading the written CHGCAR. The result is the post-one-mix
density, which is what hybrid-init experiments need: a CHGCAR with the right
grid + augmentation shapes that represents the (near-)atomic guess.

Reused by the decomposition runner (not included) to build the variant 3 / variant 4
hybrid CHGCARs (converged-total + SAD-diff, and SAD-total + converged-diff).
"""
import os
import shutil
import subprocess
import time

from pymatgen.io.vasp.inputs import Incar

from .sources.mp import write_mp_inputs_for_mpid


def cached_sad_chgcar_path(sad_cache_dir: str, mpid: str) -> str:
    return os.path.join(sad_cache_dir, f"{mpid}_SAD_CHGCAR")


def extract_sad_chgcar(
    mpid: str,
    sad_cache_dir: str,
    vasp_cmd,
    workdir: str | None = None,
) -> str:
    """Produce a SAD-equivalent CHGCAR for `mpid` and return its path.

    If ``<sad_cache_dir>/<mpid>_SAD_CHGCAR`` already exists, returns it
    without re-running VASP. Otherwise creates a scratch workdir, writes
    exact MP inputs, patches the INCAR for a one-step LCHARG run, calls
    VASP, and moves CHGCAR into the cache.

    Args:
        mpid: Materials Project id (e.g. ``mp-19009``).
        sad_cache_dir: Directory that holds reusable SAD CHGCARs.
        vasp_cmd: Tuple/list of argv used to launch VASP.
        workdir: Optional scratch dir for the 1-step run. Defaults to
            ``<sad_cache_dir>/extract_<mpid>``.

    Returns:
        Absolute path to the cached SAD CHGCAR.
    """
    os.makedirs(sad_cache_dir, exist_ok=True)
    cached = cached_sad_chgcar_path(sad_cache_dir, mpid)
    if os.path.isfile(cached) and os.path.getsize(cached) > 0:
        return cached

    if workdir is None:
        workdir = os.path.join(sad_cache_dir, f"extract_{mpid}")
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir, exist_ok=True)

    write_mp_inputs_for_mpid(mpid, workdir)

    # K-point sampling is irrelevant for SAD: ICHARG=2 + NELM=1 just writes the
    # atomic-superposition density on the FFT grid. The FFT grid is set by
    # ENCUT/PREC, not k-points, and augmentation occupancy shapes don't depend
    # on k-sampling either. Force Γ-only to sidestep IBZKPT's k-mesh-type
    # detection, which fails on some MP KPOINTS (seen for mp-1186919:
    # "VERY BAD NEWS! ... Fatal error detecting k-mesh type! 60001").
    with open(os.path.join(workdir, "KPOINTS"), "w") as fh:
        fh.write("Gamma-only for SAD extract\n0\nGamma\n1 1 1\n0 0 0\n")

    incar_path = os.path.join(workdir, "INCAR")
    incar = Incar.from_file(incar_path)
    # NPAR/NCORE/KPAR/NSIM: parallelism knobs from the original run, often
    # invalid for our core count. POMASS: MP-fetched INCARs sometimes carry a
    # single-value POMASS that VASP 5.4.4 rejects against multi-species POSCARs
    # ("Error reading item 'POMASS' from file INCAR. Found N=1 data."). Drop
    # it so VASP falls back to the per-species POMASS in POTCAR.
    for bad in ("NPAR", "NCORE", "KPAR", "NSIM", "POMASS"):
        incar.pop(bad, None)
    incar["ICHARG"] = 2     # SAD start
    incar["ISTART"] = 0
    incar["NELM"] = 1       # one electronic step → write density and stop
    incar["LCHARG"] = True
    incar["LWAVE"] = False
    # MP static INCARs default to ISMEAR=-5 (tetrahedron + Blöchl), which
    # aborts in IBZKPT with "Tetrahedron method fails for NKPT<4" once we force
    # the Γ-only KPOINTS above (NKPT=1). Smearing is irrelevant for a one-step
    # ICHARG=2/NELM=1 density write, so use Gaussian smearing, which has no
    # k-point-count requirement.
    incar["ISMEAR"] = 0
    incar["SIGMA"] = 0.05
    incar.write_file(incar_path)

    start = time.time()
    # NELM=1 hits "electronic step limit" → non-zero return is expected.
    # We rely on CHGCAR existing rather than VASP's exit code.
    # Capture stdout+stderr into workdir/vasp.out so failures are diagnosable
    # without having to re-run interactively (this VASP build prints all its
    # diagnostics — INCAR-parse errors, POTCAR mismatches, etc. — to stdout).
    stdout_path = os.path.join(workdir, "vasp.out")
    with open(stdout_path, "w") as fh:
        result = subprocess.run(
            vasp_cmd, cwd=workdir, check=False,
            stdout=fh, stderr=subprocess.STDOUT,
        )
    wall = time.time() - start

    produced = os.path.join(workdir, "CHGCAR")
    if not (os.path.isfile(produced) and os.path.getsize(produced) > 0):
        # Surface the tail of vasp.out in the exception so the SLURM log
        # shows what VASP actually complained about.
        tail = ""
        try:
            with open(stdout_path) as fh:
                lines = fh.readlines()
            tail = "".join(lines[-30:]).rstrip()
        except OSError:
            pass
        raise RuntimeError(
            f"SAD extraction for {mpid} did not produce a CHGCAR "
            f"(workdir={workdir}, wall={wall:.1f}s, "
            f"vasp_rc={result.returncode}). "
            f"Last lines of vasp.out:\n{tail}"
        )

    shutil.move(produced, cached)
    return cached
