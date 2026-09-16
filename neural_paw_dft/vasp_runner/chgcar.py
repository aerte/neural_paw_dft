"""CHGCAR helpers: structure loading and FFT grid parsing."""
import os
import re
import shutil
import tempfile

import lz4.frame
import numpy as np


# --noisemag diff-grid noise bounds: values are drawn log-uniformly in
# magnitude across [NOISE_MAG_LO, NOISE_MAG_HI] with a random sign, so the
# seeded 'diff' grid spans ~1e-7 .. 1e-5 instead of the strict zeros of
# --zeromag (a symmetry-broken but still near-zero magnetization seed).
NOISE_MAG_LO = 1e-7
NOISE_MAG_HI = 1e-5


def get_chgcar_grid_dims_textparse(chgcar_path):
    """Parse (NGXF, NGYF, NGZF) from a CHGCAR(.lz4) via line-by-line scan.

    Used by the GNOME / true-init / default scripts where pymatgen's
    Chgcar.from_file is too heavy / fails on lz4-only files.
    """
    path = chgcar_path
    tmp_created = False

    if chgcar_path.endswith(".lz4"):
        tmpfd, tmppath = tempfile.mkstemp(prefix="tmp_chgcar_grid_")
        os.close(tmpfd)
        with lz4.frame.open(chgcar_path, "rb") as src, open(tmppath, "wb") as dst:
            shutil.copyfileobj(src, dst)
        path = tmppath
        tmp_created = True

    nx = ny = nz = None
    try:
        with open(path, "r") as f:
            for _ in range(7):
                try:
                    next(f)
                except StopIteration:
                    break
            for line in f:
                toks = line.split()
                if len(toks) == 3 and all(t.isdigit() for t in toks):
                    nx, ny, nz = map(int, toks)
                    break
    finally:
        if tmp_created:
            os.remove(path)

    if nx is None:
        raise RuntimeError(
            f"Could not find NGXF/NGYF/NGZF line in {chgcar_path} "
            f"(search failed after line 7 skip)"
        )

    return nx, ny, nz


def get_chgcar_dim(chg):
    """Robust (nx, ny, nz) for a Chgcar across pymatgen versions."""
    dim = getattr(chg, "dim", None)
    if dim is not None:
        dim = tuple(dim)
        if len(dim) == 3:
            return dim

    data = chg.data
    if isinstance(data, dict) and len(data) > 0:
        arr = np.asarray(next(iter(data.values())))
    else:
        arr = np.asarray(data)
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D grid, got shape {arr.shape}")
    return tuple(arr.shape)


def _rescale_ml_to_true_charge(ml_grid, true_chg):
    """Rescale ml_grid so its integrated charge matches true_chg."""
    from pymatgen.electronic_structure.core import Spin

    data = true_chg.data
    if isinstance(data, dict):
        if (Spin.up in data) and (Spin.down in data):
            true_tot = np.asarray(data[Spin.up]) + np.asarray(data[Spin.down])
        elif "total" in data:
            true_tot = np.asarray(data["total"])
        else:
            true_tot = np.asarray(next(iter(data.values())))
    else:
        true_tot = np.asarray(data)

    dim = true_tot.shape
    assert ml_grid.shape == dim, f"ML grid shape {ml_grid.shape} != template shape {dim}"

    cell_vol = true_chg.structure.lattice.volume
    npoints = int(dim[0] * dim[1] * dim[2])
    dv = cell_vol / npoints

    q_true = true_tot.sum() * dv
    q_ml = ml_grid.sum() * dv
    if q_ml == 0:
        raise RuntimeError("ML grid has zero integrated charge, cannot rescale")
    scale = q_true / q_ml
    return ml_grid * scale


def _renorm_grid_to_ref(ml_grid, ref_chg, label: str):
    """Scale ``ml_grid`` so its integrated charge matches ``ref_chg``'s total grid.

    VASP adds the PAW compensation charge onto the FFT grid, so a converged
    CHGCAR grid integrates to NELECT; the reference is therefore the same target
    as the valence electron count, and taking it from the reference avoids
    needing the POTCAR to look NELECT up.

    Purely multiplicative — it fixes the integrated charge, not the shape.
    """
    scaled = _rescale_ml_to_true_charge(ml_grid, ref_chg)
    scale = float(scaled.sum() / ml_grid.sum()) if ml_grid.sum() else float("nan")
    print(f"[renorm] {label}: scaling predicted grid by {scale:.6f}")
    return scaled


def dims_line(chg) -> str:
    """The 'NGX NGY NGZ' grid header exactly as pymatgen's write_file emits it.

    Each density channel is preceded by one of these, so counting occurrences
    locates the start of the spin block.
    """
    nx, ny, nz = chg.dim
    return f"   {nx}   {ny}   {nz}"


_AUG_HDR = re.compile(r"^\s*augmentation occupancies\s+(\d+)\s+(\d+)\s*$")


def rewrite_aug_headers(path: str) -> int:
    """Re-emit every ``augmentation occupancies`` header in VASP's own spelling.

    VASP right-aligns both integers in width 4; pymatgen's writer pads one space
    wider as soon as the ion index reaches two digits::

        VASP      'augmentation occupancies  10  33'
        pymatgen  'augmentation occupancies   10  33'

    VASP reads that header with a fixed-width format, so the extra space shifts
    the integer fields and the read fails. It does not fail loudly: it discards
    the *entire* CHGCAR and silently starts from a superposition of atomic
    charge densities ("charge density of overlapping atoms calculated" in the
    OUTCAR), which is indistinguishable from a bad seed in the SCF statistics.
    Structures with <= 9 ions never have a two-digit index and so were unaffected,
    which is what made this look like a physics result for so long.

    Verified by controlled experiment: rewriting
    only these headers makes VASP ingest the seed and reproduce the reference
    run exactly; reformatting the augmentation *values* instead changes nothing,
    as VASP parses those list-directed.

    Returns the number of headers rewritten.
    """
    tmp = path + ".aughdr.tmp"
    n = 0
    with open(path, errors="ignore") as src, open(tmp, "w") as dst:
        for line in src:
            m = _AUG_HDR.match(line)
            if m:
                dst.write(f"augmentation occupancies"
                          f"{int(m.group(1)):4d}{int(m.group(2)):4d}\n")
                n += 1
            else:
                dst.write(line)
    os.replace(tmp, path)
    return n


def read_spin_moment_line(src_path: str, dims_line: str, n_ions: int):
    """Return the per-ion moment block a spin-polarized VASP CHGCAR carries
    between the total-augmentation block and the spin density grid, as a list of
    lines, or None if ``src_path`` has no spin block.

    The block holds one value per ion, five per line, and so spans
    ``ceil(n_ions / 5)`` lines immediately preceding the SECOND occurrence of the
    grid header (the first belongs to the total grid itself). Reading only the
    last of those lines truncates the block for every structure with more than
    five ions, which VASP rejects — it then discards the whole CHGCAR and falls
    back to a superposition of atomic charge densities.
    """
    want = " ".join(dims_line.split())
    n_lines = -(-n_ions // 5)
    prev: list[str] = []
    seen = 0
    with open(src_path, errors="ignore") as f:
        for line in f:
            if " ".join(line.split()) == want:
                seen += 1
                if seen == 2:
                    if len(prev) < n_lines:
                        return None
                    return [ln.rstrip("\n") for ln in prev[-n_lines:]]
            prev.append(line)
            if len(prev) > n_lines:
                prev.pop(0)
    return None


def splice_spin_moment_line(path: str, dims_line: str,
                             moment_lines: list) -> bool:
    """Insert ``moment_lines`` immediately before the second grid header of the
    CHGCAR at ``path``, in place. Returns True if they were inserted."""
    want = " ".join(dims_line.split())
    tmp = path + ".momentline.tmp"
    seen = 0
    done = False
    with open(path, errors="ignore") as src, open(tmp, "w") as dst:
        for line in src:
            if not done and " ".join(line.split()) == want:
                seen += 1
                if seen == 2:
                    for ml in moment_lines:
                        dst.write(ml + "\n")
                    done = True
            dst.write(line)
    os.replace(tmp, path)
    return done


def write_chgcar(chg, out_path: str, moment_src: str) -> bool:
    """``Chgcar.write_file`` with the spin block repaired.

    pymatgen's ``VolumetricData.write_file`` emits write_spin('total'),
    write_aug('total'), write_spin('diff') back to back — it has no code path
    for the per-ion moment line a real VASP spin-polarized CHGCAR carries
    between the total augmentation and the spin grid. Without that line VASP
    does not pick up the CHGCAR's spin channel at all and silently falls back
    to the INCAR MAGMOM, so a seed built to carry a converged (or zero, or
    noisy) magnetization instead starts from full atomic moments.

    The block is copied verbatim from ``moment_src``. VASP does not appear to use
    its values as the initial moments — the converged reference CHGCARs carry
    large MAGMOM-like values there yet start their SCF from ~0 µB — it is needed
    for the file to parse.

    Returns True if the block was spliced (False for a total-only output, which
    has no spin block and needs none). Raises ``RuntimeError`` if the output IS
    spin-polarized but ``moment_src`` cannot supply the block, rather than
    writing a CHGCAR whose spin channel VASP would silently ignore.
    """
    chg.write_file(out_path)
    # Applies to every rebuilt CHGCAR, spin-polarized or not: a total-only seed
    # carries augmentation headers too and VASP rejects it just the same.
    rewrite_aug_headers(out_path)
    if not chg.is_spin_polarized:
        return False

    dl = dims_line(chg)
    moment_lines = read_spin_moment_line(moment_src, dl,
                                         chg.structure.num_sites)
    if moment_lines is None:
        raise RuntimeError(
            f"moment_src={moment_src} has no per-ion moment block for grid "
            f"{dl.strip()}; refusing to write a spin-polarized CHGCAR "
            f"whose spin channel VASP would ignore."
        )
    if not splice_spin_moment_line(out_path, dl, moment_lines):
        raise RuntimeError(
            f"could not locate the spin grid header in {out_path}; "
            f"the per-ion moment block was not restored."
        )
    return True


def build_hybrid_chgcar(total_src: str, diff_src: str, out_path: str):
    """Compose a CHGCAR by taking the 'total' channel + total augmentation
    occupancies from ``total_src`` and the 'diff' (magnetization) channel +
    diff augmentation occupancies from ``diff_src``.

    Used by the decomposition runner (not included) to build:
      - variant 3 (sad_diff_hybrid): total_src=converged, diff_src=SAD
      - variant 4 (sad_total_hybrid): total_src=SAD,        diff_src=converged

    Both inputs must come from runs on the same structure with identical
    FFT grid, atom ordering, and POTCAR set — that is guaranteed because
    the SAD extraction reuses the same MP TaskDoc inputs as the converged
    (true) run.

    Raises ``RuntimeError`` if either source lacks a 'diff' channel or if
    grid / augmentation shapes don't line up between sources (would yield
    a silently corrupted CHGCAR otherwise).
    """
    from pymatgen.io.vasp.outputs import Chgcar

    total_chg = Chgcar.from_file(total_src)
    diff_chg = Chgcar.from_file(diff_src)

    for label, chg, path in (
        ("total_src", total_chg, total_src),
        ("diff_src", diff_chg, diff_src),
    ):
        if not (isinstance(chg.data, dict)
                and "total" in chg.data and "diff" in chg.data):
            raise RuntimeError(
                f"{label}={path} has no 'diff' channel; "
                f"hybrid build requires spin-polarized CHGCARs."
            )

    if total_chg.data["total"].shape != diff_chg.data["total"].shape:
        raise RuntimeError(
            f"Grid shape mismatch: total_src={total_chg.data['total'].shape}, "
            f"diff_src={diff_chg.data['total'].shape}"
        )
    if total_chg.structure.num_sites != diff_chg.structure.num_sites:
        raise RuntimeError(
            f"Site count mismatch: total_src={total_chg.structure.num_sites}, "
            f"diff_src={diff_chg.structure.num_sites}"
        )

    # Replace total_chg's 'diff' payload with diff_chg's.
    total_chg.data = {
        "total": total_chg.data["total"],
        "diff": diff_chg.data["diff"],
    }

    aug_t = total_chg.data_aug if isinstance(total_chg.data_aug, dict) else {}
    aug_d = diff_chg.data_aug if isinstance(diff_chg.data_aug, dict) else {}
    merged_aug = {}
    if "total" in aug_t:
        merged_aug["total"] = aug_t["total"]
    if "diff" in aug_d:
        merged_aug["diff"] = aug_d["diff"]
    total_chg.data_aug = merged_aug

    # diff_src is the provenance of the spin channel, so take its moment line.
    write_chgcar(total_chg, out_path, diff_src)


def build_aug_swap_chgcar(grid_src: str, aug_src: str, out_path: str,
                          total_only: bool = False):
    """Compose a CHGCAR by taking pseudo density grid data (`total`+`diff`
    3D arrays) from ``grid_src`` and augmentation occupancies
    (`data_aug['total']`+`data_aug['diff']`) from ``aug_src``.

    Used by experiments/vasp_runner/run_sad_grid_conv_aug_mp.py
    to decompose the contribution of the pseudo charge density versus the
    PAW augmentation occupancies to SCF convergence.

    With ``total_only=True`` only the total channels are used — total pseudo
    grid from ``grid_src`` + ``data_aug['total']`` from ``aug_src``, no diff
    grid or diff aug — the strictly total-only seed for an ISPIN=1 run
    (run_sad_grid_conv_aug_mp.py --ispin1). Neither source then needs a 'diff' channel.

    Both inputs must come from runs on the same structure with identical
    FFT grid, atom ordering, and POTCAR set — guaranteed because the SAD
    extraction reuses the same MP TaskDoc inputs as the converged run.

    Raises ``RuntimeError`` if either source lacks a 'diff' channel, if
    ``aug_src.data_aug`` is missing 'total' or 'diff' keys, or if grid
    shapes / site counts don't match between sources.
    """
    from pymatgen.io.vasp.outputs import Chgcar

    grid_chg = Chgcar.from_file(grid_src)
    aug_chg = Chgcar.from_file(aug_src)

    need = ("total",) if total_only else ("total", "diff")
    for label, chg, path in (
        ("grid_src", grid_chg, grid_src),
        ("aug_src", aug_chg, aug_src),
    ):
        if not (isinstance(chg.data, dict)
                and all(k in chg.data for k in need)):
            raise RuntimeError(
                f"{label}={path} is missing a {need} channel; "
                f"aug-swap build requires spin-polarized CHGCARs "
                f"(total-only ones with total_only=True)."
            )

    if grid_chg.data["total"].shape != aug_chg.data["total"].shape:
        raise RuntimeError(
            f"Grid shape mismatch: grid_src={grid_chg.data['total'].shape}, "
            f"aug_src={aug_chg.data['total'].shape}"
        )
    if grid_chg.structure.num_sites != aug_chg.structure.num_sites:
        raise RuntimeError(
            f"Site count mismatch: grid_src={grid_chg.structure.num_sites}, "
            f"aug_src={aug_chg.structure.num_sites}"
        )

    aug_payload = aug_chg.data_aug if isinstance(aug_chg.data_aug, dict) else {}
    for key in need:
        if key not in aug_payload:
            raise RuntimeError(
                f"aug_src={aug_src} data_aug is missing '{key}' key; "
                f"got keys={sorted(aug_payload.keys())}"
            )

    if total_only:
        grid_chg.data = {"total": grid_chg.data["total"]}
        grid_chg.data_aug = {"total": aug_payload["total"]}
        # is_spin_polarized is fixed at load time from grid_src's channel count;
        # a spin-polarized grid_src leaves it True and write_file would then try
        # to emit the 'diff' grid just dropped.
        grid_chg.is_spin_polarized = False
    else:
        grid_chg.data = {
            "total": grid_chg.data["total"],
            "diff": grid_chg.data["diff"],
        }
        grid_chg.data_aug = {
            "total": aug_payload["total"],
            "diff": aug_payload["diff"],
        }

    write_chgcar(grid_chg, out_path, aug_src)


def build_total_pseudo_grid_chgcar(grid_src: str, aug_src: str, out_path: str,
                                   zero_mag: bool = False,
                                   noise_mag: bool = False,
                                   conv_spin: bool = False,
                                   noise_seed: int = 0,
                                   renorm: bool = False,
                                   spin_grid_src: str = None):
    """Compose a TOTAL CHGCAR: the ``total`` pseudo density grid (3D array) from
    ``grid_src`` and the ``data_aug['total']`` augmentation occupancies from
    ``aug_src``.

    ``spin_grid_src`` (only with ``conv_spin=True``) replaces the converged
    ``diff`` GRID with an ML spin-density prediction — a ``*_spin.npy.lz4``
    file in e/A^3, scaled by the cell volume to CHGCAR units (see
    ``vasp_runner.spin_npy``). The ``diff`` augmentation stays converged.
    ``grid_src`` may be the reference itself, in which case the total grid is
    converged too and the seed differs from the reference only in the diff grid.

    By default (all mag flags ``False``) the result is strictly total-only — no
    ``diff`` grid, no ``diff`` augmentation — so under ISPIN=2 VASP seeds the
    initial magnetization from ``MAGMOM`` (superposition of atomic moments).
    With ``zero_mag=True`` an explicit ZERO magnetization channel is written
    (``diff`` grid and per-atom ``diff`` augmentation both all-zeros), so with
    ICHARG=1 the ISPIN=2 initial moments are 0 and ``MAGMOM`` is not consulted.
    With ``noise_mag=True`` both the ``diff`` grid and the per-atom ``diff``
    augmentation are instead seeded with small random values (log-uniform
    magnitude in [NOISE_MAG_LO, NOISE_MAG_HI] with random sign,
    ``noise_seed``-reproducible) — a symmetry-broken near-zero magnetization
    seed rather than strict zeros.
    With ``conv_spin=True`` the CONVERGED spin-difference channel is carried
    over from ``aug_src``: the ``diff`` grid and per-atom ``diff`` augmentation
    are taken verbatim from the reference CHGCAR, so only the ``total`` pseudo
    grid is the ML prediction while the magnetization channel (grid + aug) and
    the total augmentation are all converged. Requires ``aug_src`` to carry a
    ``diff`` grid channel and a ``data_aug['diff']`` dict.
    ``zero_mag``, ``noise_mag`` and ``conv_spin`` are mutually exclusive.
    With ``renorm=True`` the predicted total grid is first scaled so its
    integrated charge matches ``aug_src``'s (i.e. NELECT); off by default, since
    every run to date used the prediction verbatim.

    Used by experiments/vasp_runner/run_ml_seed_{mp,gnome}.py to seed VASP with an ML-predicted total
    pseudo grid plus the converged (reference) total augmentation occupancies.
    Magnetic structures are skipped upstream, so the total-only channel is
    always sufficient here.

    Both inputs must come from the same structure with identical FFT grid, atom
    ordering, and POTCAR set. Raises ``RuntimeError`` on any grid-shape /
    site-count mismatch or a missing/empty ``aug_src.data_aug['total']`` (silent
    corruption guards mirroring ``build_aug_swap_chgcar``).
    """
    if sum(bool(x) for x in (zero_mag, noise_mag, conv_spin)) > 1:
        raise ValueError(
            "zero_mag, noise_mag and conv_spin are mutually exclusive."
        )
    if spin_grid_src is not None and not conv_spin:
        raise ValueError("spin_grid_src requires conv_spin=True.")

    from pymatgen.io.vasp.outputs import Chgcar

    grid_chg = Chgcar.from_file(grid_src)
    aug_chg = Chgcar.from_file(aug_src)

    for label, chg, path in (
        ("grid_src", grid_chg, grid_src),
        ("aug_src", aug_chg, aug_src),
    ):
        if not (isinstance(chg.data, dict) and "total" in chg.data):
            raise RuntimeError(f"{label}={path} has no 'total' density channel.")

    if grid_chg.data["total"].shape != aug_chg.data["total"].shape:
        raise RuntimeError(
            f"Grid shape mismatch: grid_src={grid_chg.data['total'].shape}, "
            f"aug_src={aug_chg.data['total'].shape}"
        )
    if grid_chg.structure.num_sites != aug_chg.structure.num_sites:
        raise RuntimeError(
            f"Site count mismatch: grid_src={grid_chg.structure.num_sites}, "
            f"aug_src={aug_chg.structure.num_sites}"
        )

    aug_payload = aug_chg.data_aug if isinstance(aug_chg.data_aug, dict) else {}
    aug_total = aug_payload.get("total")
    if not isinstance(aug_total, dict) or not aug_total:
        raise RuntimeError(
            f"aug_src={aug_src} has no non-empty 'total' augmentation dict; "
            f"cannot supply converged augmentation occupancies."
        )

    total = grid_chg.data["total"]
    if renorm:
        total = _renorm_grid_to_ref(total, aug_chg, os.path.basename(out_path))
    if conv_spin:
        import numpy as np
        conv_diff = aug_chg.data.get("diff") if isinstance(aug_chg.data, dict) else None
        if conv_diff is None:
            raise RuntimeError(
                f"aug_src={aug_src} has no 'diff' density channel; "
                f"conv_spin requires a spin-polarized reference CHGCAR."
            )
        if conv_diff.shape != total.shape:
            raise RuntimeError(
                f"Diff grid shape mismatch: aug_src diff={conv_diff.shape}, "
                f"grid_src total={total.shape}"
            )
        aug_diff = aug_payload.get("diff")
        if not isinstance(aug_diff, dict) or not aug_diff:
            raise RuntimeError(
                f"aug_src={aug_src} has no non-empty 'diff' augmentation dict; "
                f"conv_spin requires converged diff augmentation occupancies."
            )
        diff = np.asarray(conv_diff)
        if spin_grid_src is not None:
            from .spin_npy import load_spin_grid, spin_grid_as_chgcar_diff
            diff = spin_grid_as_chgcar_diff(load_spin_grid(spin_grid_src), aug_chg)
            print(f"[spin] {os.path.basename(out_path)}: ML diff grid from "
                  f"{spin_grid_src} (M_pred={diff.sum() / diff.size:.3f}, "
                  f"M_ref={conv_diff.sum() / conv_diff.size:.3f} mu_B)")
        grid_chg.data = {"total": total, "diff": diff}
        grid_chg.data_aug = {"total": aug_total, "diff": aug_diff}
        # is_spin_polarized is fixed at load time from the (total-only) grid
        # source's key count, so it is False here; force it True or write_file
        # skips the 'diff' grid + augmentation we just added.
        grid_chg.is_spin_polarized = True
    elif zero_mag or noise_mag:
        import numpy as np
        if noise_mag:
            rng = np.random.default_rng(noise_seed)

            def _noise(shape, dtype):
                # log-uniform magnitude in [LO, HI] with random sign
                mag = np.exp(rng.uniform(np.log(NOISE_MAG_LO),
                                         np.log(NOISE_MAG_HI), size=shape))
                sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
                return (sign * mag).astype(dtype)

            diff = _noise(total.shape, total.dtype)
            diff_aug = {k: _noise((a := np.asarray(v)).shape, a.dtype)
                        for k, v in aug_total.items()}
        else:
            diff = np.zeros_like(total)
            diff_aug = {k: np.zeros_like(np.asarray(v))
                        for k, v in aug_total.items()}
        grid_chg.data = {"total": total, "diff": diff}
        grid_chg.data_aug = {"total": aug_total, "diff": diff_aug}
        # is_spin_polarized is fixed at load time from the (total-only) grid
        # source's key count, so it is False here; force it True or write_file
        # skips the 'diff' grid + augmentation we just added.
        grid_chg.is_spin_polarized = True
    else:
        grid_chg.data = {"total": total}
        grid_chg.data_aug = {"total": aug_total}
        # is_spin_polarized is fixed at load time from grid_src's channel count;
        # a spin-polarized grid_src leaves it True and write_file would then try
        # to emit the 'diff' grid just dropped.
        grid_chg.is_spin_polarized = False

    # grid_src carries only the total pseudo grid here; the converged aug_src is
    # the only source of the moment line.
    write_chgcar(grid_chg, out_path, aug_src)


def build_total_only_chgcar(src: str, out_path: str, spin: str = "none",
                            noise_seed: int = 0):
    """Keep ``src``'s converged ``total`` pseudo density grid and its
    ``data_aug['total']`` augmentation occupancies, and replace the spin channel
    according to ``spin``:

      - ``"none"``  (default): no ``diff`` grid, no ``diff`` augmentation. Used
        by run_true_init_mp.py --ispin1 to seed the ISPIN=1 Oracle, and by
        --magmom, where ISPIN=2 then builds the initial moments from the INCAR
        MAGMOM instead. Same construction as the total-only build in
        ``build_total_pseudo_grid_chgcar`` (the total-pseudo-grid runner (not included) --ispin1),
        with the converged pseudo grid in place of the ML-predicted one.
      - ``"zero"``: an explicit all-zero ``diff`` grid + per-atom ``diff``
        augmentation, so under ICHARG=1/ISPIN=2 the initial moments are 0 and
        MAGMOM is not consulted.
      - ``"noise"``: a symmetry-broken near-zero ``diff``, log-uniform magnitude
        in [NOISE_MAG_LO, NOISE_MAG_HI] with random sign (``noise_seed``-
        reproducible), i.e. the same scheme as
        ``build_total_pseudo_grid_chgcar(noise_mag=True)``. Here the magnitudes
        are read as µB/Å³ and the ``diff`` GRID is written scaled by the cell
        volume, because CHGCAR grid values are ρ·V. The augmentation occupancies
        are on-site numbers rather than a ρ·V density, so they stay unscaled —
        which makes this grid noise ~V× stronger than the raw-units noise in
        ``build_total_pseudo_grid_chgcar``.

    Raises ``RuntimeError`` if ``src`` has no 'total' density channel or no
    non-empty ``data_aug['total']``.
    """
    if spin not in ("none", "zero", "noise"):
        raise ValueError(f"spin must be 'none', 'zero' or 'noise'; got {spin!r}.")

    from pymatgen.io.vasp.outputs import Chgcar

    chg = Chgcar.from_file(src)
    if not (isinstance(chg.data, dict) and "total" in chg.data):
        raise RuntimeError(f"src={src} has no 'total' density channel.")

    aug_payload = chg.data_aug if isinstance(chg.data_aug, dict) else {}
    aug_total = aug_payload.get("total")
    if not isinstance(aug_total, dict) or not aug_total:
        raise RuntimeError(
            f"src={src} has no non-empty 'total' augmentation dict; "
            f"cannot supply converged augmentation occupancies."
        )

    total = chg.data["total"]

    if spin == "none":
        chg.data = {"total": total}
        chg.data_aug = {"total": aug_total}
        # is_spin_polarized is fixed at load time from the source's channel count;
        # left True, write_file would try to emit the 'diff' grid just dropped.
        chg.is_spin_polarized = False
    else:
        import numpy as np

        if spin == "noise":
            rng = np.random.default_rng(noise_seed)

            def _noise(shape, dtype, scale):
                # log-uniform magnitude in [LO, HI] with random sign
                mag = np.exp(rng.uniform(np.log(NOISE_MAG_LO),
                                         np.log(NOISE_MAG_HI), size=shape))
                sign = np.where(rng.random(shape) < 0.5, -1.0, 1.0)
                return (scale * sign * mag).astype(dtype)

            diff = _noise(total.shape, total.dtype, chg.structure.volume)
            diff_aug = {k: _noise((a := np.asarray(v)).shape, a.dtype, 1.0)
                        for k, v in aug_total.items()}
        else:
            diff = np.zeros_like(total)
            diff_aug = {k: np.zeros_like(np.asarray(v))
                        for k, v in aug_total.items()}

        chg.data = {"total": total, "diff": diff}
        chg.data_aug = {"total": aug_total, "diff": diff_aug}
        # Source is spin-polarized so this is already True, but the seed must
        # carry a 'diff' grid + augmentation regardless of the source's shape.
        chg.is_spin_polarized = True

    write_chgcar(chg, out_path, src)


def aug_dict_from_npz(npz_path: str, structure, ref_aug: dict):
    """Build a pymatgen ``data_aug`` channel dict ``{atom_idx: np.ndarray}``
    from an ML-prediction ``.npz`` (keys ``atomic_numbers``,
    ``aug_sanvito_padded`` (n_atoms, P), ``mask`` (n_atoms, P)).

    ``aug_padded[i][mask[i]]`` is the per-atom augmentation-occupancy block in
    CHGCAR atom order (1-based keys). Validates that the npz describes the same
    structure (atom count + atomic numbers, in order) and that each atom's block
    length matches the reference CHGCAR's ``ref_aug`` block — these are silent
    corruption guards mirroring ``build_aug_swap_chgcar``.

    ``ref_aug`` is the reference channel dict (e.g. ``chg.data_aug['total']``).
    """
    import numpy as np

    with np.load(npz_path) as d:
        Z = np.asarray(d["atomic_numbers"]).astype(int)
        aug = np.asarray(d["aug_sanvito_padded"])
        mask = np.asarray(d["mask"]).astype(bool)
        # AugNet npz files carry `schema_mask` (the full POTCAR block, a
        # contiguous prefix) alongside `mask` (= schema_mask & lmaxmix_mask,
        # with the L > LMAXMIX entries knocked out IN PLACE and therefore
        # scattered). Order/length must be validated against the schema mask;
        # `mask` describes zeroing, not layout.
        schema_mask = (np.asarray(d["schema_mask"]).astype(bool)
                       if "schema_mask" in d else None)

    n = structure.num_sites
    if len(Z) != n:
        raise RuntimeError(
            f"{npz_path}: atom count {len(Z)} != CHGCAR structure sites {n}"
        )
    struct_Z = [site.specie.Z for site in structure]
    if list(Z) != struct_Z:
        raise RuntimeError(
            f"{npz_path}: atomic_numbers {list(Z)} do not match CHGCAR "
            f"structure {struct_Z} (order matters)"
        )
    if aug.shape[0] != n or mask.shape[0] != n:
        raise RuntimeError(
            f"{npz_path}: aug/mask first dim {aug.shape[0]}/{mask.shape[0]} "
            f"!= atom count {n}"
        )

    out = {}
    for i in range(n):
        key = i + 1
        ref_size = np.asarray(ref_aug[key]).size

        # The block length is fixed by the POTCAR projectors; LMAXMIX only
        # zeroes the L > LMAXMIX entries *in place* and never shortens the
        # block. Some prediction dirs nevertheless write `mask` at the
        # LMAXMIX-truncated count (e.g. 83 instead of 138 for a 2s2p2d PAW)
        # while `aug_sanvito_padded` stays in full CHGCAR order, so the mask is
        # a prefix that is short of the block VASP expects. Take the reference
        # length and treat the mask as a lower bound, not as the block.
        order_mask = mask if schema_mask is None else schema_mask
        nz = np.flatnonzero(order_mask[i])
        if nz.size and nz[-1] + 1 != nz.size:
            raise RuntimeError(
                f"{npz_path}: atom {key} mask is not a contiguous prefix "
                f"(last set index {nz[-1]}, {nz.size} set) — the padded vector "
                f"is not in CHGCAR order as assumed."
            )
        if ref_size > aug.shape[1]:
            raise RuntimeError(
                f"{npz_path}: atom {key} reference CHGCAR block length "
                f"{ref_size} exceeds the padded prediction width "
                f"{aug.shape[1]} (POTCAR/grid mismatch?)"
            )
        if nz.size > ref_size:
            raise RuntimeError(
                f"{npz_path}: atom {key} predicted block length {nz.size} "
                f"> reference CHGCAR block length {ref_size} "
                f"(POTCAR/grid mismatch?)"
            )
        out[key] = np.asarray(aug[i][:ref_size], dtype=float)
    return out


def build_ml_grid_ml_aug_chgcar(grid_src: str, aug_ref_src: str,
                                total_aug_npz: str, out_path: str,
                                renorm: bool = False):
    """Compose a strictly TOTAL-only CHGCAR that is ML end to end: the ``total``
    pseudo density grid from the predicted CHGCAR ``grid_src`` and the PAW
    augmentation occupancies from the ML prediction ``total_aug_npz``.

    This is the union of the two existing single-channel builders —
    ``build_total_pseudo_grid_chgcar`` (ML grid, converged augmentation) and
    ``build_ml_aug_chgcar(total_only=True)`` (converged grid, ML augmentation) —
    so nothing in the seed comes from the converged reference except the two
    things that are not predictions: the per-atom augmentation *block lengths*
    (a POTCAR property, used to validate and size the npz) and the per-ion
    moment line the file format requires. There is no ``diff`` grid and no
    ``diff`` augmentation, so under ISPIN=2 VASP builds the initial
    magnetization from the INCAR MAGMOM — which the caller sets to a small
    uniform value.

    ``aug_ref_src`` is the converged reference CHGCAR for the same structure;
    predicted CHGCARs carry no usable augmentation blocks of their own.

    With ``renorm=True`` the predicted total grid is first scaled so its
    integrated charge matches ``aug_ref_src``'s (i.e. NELECT); off by default,
    since every run to date used the prediction verbatim. The predicted
    augmentation blocks are never rescaled.

    Raises ``RuntimeError`` on any grid-shape, site-count or block-length
    mismatch (the same silent-corruption guards as the builders above).
    """
    from pymatgen.io.vasp.outputs import Chgcar

    grid_chg = Chgcar.from_file(grid_src)
    ref_chg = Chgcar.from_file(aug_ref_src)

    for label, chg, path in (
        ("grid_src", grid_chg, grid_src),
        ("aug_ref_src", ref_chg, aug_ref_src),
    ):
        if not (isinstance(chg.data, dict) and "total" in chg.data):
            raise RuntimeError(f"{label}={path} has no 'total' density channel.")

    if grid_chg.data["total"].shape != ref_chg.data["total"].shape:
        raise RuntimeError(
            f"Grid shape mismatch: grid_src={grid_chg.data['total'].shape}, "
            f"aug_ref_src={ref_chg.data['total'].shape}"
        )
    if grid_chg.structure.num_sites != ref_chg.structure.num_sites:
        raise RuntimeError(
            f"Site count mismatch: grid_src={grid_chg.structure.num_sites}, "
            f"aug_ref_src={ref_chg.structure.num_sites}"
        )

    ref_aug = ref_chg.data_aug if isinstance(ref_chg.data_aug, dict) else {}
    if not isinstance(ref_aug.get("total"), dict) or not ref_aug["total"]:
        raise RuntimeError(
            f"aug_ref_src={aug_ref_src} has no non-empty 'total' augmentation "
            f"dict; cannot size or validate the predicted augmentation blocks."
        )

    total_dict = aug_dict_from_npz(total_aug_npz, ref_chg.structure,
                                    ref_aug["total"])

    total = grid_chg.data["total"]
    if renorm:
        total = _renorm_grid_to_ref(total, ref_chg, os.path.basename(out_path))

    grid_chg.data = {"total": total}
    grid_chg.data_aug = {"total": total_dict}
    # is_spin_polarized is fixed at load time from grid_src's channel count; a
    # spin-polarized prediction leaves it True and write_file would then try to
    # emit the 'diff' grid just dropped.
    grid_chg.is_spin_polarized = False

    # grid_src carries only a predicted grid, so the converged reference is the
    # only possible source of the moment line (unused for a total-only seed,
    # but write_chgcar takes it uniformly).
    write_chgcar(grid_chg, out_path, aug_ref_src)


def build_ml_aug_chgcar(grid_src: str, total_aug_npz: str, out_path: str,
                        magnetic_aug_npz: str = None,
                        total_only: bool = False):
    """Compose a CHGCAR by keeping the pseudo density grid (``total`` and, if
    present, ``diff`` 3D arrays) from the converged ``grid_src`` while replacing
    the PAW augmentation occupancies with ML predictions.

    - ``data_aug['total']`` is replaced from ``total_aug_npz``; with
      ``total_aug_npz=None`` the converged total augmentation is kept.
    - If ``magnetic_aug_npz`` is given, ``data_aug['diff']`` is replaced from it
      (the reference must already carry a ``diff`` augmentation channel);
      otherwise the converged ``diff`` augmentation is left untouched.
    At least one of the two npz paths must be given.

    Used by ``run_ml_seed_mp.py`` to seed VASP with the converged pseudo
    charge density plus ML-predicted augmentation occupancies.

    With ``total_only=True`` the ``diff`` pseudo grid and ``diff`` augmentation
    are dropped entirely, leaving a strictly total-only seed: the converged
    total valence density plus the ML total augmentation, with the spin channel
    left to the INCAR ``MAGMOM``. Mutually exclusive with ``magnetic_aug_npz``.

    Raises ``RuntimeError`` on any structure / block-length mismatch between the
    npz and the reference CHGCAR (silent corruption guard).
    """
    if total_only and magnetic_aug_npz is not None:
        raise ValueError("total_only and magnetic_aug_npz are mutually exclusive.")
    if total_aug_npz is None and magnetic_aug_npz is None:
        raise ValueError("at least one of total_aug_npz / magnetic_aug_npz is required.")
    if total_only and total_aug_npz is None:
        raise ValueError("total_only requires total_aug_npz.")

    from pymatgen.io.vasp.outputs import Chgcar

    chg = Chgcar.from_file(grid_src)

    ref_aug = chg.data_aug if isinstance(chg.data_aug, dict) else {}
    if not isinstance(ref_aug.get("total"), dict):
        raise RuntimeError(
            f"grid_src={grid_src} has no parsed 'total' augmentation dict; "
            f"cannot inject ML augmentation occupancies."
        )

    new_aug = dict(ref_aug)
    if total_aug_npz is not None:
        total_dict = aug_dict_from_npz(total_aug_npz, chg.structure, ref_aug["total"])
        new_aug["total"] = total_dict

    if magnetic_aug_npz is not None:
        if not isinstance(ref_aug.get("diff"), dict):
            raise RuntimeError(
                f"grid_src={grid_src} has no 'diff' augmentation channel to "
                f"replace, but magnetic_aug_npz={magnetic_aug_npz} was given."
            )
        new_aug["diff"] = aug_dict_from_npz(
            magnetic_aug_npz, chg.structure, ref_aug["diff"]
        )

    if total_only:
        chg.data = {"total": chg.data["total"]}
        new_aug = {"total": total_dict}
        # is_spin_polarized is fixed at load time from grid_src's channel count;
        # a spin-polarized grid_src leaves it True and write_file would then try
        # to emit the 'diff' grid just dropped.
        chg.is_spin_polarized = False

    chg.data_aug = new_aug
    write_chgcar(chg, out_path, grid_src)
