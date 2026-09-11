"""GNOME_2 data-source helpers: 10-hex ids and VASP inputs from the reference runs.

The GNOME test set has no Materials Project API behind it, so there is no
`write_..._inputs_for_id` that can re-derive the exact calculation settings the
way `sources/mp.py` does from a TaskDoc. Instead every structure already has a
completed reference SCF on disk:

    <ref_runs_root>/<gid>.chgcar/run_default/{POSCAR,INCAR,KPOINTS,POTCAR,OUTCAR,…}

and its four input files are copied verbatim into each variant's workdir. That
is what makes the variants comparable: identical ENCUT/EDIFF/KPOINTS/POTCAR by
construction rather than by re-derivation, so a difference in SCF step count is
attributable to the seed alone.

Ids are the 10 lowercase hex characters that prefix every filename in this
dataset (`00139449c9`), not MP's `mp-<digits>`. They are carried in the task
CSV's `GNOME_ID` column; the path regex is only a fallback.
"""
import os
import re
import shutil
import tarfile

import lz4.frame


class MissingGnomeInputsError(RuntimeError):
    """The reference run directory has no usable VASP input set for this id."""


# A POSIX tar header is 512 bytes with the magic 'ustar' at offset 257.
_TAR_MAGIC_OFFSET = 257
_TAR_MAGIC = b"ustar"


def _is_tar(path: str) -> bool:
    with open(path, "rb") as f:
        head = f.read(512)
    return head[_TAR_MAGIC_OFFSET:_TAR_MAGIC_OFFSET + 5] == _TAR_MAGIC


def materialize_chgcar(src: str, dest: str):
    """Decompress `src` into `dest`, unwrapping the tar the GNOME references
    are stored in.

    `gnome_ecd_from_ecd_paper/<gid>.chgcar.lz4` is not an lz4-compressed CHGCAR
    but an lz4-compressed *tar archive* holding a single `CHGCAR` member. VASP
    happens to survive a naive decompression — the 512-byte tar header lands
    where CHGCAR's free-form comment line goes and is ignored — but pymatgen
    cannot parse the NUL bytes, so any variant that has to *read* the reference
    (rather than just hand it to VASP) needs the archive unwrapped. Doing it
    uniformly also keeps the Oracle's seed byte-identical to the reference
    instead of carrying a junk header and a NUL trailer.

    Falls back to a plain copy when `src` is neither compressed nor a tar.
    """
    if src.endswith(".lz4"):
        with lz4.frame.open(src, "rb") as s, open(dest, "wb") as d:
            shutil.copyfileobj(s, d)
    elif os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy(src, dest)

    if not _is_tar(dest):
        return

    tmp = dest + ".untar.tmp"
    with tarfile.open(dest, mode="r") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        if len(members) != 1:
            raise RuntimeError(
                f"{src} is a tar archive with {len(members)} file members; "
                f"expected exactly one CHGCAR."
            )
        extracted = tf.extractfile(members[0])
        if extracted is None:
            raise RuntimeError(f"{src}: could not extract {members[0].name}.")
        with extracted as s, open(tmp, "wb") as d:
            shutil.copyfileobj(s, d)
    os.replace(tmp, dest)


GID_RE = re.compile(r"^([0-9a-f]{10})(?:[._]|$)")

# Copied verbatim from the reference run into every variant workdir. All four
# are required — a partial input set would silently change the calculation.
INPUT_FILES = ("POSCAR", "INCAR", "KPOINTS", "POTCAR")

REF_SUBDIR = "run_default"


def extract_gid_from_path(path):
    """Return the 10-hex GNOME id from a basename, or None."""
    m = GID_RE.match(os.path.basename(path))
    return m.group(1) if m else None


def derive_gid(row, chgcar_path: str) -> str:
    """Pull GNOME_ID from a CSV row, or fall back to a regex on the path."""
    for col in ("GNOME_ID", "gnome_id", "MP_ID"):
        if col in row.index:
            val = str(row[col]).strip()
            if GID_RE.match(val):
                return val
    gid = extract_gid_from_path(chgcar_path)
    if gid is None:
        raise RuntimeError(f"Could not infer GNOME id for CHGCAR_PATH={chgcar_path}")
    return gid


def ref_run_dir(gid: str, ref_runs_root: str) -> str:
    """Directory holding the reference default SCF for `gid`."""
    return os.path.join(ref_runs_root, f"{gid}.chgcar", REF_SUBDIR)


def write_gnome_inputs_for_gid(gid: str, workdir: str, ref_runs_root: str):
    """Copy POSCAR/INCAR/KPOINTS/POTCAR for `gid` into `workdir`.

    Raises ``MissingGnomeInputsError`` if the reference run directory or any of
    the four inputs is absent, which the runners record as a terminal skip.
    """
    src = ref_run_dir(gid, ref_runs_root)
    if not os.path.isdir(src):
        raise MissingGnomeInputsError(f"No reference run directory for {gid}: {src}")

    missing = [n for n in INPUT_FILES if not os.path.isfile(os.path.join(src, n))]
    if missing:
        raise MissingGnomeInputsError(
            f"Reference run {src} is missing {', '.join(missing)}"
        )

    os.makedirs(workdir, exist_ok=True)
    for name in INPUT_FILES:
        shutil.copy(os.path.join(src, name), os.path.join(workdir, name))


TASK_HEADER = ["CHGCAR_PATH", "GNOME_ID"]
