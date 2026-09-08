#!/usr/bin/env python3
"""
pack_free_atom_refs.py

Collapse the free-atom reference calculations produced by free_atom_refs.py
into a single .npz, so downstream code does not have to re-parse ~2M-line
CHGCARs every time it wants a 390-float reference vector.

Layout matches what the model writes for its own predictions
(`export_test_augmentation_predictions` in train_augnet.py): the occupancy
vector is kept in raw CHGCAR order -- which is Sanvito order -- zero-padded to
`--max-dim` (390) slots, with a boolean `schema_mask` marking the `[:schema]`
prefix this element actually stores.

Difference from the per-structure export: there is no `lmaxmix_mask` here.
Deriving one needs the per-element L labels from paw_basis_transform, and this
script is deliberately standalone (numpy + stdlib only, no repo imports, no
pymatgen). The INCAR's LMAXMIX is recorded verbatim instead, so a consumer that
does have the repo on hand can build the mask itself.

Each reference is one free atom, so every array has exactly one atom per row.

Output arrays, all aligned along the first axis (n_refs):

    element        (n,) str        e.g. "Fe"
    variant        (n,) str        "plain" | "ldau_convention"
    Z              (n,) int32
    schema         (n,) int32      coefficients this POTCAR writes
    lmaxmix        (n,) int32      from the run's INCAR; -1 if the key is absent
    aug_total      (n, max_dim) float64   total-charge augmentation occupancies
    aug_mag        (n, max_dim) float64   magnetization channel; zeros if ISPIN=1
    has_mag        (n,) bool
    schema_mask    (n, max_dim) bool
    potcar_titel   (n,) str
    potcar_sha256  (n,) str
    path           (n,) str        source directory, relative to --root

Usage:
    python pack_free_atom_refs.py                      # ./free_atom_refs -> ./free_atom_refs.npz
    python pack_free_atom_refs.py --root DIR --out F.npz
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

# Index == atomic number. Index 0 is a placeholder so SYMBOLS.index(sym) == Z.
SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co "
    "Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te "
    "I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir "
    "Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
    "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def read_incar_lmaxmix(incar_path: Path) -> int:
    """LMAXMIX as written in the INCAR, or -1 if the key is not there."""
    if not incar_path.exists():
        return -1
    for line in incar_path.read_text(errors="replace").splitlines():
        key, sep, val = line.partition("=")
        if sep and key.strip().upper() == "LMAXMIX":
            return int(val.split("#")[0].split("!")[0].strip())
    return -1


def read_poscar_element(poscar_path: Path) -> str:
    """Species symbol from the POSCAR header (line 6). One species per run."""
    lines = poscar_path.read_text(errors="replace").splitlines()
    species = lines[5].split()
    if len(species) != 1:
        raise ValueError(f"{poscar_path}: expected one species, got {species}")
    return species[0]


def potcar_identity(potcar_path: Path):
    raw = potcar_path.read_bytes()
    titel = [
        ln.strip()
        for ln in raw.decode(errors="ignore").splitlines()
        if "TITEL" in ln
    ]
    return ("; ".join(titel), hashlib.sha256(raw).hexdigest())


def stream_aug_blocks(chgcar_path: Path):
    """Augmentation-occupancy blocks from a CHGCAR, as [(ion_index0, values)].

    Streams the handle: the charge grid ahead of the blocks is ~2M lines and
    holding it costs far more than the parse.
    """
    blocks = []
    with open(chgcar_path, "r", errors="replace") as f:
        for line in f:
            if "augmentation occupancies" not in line:
                continue
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"{chgcar_path}: malformed header: {line.strip()}")
            ion_index = int(parts[2]) - 1
            n_values = int(parts[3])

            vals = []
            while len(vals) < n_values:
                nxt = next(f, None)
                if nxt is None:
                    raise ValueError(
                        f"{chgcar_path}: truncated block for ion {ion_index + 1}"
                    )
                vals.extend(float(x) for x in nxt.split())
            blocks.append((ion_index, np.asarray(vals[:n_values], dtype=float)))
    return blocks


def parse_single_atom_chgcar(chgcar_path: Path):
    """(total, mag) occupancy vectors for a one-atom CHGCAR; mag is None if ISPIN=1.

    VASP writes n_atoms total blocks, then n_atoms magnetization blocks for a
    collinear spin-polarized run. Here n_atoms == 1.
    """
    blocks = stream_aug_blocks(chgcar_path)
    if len(blocks) == 1:
        return blocks[0][1], None
    if len(blocks) == 2:
        return blocks[0][1], blocks[1][1]
    raise ValueError(
        f"{chgcar_path}: {len(blocks)} augmentation blocks, expected 1 (ISPIN=1) "
        f"or 2 (ISPIN=2) for a single atom"
    )


def collect(root: Path, max_dim: int):
    records = []
    for chgcar in sorted(root.glob("*/*/CHGCAR")):
        folder = chgcar.parent
        element = read_poscar_element(folder / "POSCAR")
        if element not in SYMBOLS:
            print(f"[warn] {folder}: unknown species {element!r}, skipped")
            continue

        try:
            total, mag = parse_single_atom_chgcar(chgcar)
        except Exception as e:  # half-written CHGCAR from a killed job
            print(f"[warn] {folder}: {e}")
            continue

        if total.shape[0] > max_dim:
            raise ValueError(
                f"{folder}: schema {total.shape[0]} exceeds --max-dim {max_dim}"
            )
        if mag is not None and mag.shape[0] != total.shape[0]:
            raise ValueError(f"{folder}: mag/total length mismatch")

        titel, sha = potcar_identity(folder / "POTCAR")
        records.append(
            dict(
                element=element,
                variant=folder.name,
                Z=SYMBOLS.index(element),
                schema=total.shape[0],
                lmaxmix=read_incar_lmaxmix(folder / "INCAR"),
                total=total,
                mag=mag,
                titel=titel,
                sha256=sha,
                path=str(folder.relative_to(root)),
            )
        )
        print(f"[ok  ] {records[-1]['path']}  schema={total.shape[0]} "
              f"lmaxmix={records[-1]['lmaxmix']} "
              f"mag={'yes' if mag is not None else 'no'}")
    return records


def pack(records, max_dim: int):
    n = len(records)
    aug_total = np.zeros((n, max_dim), dtype=np.float64)
    aug_mag = np.zeros((n, max_dim), dtype=np.float64)
    schema_mask = np.zeros((n, max_dim), dtype=bool)

    for i, r in enumerate(records):
        s = r["schema"]
        aug_total[i, :s] = r["total"]
        schema_mask[i, :s] = True
        if r["mag"] is not None:
            aug_mag[i, :s] = r["mag"]

    return dict(
        element=np.array([r["element"] for r in records]),
        variant=np.array([r["variant"] for r in records]),
        Z=np.array([r["Z"] for r in records], dtype=np.int32),
        schema=np.array([r["schema"] for r in records], dtype=np.int32),
        lmaxmix=np.array([r["lmaxmix"] for r in records], dtype=np.int32),
        aug_total=aug_total,
        aug_mag=aug_mag,
        has_mag=np.array([r["mag"] is not None for r in records], dtype=bool),
        schema_mask=schema_mask,
        potcar_titel=np.array([r["titel"] for r in records]),
        potcar_sha256=np.array([r["sha256"] for r in records]),
        path=np.array([r["path"] for r in records]),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--root", default="free_atom_refs",
                    help="directory holding <element>/<variant>/CHGCAR")
    ap.add_argument("--out", default=None,
                    help="output .npz (default: <root>.npz next to --root)")
    ap.add_argument("--max-dim", type=int, default=390,
                    help="padded width, matching the model's export (default 390)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"error: --root does not exist: {root}")
    out = Path(args.out) if args.out else root.with_suffix(".npz")

    records = collect(root, args.max_dim)
    if not records:
        sys.exit(f"error: no usable CHGCARs under {root}")

    np.savez_compressed(out, **pack(records, args.max_dim))

    n_el = len({r["element"] for r in records})
    print(f"[done] {len(records)} references ({n_el} elements) -> {out}")


if __name__ == "__main__":
    sys.exit(main())
