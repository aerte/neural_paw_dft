# run_paw_chgcar.py
from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from pymatgen.io.vasp.outputs import Chgcar
from pymatgen.core.periodic_table import Element
from src.paw_basis_transform import (
    sanvito_to_e3nn_with_basis_padded,
    physical_to_canonical_blocks,
)
from src.augnet_model import Z_TO_SCHEMA
import tempfile
import lz4.frame
from contextlib import contextmanager

# ----------------------------
# CHGCAR augmentation parsing
# ----------------------------
def read_text_maybe_lz4(path: str) -> list[str]:
    path = str(path)

    if path.endswith(".lz4"):
        with lz4.frame.open(path, mode="rt", errors="replace") as f:
            return f.readlines()

    with open(path, "r", errors="replace") as f:
        return f.readlines()


@contextmanager
def materialized_chgcar_path(path: str):
    """pymatgen cannot read .lz4 directly; decompress to a temp file if needed."""
    path = str(path)

    if not path.endswith(".lz4"):
        yield path
        return

    with tempfile.NamedTemporaryFile(suffix=".chgcar", delete=True) as tmp:
        with lz4.frame.open(path, mode="rb") as src:
            tmp.write(src.read())
            tmp.flush()
            yield tmp.name


def parse_aug_channels_from_file(
    chgcar_path: str,
    atomic_numbers: np.ndarray,
    lines: list[str] | None = None,
) -> Dict[str, List[np.ndarray] | None]:
    """Parse the augmentation occupancy blocks: {"total": [...], "mag": [...] or None}.
    ISPIN=2 files hold 2*n_atoms blocks (total, then magnetization)."""
    if lines is None:
        lines = read_text_maybe_lz4(chgcar_path)

    aug_blocks = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("augmentation occupancies"):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed augmentation header: {line}")

            ion_index = int(parts[2]) - 1
            n_values = int(parts[3])

            vals = []
            i += 1

            while i < len(lines) and len(vals) < n_values:
                vals.extend(float(x) for x in lines[i].split())
                i += 1

            vals = np.asarray(vals[:n_values], dtype=float)

            if vals.shape[0] != n_values:
                raise ValueError(
                    f"Expected {n_values} augmentation values for ion {ion_index + 1}, "
                    f"got {vals.shape[0]}"
                )

            aug_blocks.append((ion_index, vals))
            continue

        i += 1

    return _split_aug_channels(aug_blocks, atomic_numbers)


def _split_aug_channels(
    aug_blocks: List[tuple],
    atomic_numbers: np.ndarray,
) -> Dict[str, List[np.ndarray] | None]:
    n_atoms = len(atomic_numbers)

    if len(aug_blocks) == n_atoms:
        total_blocks = aug_blocks
        mag_blocks = None

    elif len(aug_blocks) == 2 * n_atoms:
        total_blocks = aug_blocks[:n_atoms]
        mag_blocks = aug_blocks[n_atoms:2 * n_atoms]

    elif len(aug_blocks) == 4 * n_atoms:
        raise NotImplementedError(
            "Found 4*n_atoms augmentation blocks, likely noncollinear CHGCAR. "
            "Extend parser to return mag_x/mag_y/mag_z."
        )

    else:
        raise ValueError(
            f"Found {len(aug_blocks)} augmentation blocks in raw CHGCAR text, "
            f"but structure has {n_atoms} atoms. Expected n_atoms, 2*n_atoms, or 4*n_atoms."
        )

    def sort_and_validate(blocks, label: str) -> List[np.ndarray]:
        sorted_vecs = [None] * n_atoms

        for ion_index, vals in blocks:
            if ion_index < 0 or ion_index >= n_atoms:
                raise ValueError(f"{label}: invalid augmentation ion index {ion_index + 1}")
            sorted_vecs[ion_index] = vals

        for atom_i, vals in enumerate(sorted_vecs):
            if vals is None:
                raise ValueError(f"{label}: missing augmentation vector for atom {atom_i + 1}")

            z = int(atomic_numbers[atom_i])
            expected = Z_TO_SCHEMA[z]

            if vals.shape[0] != expected:
                raise ValueError(
                    f"{label}: atom {atom_i + 1} Z={z}: CHGCAR has {vals.shape[0]} values, "
                    f"but schema expects {expected}."
                )

        return sorted_vecs

    total = sort_and_validate(total_blocks, "total")
    mag = sort_and_validate(mag_blocks, "mag") if mag_blocks is not None else None

    return {
        "total": total,
        "mag": mag,
    }


def stream_aug_channels_from_file(chgcar_path: str):
    """Read the POSCAR header and the augmentation blocks by streaming the file,
    never holding the density grid. Returns (atomic_numbers, channels)."""
    path = str(chgcar_path)
    opener = (lambda: lz4.frame.open(path, mode="rt", errors="replace")) \
        if path.endswith(".lz4") else (lambda: open(path, "r", errors="replace"))

    header: list[str] = []
    aug_blocks: List[tuple] = []

    with opener() as f:
        for line in f:
            if len(header) < 7:
                header.append(line)
                continue

            if "augmentation occupancies" not in line:
                continue

            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed augmentation header: {line.strip()}")

            ion_index = int(parts[2]) - 1
            n_values = int(parts[3])

            vals: List[float] = []
            while len(vals) < n_values:
                nxt = next(f, None)
                if nxt is None:
                    raise ValueError(
                        f"Truncated augmentation block for ion {ion_index + 1}"
                    )
                vals.extend(float(x) for x in nxt.split())

            aug_blocks.append((ion_index, np.asarray(vals[:n_values], dtype=float)))

    atomic_numbers = _atomic_numbers_from_chgcar_header(header)
    return atomic_numbers, _split_aug_channels(aug_blocks, atomic_numbers)


def parse_aug_from_file(chgcar_path: str, atomic_numbers: np.ndarray) -> List[np.ndarray]:
    return parse_aug_channels_from_file(chgcar_path, atomic_numbers)["total"]


def pack_aug_padded_sanvito(
    aug_vectors: List[np.ndarray],
    atomic_numbers: np.ndarray,
    max_dim: int = 390,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad raw Sanvito-order vectors to [n_atoms, max_dim] plus a schema mask."""
    n_atoms = len(atomic_numbers)
    y = torch.zeros((n_atoms, max_dim), dtype=torch.float32)
    mask = torch.zeros((n_atoms, max_dim), dtype=torch.bool)

    for i, vals in enumerate(aug_vectors):
        schema = Z_TO_SCHEMA[int(atomic_numbers[i])]
        assert len(vals) == schema
        y[i, :schema] = torch.tensor(vals, dtype=torch.float32)
        mask[i, :schema] = True

    return y, mask


def pack_optional_aug_padded_sanvito(
    aug_vectors: List[np.ndarray] | None,
    atomic_numbers: np.ndarray,
    max_dim: int = 390,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Like pack_aug_padded_sanvito; None -> zeros, all-False mask, present=False."""
    n_atoms = len(atomic_numbers)

    if aug_vectors is None:
        y = torch.zeros((n_atoms, max_dim), dtype=torch.float32)
        mask = torch.zeros((n_atoms, max_dim), dtype=torch.bool)
        return y, mask, False

    y, mask = pack_aug_padded_sanvito(aug_vectors, atomic_numbers, max_dim=max_dim)
    return y, mask, True


# ----------------------------
# LMAXMIX recovery
# ----------------------------

def detect_lmaxmix(
    atomic_numbers: np.ndarray,
    aug_vectors: List[np.ndarray],
) -> Dict:
    """Recover LMAXMIX from the blocks: VASP stores a literal 0.0 for L > LMAXMIX, so
    LMAXMIX = max L with a nonzero block on any atom. When nothing is truncated,
    "lmaxmix" is None and only max_L_representable is a lower bound."""
    max_abs: Dict[int, float] = {}
    n_blocks: Dict[int, int] = {}
    z_max_abs: Dict[Tuple[int, int], float] = {}

    for a, z in enumerate(np.asarray(atomic_numbers).tolist()):
        z = int(z)
        v = np.asarray(aug_vectors[a], dtype=float)
        schema = Z_TO_SCHEMA[z]
        if v.shape[0] != schema:
            raise ValueError(f"atom {a}: {v.shape[0]} values, schema expects {schema}")

        for sanvito_slice, _e3nn_slice, L in physical_to_canonical_blocks(z):
            m = float(np.abs(v[sanvito_slice]).max())
            max_abs[L] = max(max_abs.get(L, 0.0), m)
            n_blocks[L] = n_blocks.get(L, 0) + 1
            z_max_abs[(z, L)] = max(z_max_abs.get((z, L), 0.0), m)

    max_L_representable = max(max_abs)
    nonzero_ls = [L for L, m in max_abs.items() if m > 0.0]
    max_L_nonzero = max(nonzero_ls) if nonzero_ls else 0

    truncated = max_L_nonzero < max_L_representable

    return {
        "lmaxmix": max_L_nonzero if truncated else None,
        "max_L_representable": int(max_L_representable),
        "max_L_nonzero": int(max_L_nonzero),
        "truncated": bool(truncated),
        # Real LMAXMIX values are even; odd means the top L was symmetry-dead everywhere.
        "suspicious": bool(truncated and max_L_nonzero % 2 == 1),
        "max_abs_by_L": {L: max_abs[L] for L in sorted(max_abs)},
        "n_blocks_by_L": {L: n_blocks[L] for L in sorted(n_blocks)},
        "z_max_abs": z_max_abs,
    }


def resolve_lmaxmix(detected: Dict | None) -> int | None:
    """The cut to mask at, or None for "mask nothing". Odd inferences are rounded up
    to the next even value (real LMAXMIX settings are even)."""
    if detected is None or not detected.get("truncated"):
        return None

    lm = int(detected["lmaxmix"])
    if lm % 2 == 1:
        lm += 1
    return min(lm, int(detected["max_L_representable"]))


# ----------------------------
# Structure loading
# ----------------------------

class SimpleBatch:
    def __init__(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        cell: torch.Tensor,
        pbc: torch.Tensor,
    ):
        self.atomic_numbers = atomic_numbers
        self.positions = positions
        self.cell = cell
        self.pbc = pbc


def load_structure_from_chgcar(chgcar_path: str):
    with materialized_chgcar_path(chgcar_path) as readable_path:
        chgcar = Chgcar.from_file(readable_path)

    structure = chgcar.structure

    atomic_numbers = np.asarray([site.specie.Z for site in structure], dtype=int)
    positions = np.asarray(structure.cart_coords, dtype=float)
    cell = np.asarray(structure.lattice.matrix, dtype=float)
    pbc = np.asarray([True, True, True], dtype=bool)

    return structure, atomic_numbers, positions, cell, pbc


# ----------------------------
# Formatting predicted augmentation blocks
# ----------------------------

def format_aug_block(atom_index_1based: int, values: np.ndarray, values_per_line: int = 5) -> str:
    lines = [f"augmentation occupancies   {atom_index_1based:4d} {len(values):5d}\n"]

    for i in range(0, len(values), values_per_line):
        chunk = values[i : i + values_per_line]
        lines.append(" ".join(f"{x: .10E}" for x in chunk) + "\n")

    return "".join(lines)


def predicted_aug_blocks_text(
    y_total_sanvito: torch.Tensor,
    atomic_numbers: torch.Tensor,
    y_mag_sanvito: torch.Tensor | None = None,
) -> str:
    """VASP-style augmentation blocks: total, then magnetization if given."""
    blocks = []

    total_np = y_total_sanvito.detach().cpu().numpy()

    for i, z in enumerate(atomic_numbers.tolist()):
        schema = Z_TO_SCHEMA[int(z)]
        vals = total_np[i, :schema]
        blocks.append(format_aug_block(i + 1, vals))

    if y_mag_sanvito is not None:
        mag_np = y_mag_sanvito.detach().cpu().numpy()

        for i, z in enumerate(atomic_numbers.tolist()):
            schema = Z_TO_SCHEMA[int(z)]
            vals = mag_np[i, :schema]
            blocks.append(format_aug_block(i + 1, vals))

    return "".join(blocks)


# ----------------------------
# Training examples
# ----------------------------

def build_training_example_from_chgcar_dual(chgcar_path: str, target_order: str = "e3nn",
                                            with_lmaxmix: bool = False):
    """Returns (batch_stub, targets) with total_target/total_mask, mag_target/mag_mask,
    has_mag, and the raw detect_lmaxmix() result when with_lmaxmix."""
    structure, atomic_numbers_np, positions_np, cell_np, pbc_np = load_structure_from_chgcar(chgcar_path)

    aug = parse_aug_channels_from_file(chgcar_path, atomic_numbers_np)

    total_sanvito, total_mask_sanvito = pack_aug_padded_sanvito(
        aug["total"],
        atomic_numbers_np,
    )

    mag_sanvito, mag_mask_sanvito, has_mag = pack_optional_aug_padded_sanvito(
        aug["mag"],
        atomic_numbers_np,
    )

    atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)

    if target_order == "sanvito":
        total_target, total_mask = total_sanvito, total_mask_sanvito
        mag_target, mag_mask = mag_sanvito, mag_mask_sanvito

    elif target_order == "e3nn":
        total_target, total_mask = sanvito_to_e3nn_with_basis_padded(
            total_sanvito,
            total_mask_sanvito,
            atomic_numbers,
        )

        if has_mag:
            mag_target, mag_mask = sanvito_to_e3nn_with_basis_padded(
                mag_sanvito,
                mag_mask_sanvito,
                atomic_numbers,
            )
        else:
            mag_target, mag_mask = mag_sanvito, mag_mask_sanvito

    else:
        raise ValueError(f"Unknown target_order={target_order}")

    batch_stub = SimpleBatch(
        atomic_numbers=atomic_numbers,
        positions=torch.tensor(positions_np, dtype=torch.float32),
        cell=torch.tensor(cell_np, dtype=torch.float32),
        pbc=torch.tensor(pbc_np, dtype=torch.bool),
    )

    targets = {
        "total_target": total_target,
        "total_mask": total_mask,
        "mag_target": mag_target,
        "mag_mask": mag_mask,
        "has_mag": has_mag,
    }

    if with_lmaxmix:
        targets["lmaxmix"] = detect_lmaxmix(atomic_numbers_np, aug["total"])

    return batch_stub, targets


def build_training_example_from_chgcar(chgcar_path: str, target_order: str = "e3nn"):
    batch_stub, targets = build_training_example_from_chgcar_dual(
        chgcar_path,
        target_order=target_order,
    )
    return batch_stub, targets["total_target"], targets["total_mask"]


def _atomic_numbers_from_chgcar_header(lines: list[str]) -> np.ndarray:
    """Atomic numbers in POSCAR order from the VASP5 header (symbols line 5, counts
    line 6), matching pymatgen's structure exactly without parsing the grid."""
    if len(lines) < 7:
        raise ValueError("CHGCAR header too short to contain a POSCAR block")
    symbols = lines[5].split()
    counts_tok = lines[6].split()
    if not symbols or all(t.lstrip("-").isdigit() for t in symbols):
        raise ValueError("CHGCAR has no element-symbol line (VASP4 format); "
                         "cannot resolve elements from header alone")
    counts = [int(t) for t in counts_tok]
    if len(counts) != len(symbols):
        raise ValueError(f"CHGCAR header symbol/count mismatch: "
                         f"{len(symbols)} symbols vs {len(counts)} counts")
    z_by_symbol = {s: Element(s).Z for s in set(symbols)}
    z_list: List[int] = []
    for sym, n in zip(symbols, counts):
        z_list.extend([z_by_symbol[sym]] * n)
    return np.asarray(z_list, dtype=int)


def build_stats_example_from_chgcar(chgcar_path: str, target_order: str = "e3nn",
                                    with_lmaxmix: bool = False):
    """Stats-only loader: same atomic_numbers / total_target / mag_target / has_mag as
    build_training_example_from_chgcar_dual, but streams the file and skips the
    pymatgen grid parse. No masks; lmaxmix when with_lmaxmix."""
    atomic_numbers_np, aug = stream_aug_channels_from_file(chgcar_path)

    total_sanvito, total_mask_sanvito = pack_aug_padded_sanvito(
        aug["total"],
        atomic_numbers_np,
    )
    mag_sanvito, mag_mask_sanvito, has_mag = pack_optional_aug_padded_sanvito(
        aug["mag"],
        atomic_numbers_np,
    )

    atomic_numbers = torch.tensor(atomic_numbers_np, dtype=torch.long)

    if target_order == "sanvito":
        total_target = total_sanvito
        mag_target = mag_sanvito

    elif target_order == "e3nn":
        total_target, _ = sanvito_to_e3nn_with_basis_padded(
            total_sanvito,
            total_mask_sanvito,
            atomic_numbers,
        )
        if has_mag:
            mag_target, _ = sanvito_to_e3nn_with_basis_padded(
                mag_sanvito,
                mag_mask_sanvito,
                atomic_numbers,
            )
        else:
            mag_target = mag_sanvito

    else:
        raise ValueError(f"Unknown target_order={target_order}")

    targets = {
        "total_target": total_target,
        "mag_target": mag_target,
        "has_mag": has_mag,
    }
    if with_lmaxmix:
        targets["lmaxmix"] = detect_lmaxmix(atomic_numbers_np, aug["total"])

    return atomic_numbers, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chgcar", type=str, required=True)
    args = parser.parse_args()

    batch, y, mask = build_training_example_from_chgcar(args.chgcar, target_order="e3nn")
    print("Atomic numbers:", batch.atomic_numbers.tolist())
    print("Target padded shape:", tuple(y.shape))
    print("Valid coeffs per atom:", mask.sum(dim=1).tolist())
    _, aug_targets = build_training_example_from_chgcar_dual(
        args.chgcar, target_order="e3nn", with_lmaxmix=True)
    det = aug_targets["lmaxmix"]
    print(f"Detected LMAXMIX: {det['lmaxmix']} (truncated={det['truncated']}, "
          f"max representable L={det['max_L_representable']}) "
          f"-> masking at {resolve_lmaxmix(det)}")


if __name__ == "__main__":
    main()
