# neighbor_stats.py
"""
Precompute the TRUE average number of neighbors per r_max over a train split.

Why this exists : the model
divides each atom's summed neighbor-messages by a single fixed scalar
`avg_num_neighbors` (mace/modules/blocks.py). For the r_max sweep axis this scalar
must track the dataset's actual mean neighbor count at each r_max, otherwise the
r_max effect is confounded with a normalization artifact.

The mean is computed EXACTLY as training does it (mace.modules.utils
.compute_avg_num_neighbors): build the same matscipy neighbor list
(get_neighborhood) at the given cutoff, count edges per receiver atom, and average
those per-atom counts across every atom in every train structure.

Speed: the neighbor count needs ONLY positions + cell + pbc -- the "absolute top"
of the CHGCAR (comment, scale, 3 lattice lines, symbols, counts, coord mode, then
n_atoms coordinate lines). We read just those lines and never touch the huge charge
grid or the augmentation blocks, so a full-split pass is minutes, not hours. Runs
one file per worker/core, mirroring python -m src.paw_stats.
"""
from __future__ import annotations

import os
# One BLAS/OMP thread per process: workers are one-file-per-core, so per-process
# threading would oversubscribe the node. Set before numpy import.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import get_neighborhood straight from its file. Going through the `mace` package
# would drag in torch/e3nn (mace/data/__init__.py -> atomic_data), which every
# worker would pay on startup; neighborhood.py itself only needs matscipy + numpy.
# find_spec locates the installed mace-torch package without executing it.
import importlib.util as _ilu
_NB = Path(_ilu.find_spec("mace").origin).parent / "data" / "neighborhood.py"
_spec = _ilu.spec_from_file_location("_paw_neighborhood", _NB)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
get_neighborhood = _mod.get_neighborhood

# Default cutoffs: every distinct r_max the sweep touches.
DEFAULT_R_MAXES = [4.0, 6.0, 8.0, 10.0, 12.0]


def _safe_print(*args, **kwargs):
    """Best-effort logging: a stale SLURM .out handle must never abort the pass."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


# ----------------------------
# Header-only structure reader
# ----------------------------

def _open_text_stream(path: str):
    """Line iterator over a CHGCAR, transparently handling .lz4. Streaming, so a
    caller that stops early only decompresses/reads the first block(s)."""
    if str(path).endswith(".lz4"):
        import lz4.frame
        return lz4.frame.open(str(path), mode="rt", errors="replace")
    return open(str(path), "r", errors="replace")


def read_structure_header(path: str) -> Tuple[np.ndarray, np.ndarray, Tuple[bool, bool, bool]]:
    """
    Read ONLY the POSCAR block at the top of a (VASP5) CHGCAR and return
    (cart_positions [n,3], cell [3,3], pbc) as pymatgen's Chgcar.structure would.

    Layout (fixed line offsets):
      0 comment | 1 scale | 2-4 lattice | 5 element symbols | 6 counts
      7 coord mode (optional "Selective dynamics" line first) | 8.. coordinates

    Only n_atoms coordinate lines are read; the density grid is never touched.
    """
    with _open_text_stream(path) as f:
        f.readline()  # 0: comment
        scale = float(f.readline().split()[0])  # 1: scale factor
        lattice = np.array(
            [[float(x) for x in f.readline().split()[:3]] for _ in range(3)],
            dtype=float,
        )  # 2-4: lattice vectors

        # pymatgen scale convention: >0 multiplies the lattice; <0 is a target
        # volume, so the lattice is scaled to that |volume|.
        if scale < 0:
            vol = abs(np.linalg.det(lattice))
            lattice *= (abs(scale) / vol) ** (1.0 / 3.0)
        else:
            lattice *= scale

        symbols = f.readline().split()  # 5: element symbols (VASP5)
        if not symbols or all(t.lstrip("-").isdigit() for t in symbols):
            # VASP4 header (no symbol line): line 5 was already the counts. We only
            # need n_atoms, so treat it as counts and proceed.
            counts = [int(t) for t in symbols]
        else:
            counts = [int(t) for t in f.readline().split()]  # 6: per-symbol counts
        n_atoms = sum(counts)

        mode = f.readline().strip()  # 7: coord mode (or "Selective dynamics")
        if mode[:1] in ("s", "S"):
            mode = f.readline().strip()
        direct = mode[:1] in ("d", "D")

        coords = np.array(
            [[float(x) for x in f.readline().split()[:3]] for _ in range(n_atoms)],
            dtype=float,
        )

    positions = coords @ lattice if direct else coords * scale
    pbc = (True, True, True)
    return positions, lattice, pbc


# ----------------------------
# Per-file neighbor counting
# ----------------------------

def _counts_for(positions: np.ndarray, cell: np.ndarray, pbc, r_max: float) -> np.ndarray:
    """Per-receiver-atom neighbor counts at `r_max`, matching training exactly.

    get_neighborhood mutates `cell` in place for non-periodic directions, so pass a
    copy. Atoms with zero neighbors do not appear here -- this reproduces
    compute_avg_num_neighbors, which averages torch.unique(receivers) counts and so
    also excludes isolated atoms."""
    edge_index, _, _, _ = get_neighborhood(
        positions=positions, cutoff=float(r_max), pbc=pbc, cell=cell.copy()
    )
    receivers = edge_index[1]
    _, counts = np.unique(receivers, return_counts=True)
    return counts


# Set once per worker so the pool map function stays picklable / arg-light.
_R_MAXES: List[float] = []


def _init_worker(r_maxes: List[float]):
    global _R_MAXES
    _R_MAXES = r_maxes


def _process_one(path: str):
    """Worker: read one header, return per-r_max (sum_counts, n_receivers).

    Returns (ok, skip_msg, partial) where partial maps r_max -> (sum, n)."""
    try:
        positions, cell, pbc = read_structure_header(path)
        part: Dict[float, Tuple[int, int]] = {}
        for r in _R_MAXES:
            counts = _counts_for(positions, cell, pbc, r)
            part[r] = (int(counts.sum()), int(counts.size))
        return True, None, part
    except Exception as e:
        return False, f"{path}: {type(e).__name__}: {e}", None


def compute_neighbor_stats(files: List[str], r_maxes: List[float],
                           num_workers: Optional[int] = None) -> dict:
    """One streaming pass over the train files. Returns a result dict with the
    measured avg_num_neighbors per r_max plus provenance meta."""
    workers = num_workers or os.cpu_count() or 1
    paths = [str(p) for p in files]
    total = len(paths)
    _safe_print(f"  counting neighbors over {total} files at r_max={r_maxes} "
                f"with {workers} worker(s)", flush=True)

    sum_counts = {r: 0 for r in r_maxes}
    n_recv = {r: 0 for r in r_maxes}
    n_ok = n_skipped = 0
    t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init_worker, initargs=(r_maxes,)) as pool:
        for i, (ok, skip_msg, part) in enumerate(
            pool.imap_unordered(_process_one, paths, chunksize=8), start=1
        ):
            if ok:
                for r in r_maxes:
                    s, n = part[r]
                    sum_counts[r] += s
                    n_recv[r] += n
                n_ok += 1
            else:
                n_skipped += 1
                _safe_print(f"  [skip] {skip_msg}", file=sys.stderr)
            if i % 500 == 0 or i == total:
                el = time.perf_counter() - t0
                rate = i / el if el > 0 else 0.0
                _safe_print(f"  {i}/{total} files ({el:.0f}s, {rate:.1f} files/s)", flush=True)

    if n_skipped:
        _safe_print(f"  skipped {n_skipped} unreadable/invalid file(s); used {n_ok}",
                    file=sys.stderr)

    avg = {r: (sum_counts[r] / n_recv[r]) if n_recv[r] else None for r in r_maxes}
    return {
        # String keys so this round-trips cleanly through JSON.
        "avg_num_neighbors": {f"{r:g}": avg[r] for r in r_maxes},
        "meta": {
            "r_maxes": list(r_maxes),
            "n_files_used": n_ok,
            "n_files_skipped": n_skipped,
            "n_atoms_counted": {f"{r:g}": n_recv[r] for r in r_maxes},
        },
    }


# ----------------------------
# File discovery + CLI (mirrors paw_stats.py)
# ----------------------------

def material_id_from_file(path: Path) -> str:
    return path.name.split(".chgcar")[0]


def train_ids_from_split(split_file: str) -> set:
    with open(split_file) as f:
        return set(json.load(f).get("train", []))


def find_chgcar_files(data_dir: str) -> List[Path]:
    d = Path(data_dir)
    files: List[Path] = []
    for pat in ("*.chgcar.lz4", "*.chgcar", "CHGCAR*"):
        files.extend(sorted(d.rglob(pat)))
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description="Precompute avg_num_neighbors per r_max.")
    ap.add_argument("--data-dir", type=str, default="data",
                    help="Directory of *.chgcar.lz4 train files.")
    ap.add_argument("--out", type=str, default="neighbor_stats.json")
    ap.add_argument("--r-maxes", type=float, nargs="+", default=DEFAULT_R_MAXES,
                    help=f"Cutoffs to measure (default: {DEFAULT_R_MAXES}).")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Parallel worker processes (default: os.cpu_count()).")
    ap.add_argument("--split-file", type=str, default=None,
                    help="datasplits_*.json; restrict to its 'train' IDs present in --data-dir.")
    args = ap.parse_args()

    files = find_chgcar_files(args.data_dir)
    if args.split_file:
        train_ids = train_ids_from_split(args.split_file)
        matched = [f for f in files if material_id_from_file(f) in train_ids]
        _safe_print(f"Split '{Path(args.split_file).name}': {len(train_ids)} train IDs, "
                    f"{len(matched)}/{len(files)} present in {args.data_dir}")
        files = matched
    if not files:
        # Exit 3 == "no files matched" (matches get_stats.sh's fallback contract).
        _safe_print(f"No CHGCAR train files found under {args.data_dir}"
                    + (f" for split {args.split_file}" if args.split_file else ""),
                    file=sys.stderr)
        sys.exit(3)
    _safe_print(f"Using {len(files)} train files")

    label = Path(args.split_file).name if args.split_file else args.data_dir
    t_start = time.perf_counter()
    result = compute_neighbor_stats(files, args.r_maxes, num_workers=args.num_workers)
    elapsed = time.perf_counter() - t_start

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    mm, ss = divmod(int(elapsed), 60)
    _safe_print(f"Saved neighbor stats to {args.out}")
    _safe_print(f"Elapsed for {label}: {mm:d}m{ss:02d}s ({elapsed:.1f}s)")
    for r in args.r_maxes:
        v = result["avg_num_neighbors"][f"{r:g}"]
        _safe_print(f"  r_max={r:g}: avg_num_neighbors={v:.2f}" if v is not None
                    else f"  r_max={r:g}: no data")


if __name__ == "__main__":
    main()
