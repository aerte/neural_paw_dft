# paw_stats.py
"""
Per-element, per-irrep-copy standardization of PAW targets.

shift: only on 0e copies (0 for L>0, whose mean is 0 by symmetry); scale: one scalar
per irrep copy. Both are equivariance-safe. Per atomic number Z, with a schema-pooled
fallback for elements with fewer than min_atoms training atoms. Computed on the train
split only and saved to paw_stats.pt.
"""
from __future__ import annotations

import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")  # e3nn constants under torch>=2.6

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .augnet_model import (
    SCHEMA_COPY_SLICES,
    Z_TO_SCHEMA,
    lmaxmix_component_mask,
)
from .run_paw_chgcar import (
    build_training_example_from_chgcar_dual,
    build_stats_example_from_chgcar,
    resolve_lmaxmix,
)

MAX_DIM = 390
CHANNELS = ("total", "mag")


def _safe_print(*args, **kwargs):
    """Swallow stale-file-handle errors on the SLURM .out; a log write must not abort the job."""
    try:
        print(*args, **kwargs)
    except OSError:
        pass


# ----------------------------
# Accumulator
# ----------------------------

def _zeros() -> torch.Tensor:
    return torch.zeros(MAX_DIM, dtype=torch.double)


class _ChannelAccum:
    """Streaming component-wise sums keyed by Z and by schema (pooled fallback).
    Counts are per component, since LMAXMIX masking removes components per atom."""

    def __init__(self):
        self.n_atoms: Dict[int, int] = {}
        self.count_vec: Dict[int, torch.Tensor] = {}
        self.sum_vec: Dict[int, torch.Tensor] = {}
        self.sumsq_vec: Dict[int, torch.Tensor] = {}
        self.n_atoms_s: Dict[int, int] = {}
        self.count_vec_s: Dict[int, torch.Tensor] = {}
        self.sum_vec_s: Dict[int, torch.Tensor] = {}
        self.sumsq_vec_s: Dict[int, torch.Tensor] = {}

    def update(self, target: torch.Tensor, atomic_numbers: torch.Tensor,
               mask: Optional[torch.Tensor] = None):
        """target [n_atoms, MAX_DIM] e3nn order. Masked-out components contribute to
        neither sums nor counts; mask=None counts every component of the schema."""
        target = target.double()
        if mask is None:
            keep = torch.zeros_like(target)
            for z in atomic_numbers.unique().tolist():
                keep[atomic_numbers == z, :Z_TO_SCHEMA[int(z)]] = 1.0
        else:
            keep = mask.double()
            target = target * keep

        sq = target * target
        for z in atomic_numbers.unique().tolist():
            z = int(z)
            rows = atomic_numbers == z
            n = int(rows.sum())
            c = keep[rows].sum(dim=0)
            s = target[rows].sum(dim=0)
            ss = sq[rows].sum(dim=0)

            self.n_atoms[z] = self.n_atoms.get(z, 0) + n
            self.count_vec[z] = self.count_vec.get(z, _zeros()) + c
            self.sum_vec[z] = self.sum_vec.get(z, _zeros()) + s
            self.sumsq_vec[z] = self.sumsq_vec.get(z, _zeros()) + ss

            schema = Z_TO_SCHEMA[z]
            self.n_atoms_s[schema] = self.n_atoms_s.get(schema, 0) + n
            self.count_vec_s[schema] = self.count_vec_s.get(schema, _zeros()) + c
            self.sum_vec_s[schema] = self.sum_vec_s.get(schema, _zeros()) + s
            self.sumsq_vec_s[schema] = self.sumsq_vec_s.get(schema, _zeros()) + ss

    def merge_from(self, other: "_ChannelAccum"):
        for z, n in other.n_atoms.items():
            self.n_atoms[z] = self.n_atoms.get(z, 0) + n
            self.count_vec[z] = self.count_vec.get(z, _zeros()) + other.count_vec[z]
            self.sum_vec[z] = self.sum_vec.get(z, _zeros()) + other.sum_vec[z]
            self.sumsq_vec[z] = self.sumsq_vec.get(z, _zeros()) + other.sumsq_vec[z]
        for schema, n in other.n_atoms_s.items():
            self.n_atoms_s[schema] = self.n_atoms_s.get(schema, 0) + n
            self.count_vec_s[schema] = self.count_vec_s.get(schema, _zeros()) + other.count_vec_s[schema]
            self.sum_vec_s[schema] = self.sum_vec_s.get(schema, _zeros()) + other.sum_vec_s[schema]
            self.sumsq_vec_s[schema] = self.sumsq_vec_s.get(schema, _zeros()) + other.sumsq_vec_s[schema]


def _copy_mu_sig(count_vec: torch.Tensor, sum_vec: torch.Tensor, sumsq_vec: torch.Tensor,
                 sl: slice, L: int, eps: float) -> Optional[tuple]:
    """Per-copy (mu, sig) pooled over all 2L+1 components and atoms; None if no data."""
    n = float(count_vec[sl].sum())
    if n == 0:
        return None
    s = float(sum_vec[sl].sum())
    ss = float(sumsq_vec[sl].sum())
    mu = s / n if L == 0 else 0.0
    var = ss / n - mu * mu
    sig = (max(var, eps * eps)) ** 0.5
    return mu, sig


# ----------------------------
# PAWStats
# ----------------------------

class PAWStats:
    """Dense per-Z tables per channel: shift[channel], scale[channel] of shape
    [119, MAX_DIM]. The min_atoms fallback is baked in, so applying is a row gather."""

    def __init__(self, shift: Dict[str, torch.Tensor], scale: Dict[str, torch.Tensor],
                 meta: Optional[dict] = None):
        self.shift = shift
        self.scale = scale
        self.meta = meta or {}

    def _rows(self, atomic_numbers: torch.Tensor, channel: str, dtype: torch.dtype):
        # Index on CPU, cast, then move: keeps float64 off devices without it (MPS).
        idx = atomic_numbers.detach().to("cpu").long()
        shift = self.shift[channel].index_select(0, idx)
        scale = self.scale[channel].index_select(0, idx)
        shift = shift.to(device=atomic_numbers.device, dtype=dtype)
        scale = scale.to(device=atomic_numbers.device, dtype=dtype)
        return shift, scale

    def standardize(self, y: torch.Tensor, atomic_numbers: torch.Tensor, channel: str) -> torch.Tensor:
        shift, scale = self._rows(atomic_numbers, channel, y.dtype)
        return (y - shift) / scale

    def de_standardize(self, y: torch.Tensor, atomic_numbers: torch.Tensor, channel: str) -> torch.Tensor:
        shift, scale = self._rows(atomic_numbers, channel, y.dtype)
        return y * scale + shift

    def save(self, path: str):
        torch.save({"shift": self.shift, "scale": self.scale, "meta": self.meta}, path)

    @classmethod
    def load(cls, path: str) -> "PAWStats":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        return cls(blob["shift"], blob["scale"], blob.get("meta", {}))


# ----------------------------
# Build from train files
# ----------------------------

_LMAXMIX_MODE = "off"  # worker-side; re-set in _init_worker for spawned workers


def _init_worker(l_channel_overrides: Optional[Dict] = None, lmaxmix_mode: str = "off"):
    torch.set_num_threads(1)
    global _LMAXMIX_MODE
    _LMAXMIX_MODE = lmaxmix_mode
    # Spawned workers do not inherit module state, so re-apply the projector overrides.
    if l_channel_overrides:
        from .paw_basis_transform import set_l_channel_overrides
        set_l_channel_overrides(l_channel_overrides)


def _accum_from(atomic_numbers, targets):
    mask = None
    lm = resolve_lmaxmix(targets.get("lmaxmix"))
    if lm is not None:
        mask = lmaxmix_component_mask(atomic_numbers, lm, max_dim=MAX_DIM)

    accum = {c: _ChannelAccum() for c in CHANNELS}
    accum["total"].update(targets["total_target"], atomic_numbers, mask=mask)
    if bool(targets["has_mag"]):
        accum["mag"].update(targets["mag_target"], atomic_numbers, mask=mask)
    return accum, lm


def _process_one(path: str):
    """Worker: fast header+aug loader first, pymatgen fallback second.
    Returns (ok, skip_msg, (accum, lmaxmix), fell_back)."""
    detect = _LMAXMIX_MODE == "auto"
    try:
        atomic_numbers, targets = build_stats_example_from_chgcar(
            str(path), target_order="e3nn", with_lmaxmix=detect)
        return True, None, _accum_from(atomic_numbers, targets), False
    except Exception as e_fast:
        try:
            batch_stub, targets = build_training_example_from_chgcar_dual(
                str(path), target_order="e3nn", with_lmaxmix=detect)
            return True, None, _accum_from(batch_stub.atomic_numbers, targets), True
        except Exception as e_full:
            return (False,
                    f"{path}: custom loader {type(e_fast).__name__}: {e_fast}; "
                    f"default loader {type(e_full).__name__}: {e_full}",
                    None, False)


def compute_paw_stats(train_files: List[str], min_atoms: int = 50, eps: float = 1e-6,
                      num_workers: Optional[int] = None,
                      l_channel_overrides: Optional[Dict] = None,
                      lmaxmix_mode: str = "auto") -> PAWStats:
    """One parallel streaming pass over the train files. lmaxmix_mode='auto' drops each
    file's L>LMAXMIX components (otherwise the L>=3 scale is under-estimated)."""
    if lmaxmix_mode not in {"auto", "off"}:
        raise ValueError(f"lmaxmix_mode must be auto or off, got {lmaxmix_mode!r}")

    accum = {c: _ChannelAccum() for c in CHANNELS}

    workers = num_workers or os.cpu_count() or 1
    paths = [str(p) for p in train_files]
    total = len(paths)
    _safe_print(f"  computing stats over {total} files with {workers} worker(s) "
                f"(lmaxmix_mode={lmaxmix_mode})", flush=True)

    n_ok = 0
    n_skipped = 0
    n_fallback = 0
    lmaxmix_hist: Dict[Optional[int], int] = {}
    t0 = time.perf_counter()
    with Pool(processes=workers, initializer=_init_worker,
              initargs=(l_channel_overrides, lmaxmix_mode)) as pool:
        for i, (ok, skip_msg, part, fell_back) in enumerate(
            pool.imap_unordered(_process_one, paths, chunksize=8), start=1
        ):
            if ok:
                part, lm = part
                for c in CHANNELS:
                    accum[c].merge_from(part[c])
                lmaxmix_hist[lm] = lmaxmix_hist.get(lm, 0) + 1
                n_ok += 1
                if fell_back:
                    n_fallback += 1
            else:
                n_skipped += 1
                _safe_print(f"  [skip] {skip_msg}", file=sys.stderr)
            if i % 500 == 0 or i == total:
                el = time.perf_counter() - t0
                rate = i / el if el > 0 else 0.0
                _safe_print(f"  {i}/{total} files ({el:.0f}s, {rate:.1f} files/s)", flush=True)

    if n_fallback:
        _safe_print(f"  {n_fallback} file(s) used the default pymatgen loader "
                    f"(custom loader failed on them)", file=sys.stderr)
    if n_skipped:
        _safe_print(f"  skipped {n_skipped} unreadable/invalid file(s); used {n_ok}", file=sys.stderr)

    shift = {c: torch.zeros(119, MAX_DIM, dtype=torch.float64) for c in CHANNELS}
    scale = {c: torch.ones(119, MAX_DIM, dtype=torch.float64) for c in CHANNELS}
    used_fallback = {c: [] for c in CHANNELS}
    # (z, copy_idx) never observed (truncated in every train frame): kept at identity.
    no_data = {c: [] for c in CHANNELS}

    for c in CHANNELS:
        a = accum[c]
        for z, schema in Z_TO_SCHEMA.items():
            per_z = z in a.n_atoms and a.n_atoms[z] >= min_atoms
            if not per_z:
                used_fallback[c].append(z)
            for copy_idx, sl, ir in SCHEMA_COPY_SLICES[schema]:
                if per_z:
                    res = _copy_mu_sig(a.count_vec[z], a.sum_vec[z], a.sumsq_vec[z], sl, ir.l, eps)
                else:
                    if a.n_atoms_s.get(schema, 0) == 0:
                        res = None
                    else:
                        res = _copy_mu_sig(a.count_vec_s[schema], a.sum_vec_s[schema],
                                           a.sumsq_vec_s[schema], sl, ir.l, eps)
                if res is None:
                    if z in a.n_atoms:
                        no_data[c].append((z, copy_idx))
                    continue
                mu, sig = res
                shift[c][z, sl] = mu
                scale[c][z, sl] = sig

    if lmaxmix_mode == "auto":
        _safe_print(f"  LMAXMIX over {n_ok} train files: "
                    + ", ".join(f"{'none' if k is None else k}: {v}" for k, v in
                                sorted(lmaxmix_hist.items(),
                                       key=lambda kv: (kv[0] is not None, kv[0])))
                    + "  ('none' == no truncation visible, nothing masked)")

    for c in CHANNELS:
        if no_data[c]:
            zs = sorted({z for z, _ in no_data[c]})
            _safe_print(f"  [{c}] {len(no_data[c])} irrep copies over {len(zs)} element(s) "
                        f"had no observation (LMAXMIX-truncated in every train frame); "
                        f"left at identity: Z={zs}", file=sys.stderr)

    meta = {
        "min_atoms": min_atoms,
        "eps": eps,
        "lmaxmix_mode": lmaxmix_mode,
        "no_data": no_data,
        "lmaxmix_hist": {("none" if k is None else k): v for k, v in sorted(
            lmaxmix_hist.items(), key=lambda kv: (kv[0] is not None, kv[0]))},
        "n_train_files": len(train_files),
        "n_files_used": n_ok,
        "n_files_skipped": n_skipped,
        "n_files_loader_fallback": n_fallback,
        "n_atoms_total": {c: sum(accum[c].n_atoms.values()) for c in CHANNELS},
        "used_fallback": used_fallback,
    }
    return PAWStats(shift=shift, scale=scale, meta=meta)


def standardize(y: torch.Tensor, atomic_numbers: torch.Tensor, channel: str, stats: PAWStats) -> torch.Tensor:
    return stats.standardize(y, atomic_numbers, channel)


def de_standardize(y: torch.Tensor, atomic_numbers: torch.Tensor, channel: str, stats: PAWStats) -> torch.Tensor:
    return stats.de_standardize(y, atomic_numbers, channel)


# ----------------------------
# File discovery + CLI
# ----------------------------

def material_id_from_file(path: Path) -> str:
    return path.name.split(".chgcar")[0]


def train_ids_from_split(split_file: str) -> set:
    import json
    with open(split_file) as f:
        split = json.load(f)
    return set(split.get("train", []))


def find_chgcar_files(data_dir: str) -> List[Path]:
    d = Path(data_dir)
    files: List[Path] = []
    for pat in ("*.chgcar.lz4", "*.chgcar", "CHGCAR*"):
        files.extend(sorted(d.rglob(pat)))
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def _verify(stats: PAWStats, train_files: List[Path]):
    """Check the standardized distribution (mean ~0 on 0e, RMS ~1) and the round-trip."""
    print("\n=== Verifying standardized statistics ===")
    for channel in CHANNELS:
        acc = _ChannelAccum()
        any_data = False
        detect = stats.meta.get("lmaxmix_mode", "off") == "auto"
        for path in train_files:
            try:
                batch_stub, targets = build_training_example_from_chgcar_dual(
                    str(path), target_order="e3nn", with_lmaxmix=detect)
            except Exception:
                continue
            if channel == "mag" and not bool(targets["has_mag"]):
                continue
            any_data = True
            y = targets[f"{channel}_target"].double()
            z = batch_stub.atomic_numbers
            y_std = stats.standardize(y, z, channel)
            lm = resolve_lmaxmix(targets.get("lmaxmix"))
            mask = lmaxmix_component_mask(z, lm, max_dim=MAX_DIM) if lm is not None else None
            acc.update(y_std, z, mask=mask)
        if not any_data:
            print(f"[{channel}] no data (skipped)")
            continue

        max_abs_mean_0e = 0.0
        max_rms_dev = 0.0
        max_shift_L = 0.0
        min_scale = float("inf")
        # Schema-pooled elements are not per-Z unit/zero; exclude them from the checks.
        fb = set(stats.meta.get("used_fallback", {}).get(channel, []))
        for z, schema in Z_TO_SCHEMA.items():
            if acc.n_atoms.get(z, 0) == 0 or z in fb:
                continue
            for copy_idx, sl, ir in SCHEMA_COPY_SLICES[schema]:
                res = _copy_mu_sig(acc.count_vec[z], acc.sum_vec[z], acc.sumsq_vec[z], sl, 0, 0.0)
                if res is None:
                    continue
                mean, _ = res
                n = float(acc.count_vec[z][sl].sum())
                rms = (float(acc.sumsq_vec[z][sl].sum()) / n) ** 0.5
                # Structural-constant copies (scale floored to eps) are excluded.
                floored = stats.scale[channel][z, sl][0].item() <= 10 * stats.meta["eps"]
                if not floored:
                    if ir.l == 0:
                        max_abs_mean_0e = max(max_abs_mean_0e, abs(mean))
                    max_rms_dev = max(max_rms_dev, abs(rms - 1.0))
                max_shift_L = max(max_shift_L, abs(stats.shift[channel][z, sl].max().item()) if ir.l > 0 else 0.0)
                min_scale = min(min_scale, stats.scale[channel][z, sl].min().item())

        print(f"[{channel}] max|mean(0e)|={max_abs_mean_0e:.2e}  "
              f"max|RMS-1|={max_rms_dev:.2e}  max shift(L>0)={max_shift_L:.2e}  "
              f"min scale={min_scale:.2e}")
        assert max_abs_mean_0e < 1e-6, f"0e mean not ~0: {max_abs_mean_0e}"
        assert max_rms_dev < 1e-4, f"RMS not ~1: 1+{max_rms_dev}"
        assert max_shift_L == 0.0, f"L>0 shift not exactly 0: {max_shift_L}"
        assert min_scale > 0.0, f"non-positive scale: {min_scale}"

    print("\n=== Round-trip standardize -> de_standardize ===")
    for path in train_files:
        try:
            batch_stub, targets = build_training_example_from_chgcar_dual(str(path), target_order="e3nn")
        except Exception:
            continue
        for channel in CHANNELS:
            if channel == "mag" and not bool(targets["has_mag"]):
                continue
            y = targets[f"{channel}_target"]
            z = batch_stub.atomic_numbers
            back = stats.de_standardize(stats.standardize(y, z, channel), z, channel)
            err = (back - y).abs().max().item()
            assert err < 1e-6, f"round-trip {channel} {path}: {err}"
    print("round-trip max error < 1e-6  OK")
    print("\nAll verification checks passed.")


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")  # one file per core; no per-process threading
    ap = argparse.ArgumentParser(description="Compute PAW target standardization stats.")
    ap.add_argument("--data-dir", type=str, default="data",
                    help="Directory of *.chgcar.lz4 train files (all files treated as train).")
    ap.add_argument("--out", type=str, default="paw_stats.pt")
    ap.add_argument("--min-atoms", type=int, default=50)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="Parallel worker processes (default: os.cpu_count()).")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--lmaxmix-mode", choices=["auto", "off"], default="auto",
                    help="auto (default): exclude the L>LMAXMIX components VASP never "
                         "wrote. off: count every component.")
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
        # Exit 3 == "no files matched", so the shell wrapper can fall back only then.
        _safe_print(f"No CHGCAR train files found under {args.data_dir}"
                    + (f" for split {args.split_file}" if args.split_file else ""),
                    file=sys.stderr)
        sys.exit(3)
    _safe_print(f"Using {len(files)} train files")

    split_label = Path(args.split_file).name if args.split_file else args.data_dir
    t_start = time.perf_counter()
    stats = compute_paw_stats(files, min_atoms=args.min_atoms, eps=args.eps,
                              num_workers=args.num_workers,
                              lmaxmix_mode=args.lmaxmix_mode)
    elapsed = time.perf_counter() - t_start
    stats.save(args.out)
    mm, ss = divmod(int(elapsed), 60)
    _safe_print(f"Saved stats to {args.out}")
    _safe_print(f"Elapsed for {split_label}: {mm:d}m{ss:02d}s ({elapsed:.1f}s)")
    _safe_print(f"  n_atoms_total: {stats.meta['n_atoms_total']}")
    _safe_print(f"  fallback (total) elements: {len(stats.meta['used_fallback']['total'])}")

    if args.verify:
        _verify(stats, files)


if __name__ == "__main__":
    main()
