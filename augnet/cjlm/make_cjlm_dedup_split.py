"""
Rebuild the CJLM MoS2 split with duplicate structures collapsed, removing the
train/test leakage.

All 10 `ts_` files are byte-identical copies of one geometry (upstream: the Zenodo
bundle ships ten identical CHGCARs in data_scf_middle/scf/{0..9}), and the stock
split spreads them across train (4), validation (1) and test (5). Five test frames
are therefore byte-identical to four training frames.

Where the surviving copy goes is not arbitrary: **the `ts_` structure is the only
LMAXMIX=2 frame in the dataset**, so it must stay in TRAIN or the LMAXMIX mask
becomes a no-op during training and there is nothing left to measure. It is
dropped from validation and test.

Resulting split (all distinct, no leakage):

    train       9   4x 1H + 4x 1T + 1x ts   (the only truncated frame)
    validation  2   1x 1H + 1x 1T
    test       10   5x 1H + 5x 1T           (no truncation -> no test-side mask)

Test is the paper's 1H/1T test frames exactly; the midpoint category leaves the
test set with the leakage, so metrics from this split are NOT comparable with the
15-frame numbers in §3/§8 -- only with each other.

Usage:
    python make_cjlm_dedup_split.py [--data-dir data_cjlm_mos2]
                                    [--split data_splits/datasplits_cjlm_mos2.json]
                                    [--out data_splits/datasplits_cjlm_mos2_dedup.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Bootstrap: repo root (parent of cjlm/) must be importable for the src.*
# imports below when this file is run directly.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def content_hash(path: Path) -> str:
    """Hash the physics, not the file: atomic numbers + augmentation occupancies.

    Compression metadata and byte-level framing are irrelevant here, and this is
    the same quantity the model is trained on.
    """
    from src.run_paw_chgcar import stream_aug_channels_from_file

    zs, channels = stream_aug_channels_from_file(str(path))
    h = hashlib.md5(zs.tobytes())
    for vec in channels["total"]:
        h.update(vec.tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data_cjlm_mos2"))
    ap.add_argument("--split", type=Path,
                    default=Path("data_splits/datasplits_cjlm_mos2.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("data_splits/datasplits_cjlm_mos2_dedup.json"))
    ap.add_argument("--keep-in", default="train", choices=["train", "validation", "test"],
                    help="which split keeps the single surviving copy of a duplicate "
                         "group (default train: the ts_ frame is the only LMAXMIX=2 "
                         "structure, and masking only bites if it is trained on)")
    args = ap.parse_args()

    split = json.loads(args.split.read_text())
    where = {m: k for k, v in split.items() for m in v}

    groups: dict[str, list[str]] = defaultdict(list)
    for mid in sorted(where):
        groups[content_hash(args.data_dir / f"{mid}.chgcar.lz4")].append(mid)

    n_files = len(where)
    print(f"{n_files} files -> {len(groups)} distinct structures")

    drop: set[str] = set()
    for h, members in groups.items():
        if len(members) == 1:
            continue
        # Prefer a member already in --keep-in; otherwise keep the first.
        keep = next((m for m in members if where[m] == args.keep_in), members[0])
        drop.update(m for m in members if m != keep)
        print(f"  duplicate group ({len(members)}): keeping {keep} [{where[keep]}], "
              f"dropping {len(members) - 1} "
              f"({', '.join(sorted(m for m in members if m != keep))})")

    out = {k: [m for m in v if m not in drop] for k, v in split.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"\nwrote {args.out}")
    for k, v in out.items():
        cats = defaultdict(int)
        for m in v:
            cats[m.split("_")[0]] += 1
        print(f"  {k:11s} {len(v):3d}  {dict(cats)}")

    leaked = set(out["train"]) & (set(out["test"]) | set(out["validation"]))
    assert not leaked, leaked
    seen = {}
    for k, v in out.items():
        for m in v:
            h = content_hash(args.data_dir / f"{m}.chgcar.lz4")
            assert h not in seen, f"{m} duplicates {seen[h]} across splits"
            seen[h] = m
    print("\nverified: every remaining structure is distinct, no cross-split leakage")


if __name__ == "__main__":
    main()
