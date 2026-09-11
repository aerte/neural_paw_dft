"""
Assemble the CJLM MoS2 PAW-occupancy dataset (Focassio et al., arXiv:2408.08876)
into the layout train_augnet.py consumes: one <id>.chgcar.lz4 per frame plus a
datasplits_*.json with train/validation/test ID lists.

Source: the zenodo_ml_paw bundle. 30 SCF frames of a 3x3x1 MoS2 monolayer
(9 Mo + 18 S), split 1:1 by the paper into 15 test / 15 train
(ml_models/model_paw/data_ml/{test,train}_frames.npy).

  test        = the paper's 15 test frames, verbatim (comparable to Fig. 2b)
  validation  = 3 frames carved from the paper's 15 train frames, one per
                category (1H AIMD / 1T AIMD / distorted midpoint)
  train       = the remaining 12

The 7 NEB transition-path images (data_scf_img/, ml_neb_final_images/) are the
paper's application set, not part of its train/test, and are not included.

CHGCARs are copied verbatim -- the Mo_sv projector order they use is reconciled
at read time by data.potcar_l_channels in the config (see the README this writes).

Usage:
    python make_cjlm_mos2_split.py [--source DIR] [--data-dir DIR]
                                   [--split-file PATH] [--seed 42] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import lz4.frame
import numpy as np

DEFAULT_SOURCE = Path.home() / "Desktop" / "reference" / "zenodo_ml_paw"

# Per-atom augmentation block sizes, from the paper's POTCARs (Mo_sv, S).
EXPECTED_AUG = {"Mo": 138, "S": 33}
EXPECTED_COUNTS = {"Mo": 9, "S": 18}

CATEGORIES = {"1H": "1H AIMD @ 300K", "1T": "1T AIMD @ 300K", "ts": "1H/1T midpoint, sigma=0.1A"}


def frame_to_scf_dir(frame: str, source: Path) -> Path:
    """'1H_mos2-1072' -> <source>/data_scf_md_snapshots/1H_scf/1072, etc."""
    prefix, num = frame.split("_mos2-")

    if prefix in ("1H", "1T"):
        return source / "data_scf_md_snapshots" / f"{prefix}_scf" / num
    if prefix == "ts":
        return source / "data_scf_middle" / "scf" / num

    raise ValueError(f"Unrecognized frame name: {frame}")


def check_chgcar(path: Path) -> dict:
    """Composition + augmentation block sizes, read from the CHGCAR header/blocks."""
    species: list[str] = []
    counts: list[int] = []
    aug: dict[str, int] = {}

    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i == 5:
                species = line.split()
            elif i == 6:
                counts = [int(x) for x in line.split()]
            elif line.startswith("augmentation occupancies"):
                parts = line.split()
                ion, n = int(parts[2]), int(parts[3])
                # Ions are listed species-block by species-block, in POSCAR order.
                offset = 0
                for sp, c in zip(species, counts):
                    if ion <= offset + c:
                        aug.setdefault(sp, n)
                        if aug[sp] != n:
                            raise ValueError(
                                f"{path}: inconsistent block size for {sp}: {aug[sp]} vs {n}"
                            )
                        break
                    offset += c

    info = {"species": species, "counts": counts, "aug": aug, "n_atoms": sum(counts)}

    for sp, n in EXPECTED_AUG.items():
        if aug.get(sp) != n:
            raise ValueError(f"{path}: expected {n} values/atom for {sp}, got {aug.get(sp)}")
    for sp, c in EXPECTED_COUNTS.items():
        if dict(zip(species, counts)).get(sp) != c:
            raise ValueError(f"{path}: expected {c} {sp} atoms, got {dict(zip(species, counts))}")

    return info


def compress(src: Path, dst: Path, chunk: int = 8 << 20) -> tuple[int, int]:
    """Stream src -> lz4-framed dst. Returns (raw_bytes, compressed_bytes)."""
    raw = 0
    tmp = dst.with_suffix(dst.suffix + ".partial")

    with open(src, "rb") as fin, lz4.frame.open(tmp, mode="wb") as fout:
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            raw += len(buf)
            fout.write(buf)

    tmp.replace(dst)
    return raw, dst.stat().st_size


def pick_validation(train_frames: list[str], seed: int) -> list[str]:
    """One frame per category, deterministic given the seed."""
    rng = random.Random(seed)
    val = []
    for prefix in ("1H", "1T", "ts"):
        candidates = sorted(f for f in train_frames if f.startswith(f"{prefix}_"))
        if not candidates:
            raise RuntimeError(f"No train frames in category {prefix}")
        val.append(rng.choice(candidates))
    return val


README = """# CJLM MoS2 PAW-occupancy dataset

30 VASP SCF frames of a 3x3x1 MoS2 monolayer (9 Mo + 18 S, 27 atoms), from
Focassio et al., *Covariant Jacobi-Legendre expansion for total energy
calculations within the PAW formalism*, arXiv:2408.08876 (zenodo_ml_paw bundle).
CHGCARs copied verbatim, lz4-compressed. Assembled by `make_cjlm_mos2_split.py`.

DFT: VASP/PBE, ENCUT 600 eV, PREC Accurate, EDIFF 1e-6, ISMEAR 0 / SIGMA 0.05,
Gamma-centred 6x6x1, FFT 160x160x320, non-spin-polarised (no 'diff' aug block),
LMAXMIX unset -> default 2.

## Split (`data_splits/{split_name}`)

| set | n | contents |
|---|---|---|
| test | 15 | the paper's exact test frames (`model_paw/data_ml/test_frames.npy`) -- 5x 1H AIMD, 5x 1T AIMD, 5x distorted midpoint |
| validation | 3 | one per category, carved from the paper's 15 train frames (seed {seed}) |
| train | 12 | the remaining paper train frames |

The paper trained on all 15 at a strict 1:1 train:test ratio; the 3 validation
frames are held out of that 15 because MACE training needs a validation set.

## Required config: the Mo POTCAR differs from MP's

These calculations use `PAW_PBE Mo_sv` (ZVAL 14, projector order [0,0,1,1,2,2]).
`mp_potcar_map.py` carries MP's `Mo_pv` (ZVAL 12, order (1,1,2,2,0,0)) because
that is what Materials Project uses. Both are n=138 with l-multiset {0,0,1,1,2,2},
so only the projector ORDER differs -- but reading these CHGCARs with the MP order
puts the Mo blocks in the wrong canonical slots. Set in your config:

```yaml
data:
  potcar_l_channels:
    Mo: [0, 0, 1, 1, 2, 2]
```

S is fully compatible (identical `PAW_PBE S`, ZVAL 6, (0,0,1,1), n=33) and needs
no override.

This fixes the layout only. Mo_sv keeps the 4s2 semicore in valence, so these Mo
occupancies are numerically a different quantity from MP's Mo_pv ones -- do not
mix this dataset with MP data in one split, and do not score an MP-trained model
against this Mo ground truth (see `reference/paw_eval/MACE_COMPARISON_NOTE.md`).

## Not included

The 7 NEB transition-path images (`data_scf_img/`, `ml_neb_final_images/`) are the
paper's application set, outside its train/test split. Re-run this script with a
modified frame list if you want them as an extra held-out probe.

## Frames

{frame_table}
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                   help="zenodo_ml_paw bundle root")
    p.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data_cjlm_mos2",
                   help="output CHGCAR directory")
    p.add_argument("--split-file", type=Path,
                   default=Path(__file__).parent / "data_splits" / "datasplits_cjlm_mos2.json",
                   help="output split JSON")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for the validation carve-out")
    p.add_argument("--dry-run", action="store_true",
                   help="verify sources and print the split without writing CHGCARs")
    args = p.parse_args()

    ml = args.source / "ml_models" / "model_paw" / "data_ml"
    test_frames = [str(x) for x in np.load(ml / "test_frames.npy", allow_pickle=True)]
    paper_train = [str(x) for x in np.load(ml / "train_frames.npy", allow_pickle=True)]

    overlap = set(test_frames) & set(paper_train)
    if overlap:
        raise RuntimeError(f"Paper train/test frames overlap: {sorted(overlap)}")

    val_frames = pick_validation(paper_train, args.seed)
    train_frames = [f for f in paper_train if f not in set(val_frames)]

    split = {
        "train": sorted(train_frames),
        "validation": sorted(val_frames),
        "test": sorted(test_frames),
    }

    print(f"train={len(split['train'])} validation={len(split['validation'])} "
          f"test={len(split['test'])}")
    print(f"  validation frames: {split['validation']}")

    rows = []
    total_raw = total_lz4 = 0

    args.data_dir.mkdir(parents=True, exist_ok=True)

    for subset in ("train", "validation", "test"):
        for frame in split[subset]:
            src = frame_to_scf_dir(frame, args.source) / "CHGCAR"
            if not src.exists():
                raise FileNotFoundError(f"{frame}: missing {src}")

            info = check_chgcar(src)
            dst = args.data_dir / f"{frame}.chgcar.lz4"

            if args.dry_run:
                print(f"  [dry] {subset:10s} {frame:14s} {info['n_atoms']} atoms  {src}")
                rows.append((frame, subset, src))
                continue

            raw, comp = compress(src, dst)
            total_raw += raw
            total_lz4 += comp
            print(f"  {subset:10s} {frame:14s} {raw/1e6:7.1f} MB -> {comp/1e6:6.1f} MB  {dst.name}")
            rows.append((frame, subset, src))

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    args.split_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.split_file, "w") as f:
        json.dump(split, f, indent=2)
    print(f"\nWrote split file: {args.split_file}")

    frame_table = "\n".join(
        ["| frame | set | category | source |", "|---|---|---|---|"]
        + [
            f"| `{fr}` | {sub} | {CATEGORIES[fr.split('_')[0]]} | "
            f"`{src.relative_to(args.source)}` |"
            for fr, sub, src in rows
        ]
    )
    # str.replace, not str.format: the README body contains literal braces.
    readme = (
        README.replace("{split_name}", args.split_file.name)
        .replace("{seed}", str(args.seed))
        .replace("{frame_table}", frame_table)
    )
    (args.data_dir / "README.md").write_text(readme)

    print(f"Wrote {len(rows)} CHGCARs to {args.data_dir} "
          f"({total_raw/1e9:.2f} GB -> {total_lz4/1e9:.2f} GB, "
          f"{100*total_lz4/max(total_raw,1):.0f}%)")
    print(f"Wrote {args.data_dir / 'README.md'}")


if __name__ == "__main__":
    main()
