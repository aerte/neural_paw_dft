"""Score every cjlm_ablate10k eval dir under the paper's L <= 2 filter.

Reuses cjlm_paper_comparison.collect()/metrics() (same truth parsing, same
LMAXMIX handling, same pooled physical-unit protocol) over the zeroshot dir
and every <arm>/eval_s<step>/ of one ablation output tree, emitting one JSON
with (n, R2, MAE, RMSE, MaxAE) per (arm, step). Truth and LMAXMIX parses are
cached per CHGCAR: all 41 evals score the same 10 test frames.

    python cjlm/score_ablation_l2.py augnet_runs/cjlm_ablate10k > l2.json
"""

from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import cjlm.cjlm_paper_comparison as cpc
import src.run_paw_chgcar as rpc
import scripts.analyze_l_channels as alc

# collect() re-parses each frame's CHGCAR for truth and for LMAXMIX detection.
# The frames repeat identically across eval dirs, so cache both by source path
# (the atomic numbers are a function of the frame and go along for the ride).
_read_truth = alc.read_truth
_parse_aug = rpc.parse_aug_from_file


@lru_cache(maxsize=None)
def _truth_cached(source: str):
    d = np.load(_SOURCE_TO_NPZ[source], allow_pickle=True)
    return _read_truth(source, d["atomic_numbers"])


@lru_cache(maxsize=None)
def _aug_cached(source: str):
    d = np.load(_SOURCE_TO_NPZ[source], allow_pickle=True)
    return _parse_aug(source, d["atomic_numbers"])


_SOURCE_TO_NPZ: dict[str, Path] = {}


def _register_sources(run_dir: Path):
    for npz_path in (run_dir / "aug_pred_dir_test_total").glob("*_total_aug.npz"):
        d = np.load(npz_path, allow_pickle=True)
        _SOURCE_TO_NPZ.setdefault(str(d["source_path"]), npz_path)


cpc.read_truth = lambda source, zs: _truth_cached(str(source))
rpc.parse_aug_from_file = lambda source, zs: _aug_cached(str(source))


def score_dir(run_dir: Path) -> list:
    _register_sources(run_dir)
    d = cpc.collect(run_dir)
    sel = d["L"] <= 2
    return list(cpc.metrics(d["pred"][sel], d["truth"][sel]))


def main():
    base = Path(sys.argv[1])
    out = {"zeroshot": {0: score_dir(base / "zeroshot")}}
    for arm in ["ft_full", "ft_head", "ft_freshhead", "scratch"]:
        out[arm] = {}
        for ed in sorted((base / arm).glob("eval_s*")):
            step = int(re.search(r"eval_s(\d+)", ed.name).group(1))
            out[arm][step] = score_dir(ed)
            print(f"{arm} s{step} done", file=sys.stderr)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
