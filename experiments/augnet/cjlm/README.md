# CJLM MoS2 comparison (Focassio et al. 2024)

Everything for the CJLM MoS2 benchmark lives here. Workflow:

1. `make_cjlm_mos2_split.py` — convert the CJLM MoS2 frames into
   `data_cjlm_mos2/` (one `<id>.chgcar.lz4` per frame) and write
   `data_splits/datasplits_cjlm_mos2.json`.
2. `make_cjlm_dedup_split.py` — collapse duplicate frames (hashes the physics,
   not the bytes; the 10 `ts_` frames are one structure) into
   `datasplits_cjlm_mos2_dedup.json`. Use the dedup split for anything pooled —
   the duplicates leak across train/val/test otherwise.
3. Fine-tune / zero-shot eval with `train.py` / `eval.py` and
   `configs/config_cjlm_ablate10k_*.yaml` (see the top-level README).
4. `cjlm_paper_comparison.py --run-dir <out_dir>` — reproduce the paper's
   Fig 2(b)/table rows (its L<=2 filter and metric conventions) from a
   finished run dir; `score_ablation_l2.py <out_dir>` does the same for every
   `eval_s<step>/` directory of an ablation tree.
