#!/usr/bin/env python3
"""Per-variant SCF step + wall-time statistics from runs/runs.db.

Reports, for each variant: mean ± std of `scf_total` and `wall` over all OK runs,
and the same split into magnetic vs non-magnetic by |total_mag| >= MAG_THRESHOLD.

Usage:
    python scf_stats.py                                  # all variants
    python scf_stats.py --variants default true_init     # subset
    python scf_stats.py --mag-threshold 0.1              # default 0.5
    python scf_stats.py --runs_db runs/runs.db
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
# Non-magnetic iff |total_mag| < threshold AND max_site |m.tot| < threshold.
DEFAULT_MAG_THRESHOLD = 0.1


# runs.db variant name -> filename prefix for its result CSV / magnetization JSONL.
_VARIANT_PREFIX: dict[str, str] = {
    "default":           "MP_DEFAULT_SCF",
    "true_init":         "MP_TRUE_INIT_SCF",
    "sad_total_hybrid":  "MP_SAD_TOTAL_HYBRID_SCF",
    "sad_diff_hybrid":   "MP_SAD_DIFF_HYBRID_SCF",
    "conv_grid_sad_aug": "MP_CONV_GRID_SAD_AUG",
    "sad_grid_conv_aug": "MP_SAD_GRID_CONV_AUG",
    "ml_aug_1k":         "MP_ML_AUG_1K",
    "ml_aug_10k":        "MP_ML_AUG_10K",
    "ml_aug_50k":        "MP_ML_AUG_50K",
    # AugNet (ICLR): submitted with --tag augnet_<size>, which the runner
    # prefixes to ml_aug_augnet_<size>, while the result files keep the
    # submitter's MP_AUGNET_<SIZE> naming — so the prefix cannot be derived.
    "ml_aug_augnet_1k":  "MP_AUGNET_1K",
    "ml_aug_augnet_10k": "MP_AUGNET_10K",
    "ml_aug_augnet_50k": "MP_AUGNET_50K",
    "ml_aug_augnet_full": "MP_AUGNET_FULL",
    "ml_aug_augnet_full_electrafi_ref_chgnetmag":
                         "MP_AUGNET_FULL_ELECTRAFI_REF_CHGNETMAG",
    "ml_aug_augnet_full_c3n4k_chgnetmag":
                         "MP_AUGNET_FULL_C3N4K_CHGNETMAG",
    "ml_aug_augnet_full_e10_electrafi_ref_chgnetmag":
                         "MP_AUGNET_FULL_E10_ELECTRAFI_REF_CHGNETMAG",
    "ml_aug_augnet_full_e10_c3n4k_chgnetmag":
                         "MP_AUGNET_FULL_E10_C3N4K_CHGNETMAG",
    "ml_aug_augnet_full_electrafi_ref_convspin":
                         "MP_AUGNET_FULL_ELECTRAFI_REF_CONVSPIN",
    "ml_aug_augnet_full_c3n4k_convspin":
                         "MP_AUGNET_FULL_C3N4K_CONVSPIN",
    "ml_aug_augnet_full_cords_convspin":
                         "MP_AUGNET_FULL_CORDS_CONVSPIN",
    "ml_aug_augnet_full_electrafi_ref_lorbit":
                         "MP_AUGNET_FULL_ELECTRAFI_REF_LORBIT",
    "ml_aug_augnet_full_c3n4k_lorbit":
                         "MP_AUGNET_FULL_C3N4K_LORBIT",
    "ml_aug_augnet_1k_electrafi_ref_convspin":
                         "MP_AUGNET_1K_ELECTRAFI_REF_CONVSPIN",
    "ml_aug_augnet_10k_electrafi_ref_convspin":
                         "MP_AUGNET_10K_ELECTRAFI_REF_CONVSPIN",
    "ml_aug_augnet_50k_electrafi_ref_convspin":
                         "MP_AUGNET_50K_ELECTRAFI_REF_CONVSPIN",
    "ml_aug_augnet_full_nmae_full_e15_chgnetmag":
                         "MP_AUGNET_FULL_NMAE_FULL_E15_CHGNETMAG",
    "ml_aug_augnet_full_electrafi_ref_oraclemag":
                         "MP_AUGNET_FULL_ELECTRAFI_REF_ORACLEMAG",
    "ml_aug_augnet_full_c3n4k_oraclemag":
                         "MP_AUGNET_FULL_C3N4K_ORACLEMAG",
    "total_ml_pg":       "MP_TOTAL_ML_PG",
}


def _load_site_mags(variant: str, repo_root: Path) -> dict[str, float]:
    """Return {mp_id: max(|site.tot|)} for the variant's magnetization JSONL files.

    Empty magnetization list (ISPIN=1, no spin polarization) yields 0.0.
    Missing entries mean we have no site-resolved data for that mp_id.
    """
    prefix = _VARIANT_PREFIX.get(variant)
    if not prefix and (variant.startswith("ml_aug_")
                       or variant.startswith("augnet_")
                       or variant.startswith("total_ml_pg")):
        # Suffixed tags (e.g. ml_aug_1k_v2, augnet_1k, total_ml_pg_mae_10k)
        # follow the same MP_<TAG> result naming, so derive the prefix instead
        # of requiring a per-tag entry.
        prefix = f"MP_{variant.upper()}"
    if not prefix:
        return {}
    out: dict[str, float] = {}
    for path in sorted(repo_root.glob(f"{prefix}_*_magnetization.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                mp_id = rec.get("MP_ID")
                if not mp_id:
                    continue
                sites = rec.get("magnetization") or []
                if not isinstance(sites, list):
                    continue
                m = 0.0
                for site in sites:
                    if isinstance(site, dict):
                        tot = site.get("tot")
                        if tot is not None:
                            try:
                                m = max(m, abs(float(tot)))
                            except (TypeError, ValueError):
                                pass
                out[mp_id] = m
    return out


def _mean_std(xs: list[float]) -> tuple[float, float, int]:
    n = len(xs)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = sum(xs) / n
    if n == 1:
        return (mean, float("nan"), 1)
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)  # sample std
    return (mean, math.sqrt(var), n)


def _fmt(mean: float, std: float, n: int, unit: str = "") -> str:
    if n == 0:
        return f"   n=0"
    if math.isnan(std):
        return f"{mean:>10.2f}{unit} ± {'—':>10}   n={n}"
    return f"{mean:>10.2f}{unit} ± {std:>10.2f}{unit}   n={n}"


def variant_rows(
    conn: sqlite3.Connection, variant: str, repo_root: Path,
) -> list[tuple[str, float, float, float, float | None]]:
    """Return (mp_id, scf_total, wall, |total_mag|, max_site_mag) for OK rows.

    `max_site_mag` is None when no site-resolved data exists for the mp_id
    (e.g. the JSONL was lost) — caller can decide how to handle that.
    """
    site_mags = _load_site_mags(variant, repo_root)
    rows = conn.execute(
        "SELECT mp_id, scf_total, wall, total_mag FROM runs "
        "WHERE variant=? AND status='ok'",
        (variant,),
    ).fetchall()
    out = []
    for mp_id, scf, wall, mag in rows:
        if scf is None or wall is None:
            continue
        abs_total = abs(float(mag)) if mag is not None else 0.0
        max_site = site_mags.get(mp_id)  # None if missing
        out.append((mp_id, float(scf), float(wall), abs_total, max_site))
    return out


def list_variants(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT variant FROM runs WHERE status='ok' ORDER BY variant"
    )]


def _is_nonmag(abs_total: float, max_site: float | None, thresh: float) -> bool:
    """Non-magnetic iff |total_mag| < thresh AND max per-site |m.tot| < thresh.

    If `max_site` is None (no site-resolved data on file), fall back to the
    total-only check — this matches the legacy behavior for rows whose JSONL
    is missing, rather than dropping them.
    """
    if abs_total >= thresh:
        return False
    if max_site is None:
        return True
    return max_site < thresh


def magnetic_map(
    conn: sqlite3.Connection,
    repo_root: Path,
    variant: str = "true_init",
    thresh: float = DEFAULT_MAG_THRESHOLD,
) -> dict[str, bool]:
    """Return {mp_id: is_magnetic} for every OK `variant` run, using the
    `_is_nonmag` criterion (|total_mag| < thresh AND max per-site |m.tot| <
    thresh). mp_ids with no OK `variant` row are simply absent from the map —
    callers treat that as "classification unknown".

    Shared by scf_stats' own reporting, the total_ml_pg runner (magnetic skip),
    and compare_magnetic_classifications.py so the definition lives in one place.
    """
    return {
        mp_id: not _is_nonmag(abs_total, max_site, thresh)
        for mp_id, _scf, _wall, abs_total, max_site
        in variant_rows(conn, variant, repo_root)
    }


def _paired_savings(rows, default_map, idx):
    """Mean over per-sample (default - variant)/default * 100 for paired mp_ids.

    `rows` is a list of (mp_id, scf, wall, abs_total, max_site). `idx` selects
    which metric column to compare (1=scf, 2=wall). `default_map` is
    {mp_id: (scf, wall)}. Returns (mean_pct, n_paired) or (nan, 0).
    """
    pct = []
    for r in rows:
        mp_id = r[0]
        base = default_map.get(mp_id)
        if base is None:
            continue
        d = base[idx - 1]
        v = r[idx]
        if d is None or d == 0:
            continue
        pct.append((d - v) / d * 100.0)
    if not pct:
        return (float("nan"), 0)
    return (sum(pct) / len(pct), len(pct))


def _achievable(rows, default_map, oracle_map, idx):
    """Fraction of the Oracle's achievable reduction realized, in %:
    (mean_default - mean_variant) / (mean_default - mean_oracle) * 100.

    Aggregate (ratio of mean differences, not a per-sample ratio — the
    per-sample denominator default-oracle is ~0 whenever default already
    matches the Oracle, which blows up). Means are over mp_ids present in the
    variant, `default_map`, AND `oracle_map`. Oracle itself → 100%, Default → 0%.
    Returns (pct, n_triple) or (nan, 0).
    """
    ds, vs, ts = [], [], []
    for r in rows:
        mp_id = r[0]
        base = default_map.get(mp_id)
        orc = oracle_map.get(mp_id)
        if base is None or orc is None:
            continue
        ds.append(base[idx - 1])
        vs.append(r[idx])
        ts.append(orc[idx - 1])
    n = len(ds)
    if n == 0:
        return (float("nan"), 0)
    md, mv, mt = sum(ds) / n, sum(vs) / n, sum(ts) / n
    denom = md - mt
    if denom == 0:
        return (float("nan"), n)
    return ((md - mv) / denom * 100.0, n)


def _collect(conn: sqlite3.Connection, variants: list[str], mag_thresh: float,
             repo_root: Path, align: bool = False,
             default_variant: str = "default", oracle_variant: str = "true_init"):
    """For each variant, compute (mean,std,n) tuples for SCF and wall, split into
    all / nonmag / mag. Also compute paired %-savings vs `default_variant` per
    bucket (and vs `oracle_variant`).
    Returns {variant: {"scf": {bucket: (m,s,n)},
                        "wall": ...,
                        "savings_scf": {bucket: (mean_pct, n_paired)},
                        "savings_wall": ...}}.

    `default_variant` / `oracle_variant` select which runs act as the Default and
    Oracle baselines (e.g. the ISPIN=1 experiments are scored against
    `default_ispin1` / `true_init_ispin1`).

    With `align`, every variant is restricted to the mp_ids common to all of
    `variants` (plus the baselines), and the magnetic/non-magnetic split comes
    from the `oracle_variant` classification rather than each run's own moments
    — so all rows and all buckets sit on the identical sample.
    """
    default_rows = variant_rows(conn, default_variant, repo_root)
    default_map = {r[0]: (r[1], r[2]) for r in default_rows}
    oracle_rows = variant_rows(conn, oracle_variant, repo_root)
    oracle_map = {r[0]: (r[1], r[2]) for r in oracle_rows}

    common: set[str] | None = None
    mag_ref: dict[str, bool] | None = None
    if align:
        id_sets = [{r[0] for r in variant_rows(conn, v, repo_root)}
                   for v in variants]
        id_sets = [s for s in id_sets if s]
        common = set.intersection(*id_sets) if id_sets else set()
        common &= set(default_map) & set(oracle_map)
        mag_ref = magnetic_map(conn, repo_root, oracle_variant, mag_thresh)

    out: dict[str, dict] = {}
    for v in variants:
        rows = variant_rows(conn, v, repo_root)
        if common is not None:
            rows = [r for r in rows if r[0] in common]
        if not rows:
            continue
        if mag_ref is not None:
            nm = [r for r in rows if not mag_ref.get(r[0], True)]
            mg = [r for r in rows if     mag_ref.get(r[0], True)]
        else:
            nm = [r for r in rows if     _is_nonmag(r[3], r[4], mag_thresh)]
            mg = [r for r in rows if not _is_nonmag(r[3], r[4], mag_thresh)]
        out[v] = {
            "scf": {
                "all":    _mean_std([r[1] for r in rows]),
                "nonmag": _mean_std([r[1] for r in nm]),
                "mag":    _mean_std([r[1] for r in mg]),
            },
            "wall": {
                "all":    _mean_std([r[2] for r in rows]),
                "nonmag": _mean_std([r[2] for r in nm]),
                "mag":    _mean_std([r[2] for r in mg]),
            },
            "savings_scf": {
                "all":    _paired_savings(rows, default_map, idx=1),
                "nonmag": _paired_savings(nm,   default_map, idx=1),
                "mag":    _paired_savings(mg,   default_map, idx=1),
            },
            "savings_wall": {
                "all":    _paired_savings(rows, default_map, idx=2),
                "nonmag": _paired_savings(nm,   default_map, idx=2),
                "mag":    _paired_savings(mg,   default_map, idx=2),
            },
            "savings_scf_oracle": {
                "all":    _paired_savings(rows, oracle_map, idx=1),
                "nonmag": _paired_savings(nm,   oracle_map, idx=1),
                "mag":    _paired_savings(mg,   oracle_map, idx=1),
            },
            "savings_wall_oracle": {
                "all":    _paired_savings(rows, oracle_map, idx=2),
                "nonmag": _paired_savings(nm,   oracle_map, idx=2),
                "mag":    _paired_savings(mg,   oracle_map, idx=2),
            },
            "achievable_scf": {
                "all":    _achievable(rows, default_map, oracle_map, idx=1),
                "nonmag": _achievable(nm,   default_map, oracle_map, idx=1),
                "mag":    _achievable(mg,   default_map, oracle_map, idx=1),
            },
        }
    return out


def _print_tables(data: dict, variants: list[str], align: bool,
                  show_wall_savings: bool) -> None:
    """Print the SCF/Wall mean±std tables and the paired-savings table for the
    variants present in `data`. Baselines are whatever `_collect` used."""
    W_NAME, W_COL = 20, 26

    def _stat_cell(stat, fmt):
        mean, std, n = stat
        if n == 0:
            return "--".center(W_COL)
        return f"{mean:{fmt}} ± {std:{fmt}}".rjust(W_COL)

    def _sv(stat):
        m, n = stat
        return "    --" if n == 0 or math.isnan(m) else f"{m:+5.1f}%"

    def _skip_bucket(v, bucket):
        if align:  # aligned samples make every bucket comparable
            return False
        if bucket in ("all", "nonmag") and v in _MAG_ONLY_VARIANTS:
            return True
        return False

    # --- SCF & Wall stats ---
    for metric, title, fmt in [("scf", "SCF steps", ".2f"),
                                ("wall", "Wall time (s)", ".1f")]:
        hdr = (f"{'Variant':<{W_NAME}} {'All':>{W_COL}} "
               f"{'Non-mag':>{W_COL}} {'Magnetic':>{W_COL}} {'N':>6}")
        print(f"== {title} ==")
        print(f"  {hdr}")
        print(f"  {'-'*len(hdr)}")
        for v in variants:
            if v not in data:
                continue
            d = data[v][metric]
            name = _VARIANT_LABEL.get(v, v)
            cells = []
            for b in ("all", "nonmag", "mag"):
                if _skip_bucket(v, b):
                    cells.append("--".center(W_COL))
                else:
                    cells.append(_stat_cell(d[b], fmt))
            n_shown = (d["all"][2] if align or v not in _MAG_ONLY_VARIANTS
                       else d["mag"][2])
            print(f"  {name:<{W_NAME}} {cells[0]} {cells[1]} {cells[2]} {n_shown:>6}")
        print()

    # --- Savings ---
    if show_wall_savings:
        hdr = (f"{'Variant':<{W_NAME}} {'SCF vs Def':>12} {'SCF vs Orc':>12} "
               f"{'SCF %ach':>12} {'Wall vs Def':>12} {'Wall vs Orc':>12}")
    else:
        hdr = (f"{'Variant':<{W_NAME}} {'SCF vs Def':>12} {'SCF vs Orc':>12} "
               f"{'SCF %ach':>12} {'N':>8}")
    print("== Savings (paired % reduction) ==")
    print("   SCF %ach = (default-variant)/(default-oracle): "
          "% of the Oracle's SCF reduction that the variant realizes.")
    for bucket in ("all", "nonmag", "mag"):
        label = {"all": "All", "nonmag": "Non-mag", "mag": "Magnetic"}[bucket]
        print(f"\n  [{label}]")
        print(f"  {hdr}")
        print(f"  {'-'*len(hdr)}")
        for v in variants:
            if v not in data:
                continue
            if _skip_bucket(v, bucket):
                continue
            name = _VARIANT_LABEL.get(v, v)
            ss = data[v]["savings_scf"]
            so = data[v]["savings_scf_oracle"]
            sa = data[v]["achievable_scf"]
            row = (f"  {name:<{W_NAME}} {_sv(ss[bucket]):>12} {_sv(so[bucket]):>12} "
                   f"{_sv(sa[bucket]):>12}")
            if show_wall_savings:
                sw = data[v]["savings_wall"]
                wo = data[v]["savings_wall_oracle"]
                row += f" {_sv(sw[bucket]):>12} {_sv(wo[bucket]):>12}"
            else:
                row += f" {ss[bucket][1]:>8}"
            print(row)


def report(conn: sqlite3.Connection, variants: list[str], mag_thresh: float,
           repo_root: Path, align: bool = False,
           show_wall_savings: bool = False) -> None:
    """Human-readable report. ISPIN=1 variants (name contains 'ispin1') are split
    into their own section, scored against the ISPIN=1 baselines rather than the
    ISPIN=2 Default/Oracle."""
    ispin1 = [v for v in variants if "ispin1" in v]
    main = [v for v in variants if v not in ispin1]

    print(f"Non-magnetic: |total_mag| < {mag_thresh} µB AND "
          f"max|site m.tot| < {mag_thresh} µB.\n")

    if main:
        data = _collect(conn, main, mag_thresh, repo_root, align=align)
        if align:
            n_common = max((d["scf"]["all"][2] for d in data.values()), default=0)
            print(f"Aligned: all variants restricted to the {n_common} mp_ids common "
                  f"to them and to default/true_init; buckets from `true_init`.\n")
        _print_tables(data, main, align, show_wall_savings)

    if ispin1:
        idata = _collect(conn, ispin1, mag_thresh, repo_root,
                         default_variant="default_ispin1",
                         oracle_variant="true_init_ispin1")
        if idata:
            print("=" * 68)
            print("ISPIN=1 experiments — Default = default_ispin1, "
                  "Oracle = true_init_ispin1.")
            print("=" * 68 + "\n")
            _print_tables(idata, ispin1, align=False,
                          show_wall_savings=show_wall_savings)


_VARIANT_LABEL = {
    "default":           "Default",
    "true_init":         "Oracle",
    "sad_total_hybrid":  "Hyb-T",
    "sad_diff_hybrid":   "Hyb-D",
    "conv_grid_sad_aug": "GA-A",
    "sad_grid_conv_aug": "GA-B",
    "true_init_zeromag":   "Conv-Zero",
    "true_init_noisemag":  "Conv-Noise",
    "true_init_magmom0p1": "Conv-M0.1",
    "ml_aug_1k":         "ML-Aug 1K",
    "ml_aug_10k":        "ML-Aug 10K",
    "ml_aug_50k":        "ML-Aug 50K",
    "default_ispin1":              "Default (S1)",
    "true_init_ispin1":            "Oracle (S1)",
    "total_ml_pg_ispin1_mae_50k":  "PG MAE 50K (S1)",
}

# Section blocks shown in the LaTeX tables, separated by \addlinespace.
# Order inside each block matters; blocks are emitted in this order too.
_LATEX_SECTIONS: list[list[str]] = [
    ["default", "true_init"],
    ["sad_total_hybrid", "sad_diff_hybrid"],
    ["conv_grid_sad_aug", "sad_grid_conv_aug"],
    ["ml_aug_1k", "ml_aug_10k", "ml_aug_50k"],
]

# Component decomposition for the checkmark-style savings table: for each
# variant, is each of the four density components initialized from the converged
# density (True, \cm) or from SAD (False, \sad)?
#   (pseudo-grid total, pseudo-grid spin, augmentation total, augmentation spin)
_COMPONENTS: dict[str, tuple[bool, bool, bool, bool]] = {
    "default":           (False, False, False, False),
    "true_init":         (True,  True,  True,  True),
    "sad_diff_hybrid":   (True,  False, True,  False),  # converged total, SAD spin
    "sad_total_hybrid":  (False, True,  False, True),   # SAD total, converged spin
    "conv_grid_sad_aug": (True,  True,  False, False),  # converged grid, SAD aug
    "sad_grid_conv_aug": (False, False, True,  True),   # SAD grid, converged aug
}

# Row order / blocking for the component-checkmark table (matches _COMPONENTS).
_COMPONENT_SECTIONS: list[list[str]] = [
    ["default", "true_init"],
    ["sad_diff_hybrid", "sad_total_hybrid"],
    ["conv_grid_sad_aug", "sad_grid_conv_aug"],
]

# Variants where only the spin channel differs from the converged density: the
# init reduces to a trivial case for non-magnetic structures, so All/Non-mag
# columns are not meaningful. Only the Magnetic column is reported.
_MAG_ONLY_VARIANTS: set[str] = {"sad_total_hybrid", "sad_diff_hybrid"}


def _cell(stat: tuple[float, float, int], decimals: int) -> str:
    mean, std, n = stat
    if n == 0:
        return "--"
    if math.isnan(std):
        return f"${mean:.{decimals}f}$"
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"


def _savings_cell(stat: tuple[float, int]) -> str:
    mean_pct, n = stat
    if n == 0 or math.isnan(mean_pct):
        return "--"
    color = "green!50!black" if mean_pct >= 0 else "red!70!black"
    return f"\\textcolor{{{color}}}{{${mean_pct:+.1f}\\%$}}"


def _mark(converged: bool) -> str:
    """Component cell for the checkmark table: \\cm converged / \\sad SAD."""
    return r"\cm" if converged else r"\sad"


def _savings_quad(data: dict, v: str, savings_key: str,
                  savings_oracle_key: str) -> tuple[str, str, str, str]:
    """Return the four savings cells (Default non-mag/mag, Oracle non-mag/mag)
    for variant `v`, applying the baseline / self-comparison / mag-only rules
    shared by the component and ML savings tables."""
    sv = data[v][savings_key]
    sv_o = data[v][savings_oracle_key]
    if v == "default":
        return ("--", "--", "--", "--")
    if v == "true_init":  # Oracle: vs-default only, vs-oracle is self
        return (_savings_cell(sv["nonmag"]), _savings_cell(sv["mag"]), "--", "--")
    mag_only = v in _MAG_ONLY_VARIANTS
    d_nm = "--" if mag_only else _savings_cell(sv["nonmag"])
    o_nm = "--" if mag_only else _savings_cell(sv_o["nonmag"])
    return (d_nm, _savings_cell(sv["mag"]), o_nm, _savings_cell(sv_o["mag"]))


def _latex_component_savings_table(data: dict, savings_key: str,
                                   savings_oracle_key: str,
                                   caption: str, label: str) -> str:
    """Checkmark-style savings table: four component columns (\\cm / \\sad) plus
    Diff vs Default and Diff vs Oracle, each split non-magnetic / magnetic."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{ccccrrrr}",
        r"\toprule",
        r"\multicolumn{4}{c}{Components}",
        r"& \multicolumn{2}{c}{Diff vs Default}",
        r"& \multicolumn{2}{c}{Diff vs Oracle} \\",
        r"\cmidrule(lr){1-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"\multicolumn{2}{c}{Pseudo Grid}",
        r"& \multicolumn{2}{c}{Augmentation Occ.}",
        r"& \multicolumn{2}{c}{}",
        r"& \multicolumn{2}{c}{} \\",
        r"\cmidrule(lr){1-2} \cmidrule(lr){3-4}",
        r"$\rho_{\uparrow+\downarrow}$",
        r"& $\rho_{\uparrow-\downarrow}$",
        r"& $\rho_{\uparrow+\downarrow}$",
        r"& $\rho_{\uparrow-\downarrow}$",
        r"& Non-mag. & Mag. & Non-mag. & Mag. \\",
        r"\midrule",
    ]
    sections = [[v for v in s if v in data and v in _COMPONENTS]
                for s in _COMPONENT_SECTIONS]
    sections = [s for s in sections if s]
    for i, section in enumerate(sections):
        if i > 0:
            lines.append(r"\addlinespace[0.6em]")
        for v in section:
            marks = " & ".join(_mark(c) for c in _COMPONENTS[v])
            d_nm, d_mg, o_nm, o_mg = _savings_quad(
                data, v, savings_key, savings_oracle_key)
            lines.append(f"{marks} & {d_nm} & {d_mg} & {o_nm} & {o_mg} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def _latex_ml_savings_table(data: dict, ml_variants: list[str], savings_key: str,
                            savings_oracle_key: str, caption: str, label: str,
                            size_label) -> str:
    """Savings table for a family of ML variants: a leftmost model-size column
    (in place of the component columns) plus Default / Oracle baseline rows."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (r"& \multicolumn{2}{c}{Diff vs Default}"
         r" & \multicolumn{2}{c}{Diff vs Oracle} \\"),
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Model & Non-mag. & Mag. & Non-mag. & Mag. \\",
        r"\midrule",
    ]
    baselines = [v for v in ("default", "true_init") if v in data]
    for v in baselines:
        d_nm, d_mg, o_nm, o_mg = _savings_quad(
            data, v, savings_key, savings_oracle_key)
        name = _VARIANT_LABEL.get(v, v.replace("_", r"\_"))
        lines.append(f"{name} & {d_nm} & {d_mg} & {o_nm} & {o_mg} \\\\")
    ml = [v for v in ml_variants if v in data]
    if baselines and ml:
        lines.append(r"\addlinespace[0.6em]")
    for v in ml:
        d_nm, d_mg, o_nm, o_mg = _savings_quad(
            data, v, savings_key, savings_oracle_key)
        lines.append(f"{size_label(v)} & {d_nm} & {d_mg} & {o_nm} & {o_mg} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def _ml_size_label(v: str) -> str:
    """ml_aug_10k -> 10K (leftmost column label for the ML-Aug table)."""
    return v.replace("ml_aug_", "").upper()


def _pg_size_label(v: str) -> str:
    """total_ml_pg / total_ml_pg_mae_10k -> PG / MAE 10K (pseudo-grid table)."""
    suffix = v.replace("total_ml_pg", "").lstrip("_")
    return suffix.upper().replace("_", " ") if suffix else "PG"


def _latex_stats_table(data: dict, metric: str,
                       caption: str, label: str, decimals: int) -> str:
    """Mean ± std table (no savings)."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (r"Variant & All & Non-magnetic & Magnetic & $N_{\text{ok}}$ \\"),
        r"\midrule",
    ]
    sections = [
        [v for v in section if v in data]
        for section in _LATEX_SECTIONS
    ]
    sections = [s for s in sections if s]
    for i, section in enumerate(sections):
        if i > 0:
            lines.append(r"\addlinespace[0.6em]")
        for v in section:
            d = data[v][metric]
            name = _VARIANT_LABEL.get(v, v.replace("_", r"\_"))
            if v in _MAG_ONLY_VARIANTS:
                n_shown = d["mag"][2]
                lines.append(
                    f"{name} & -- & -- & {_cell(d['mag'], decimals)} & {n_shown} \\\\"
                )
            else:
                n_shown = d["all"][2]
                lines.append(
                    f"{name} & {_cell(d['all'], decimals)} & "
                    f"{_cell(d['nonmag'], decimals)} & {_cell(d['mag'], decimals)} & "
                    f"{n_shown} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def _latex_savings_table(data: dict, savings_key: str, savings_oracle_key: str,
                         caption: str, label: str) -> str:
    """Separate savings table with Default and Oracle as own columns."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        (r" & \multicolumn{2}{c}{Diff vs Default}"
         r" & \multicolumn{2}{c}{Diff vs Oracle} \\"),
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        r"Variant & Non-magnetic & Magnetic & Non-magnetic & Magnetic \\",
        r"\midrule",
    ]
    sections = [
        [v for v in section if v in data]
        for section in _LATEX_SECTIONS
    ]
    sections = [s for s in sections if s]
    for i, section in enumerate(sections):
        if i > 0:
            lines.append(r"\addlinespace[0.6em]")
        for v in section:
            sv = data[v][savings_key]
            sv_o = data[v][savings_oracle_key]
            name = _VARIANT_LABEL.get(v, v.replace("_", r"\_"))
            if v == "default":
                d_nm = d_mg = o_nm = o_mg = "--"
            elif v == "true_init":
                d_nm = _savings_cell(sv["nonmag"])
                d_mg = _savings_cell(sv["mag"])
                o_nm = o_mg = "--"  # self-comparison
            else:
                d_nm = "--" if v in _MAG_ONLY_VARIANTS else _savings_cell(sv["nonmag"])
                d_mg = _savings_cell(sv["mag"])
                o_nm = "--" if v in _MAG_ONLY_VARIANTS else _savings_cell(sv_o["nonmag"])
                o_mg = _savings_cell(sv_o["mag"])
            lines.append(f"{name} & {d_nm} & {d_mg} & {o_nm} & {o_mg} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    return "\n".join(lines)


def report_latex(conn: sqlite3.Connection, variants: list[str], mag_thresh: float,
                 repo_root: Path) -> None:
    data = _collect(conn, variants, mag_thresh, repo_root)
    print("% Non-magnetic = |total_mag| < "
          f"{mag_thresh} muB AND max|site m.tot| < {mag_thresh} muB.")
    print("% Mean +/- sample std over status='ok' rows.")
    print("% Requires: \\usepackage{booktabs}, \\usepackage{xcolor}.")
    print("% Component tables also need \\cm and \\sad, e.g. (pifont):")
    print("%   \\newcommand{\\cm}{\\ding{51}}  \\newcommand{\\sad}{\\ding{55}}")
    print("% Variant labels: Default=MP-default SCF init; Oracle=ICHARG=1 from "
          "converged CHGCAR;")
    print("% Hyb-T = SAD total + converged spin-difference; "
          "Hyb-D = converged total + SAD spin-difference;")
    print("% GA-A = converged grid + SAD augmentation; "
          "GA-B = SAD grid + converged augmentation.\n")
    mag_caption = (
        "split into non-magnetic ($|m_\\text{tot}| < "
        f"{mag_thresh}\\,\\mu_B$ and $\\max_i |m_i| < {mag_thresh}\\,\\mu_B$) "
        "and magnetic"
    )
    savings_caption = (
        "Mean per-sample percentage reduction vs.\\ Default / Oracle, "
        "averaged over paired mp\\_ids within each bucket."
    )
    comp_note = (
        "\\cm{} indicates converged initialization for that component, while "
        "\\sad{} indicates SAD initialization."
    )

    # --- New component-checkmark savings tables ---
    print("% ===== Component savings tables =====")
    print(_latex_component_savings_table(
        data, "savings_scf", "savings_scf_oracle",
        caption=f"Paired SCF savings (\\%), {mag_caption}. {comp_note} {savings_caption}",
        label="tab:scf_savings_components",
    ))
    print()
    print(_latex_component_savings_table(
        data, "savings_wall", "savings_wall_oracle",
        caption=(f"Paired wall-time savings (\\%), {mag_caption}. "
                 f"{comp_note} {savings_caption}"),
        label="tab:wall_savings_components",
    ))
    print()

    # --- ML-Aug savings tables (model size + Default/Oracle baselines) ---
    ml_aug_variants = ["ml_aug_1k", "ml_aug_10k", "ml_aug_50k"]
    if any(v in data for v in ml_aug_variants):
        ml_note = (
            "ML-Aug rows use the converged pseudo grid with ML-predicted "
            "augmentation occupancies; the leftmost column is the "
            "training-set size."
        )
        print("% ===== ML-Aug savings tables =====")
        print(_latex_ml_savings_table(
            data, ml_aug_variants, "savings_scf", "savings_scf_oracle",
            caption=(f"Paired SCF savings (\\%) for ML-augmentation "
                     f"initialization, {mag_caption}. {ml_note} {savings_caption}"),
            label="tab:scf_savings_ml_aug", size_label=_ml_size_label,
        ))
        print()
        print(_latex_ml_savings_table(
            data, ml_aug_variants, "savings_wall", "savings_wall_oracle",
            caption=(f"Paired wall-time savings (\\%) for ML-augmentation "
                     f"initialization, {mag_caption}. {ml_note} {savings_caption}"),
            label="tab:wall_savings_ml_aug", size_label=_ml_size_label,
        ))
        print()

    # --- Pseudo-grid savings tables (same layout; emitted once data exists) ---
    pg_variants = sorted(v for v in data if v.startswith("total_ml_pg"))
    if pg_variants:
        pg_note = (
            "Pseudo-grid rows use an ML-predicted total pseudo grid with "
            "converged augmentation occupancies; the leftmost column is the "
            "density model."
        )
        print("% ===== Pseudo-grid savings tables =====")
        print(_latex_ml_savings_table(
            data, pg_variants, "savings_scf", "savings_scf_oracle",
            caption=(f"Paired SCF savings (\\%) for ML total pseudo-grid "
                     f"initialization, {mag_caption}. {pg_note} {savings_caption}"),
            label="tab:scf_savings_pg", size_label=_pg_size_label,
        ))
        print()
        print(_latex_ml_savings_table(
            data, pg_variants, "savings_wall", "savings_wall_oracle",
            caption=(f"Paired wall-time savings (\\%) for ML total pseudo-grid "
                     f"initialization, {mag_caption}. {pg_note} {savings_caption}"),
            label="tab:wall_savings_pg", size_label=_pg_size_label,
        ))
        print()

    # --- Legacy tables (kept for now) ---
    print("% ===== Legacy tables (previous format) =====")
    print(_latex_stats_table(
        data, "scf",
        caption=("SCF iterations per structure (mean $\\pm$ std) across all "
                 f"successful runs, {mag_caption}."),
        label="tab:scf_steps",
        decimals=2,
    ))
    print()
    print(_latex_savings_table(
        data, "savings_scf", "savings_scf_oracle",
        caption=f"Paired SCF savings (\\%),  {mag_caption}. {savings_caption}",
        label="tab:scf_savings",
    ))
    print()
    print(_latex_stats_table(
        data, "wall",
        caption=("Wall time per structure in seconds (mean $\\pm$ std) across all "
                 f"successful runs, {mag_caption}."),
        label="tab:wall_time",
        decimals=1,
    ))
    print()
    print(_latex_savings_table(
        data, "savings_wall", "savings_wall_oracle",
        caption=f"Paired wall-time savings (\\%), {mag_caption}. {savings_caption}",
        label="tab:wall_savings",
    ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs_db", default="runs/runs.db")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Restrict to these variants (default: all with OK rows).")
    ap.add_argument("--mag-threshold", type=float, default=DEFAULT_MAG_THRESHOLD,
                    help=f"|total_mag| cutoff in µB (default {DEFAULT_MAG_THRESHOLD}).")
    ap.add_argument("--align", action="store_true",
                    help="Restrict every variant to the mp_ids common to all of "
                         "them (and to default/true_init), and bucket "
                         "magnetic/non-magnetic by `true_init` for all rows.")
    ap.add_argument("--wall-savings", action="store_true",
                    help="Show the Wall vs Def / Wall vs Orc savings columns "
                         "instead of the sample-count (N) column.")
    ap.add_argument("--latex", action="store_true",
                    help="Also write LaTeX tables (booktabs) to --output.")
    ap.add_argument("-o", "--output", default='random/latex_tables.txt',
                    help="File to write LaTeX tables to when --latex is set.")
    args = ap.parse_args()

    db = Path(args.runs_db)
    if not db.is_absolute():
        db = REPO_ROOT / db
    if not db.exists():
        print(f"FATAL: runs manifest not found: {db}")
        return 1

    import sys, io
    conn = sqlite3.connect(str(db))
    variants = args.variants or list_variants(conn)

    # The human-readable report always prints to the terminal.
    report(conn, variants, args.mag_threshold, REPO_ROOT, align=args.align,
           show_wall_savings=args.wall_savings)

    # LaTeX tables are optional and go to --output.
    if args.latex:
        buf = io.StringIO()
        sys.stdout = buf
        try:
            report_latex(conn, variants, args.mag_threshold, REPO_ROOT)
        finally:
            sys.stdout = sys.__stdout__
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = REPO_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(buf.getvalue())
        print(f"\nSaved LaTeX tables to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
