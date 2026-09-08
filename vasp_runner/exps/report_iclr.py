#!/usr/bin/env python3
"""ICLR AugNet results, chgnetmag_results.txt style.

Writes notes/results_iclr.txt from runs.db / runs_gnome.db snapshots. Sections
(non-magnetic / magnetic / total) are pointwise-aligned: every row in a section
is averaged over the SAME ids -- the ids where all FINISHED rows in that section
have an OK run with steps and wall time.

Usage:
    sqlite3 runs/runs.db ".backup /tmp/snap.db"
    sqlite3 runs/runs_gnome.db ".backup /tmp/snap_gnome.db"
    python exps/report_iclr.py [--db /tmp/snap.db] [--out notes/results_iclr.txt]
"""
import argparse
import os
import sqlite3
import statistics
from datetime import date, datetime, timezone

DEFAULT = "default"
ORACLE = "true_init"

# Dropped from every MP population: the ElectraFi-ref + AugNet-full run for
# mp-1219533 "converged" (status ok, 30 steps) to -1193395.6 eV against the
# ~-21.68 eV every other variant reaches, i.e. a diverged SCF that survived the
# convergence check. Left in, its 1.19e6 eV offset swamps mean dE_min for every
# other row in sections 4-5.
EXCLUDE_MP = {"mp-1219533"}

ROWS_NONMAG = [
    ("default",                                     "Default (SAD)"),
    ("true_init",                                   "Oracle (converged seed)"),
    ("ml_aug_augnet_1k",                            "AugNet 1k (conv grid, conv spin)"),
    ("ml_aug_augnet_10k",                           "AugNet 10k (conv grid, conv spin)"),
    ("ml_aug_augnet_50k",                           "AugNet 50k (conv grid, conv spin)"),
    ("ml_aug_augnet_full",                          "AugNet full (conv grid, conv spin)"),
    ("ml_aug_augnet_full_e10",                      "AugNet full/10ep (conv grid, conv spin)"),
    ("ml_aug_augnet_full_electrafi_ref_chgnetmag",  "ElectraFi-ref + AugNet full (CHGNet MAGMOM)"),
    ("ml_aug_augnet_full_c3n4k_chgnetmag",          "charge3net + AugNet full (CHGNet MAGMOM)"),
    ("ml_aug_augnet_full_e10_electrafi_ref_chgnetmag", "ElectraFi-ref + AugNet full/10ep (CHGNet MAGMOM)"),
    ("ml_aug_augnet_full_e10_c3n4k_chgnetmag",      "charge3net + AugNet full/10ep (CHGNet MAGMOM)"),
    ("ml_aug_augnet_full_nmae_full_e15_chgnetmag",  "NMAE full/15ep + AugNet full (CHGNet MAGMOM)"),
    ("total_ml_pg_convspin_nmae_1k",                "ElectraFi NMAE 1k (ML grid, conv aug+spin)"),
    ("total_ml_pg_convspin_nmae_10k",               "ElectraFi NMAE 10k (ML grid, conv aug+spin)"),
    ("total_ml_pg_convspin_nmae_50k",               "ElectraFi NMAE 50k (ML grid, conv aug+spin)"),
    ("total_ml_pg_convspin_nmae_full_e5",           "ElectraFi NMAE full/5ep (ML grid, conv aug+spin)"),
    ("total_ml_pg_convspin_nmae_full_e10",          "ElectraFi NMAE full/10ep (ML grid, conv aug+spin)"),
    ("total_ml_pg_convspin_nmae_full_e15",          "ElectraFi NMAE full/15ep (ML grid, conv aug+spin; nonmag only)"),
    ("ml_aug_augnet_1k_nmae_1k",                    "AugNet 1k + NMAE 1k (both ML, conv spin)"),
    ("ml_aug_augnet_10k_nmae_10k",                  "AugNet 10k + NMAE 10k (both ML, conv spin)"),
    ("ml_aug_augnet_50k_nmae_50k",                  "AugNet 50k + NMAE 50k (both ML, conv spin)"),
    ("ml_aug_augnet_full_nmae_full_e5",             "AugNet full + NMAE full/5ep (both ML, conv spin)"),
    ("ml_aug_augnet_1k_electrafi_ref_convspin",     "AugNet 1k + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_10k_electrafi_ref_convspin",    "AugNet 10k + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_50k_electrafi_ref_convspin",    "AugNet 50k + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_full_electrafi_ref_convspin",   "AugNet full + ElectraFi-ref (both ML, conv spin)"),
]
# The 2026 NMAE conv-spin rows pulled in from results_summary.txt section 2 ran
# BEFORE --include_magnetic existed, so they only ever ran the non-magnetic
# half (magnetic structures are recorded skipped_magnetic). Keeping them out of
# sections 2/3/5 stops them collapsing those aligned subsets to the
# non-magnetic ids.
NONMAG_ONLY = {"ml_aug_augnet_full_e10",   # submitted --nonmag-only
               "total_ml_pg_convspin_nmae_1k",
               "total_ml_pg_convspin_nmae_10k",
               "total_ml_pg_convspin_nmae_50k",
               # matched-budget both-channels-ML rows: submitted --nonmag-only
               "ml_aug_augnet_1k_nmae_1k",
               "ml_aug_augnet_10k_nmae_10k",
               "ml_aug_augnet_50k_nmae_50k",
               "ml_aug_augnet_full_nmae_full_e5",
               "ml_aug_augnet_full_electrafi_ref_convspin",
               "ml_aug_augnet_1k_electrafi_ref_convspin",
               "ml_aug_augnet_10k_electrafi_ref_convspin",
               "ml_aug_augnet_50k_electrafi_ref_convspin"}
ROWS_MAG = [r for r in ROWS_NONMAG if r[0] not in NONMAG_ONLY]

# Spin-channel seeds (2026-08-28): ONE spin component ML, everything else
# converged. Submitted --mag-only, so they exist for the magnetic half only.
# The two total-channel twins are listed for side-by-side reading.
ROWS_SPIN = [
    ("default",                            "Default (SAD)"),
    ("true_init",                          "Oracle (converged seed)"),
    ("total_ml_pg_spingrid_convrest",      "Spin NMAE grid (ML diff grid, rest conv)"),
    ("ml_aug_spinaug_convrest",            "Spin AugNet aug (ML diff aug, rest conv)"),
    ("total_ml_pg_convspin_nmae_full_e5",  "  cf. NMAE full/5ep total grid (rest conv)"),
    ("ml_aug_augnet_full",                 "  cf. AugNet full total aug (rest conv)"),
]

# All four MP components ML (2026-08-28): NMAE full/5ep total grid + AugNet
# full total aug + spin NMAE diff grid + spin AugNet diff aug; full 1943 set.
ROWS_ALLML = [
    ("default",                            "Default (SAD)"),
    ("true_init",                          "Oracle (converged seed)"),
    ("ml_aug_allml_full_e5",               "ALL ML: grid+aug+spin grid+spin aug (full/5ep)"),
    ("ml_aug_augnet_full_nmae_full_e5",    "  cf. AugNet full + NMAE full/5ep (conv spin)"),
    ("ml_aug_allml_electrafi_ref",         "ALL ML: ElectraFi-ref grid + AugNet full + spin models"),
]
ROWS_ALLML_MAG = [r for r in ROWS_ALLML if r[0] not in NONMAG_ONLY]

# Same recipe, previous-generation augmentation models -- paired deltas only.
PAIRS_SAME_RECIPE = [
    ("ml_aug_augnet_1k",   "ml_aug_1k_r6fix",   "AugNet 1k   vs old-aug 1k"),
    ("ml_aug_augnet_10k",  "ml_aug_10k_r6fix",  "AugNet 10k  vs old-aug 10k"),
    ("ml_aug_augnet_50k",  "ml_aug_50k_r6fix",  "AugNet 50k  vs old-aug 50k"),
    ("ml_aug_augnet_full", "ml_aug_full_r6fix", "AugNet full vs old-aug full"),
    ("ml_aug_augnet_full_electrafi_ref_chgnetmag",
     "ml_aug_full_electrafi_ref_chgnetmag", "ElectraFi-ref: AugNet full vs old aug"),
    ("ml_aug_augnet_full_c3n4k_chgnetmag",
     "ml_aug_full_c3n4k_chgnetmag", "charge3net:    AugNet full vs old aug"),
]

GNOME_ROWS_NONMAG = [
    ("gnome_default",                                "Default (SAD)"),
    ("gnome_true_init",                              "Oracle (converged seed)"),
    ("gnome_mlaug_electrafi_chgnetmag_augnetfull",   "ElectraFi-ref + AugNet full (CHGNet MAGMOM)"),
    ("gnome_mlaug_charge3net_chgnetmag_augnetfull",  "charge3net + AugNet full (CHGNet MAGMOM)"),
    ("gnome_mlaug_electrafi_convspin_augnet1k_nmae1k",   "AugNet 1k + NMAE 1k (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet10k_nmae10k", "AugNet 10k + NMAE 10k (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet50k_nmae50k", "AugNet 50k + NMAE 50k (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet1k",      "AugNet 1k + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet10k",     "AugNet 10k + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet50k",     "AugNet 50k + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnetfull",    "AugNet full + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnetfull_nmaefull_e15",
                                                     "AugNet full + NMAE full/15ep (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_chgnetmag_augnetfull_nmaefull_e15",
                                                     "NMAE full/15ep + AugNet full (CHGNet MAGMOM)"),
    ("gnome_mlaug_electrafi_allml",                  "ALL ML: ElectraFi-ref + AugNet full + spin models"),
    ("gnome_augnet_convrest_1k",                     "AugNet 1k (conv grid, conv spin)"),
    ("gnome_augnet_convrest_10k",                    "AugNet 10k (conv grid, conv spin)"),
    ("gnome_augnet_convrest_50k",                    "AugNet 50k (conv grid, conv spin)"),
    ("gnome_augnet_convrest_full",                   "AugNet full (conv grid, conv spin)"),
    ("gnome_electrafi_convrest_1k",                  "ElectraFi 1k (ML grid, conv aug+spin)"),
    ("gnome_electrafi_convrest_10k",                 "ElectraFi 10k (ML grid, conv aug+spin)"),
    ("gnome_electrafi_convrest_50k",                 "ElectraFi 50k (ML grid, conv aug+spin)"),
]
# The matched-budget conv-spin rows ran the non-magnetic 1254 only.
GNOME_NONMAG_ONLY = {"gnome_mlaug_electrafi_convspin_augnetfull",
                     "gnome_mlaug_electrafi_convspin_augnet1k",
                     "gnome_mlaug_electrafi_convspin_augnet10k",
                     "gnome_mlaug_electrafi_convspin_augnet50k",
                     "gnome_mlaug_electrafi_convspin_augnetfull_nmaefull_e15",
                     "gnome_mlaug_electrafi_convspin_augnet1k_nmae1k",
                     "gnome_mlaug_electrafi_convspin_augnet10k_nmae10k",
                     "gnome_mlaug_electrafi_convspin_augnet50k_nmae50k",
                     "gnome_augnet_convrest_1k",
                     "gnome_augnet_convrest_10k",
                     "gnome_augnet_convrest_50k",
                     "gnome_augnet_convrest_full",
                     "gnome_electrafi_convrest_1k",
                     "gnome_electrafi_convrest_10k",
                     "gnome_electrafi_convrest_50k"}
GNOME_ROWS_MAG = [r for r in GNOME_ROWS_NONMAG if r[0] not in GNOME_NONMAG_ONLY]
GNOME_PAIRS = [
    ("gnome_mlaug_electrafi_chgnetmag_augnetfull",
     "gnome_mlaug_electrafi_chgnetmag", "ElectraFi-ref: AugNet full vs old aug"),
    ("gnome_mlaug_charge3net_chgnetmag_augnetfull",
     "gnome_mlaug_charge3net_chgnetmag", "charge3net:    AugNet full vs old aug"),
]

HEADER = """\
===============================================================================
ICLR AUGNET RESULTS -- MP + GNOME -- standard replayed protocols
===============================================================================
n      = completed OK runs for that row's variant in this population
         (with steps and wall time); the table means are over the
         smaller ALIGNED subset printed under each table
steps  = mean SCF steps to convergence
time   = mean VASP wall time per structure, seconds
rel%   = SCF-step reduction vs `Default (SAD)`  (higher is better)
orc%   = SCF-step reduction vs the Oracle (0 = matches a perfect seed,
         negative = more steps than the Oracle)
time%  = wall-time reduction vs `Default (SAD)` (higher is better)
+-sd   = across-structure standard deviation of the column to its
         left -- spread, not a standard error. For rel%/time% it is the
         spread of the PER-STRUCTURE reduction, while the printed mean
         stays the ratio-of-means convention, so the pair can differ.
med%   = median of the per-structure rel% / time% to its left.
dE<1meV% = share of the ALIGNED runs whose final total energy is
         within 1 meV of the default run's -- did the variant land in
         the same minimum

Every column except n is computed on the ALIGNED subset -- the ids
where all FINISHED rows (final? = yes) in that section have an OK run
(the smallest joint subset, the primary_results.txt convention). A
still-running row (final? = -) does not shrink that subset; it is
averaged on the part of the subset it has finished (rel%/time%/dE
against the default over those same ids) -- treat it as provisional.
n is that row's own coverage in the population, for context only.
final? = yes when no structure is in_progress for that variant in
         the snapshot (a slice may still hold pending rows, so `yes` is
         provisional until the queue is empty); `-` while a driver is
         actively holding a structure; `orphan` when the only in_progress
         rows are older than --stale-hours, i.e. a walltime kill or node
         failure left them claimed with no job behind them. An `orphan`
         row is scored like a finished one -- the stuck ids are simply
         missing from it -- and the STATUS block above names them so the
         slice can be resubmitted.

AugNet = the ICLR shared-head augmentation-occupancy models
(paw_mace_runs/*_sharedref_allz/total_sharedref_allz_w64_l3_r6_c3_*),
which REPLACE every earlier augmentation model. Two recipes appear:

  * `AugNet <size> (conv grid, conv spin)` -- fully converged CHGCAR
    with ONLY the total augmentation swapped for the prediction. Both
    spin channels stay converged (`diff` grid and `diff` augmentation
    untouched), so the measured effect is purely the augmentation-charge
    prediction error. Runner run_ml_seed_mp.py, no flags, full
    1943-row test set (exps/relaxmag_full_1943.csv). The `full/10ep` row
    is the same recipe with the 10-epoch AugNet-full checkpoint
    (full_h200_sharedref_allz_e10), non-magnetic only.
  * `ElectraFi NMAE full/<N>ep (ML grid, conv aug+spin)` -- the mirror
    image of the first recipe: the total pseudo grid is ML (ElectraFi
    trained on the full set to an NMAE objective, N epochs) while the
    total augmentation, the diff grid and the diff augmentation all stay
    converged. Runner the total-pseudo-grid runner (not included) --conv_spin
    --include_magnetic; the --include_magnetic gate is new (every earlier
    total_ml_pg_convspin_* row recorded magnetic structures as
    skipped_magnetic and so ran the non-magnetic half only), so these two
    are the first conv-spin ML-grid rows on the full 1943-row set. The
    1k / 10k / 50k rows of the same recipe are the earlier ElectraFi NMAE
    training budgets, pulled in from results_summary.txt section 2 so the
    whole budget ladder is scored on one shared aligned subset; being
    non-magnetic-only they appear in sections 1 and 4 alone.
  * `AugNet <size> + NMAE <size> (both ML, conv spin)` -- matched
    training budgets on BOTH ML channels: the total pseudo grid from the
    ElectraFi NMAE model of that budget and the total augmentation from
    the AugNet model of the same budget, while the diff grid and diff
    augmentation stay converged. Non-magnetic populations only. On MP
    that is run_ml_seed_mp.py with --grid_pred_dir --aug_pred_dir
    --nonmag_only; on GNoME it is run_ml_seed_gnome.py --conv_spin,
    a mode added 2026-08-27 (that runner was total-only + MAGMOM before,
    so GNoME had no converged-spin ML recipe at all).
  * `<grid model> + AugNet full (CHGNet MAGMOM)` -- total-only seed that
    is ML end to end: ML total pseudo grid (ElectraFi-ref /
    charge3net 4k, the latter with --renorm) + AugNet-full total
    augmentation, no diff channel, spin from unsigned CHGNet per-site
    moments (chgnet_preds/chgnet_pred_mp.csv). This is the 2026-08-19
    chgnetmag campaign with the augmentation model swapped, so the
    paired deltas in the Notes are one-factor comparisons.

Everything else (INCAR, ISPIN=2, ICHARG=1) is identical to the existing
default / true_init variants, so steps are directly comparable.

GNoME now also carries the two SINGLE-CHANNEL recipes against a converged
remainder, the twins of the MP rows above: `AugNet <size> (conv grid,
conv spin)` (only the total augmentation is predicted) and `ElectraFi
<size> (ML grid, conv aug+spin)` (only the total pseudo grid is). Both
run run_ml_seed_gnome.py --conv_spin with one prediction directory
omitted -- a mode added 2026-08-28, when both dirs became optional; they
cover the non-magnetic 1254 only.

Sections 6-9 are the MP spin-channel-ML and all-four-components-ML tables
(2026-08-28, see run_instructions_iclr.md). Sections 10-14 are the ML-end-to-end recipe on GNoME (runs_gnome.db,
chgnet_preds/chgnet_pred_gnome.csv, uniform-0.001 fallback for gids
without a prediction): full 1833 set = 1254 non-magnetic + 579 magnetic,
populations from exps/gnome2_{nonmag,full}.csv -- 1314 non-magnetic +
590 magnetic = 1904, the gids with a magnetic verdict and a prediction in
every dir. These CSVs replaced the *_noyb ones on 2026-08-28: the old
builder dropped all 75 Yb-containing structures because the FIRST AugNet
export had none of them, while the shared-head export used here covers
71/75. Rows scored before that date sit on the smaller 1254/1833 sets;
the alignment drops ids a finished variant never ran, so mixed-vintage
sections stay pointwise-comparable but their aligned n differs.

MP populations by the DB criterion (true_init_magmom0p1 ok /
skipped_magnetic), minus mp-1219533: its ElectraFi-ref + AugNet-full run
is recorded ok at -1193395.6 eV against the ~-21.68 eV every other
variant reaches -- a diverged SCF that passed the convergence check, and
its 1.19e6 eV offset swamped mean dE_min for every other row. Each
section is pointwise-aligned (see above).
Percentages are ratios of means over that aligned sample (the rel%
convention of results_summary.txt).
"""


def fetch(conn, variant):
    """{id: (steps, wall, energy)} for OK rows with steps and wall."""
    return {mp: (s, w, e) for mp, s, w, e in conn.execute(
        "SELECT mp_id, scf_total, wall, energy FROM runs WHERE variant=? "
        "AND status='ok' AND scf_total IS NOT NULL AND wall IS NOT NULL",
        (variant,))}


def run_state(conn, variants, stale_hours):
    """Per-variant run state: 'yes' | '-' (running) | 'orphan'.

    An in_progress row means "a driver claimed this structure and never wrote a
    terminal status". That is the live state while a slice job runs, but it also
    survives a walltime kill / node failure forever, with no job left to retry
    it -- an orphan. Age it: a variant whose newest in_progress row was written
    more than `stale_hours` ago has nothing running, so it counts as finished
    for alignment (it is simply missing those ids) and is flagged 'orphan' so
    the rows can be mopped up by resubmitting their slice.
    """
    now = datetime.now(timezone.utc)
    state, orphans = {}, {}
    for v in variants:
        rows = conn.execute(
            "SELECT mp_id, ingested_at FROM runs WHERE variant=? "
            "AND status='in_progress'", (v,)).fetchall()
        if not rows:
            state[v] = "yes"
            continue
        ages = []
        for mp, ts in rows:
            try:
                ages.append((now - datetime.fromisoformat(ts)).total_seconds() / 3600)
            except (TypeError, ValueError):
                ages.append(0.0)     # unparseable stamp -> assume live
        if min(ages) <= stale_hours:
            state[v] = "-"
        else:
            state[v] = "orphan"
            orphans[v] = [(mp, age) for (mp, _), age in zip(rows, ages)]
    return state, orphans


def present(rows, data, pop):
    """Drop rows with no OK run in this population (not yet started)."""
    return [(v, l) for v, l in rows if any(i in pop for i in data[v])]


def section(title, rows, data, pop, state, preamble=""):
    L = ["=" * 79, title, "=" * 79]
    if preamble:
        L += [preamble]
    L += [f"{'experiment':<44}{'n':>5}{'steps':>8}{'+-sd':>8}{'time(s)':>10}"
          f"{'+-sd':>10}{'rel%':>8}{'+-sd':>8}{'med%':>8}{'orc%':>8}{'time%':>8}"
          f"{'+-sd':>8}{'med%':>8}{'dE<1meV%':>10}  final?",
          "-" * 161]
    aligned = pop.copy()
    for v, _ in rows:
        if state.get(v) != "-":
            aligned &= set(data[v])
    sd = lambda a: statistics.stdev(a) if len(a) > 1 else 0.0
    for v, label in rows:
        n = sum(1 for i in data[v] if i in pop)
        ids = aligned & set(data[v]) if state.get(v) == "-" else aligned
        if not ids:
            L.append(f"{label:<44}{n:>5}{'--':>8}")
            continue
        steps = [data[v][i][0] for i in ids]
        walls = [data[v][i][1] for i in ids]
        m, w = statistics.fmean(steps), statistics.fmean(walls)
        d_steps = statistics.fmean(data[DEFAULT][i][0] for i in ids)
        d_wall = statistics.fmean(data[DEFAULT][i][1] for i in ids)
        o_steps = statistics.fmean(data[ORACLE][i][0] for i in ids)
        rels = [100 * (data[DEFAULT][i][0] - data[v][i][0]) / data[DEFAULT][i][0]
                for i in ids if data[DEFAULT][i][0]]
        wrels = [100 * (data[DEFAULT][i][1] - data[v][i][1]) / data[DEFAULT][i][1]
                 for i in ids if data[DEFAULT][i][1]]
        pairs = [i for i in ids
                 if data[v][i][2] is not None
                 and data[DEFAULT][i][2] is not None]
        agree = sum(1 for i in pairs
                    if abs(data[v][i][2] - data[DEFAULT][i][2]) < 1e-3)
        de = f"{100 * agree / len(pairs):>10.1f}" if pairs else f"{'--':>10}"
        fin = state.get(v, "yes")
        L.append(f"{label:<44}{n:>5}{m:>8.2f}{sd(steps):>8.2f}{w:>10.2f}"
                 f"{sd(walls):>10.2f}"
                 f"{100 * (d_steps - m) / d_steps:>+8.2f}{sd(rels):>8.2f}"
                 f"{statistics.median(rels):>+8.2f}"
                 f"{100 * (o_steps - m) / o_steps:>+8.2f}"
                 f"{100 * (d_wall - w) / d_wall:>+8.2f}{sd(wrels):>8.2f}"
                 f"{statistics.median(wrels):>+8.2f}"
                 f"{de}  {fin}")
    L += ["", f"aligned n = {len(aligned)}", ""]
    return L


def energy_section(title, rows, data, pop, state):
    """Which initialization reaches the lowest final energy, per structure."""
    aligned = [i for i in pop
               if all(i in data[v] and data[v][i][2] is not None
                      for v, _ in rows if state.get(v) != "-")]
    L = ["=" * 79, title, "=" * 79,
         f"Aligned on the {len(aligned)} structures where every row has an"
         " OK energy.",
         "lowest% = final energy within 1 meV of the lowest reached by any"
         " row (ties",
         "count for all); dE_min = excess above that per-structure minimum;"
         " <def% =",
         "ends >1 meV BELOW the default run (found a deeper minimum).",
         "",
         f"{'experiment':<44}{'lowest%':>9}{'med dE_min':>12}"
         f"{'mean dE_min':>13}{'<def%':>8}",
         "-" * 86]
    lo = {i: min(data[v][i][2] for v, _ in rows
                 if i in data[v] and data[v][i][2] is not None)
          for i in aligned}
    outliers = []
    for v, label in rows:
        ids = [i for i in aligned
               if i in data[v] and data[v][i][2] is not None]
        if not ids:
            L.append(f"{label:<44}{'--':>9}")
            continue
        exc = [data[v][i][2] - lo[i] for i in ids]
        low = sum(1 for e in exc if e < 1e-3)
        below = sum(1 for i in ids
                    if data[v][i][2] < data[DEFAULT][i][2] - 1e-3)
        L.append(f"{label:<44}{100 * low / len(ids):>9.1f}"
                 f"{1e3 * statistics.median(exc):>10.2f}meV"
                 f"{1e3 * statistics.fmean(exc):>11.1f}meV"
                 f"{100 * below / len(ids):>8.1f}")
        outliers += [(label, i, e) for i, e in zip(ids, exc) if e > 1.0]
    # A single diverged run (energy off by eV or more) dominates mean dE_min;
    # name them so the column is readable and the median stays the summary.
    if outliers:
        L += ["", "excess > 1 eV above the per-structure minimum (these"
              " dominate mean dE_min):"]
        for label, i, e in sorted(outliers, key=lambda t: -t[2])[:10]:
            L.append(f"    {label:<44}{i:>14}  {e:>+12.2f} eV")
        if len(outliers) > 10:
            L.append(f"    ... and {len(outliers) - 10} more")
    L += [""]
    return L


def paired(data_v, data_b, ids):
    ids = [i for i in ids if i in data_v and i in data_b]
    if len(ids) < 3:
        return None, len(ids)
    return statistics.fmean(data_v[i][0] - data_b[i][0] for i in ids), len(ids)


def gnome_populations():
    import csv as _csv
    import pathlib
    exps = pathlib.Path(__file__).resolve().parent

    def gids(p):
        with open(p) as f:
            return {r["GNOME_ID"].strip() for r in _csv.DictReader(f)}

    nonmag = gids(exps / "gnome2_nonmag.csv")
    full = gids(exps / "gnome2_full.csv")
    return nonmag, full - nonmag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/snap.db")
    ap.add_argument("--gnome_db", default="/tmp/snap_gnome.db")
    ap.add_argument("--out", default="notes/results_iclr.txt")
    ap.add_argument("--stale-hours", type=float, default=6.0,
                    help="An in_progress row older than this has no job behind "
                         "it any more (walltime kill / node failure); its "
                         "variant is reported 'orphan', not still running.")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)

    nonmag = {r[0] for r in conn.execute(
        "SELECT mp_id FROM runs WHERE variant='true_init_magmom0p1' AND status='ok'")}
    mag = {r[0] for r in conn.execute(
        "SELECT mp_id FROM runs WHERE variant='true_init_magmom0p1' "
        "AND status='skipped_magnetic'")}
    nonmag -= EXCLUDE_MP
    mag -= EXCLUDE_MP

    variants = ({v for v, _ in ROWS_NONMAG + ROWS_SPIN + ROWS_ALLML}
                | {b for _, b, _ in PAIRS_SAME_RECIPE})
    data = {v: fetch(conn, v) for v in variants}
    state, orphans = run_state(conn, variants, args.stale_hours)

    all_rows = {v: l for v, l in ROWS_NONMAG + ROWS_SPIN + ROWS_ALLML}
    running = sorted(v for v in all_rows if state.get(v) == "-")
    orphaned = sorted(v for v in all_rows if state.get(v) == "orphan")
    L = [HEADER]
    L.append(f"STATUS ({date.today()}): "
             + ("every MP table variant FINAL (no in_progress rows)."
                if not running else
                "INTERIM -- still running: " + ", ".join(running) + ".")
             + " Rerun exps/report_iclr.py on a fresh snapshot to update.")
    for v in orphaned:
        ids = ", ".join(f"{mp} ({age:.0f}h)" for mp, age in orphans[v])
        L.append(f"  ORPHAN {v}: {len(orphans[v])} row(s) left in_progress with "
                 f"no job behind them -- {ids}. Counted as final (those ids are "
                 f"simply missing); resubmit their slice to mop up.")
    L.append("")

    rows_nm = present(ROWS_NONMAG, data, nonmag)
    rows_m = present(ROWS_MAG, data, mag)
    L += section("1. NON-MAGNETIC", rows_nm, data, nonmag, state)
    nonmag_only_note = ("The ElectraFi NMAE 1k/10k/50k rows ran the "
                        "non-magnetic half only (they predate "
                        "--include_magnetic), so they are absent here.\n")
    L += section("2. MAGNETIC", rows_m, data, mag, state, nonmag_only_note)
    L += section("3. TOTAL (non-magnetic + magnetic)", rows_m, data,
                 nonmag | mag, state, nonmag_only_note)
    L += energy_section("4. LOWEST FINAL ENERGY -- NON-MAGNETIC",
                        rows_nm, data, nonmag, state)
    L += energy_section("5. LOWEST FINAL ENERGY -- MAGNETIC",
                        rows_m, data, mag, state)

    spin_note = ("One spin component is the ML prediction (diff GRID from the "
                 "ELECTRAFI spin npy, or diff AUG from the AugNet spin export); "
                 "the total grid, total aug and the other spin component are "
                 "the converged reference. Magnetic structures only "
                 "(--mag-only); the `cf.` rows are the total-channel twins.\n")
    L += section("6. SPIN CHANNEL ML -- MAGNETIC",
                 present(ROWS_SPIN, data, mag), data, mag, state, spin_note)
    allml_note = ("Every component ML: NMAE full/5ep total grid + AugNet full "
                  "total aug + spin NMAE diff grid + spin AugNet diff aug. "
                  "Nothing from the converged reference except block "
                  "lengths / the moment line. Full 1943 set.\n")
    L += section("7. ALL FOUR COMPONENTS ML -- NON-MAGNETIC",
                 present(ROWS_ALLML, data, nonmag), data, nonmag, state,
                 allml_note)
    L += section("8. ALL FOUR COMPONENTS ML -- MAGNETIC",
                 present(ROWS_ALLML_MAG, data, mag), data, mag, state,
                 allml_note)
    L += section("9. ALL FOUR COMPONENTS ML -- TOTAL",
                 present(ROWS_ALLML_MAG, data, nonmag | mag), data,
                 nonmag | mag, state, allml_note)

    gdata = None
    if os.path.isfile(args.gnome_db):
        gconn = sqlite3.connect(args.gnome_db)
        g_nonmag, g_mag = gnome_populations()
        gvariants = ({v for v, _ in GNOME_ROWS_NONMAG}
                     | {b for _, b, _ in GNOME_PAIRS})
        gdata = {v: fetch(gconn, v) for v in gvariants}
        gdata[DEFAULT] = gdata["gnome_default"]
        gdata[ORACLE] = gdata["gnome_true_init"]
        g_state, g_orphans = run_state(gconn, gvariants, args.stale_hours)
        for v in sorted(g_orphans):
            print(f"note: GNoME {v} has {len(g_orphans[v])} orphaned "
                  f"in_progress row(s)")
        g_nm = present(GNOME_ROWS_NONMAG, gdata, g_nonmag)
        g_m = present(GNOME_ROWS_MAG, gdata, g_mag)
        L += section("10. GNOME -- NON-MAGNETIC", g_nm, gdata, g_nonmag, g_state)
        g_note = ("The matched-budget and single-channel conv-spin rows ran "
                  "the non-magnetic 1254 only, so they are absent here.\n")
        L += section("11. GNOME -- MAGNETIC", g_m, gdata, g_mag, g_state, g_note)
        L += section("12. GNOME -- TOTAL (non-magnetic + magnetic)",
                     g_m, gdata, g_nonmag | g_mag, g_state, g_note)
        L += energy_section("13. GNOME -- LOWEST FINAL ENERGY -- NON-MAGNETIC",
                            g_nm, gdata, g_nonmag, g_state)
        L += energy_section("14. GNOME -- LOWEST FINAL ENERGY -- MAGNETIC",
                            g_m, gdata, g_mag, g_state)

    notes = ["Notes:",
             "  * AugNet vs the PREVIOUS-generation augmentation model, same"
             " recipe and same",
             "    grid/spin init, paired (negative = AugNet cheaper):"]
    for v, b, tag in PAIRS_SAME_RECIPE:
        for popname, pop in (("non-mag", nonmag), ("mag", mag)):
            d, n = paired(data[v], data[b], pop)
            if d is not None:
                notes.append(f"      {tag} [{popname}]: "
                             f"delta {d:+.2f} steps (n={n})")
    if gdata is not None:
        notes.append("  * GNoME, same comparison:")
        for v, b, tag in GNOME_PAIRS:
            for popname, pop in (("non-mag", g_nonmag), ("mag", g_mag)):
                d, n = paired(gdata[v], gdata[b], pop)
                if d is not None:
                    notes.append(f"      {tag} [{popname}]: "
                                 f"delta {d:+.2f} steps (n={n})")
    notes.append(
        "  * The conv-grid AugNet rows isolate the augmentation model: the"
        " total and diff")
    notes.append(
        "    pseudo grids and the diff augmentation all come from the"
        " converged reference.")
    L += notes
    L += ["", f"Last updated: {date.today()}"]

    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
