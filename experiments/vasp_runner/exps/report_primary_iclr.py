#!/usr/bin/env python3
"""Distilled ICLR tables: 6 tables (MP / GNoME x non-mag / mag / total), same
columns and alignment as exps/report_iclr.py, primary rows only.

Usage (snapshots made on the cluster with `sqlite3 runs/runs.db ".backup ..."`):
    python exps/report_primary_iclr.py --db snap_mp.db --gnome_db snap_gnome.db
"""
import argparse
import json
import os
import sqlite3
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_iclr import (DEFAULT, ORACLE, EXCLUDE_MP, fetch, run_state,
                         present, section, gnome_populations)

MP_ROWS = [
    ("default",                                    "Default (SAD)"),
    ("true_init",                                  "Oracle (converged seed)"),
    ("true_init_total_chgnetmag",                  "Oracle total + CHGNet MAGMOM"),
    # ("true_init_total_oraclemag",                  "Oracle total + oracle MAGMOM [Voronoi]"),
    ("true_init_total_lorbit",                     "Oracle total + oracle MAGMOM"),
    ("ml_aug_spinboth_convrest",                   "Oracle total + spin ML (diff grid+aug)"),
    ("ml_aug_allml_electrafi_ref",                 "ALL ML: ElectraFi-ref + AugNet full + spin models"),
    ("ml_aug_allml_c3n4k",                         "ALL ML: charge3net + AugNet full + spin models"),
    ("ml_aug_spinboth_convrest_unconstrained",     "Oracle total + unspin ML (diff grid+aug)"),
    ("ml_aug_allml_electrafi_ref_unconstrained",   "ALL ML: ElectraFi-ref + AugNet full + unspin"),
    ("ml_aug_allml_c3n4k_unconstrained",           "ALL ML: charge3net + AugNet full + unspin"),
    ("ml_aug_augnet_full_electrafi_ref_chgnetmag", "ElectraFi-ref + AugNet full (CHGNet MAGMOM)"),
    ("ml_aug_augnet_full_c3n4k_chgnetmag",         "charge3net + AugNet full (CHGNet MAGMOM)"),
    # ("ml_aug_augnet_full_electrafi_ref_oraclemag", "ElectraFi-ref + AugNet full (oracle MAGMOM) [Voronoi]"),
    # ("ml_aug_augnet_full_c3n4k_oraclemag",         "charge3net + AugNet full (oracle MAGMOM) [Voronoi]"),
    ("ml_aug_augnet_full_electrafi_ref_lorbit",    "ElectraFi-ref + AugNet full (oracle MAGMOM)"),
    ("ml_aug_augnet_full_c3n4k_lorbit",            "charge3net + AugNet full (oracle MAGMOM)"),
    ("ml_aug_augnet_full_nmae_full_e15_chgnetmag", "NMAE full/15ep + AugNet full (CHGNet MAGMOM)"),
    ("ml_aug_augnet_1k",                           "AugNet 1k (conv grid, conv spin)"),
    ("ml_aug_augnet_10k",                          "AugNet 10k (conv grid, conv spin)"),
    ("ml_aug_augnet_50k",                          "AugNet 50k (conv grid, conv spin)"),
    ("ml_aug_augnet_full",                         "AugNet full (conv grid, conv spin)"),
    ("ml_aug_augnet_1k_electrafi_ref_convspin",    "AugNet 1k + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_10k_electrafi_ref_convspin",   "AugNet 10k + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_50k_electrafi_ref_convspin",   "AugNet 50k + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_full_electrafi_ref_convspin",  "AugNet full + ElectraFi-ref (both ML, conv spin)"),
    ("ml_aug_augnet_full_c3n4k_convspin",          "AugNet full + charge3net (both ML, conv spin)"),
]
# The AugNet-full conv-spin pair was extended to the full set (magnetic
# included) on 2026-09-04; the 1k/10k/50k ElectraFi-ref rows stay non-mag only.
# While the extension is still queued those rows hold non-magnetic ids only, and
# a "final" row with no magnetic run would drag every table's aligned subset
# down to the non-magnetic ids -- so nonmag_only() below also excludes any row
# that has yet to produce a single run in the magnetic population.
MP_FULL_SET = {"ml_aug_augnet_full_electrafi_ref_convspin",
               "ml_aug_augnet_full_c3n4k_convspin"}
MP_NONMAG_ONLY = {v for v, _ in MP_ROWS
                  if v.endswith("_electrafi_ref_convspin") and v not in MP_FULL_SET}

GNOME_ROWS = [
    ("gnome_default",                               "Default (SAD)"),
    ("gnome_true_init",                             "Oracle (converged seed)"),
    # ("gnome_default_oraclemag",                     "SAD + oracle MAGMOM [Voronoi]"),
    ("gnome_default_lorbit",                        "SAD + oracle MAGMOM"),
    ("gnome_true_init_total_chgnetmag",             "Oracle total + CHGNet MAGMOM"),
    # ("gnome_true_init_total_oraclemag",             "Oracle total + oracle MAGMOM [Voronoi]"),
    ("gnome_true_init_total_lorbit",                "Oracle total + oracle MAGMOM"),
    ("gnome_spinboth_convrest",                     "Oracle total + spin ML (diff grid+aug)"),
    ("gnome_mlaug_electrafi_allml",                 "ALL ML: ElectraFi-ref + AugNet full + spin models"),
    ("gnome_mlaug_charge3net_allml",                "ALL ML: charge3net + AugNet full + spin models"),
    ("gnome_mlaug_electrafi_chgnetmag_augnetfull",  "ElectraFi-ref + AugNet full (CHGNet MAGMOM)"),
    ("gnome_mlaug_charge3net_chgnetmag_augnetfull", "charge3net + AugNet full (CHGNet MAGMOM)"),
    # ("gnome_mlaug_electrafi_oraclemag_augnetfull",  "ElectraFi-ref + AugNet full (oracle MAGMOM) [Voronoi]"),
    # ("gnome_mlaug_charge3net_oraclemag_augnetfull", "charge3net + AugNet full (oracle MAGMOM) [Voronoi]"),
    ("gnome_mlaug_electrafi_lorbit_augnetfull",     "ElectraFi-ref + AugNet full (oracle MAGMOM)"),
    ("gnome_mlaug_charge3net_lorbit_augnetfull",    "charge3net + AugNet full (oracle MAGMOM)"),
    ("gnome_mlaug_electrafi_chgnetmag_augnetfull_nmaefull_e15",
                                                    "NMAE full/15ep + AugNet full (CHGNet MAGMOM)"),
    ("gnome_mlaug_electrafi_convspin_augnetfull_nmaefull_e15",
                                                    "AugNet full + NMAE full/15ep (both ML, conv spin)"),
    ("gnome_augnet_convrest_1k",                    "AugNet 1k (conv grid, conv spin)"),
    ("gnome_augnet_convrest_10k",                   "AugNet 10k (conv grid, conv spin)"),
    ("gnome_augnet_convrest_50k",                   "AugNet 50k (conv grid, conv spin)"),
    ("gnome_augnet_convrest_full",                  "AugNet full (conv grid, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet1k",     "AugNet 1k + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet10k",    "AugNet 10k + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnet50k",    "AugNet 50k + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_electrafi_convspin_augnetfull",   "AugNet full + ElectraFi-ref (both ML, conv spin)"),
    ("gnome_mlaug_charge3net_convspin_augnetfull",  "AugNet full + charge3net (both ML, conv spin)"),
]
GNOME_FULL_SET = {"gnome_mlaug_electrafi_convspin_augnetfull",
                  "gnome_mlaug_charge3net_convspin_augnetfull"}
GNOME_NONMAG_ONLY = {v for v, _ in GNOME_ROWS
                     if ("convrest_" in v or "convspin_augnet" in v)
                     and v not in GNOME_FULL_SET}
# results_summary.txt section 1 baselines (MP only), rescored on the aligned
# subsets of tables 1-3 above.
BASELINE_ROWS = [
    ("default",                   "Default (SAD)"),
    ("true_init",                 "Oracle (converged seed)"),
    ("default_relaxmag",          "SAD + MP-relaxed MAGMOM"),
    ("default_chgnetmag",         "SAD + CHGNet MAGMOM"),
    # ("default_oraclemag",         "SAD + oracle MAGMOM [Voronoi]"),
    ("default_lorbit",            "SAD + oracle MAGMOM"),
    ("true_init_total_relaxmag",  "Oracle total + MP-relaxed MAGMOM"),
    ("true_init_total_chgnetmag", "Oracle total + CHGNet MAGMOM"),
    # ("true_init_total_oraclemag", "Oracle total + oracle MAGMOM [Voronoi]"),
    ("true_init_total_lorbit",    "Oracle total + oracle MAGMOM"),
    ("conv_grid_sad_aug_fix",     "conv grid (total+spin) + SAD aug"),
    ("conv_total_sad_rest",       "conv total ch. + SAD diff ch. (conv_total_sad_rest)"),
    ("sad_diff_hybrid_fix",       "conv total ch. + SAD diff ch. (sad_diff_hybrid)"),
    ("sad_grid_conv_aug_fix",     "SAD grid + conv aug (total+spin)"),
    ("sad_total_hybrid_fix",      "SAD total ch. + conv diff ch."),
]

HEADER = """\
===============================================================================
PRIMARY ICLR RESULTS -- MP + GNOME
===============================================================================
n        = OK runs for that variant in the population
steps    = mean SCF steps to convergence     time = mean VASP wall time, s
rel%     = SCF-step reduction vs Default     orc% = SCF-step reduction vs Oracle
time%    = wall-time reduction vs Default
+-sd     = across-structure sd of the column to its left (spread, not SE)
med%     = median of the per-structure rel% / time% (the mean rel%/time%
           columns are ratios of means, so median != mean in general)
dE<1meV% = share of runs whose final energy is within 1 meV of the default run
final?   = yes / - (still running, scored on the ids it has finished) / orphan
Every column except n is over the ALIGNED subset: the ids where every finished
row in the table has an OK run. Percentages are ratios of means.
Oracle total = converged total grid + aug; spin ML = ML diff grid + ML diff aug;
ALL ML = ML total grid + AugNet full aug + spin ML; CHGNet MAGMOM = no diff
channel, spin from unsigned CHGNet per-site moments; oracle MAGMOM = same, but
the per-site moments are the converged run's own sphere-projected (LORBIT)
moments: for MP from the reference static task's OUTCAR via mp-api
(exps/oracle_magmoms_mp_mpapi.csv), for GNoME from our gnome_true_init OUTCAR
sidecars (exps/oracle_magmoms_gnome_lorbit.csv). unspin = the unconstrained
(no charge renormalisation, norenorm200k) ElectraFi spin-grid model in place of
the constrained one; the AugNet diff aug is the same in both. The superseded
[Voronoi] oracle-MAGMOM rows (moments from integrating the reference CHGCAR's
magnetization channel over nearest-atom cells, off by up to 6.3 uB on strong
GNoME moments, notes/oracle_magmom_source_check.json) are left out of the
tables; their runs (*_oraclemag tags) are complete and can be re-added by
uncommenting them in exps/report_primary_iclr.py.
"""
NONMAG_NOTE = ("The AugNet (conv grid, conv spin) rows and the 1k/10k/50k "
               "AugNet + ElectraFi-ref (both ML, conv spin) rows ran the "
               "non-magnetic set only.\n")
PENDING_NOTE = ("Also absent here, with no magnetic run yet: %s.\n")


def nonmag_only(rows, base, data, mag):
    """`base` plus every row that has no OK run in the magnetic population.

    A row whose magnetic half is still queued would otherwise be treated as a
    finished non-magnetic row and shrink every table's aligned subset to the
    non-magnetic ids.
    """
    return base | {v for v, _ in rows if not set(data.get(v, {})) & mag}


def aligned_ids(rows, data, pop, state):
    """The aligned subset section() scores on: pop ∩ every finished row."""
    ids = set(pop)
    for v, _ in rows:
        if state.get(v) != "-":
            ids &= set(data[v])
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/tmp/snap.db")
    ap.add_argument("--gnome_db", default="/tmp/snap_gnome.db")
    ap.add_argument("--out", default="notes/primary_results_iclr.txt")
    ap.add_argument("--stale-hours", type=float, default=6.0)
    ap.add_argument("--ids-out", default=None,
                    help="Also dump the aligned id set of each table as JSON.")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    nonmag = {r[0] for r in conn.execute(
        "SELECT mp_id FROM runs WHERE variant='true_init_magmom0p1' AND status='ok'")}
    mag = {r[0] for r in conn.execute(
        "SELECT mp_id FROM runs WHERE variant='true_init_magmom0p1' "
        "AND status='skipped_magnetic'")}
    nonmag -= EXCLUDE_MP
    mag -= EXCLUDE_MP
    variants = [v for v, _ in MP_ROWS] + [v for v, _ in BASELINE_ROWS]
    data = {v: fetch(conn, v) for v in variants}
    state, _ = run_state(conn, variants, args.stale_hours)
    mp_rows = MP_ROWS
    mp_skip = nonmag_only(mp_rows, MP_NONMAG_ONLY, data, mag)
    mp_mag_rows = [r for r in mp_rows if r[0] not in mp_skip]

    gconn = sqlite3.connect(args.gnome_db)
    g_nonmag, g_mag = gnome_populations()
    gvariants = [v for v, _ in GNOME_ROWS]
    gdata = {v: fetch(gconn, v) for v in gvariants}
    gdata[DEFAULT] = gdata["gnome_default"]
    gdata[ORACLE] = gdata["gnome_true_init"]
    g_state, _ = run_state(gconn, gvariants, args.stale_hours)
    g_rows = GNOME_ROWS
    g_skip = nonmag_only(g_rows, GNOME_NONMAG_ONLY, gdata, g_mag)
    g_mag_rows = [r for r in g_rows if r[0] not in g_skip]

    running = sorted({v for v in variants if state.get(v) == "-"}) + \
        sorted({v for v in gvariants if g_state.get(v) == "-"})
    # Submitted but not yet started: no row at all in the manifest, so neither
    # run_state nor any table can show them.
    missing = sorted({v for v in variants if not data[v]}) + \
        sorted({v for v in gvariants if not gdata[v]})
    status = ("all rows final." if not running
              else "INTERIM -- still running: " + ", ".join(running) + ".")
    if missing:
        status += " Queued, no runs yet: " + ", ".join(missing) + "."
    L = [HEADER, f"STATUS ({date.today()}): " + status, ""]
    tabs = [("mp_total", present(mp_mag_rows, data, nonmag | mag), data, nonmag | mag, state),
            ("mp_magnetic", present(mp_mag_rows, data, mag), data, mag, state),
            ("mp_nonmagnetic", present(mp_rows, data, nonmag), data, nonmag, state),
            ("gnome_total", present(g_mag_rows, gdata, g_nonmag | g_mag), gdata, g_nonmag | g_mag, g_state),
            ("gnome_magnetic", present(g_mag_rows, gdata, g_mag), gdata, g_mag, g_state),
            ("gnome_nonmagnetic", present(g_rows, gdata, g_nonmag), gdata, g_nonmag, g_state)]
    if args.ids_out:
        out = {name: aligned_ids(rows, d, pop, st) for name, rows, d, pop, st in tabs}
        out["_meta"] = {"date": str(date.today()),
                        "source": "exps/report_primary_iclr.py (aligned subsets of "
                                  "notes/primary_results_iclr.txt)",
                        "n": {k: len(v) for k, v in out.items()}}
        with open(args.ids_out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.ids_out}")
    mp_note = NONMAG_NOTE + (PENDING_NOTE % ", ".join(sorted(mp_skip - MP_NONMAG_ONLY))
                             if mp_skip - MP_NONMAG_ONLY else "")
    g_note = NONMAG_NOTE + (PENDING_NOTE % ", ".join(sorted(g_skip - GNOME_NONMAG_ONLY))
                            if g_skip - GNOME_NONMAG_ONLY else "")
    L += section("1. MP -- TOTAL (non-magnetic + magnetic)",
                 present(mp_mag_rows, data, nonmag | mag), data,
                 nonmag | mag, state, mp_note)
    L += section("2. MP -- MAGNETIC",
                 present(mp_mag_rows, data, mag), data, mag, state, mp_note)
    L += section("3. MP -- NON-MAGNETIC",
                 present(mp_rows, data, nonmag), data, nonmag, state)
    L += section("4. GNOME -- TOTAL (non-magnetic + magnetic)",
                 present(g_mag_rows, gdata, g_nonmag | g_mag), gdata,
                 g_nonmag | g_mag, g_state, g_note)
    L += section("5. GNOME -- MAGNETIC",
                 present(g_mag_rows, gdata, g_mag), gdata, g_mag, g_state,
                 g_note)
    L += section("6. GNOME -- NON-MAGNETIC",
                 present(g_rows, gdata, g_nonmag), gdata, g_nonmag, g_state)
    base_note = ("Baselines from results_summary.txt section 1, rescored on the "
                 "aligned subset of the matching table above (further aligned "
                 "on the baseline rows themselves).\n")
    for k, (name, rows, d, pop, st) in enumerate(tabs[:3], start=7):
        sub = set(aligned_ids(rows, d, pop, st))
        L += section(f"{k}. MP BASELINES -- {name.split('_', 1)[1].upper()} "
                     f"(subset of table {k - 6})",
                     present(BASELINE_ROWS, data, sub), data, sub, state, base_note)
    L += [f"Last updated: {date.today()}"]
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
