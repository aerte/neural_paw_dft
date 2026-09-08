"""Result CSV schemas — exact column lists lifted from the legacy scripts.

Each (source, init) pair maps to the column order the old script wrote, so
new runs append to existing CSVs without breaking downstream consumers.
"""
from __future__ import annotations

import csv
import fcntl
import os

# Per-(source, init) header lists. Lifted verbatim from the legacy scripts.
SCHEMAS = {
    ("gnome", "default"): [
        "CHGCAR_PATH", "csv_index", "slice_start", "slice_end",
        "status_default", "energy_default",
        "scf_steps_dav_default", "scf_steps_rmm_default",
        "scf_steps_total_default", "time_default",
    ],
    ("gnome", "true"): [
        "CHGCAR_PATH", "csv_index", "slice_start", "slice_end",
        "status_true_init", "energy_true_init",
        "scf_steps_dav_true_init", "scf_steps_rmm_true_init",
        "scf_steps_total_true_init", "time_true_init",
    ],
    ("gnome", "ml"): [
        "CHGCAR_PATH", "csv_index", "slice_start", "slice_end",
        "status_ml_init", "energy_ml_init",
        "scf_steps_dav_ml_init", "scf_steps_rmm_ml_init",
        "scf_steps_other_ml_init", "scf_steps_total_ml_init",
        "time_ml_init",
        "nmae_total_ml_vs_true", "nmae_total_ml_vs_true_pct",
        "nmae_diff_ml_vs_true",  "nmae_diff_ml_vs_true_pct",
    ],
    ("mp", "default"): [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        "status_default", "energy_default",
        "scf_steps_dav_default", "scf_steps_rmm_default",
        "scf_steps_other_default", "scf_steps_total_default",
        "time_default",
    ],
    ("mp", "true"): [
        # New: mirrors the GNOME true-init schema + MP-style extras.
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        "status_true_init", "energy_true_init",
        "scf_steps_dav_true_init", "scf_steps_rmm_true_init",
        "scf_steps_other_true_init", "scf_steps_total_true_init",
        "time_true_init",
    ],
    ("mp", "ml"): [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        "status_ml_init", "energy_ml_init",
        "scf_steps_dav_ml_init", "scf_steps_rmm_ml_init",
        "scf_steps_other_ml_init", "scf_steps_total_ml_init",
        "time_ml_init",
        "nmae_total_ml_vs_true", "nmae_total_ml_vs_true_pct",
        "nmae_diff_ml_vs_true",  "nmae_diff_ml_vs_true_pct",
    ],
    ("gnome", "zero_mag"): [
        "CHGCAR_PATH", "csv_index", "slice_start", "slice_end",
        "status_zero_mag", "energy_zero_mag",
        "scf_steps_dav_zero_mag", "scf_steps_rmm_zero_mag",
        "scf_steps_total_zero_mag", "time_zero_mag",
    ],
    ("mp", "zero_mag"): [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        "status_zero_mag", "energy_zero_mag",
        "scf_steps_dav_zero_mag", "scf_steps_rmm_zero_mag",
        "scf_steps_other_zero_mag", "scf_steps_total_zero_mag",
        "time_zero_mag",
    ],
    ("mp", "sad_diff_hybrid"): [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        "status_sad_diff_hybrid", "energy_sad_diff_hybrid",
        "scf_steps_dav_sad_diff_hybrid", "scf_steps_rmm_sad_diff_hybrid",
        "scf_steps_other_sad_diff_hybrid", "scf_steps_total_sad_diff_hybrid",
        "time_sad_diff_hybrid",
    ],
    ("mp", "sad_total_hybrid"): [
        "CHGCAR_PATH", "MP_ID", "csv_index", "slice_start", "slice_end",
        "status_sad_total_hybrid", "energy_sad_total_hybrid",
        "scf_steps_dav_sad_total_hybrid", "scf_steps_rmm_sad_total_hybrid",
        "scf_steps_other_sad_total_hybrid", "scf_steps_total_sad_total_hybrid",
        "time_sad_total_hybrid",
    ],
}


def schema_for(source: str, init: str):
    """Look up a schema by (source, init).

    `init` is the *strategy family*: "default", "true", or "ml". Both
    "charge3net" and "electrafi" share the "ml" schema.
    """
    if init in ("charge3net", "electrafi"):
        family = "ml"
    else:
        family = init
    return SCHEMAS[(source, family)]


def append_result(results_csv: str, header: list, row: list):
    write_header = not os.path.exists(results_csv)
    with open(results_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def upsert_result(results_csv: str, header: list, row: list, key_col: str | None = None):
    """Write ``row`` to ``results_csv``, replacing any existing row with the
    same key instead of appending a duplicate.

    Re-running a task (e.g. a slice job restarted past walltime, or a stale
    ``performed`` flag) used to append a fresh row each time, bloating the CSV
    with N copies of the same structure. Keying on the task id makes the write
    idempotent: the row is overwritten in place and the file holds exactly one
    row per task.

    ``key_col`` defaults to ``MP_ID`` when present in ``header`` (the MP
    runners), else ``CHGCAR_PATH`` (the GNOME runners). Returns True if an
    existing row was replaced.

    A slice's loop is serial, but we still take an exclusive lock and rewrite
    via a temp file + atomic ``os.replace`` so a stray concurrent invocation
    can neither interleave a half-written row nor truncate the file.
    """
    if key_col is None:
        key_col = "MP_ID" if "MP_ID" in header else "CHGCAR_PATH"
    key_idx = header.index(key_col)
    new_key = str(row[key_idx])
    new_row = ["" if v is None else v for v in row]

    lock_path = results_csv + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)

        kept: list[list] = []
        replaced = False
        if os.path.exists(results_csv):
            with open(results_csv, "r", newline="") as f:
                reader = csv.reader(f)
                next(reader, None)  # drop old header; we always rewrite our own
                for r in reader:
                    if not r:
                        continue
                    if key_idx < len(r) and str(r[key_idx]) == new_key:
                        replaced = True
                        continue
                    kept.append(r)
        kept.append(new_row)

        tmp = results_csv + ".tmp"
        with open(tmp, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(kept)
        os.replace(tmp, results_csv)
    return replaced
