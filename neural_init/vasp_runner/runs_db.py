"""SQLite manifest for VASP run dedup and aggregation.

Organized as a collection of runs per ``mp_id``: one row per material in
``materials`` and one row per ``(mp_id, variant)`` in ``runs``. A single
``SELECT * FROM runs WHERE mp_id='mp-1234'`` returns every variant ever run
on that material side by side.

The per-slice CSVs written by the SCF drivers (the run_*_mp.py scripts) stay
as the parallel-safe write target; this manifest is populated by
``update_index.py``. Drivers consult the manifest only for the skip-check at
the top of each invocation.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar


SCHEMA = """
CREATE TABLE IF NOT EXISTS materials (
  mp_id       TEXT PRIMARY KEY,
  chgcar_path TEXT,
  formula     TEXT,
  updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS runs (
  mp_id       TEXT NOT NULL,
  variant     TEXT NOT NULL,
  status      TEXT,
  energy      REAL,
  total_mag   REAL,
  wall        REAL,
  scf_total   INTEGER,
  chgcar_path TEXT,
  source_csv  TEXT,
  source_row  INTEGER,
  ingested_at TEXT,
  result_json TEXT,
  PRIMARY KEY (mp_id, variant),
  FOREIGN KEY (mp_id) REFERENCES materials(mp_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_variant_status ON runs(variant, status);

CREATE TABLE IF NOT EXISTS ingest_log (
  path             TEXT PRIMARY KEY,
  size             INTEGER,
  mtime            REAL,
  sha256           TEXT,
  variant          TEXT,
  rows_ingested    INTEGER,
  last_ingested_at TEXT
);

CREATE TABLE IF NOT EXISTS inconsistencies (
  kind        TEXT,
  mp_id       TEXT,
  variant     TEXT,
  detail      TEXT,
  source_csv  TEXT,
  detected_at TEXT
);
"""


# Columns we promote out of result_json for direct querying.
_PROMOTED = ("status", "energy", "total_mag", "wall", "scf_total")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(db_path: str | os.PathLike) -> sqlite3.Connection:
    """Open (or create) the manifest DB and apply the schema.

    Uses a rollback journal (TRUNCATE), NOT WAL. WAL relies on mmap'd shared
    memory and POSIX byte-range locks that network filesystems (NFS/Lustre, as
    used for the cluster's shared storage) do not provide reliably; with many
    concurrent SLURM jobs across nodes opening the same file this corrupts the
    manifest ("database disk image is malformed"). A rollback journal works over
    network FS; the generous busy_timeout + run_with_retry serialize the
    infrequent writers (one row per structure, minutes apart).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=TRUNCATE")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


_T = TypeVar("_T")


def set_busy_timeout(conn: sqlite3.Connection, ms: int) -> None:
    """Raise this connection's busy timeout (used by maintenance tools that
    contend with many live SLURM writers — see ``run_with_retry``)."""
    conn.execute(f"PRAGMA busy_timeout={int(ms)}")


def run_with_retry(
    fn: Callable[[], _T],
    *,
    attempts: int = 10,
    base: float = 1.0,
    label: str = "runs.db write",
) -> _T:
    """Run a writing callable, retrying on 'database is locked'.

    Live SLURM jobs commit to the manifest constantly; SQLite serializes
    writers, so a maintenance write (DELETE/INSERT/ingest) may have to wait its
    turn beyond ``busy_timeout``. Exponential backoff (1,2,4,… s, capped at 30s)
    rides out the contention instead of crashing with OperationalError.
    """
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and i < attempts - 1:
                wait = min(base * (2 ** i), 30.0)
                print(f"  ⏳ {label}: runs.db locked (writers active); retry "
                      f"{i + 1}/{attempts - 1} in {wait:.0f}s…", flush=True)
                time.sleep(wait)
                continue
            raise
    # Unreachable: the final attempt either returns or raises above.
    raise RuntimeError("run_with_retry exhausted without returning")


# ---------------------------------------------------------------------------
# Skip-check API (read-only, called from drivers)
# ---------------------------------------------------------------------------

def is_done(conn: sqlite3.Connection, mp_id: str, variant: str) -> bool:
    """True iff a successful run for (mp_id, variant) exists in the manifest."""
    row = conn.execute(
        "SELECT 1 FROM runs WHERE mp_id=? AND variant=? AND status='ok' LIMIT 1",
        (mp_id, variant),
    ).fetchone()
    return row is not None


def done_set(
    conn: sqlite3.Connection,
    variant: str,
    mp_ids: Iterable[str],
) -> set[str]:
    """Return the subset of ``mp_ids`` with a successful run for ``variant``.

    Batched form used by the slice scan at the top of each driver.
    """
    return _select_mp_ids(conn, variant, mp_ids, where_status="status='ok'")


def attempted_set(
    conn: sqlite3.Connection,
    variant: str,
    mp_ids: Iterable[str],
) -> set[str]:
    """Return the subset of ``mp_ids`` with *any* recorded run for ``variant``.

    Used by drivers as the skip-check predicate: a row is "attempted" if the
    manifest has any status for it (``ok``, ``failed``, ``in_progress``,
    ``failed_no_taskdoc``, ``missing_chgcar``, …). Permanent failures stay
    skipped; transient failures must be cleared via reconcile_csvs.py
    before they will be retried.
    """
    return _select_mp_ids(conn, variant, mp_ids, where_status=None)


def terminal_set(
    conn: sqlite3.Connection,
    variant: str,
    mp_ids: Iterable[str],
) -> set[str]:
    """Return the subset of ``mp_ids`` with a *terminal* run for ``variant``.

    Like :func:`attempted_set` but excludes ``in_progress``. A row left
    ``in_progress`` by a sigkilled / walltime'd job never reached a terminal
    outcome and must be retryable, so it is reported as *not* done here.
    Terminal statuses (``ok``, ``failed``, ``failed_no_taskdoc``,
    ``missing_chgcar``) stay skipped. This is the skip-check the drivers use so
    orphaned mid-run rows are automatically re-picked on the next invocation.
    """
    return _select_mp_ids(
        conn, variant, mp_ids,
        where_status="status IS NOT NULL AND status<>'in_progress'",
    )


def _select_mp_ids(
    conn: sqlite3.Connection,
    variant: str,
    mp_ids: Iterable[str],
    where_status: str | None,
) -> set[str]:
    ids = [m for m in (str(x) for x in mp_ids) if m and m.lower() != "nan"]
    if not ids:
        return set()
    out: set[str] = set()
    chunk = 500  # stay under SQLite's parameter limit
    status_clause = f" AND {where_status}" if where_status else ""
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        placeholders = ",".join("?" * len(batch))
        sql = (
            f"SELECT mp_id FROM runs WHERE variant=?{status_clause} "
            f"AND mp_id IN ({placeholders})"
        )
        rows = conn.execute(sql, (variant, *batch)).fetchall()
        out.update(r["mp_id"] for r in rows)
    return out


def runs_for(conn: sqlite3.Connection, mp_id: str) -> dict[str, dict[str, Any]]:
    """Return ``{variant: row_dict}`` for the given material.

    Variant-specific fields stored in ``result_json`` are merged into the row
    so callers get one flat dict per variant.
    """
    rows = conn.execute(
        "SELECT * FROM runs WHERE mp_id=? ORDER BY variant", (mp_id,)
    ).fetchall()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        d = dict(r)
        rj = d.pop("result_json", None)
        if rj:
            try:
                extra = json.loads(rj)
                for k, v in extra.items():
                    d.setdefault(k, v)
            except json.JSONDecodeError:
                pass
        out[d["variant"]] = d
    return out


# ---------------------------------------------------------------------------
# Write API (used by update_index.py and optionally by drivers)
# ---------------------------------------------------------------------------

def _upsert_material(
    conn: sqlite3.Connection,
    *,
    mp_id: str,
    chgcar_path: str | None,
    formula: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO materials (mp_id, chgcar_path, formula, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(mp_id) DO UPDATE SET
          chgcar_path = COALESCE(excluded.chgcar_path, materials.chgcar_path),
          formula     = COALESCE(excluded.formula,     materials.formula),
          updated_at  = excluded.updated_at
        """,
        (mp_id, chgcar_path, formula, _utcnow()),
    )


def record_run(
    conn: sqlite3.Connection,
    *,
    mp_id: str,
    variant: str,
    chgcar_path: str | None,
    status: str | None,
    result: dict[str, Any],
    source_csv: str | None = None,
    source_row: int | None = None,
) -> None:
    """Insert or replace one (mp_id, variant) row.

    ``result`` is the variant-prefixed dict (e.g. {energy_<v>, total_mag_<v>,
    scf_steps_total_<v>, time_<v>, ...}). Promoted columns are extracted with
    the trailing ``_<variant>`` suffix stripped; the full dict is stored as
    JSON in ``result_json`` for forward-compat.
    """
    stripped = _strip_variant_suffix(result, variant)
    energy = _to_float(stripped.get("energy"))
    total_mag = _to_float(stripped.get("total_mag"))
    wall = _to_float(stripped.get("time", stripped.get("wall")))
    scf_total = _to_int(stripped.get("scf_steps_total", stripped.get("scf_total")))
    if status is None:
        status = stripped.get("status")

    _upsert_material(conn, mp_id=mp_id, chgcar_path=chgcar_path)

    conn.execute(
        """
        INSERT OR REPLACE INTO runs
          (mp_id, variant, status, energy, total_mag, wall, scf_total,
           chgcar_path, source_csv, source_row, ingested_at, result_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            mp_id,
            variant,
            status,
            energy,
            total_mag,
            wall,
            scf_total,
            chgcar_path,
            source_csv,
            source_row,
            _utcnow(),
            json.dumps(stripped, default=str),
        ),
    )


def write_run_if_absent(
    conn: sqlite3.Connection | None,
    *,
    mp_id: str,
    variant: str,
    chgcar_path: str | None,
    status: str,
    source_csv: str | None = None,
    source_row: int | None = None,
) -> bool:
    """Like ``write_run`` but a no-op if ``(mp_id, variant)`` already exists.

    Used for ``in_progress`` pre-registration in multi-variant drivers, so
    a prior successful run for one variant isn't downgraded when other
    variants get retried.
    """
    if conn is None:
        return False
    try:
        row = conn.execute(
            "SELECT 1 FROM runs WHERE mp_id=? AND variant=? LIMIT 1",
            (mp_id, variant),
        ).fetchone()
        if row is not None:
            return False
        return write_run(
            conn, mp_id=mp_id, variant=variant, chgcar_path=chgcar_path,
            status=status, source_csv=source_csv, source_row=source_row,
        )
    except Exception as e:
        print(f"⚠️ runs_db.write_run_if_absent({mp_id},{variant}): {e!r}")
        return False


def write_run(
    conn: sqlite3.Connection | None,
    *,
    mp_id: str,
    variant: str,
    chgcar_path: str | None,
    status: str,
    result: dict[str, Any] | None = None,
    source_csv: str | None = None,
    source_row: int | None = None,
) -> bool:
    """Driver-side: ``record_run`` + commit, swallowing errors with a warning.

    Returns True on success, False if the write failed (e.g. DB locked
    beyond ``busy_timeout``, missing dir, …). Callers may pass conn=None
    (manifest unavailable at startup) — this is a no-op.
    """
    if conn is None:
        return False
    try:
        record_run(
            conn,
            mp_id=mp_id,
            variant=variant,
            chgcar_path=chgcar_path,
            status=status,
            result=result or {},
            source_csv=source_csv,
            source_row=source_row,
        )
        conn.commit()
        return True
    except Exception as e:
        print(f"⚠️ runs_db.write_run({mp_id},{variant},{status}): {e!r}")
        return False


# ---------------------------------------------------------------------------
# CSV ingest helpers (driven by update_index.py)
# ---------------------------------------------------------------------------

def detect_variants_from_columns(columns: Iterable[str]) -> list[str]:
    """Pull variant names out of any ``status_<variant>`` columns."""
    out = []
    for c in columns:
        if c.startswith("status_") and len(c) > len("status_"):
            out.append(c[len("status_") :])
    # de-dup, preserve order
    seen = set()
    return [v for v in out if not (v in seen or seen.add(v))]


def _strip_variant_suffix(row: dict[str, Any], variant: str) -> dict[str, Any]:
    """Map e.g. ``energy_conv_grid_sad_aug`` → ``energy`` for this variant."""
    suffix = f"_{variant}"
    out: dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v == "" or (isinstance(v, float) and v != v):  # NaN
            continue
        if k.endswith(suffix):
            out[k[: -len(suffix)]] = v
        elif not k.startswith(("status_", "energy_", "total_mag_", "spin_",
                                "scf_steps_", "time_")):
            # Common columns (CHGCAR_PATH, MP_ID, csv_index, slice_*) — keep.
            out[k] = v
    return out


def _to_float(x: Any) -> float | None:
    if x is None or x == "":
        return None
    try:
        f = float(x)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _to_int(x: Any) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return None


def upsert_log(
    conn: sqlite3.Connection,
    *,
    path: str,
    size: int,
    mtime: float,
    sha256: str,
    variant: str | None,
    rows_ingested: int,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO ingest_log
          (path, size, mtime, sha256, variant, rows_ingested, last_ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (path, size, mtime, sha256, variant, rows_ingested, _utcnow()),
    )


def get_log_entry(conn: sqlite3.Connection, path: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM ingest_log WHERE path=?", (path,)
    ).fetchone()


def record_inconsistency(
    conn: sqlite3.Connection,
    *,
    kind: str,
    mp_id: str | None,
    variant: str | None,
    detail: str,
    source_csv: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO inconsistencies
          (kind, mp_id, variant, detail, source_csv, detected_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (kind, mp_id, variant, detail, source_csv, _utcnow()),
    )


def clear_inconsistencies(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM inconsistencies")
