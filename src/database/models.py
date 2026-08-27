"""SQLite database layer for instrument master data.

Schema design (2NF-normalised):
    exchanges          — lookup table for exchange segment codes
    instrument_types   — lookup table for instrument type codes
    instruments        — one row per tradable instrument (FK → exchanges, instrument_types)
    sync_log           — audit trail for data synchronisation runs

2NF compliance:
    • Every non-key column depends on the **whole** primary key.
    • The instruments table's composite dependency on exchange_seg and
      instrument_type is resolved via foreign keys to dedicated lookup
      tables, eliminating partial dependencies.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from src.config import settings
from src.logging_config import get_logger
from src.smartapi.instruments import InstrumentRecord

logger = get_logger(__name__)

# ── Schema DDL ────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
-- Exchange segment lookup (NSE, NFO, BSE, MCX, …)
CREATE TABLE IF NOT EXISTS exchanges (
    exchange_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_code TEXT    NOT NULL UNIQUE
);

-- Instrument type lookup (FUTSTK, OPTSTK, INDEX, …)
CREATE TABLE IF NOT EXISTS instrument_types (
    type_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    type_code TEXT    NOT NULL UNIQUE
);

-- Core instrument table
CREATE TABLE IF NOT EXISTS instruments (
    token           TEXT    PRIMARY KEY,
    symbol          TEXT    NOT NULL,
    name            TEXT    NOT NULL,
    exchange_id     INTEGER NOT NULL REFERENCES exchanges(exchange_id),
    type_id         INTEGER NOT NULL REFERENCES instrument_types(type_id),
    expiry          TEXT    NOT NULL DEFAULT '',
    strike_price    REAL    NOT NULL DEFAULT 0.0,
    lot_size        INTEGER NOT NULL DEFAULT 0,
    tick_size       REAL    NOT NULL DEFAULT 0.0,
    freeze_qty      INTEGER NOT NULL DEFAULT 0,
    is_cas_enabled  INTEGER NOT NULL DEFAULT 0
);

-- Sync audit log
CREATE TABLE IF NOT EXISTS sync_log (
    sync_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    NOT NULL,
    completed_at TEXT,
    record_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT    NOT NULL DEFAULT 'running',
    message     TEXT    NOT NULL DEFAULT ''
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_instruments_exchange ON instruments(exchange_id);
CREATE INDEX IF NOT EXISTS idx_instruments_type     ON instruments(type_id);
CREATE INDEX IF NOT EXISTS idx_instruments_name     ON instruments(name);
CREATE INDEX IF NOT EXISTS idx_instruments_symbol   ON instruments(symbol);
CREATE INDEX IF NOT EXISTS idx_instruments_expiry   ON instruments(expiry);
"""


# ── Connection management ─────────────────────────────────────────────────────


def _ensure_db_dir() -> None:
    """Create the parent directory for the database file if needed."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with foreign keys enabled.

    The connection is automatically committed on success or rolled back
    on exception, then closed.
    """
    _ensure_db_dir()
    conn = sqlite3.connect(str(settings.db_path), timeout=10)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema bootstrap ──────────────────────────────────────────────────────────


def init_schema() -> None:
    """Create all tables and indexes if they don't already exist."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Database schema initialised at %s", settings.db_path)


# ── Lookup table helpers ──────────────────────────────────────────────────────


def _get_or_create_exchange(conn: sqlite3.Connection, code: str) -> int:
    """Return the exchange_id for *code*, inserting if necessary."""
    row = conn.execute(
        "SELECT exchange_id FROM exchanges WHERE exchange_code = ?", (code,)
    ).fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO exchanges (exchange_code) VALUES (?)", (code,))
    return conn.execute(
        "SELECT exchange_id FROM exchanges WHERE exchange_code = ?", (code,)
    ).fetchone()[0]


def _get_or_create_instrument_type(conn: sqlite3.Connection, code: str) -> int:
    """Return the type_id for *code*, inserting if necessary."""
    row = conn.execute(
        "SELECT type_id FROM instrument_types WHERE type_code = ?", (code,)
    ).fetchone()
    if row:
        return row[0]
    conn.execute("INSERT INTO instrument_types (type_code) VALUES (?)", (code,))
    return conn.execute(
        "SELECT type_id FROM instrument_types WHERE type_code = ?", (code,)
    ).fetchone()[0]


# ── Bulk sync ─────────────────────────────────────────────────────────────────


def sync_instruments(records: list[InstrumentRecord]) -> int:
    """Replace the full instrument table with *records*.

    This is an idempotent full-refresh sync:
    1. Logs the sync start.
    2. Clears existing instrument rows.
    3. Deduplicates records by token (AngelOne CDN may contain duplicates).
    4. Bulk-inserts all records (upserting exchange/type lookups).
    5. Logs completion with the record count.

    Returns:
        The number of instruments inserted.
    """
    started_at = datetime.now(timezone.utc).isoformat()

    # Deduplicate by token — keep first occurrence
    seen_tokens: set[str] = set()
    unique_records: list[InstrumentRecord] = []
    for r in records:
        if r.token not in seen_tokens:
            seen_tokens.add(r.token)
            unique_records.append(r)

    with get_connection() as conn:
        # Start sync log entry
        conn.execute(
            "INSERT INTO sync_log (started_at, status) VALUES (?, 'running')",
            (started_at,),
        )
        sync_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        try:
            # Pre-load lookup caches for the batch
            exchange_cache: dict[str, int] = {}
            type_cache: dict[str, int] = {}

            for code in {r.exchange_seg for r in unique_records}:
                exchange_cache[code] = _get_or_create_exchange(conn, code)
            for code in {r.instrument_type for r in unique_records}:
                type_cache[code] = _get_or_create_instrument_type(conn, code)

            # Clear existing instruments for a clean refresh
            conn.execute("DELETE FROM instruments")

            # Bulk insert
            conn.executemany(
                """
                INSERT INTO instruments
                    (token, symbol, name, exchange_id, type_id,
                     expiry, strike_price, lot_size, tick_size,
                     freeze_qty, is_cas_enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.token,
                        r.symbol,
                        r.name,
                        exchange_cache[r.exchange_seg],
                        type_cache[r.instrument_type],
                        r.expiry,
                        r.strike_price,
                        r.lot_size,
                        r.tick_size,
                        r.freeze_qty,
                        int(r.is_cas_enabled),
                    )
                    for r in unique_records
                ],
            )

            # Update sync log
            completed_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE sync_log
                SET completed_at = ?, record_count = ?, status = 'success', message = ''
                WHERE sync_id = ?
                """,
                (completed_at, len(unique_records), sync_id),
            )

            logger.info(
                "Synced %d instruments (%d duplicates removed, sync_id=%d).",
                len(unique_records),
                len(records) - len(unique_records),
                sync_id,
            )
            return len(unique_records)

        except Exception as exc:
            completed_at = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                UPDATE sync_log
                SET completed_at = ?, status = 'failed', message = ?
                WHERE sync_id = ?
                """,
                (completed_at, str(exc), sync_id),
            )
            logger.exception("Sync failed (sync_id=%d).", sync_id)
            raise


# ── Query helpers ─────────────────────────────────────────────────────────────


def get_nse_instruments(
    conn: sqlite3.Connection | None = None,
    search: str = "",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return NSE cash-segment instruments, optionally filtered by name/symbol."""
    return _query_instruments(
        exchange_code="NSE",
        search=search,
        limit=limit,
        conn=conn,
    )


def get_fo_instruments(
    conn: sqlite3.Connection | None = None,
    search: str = "",
    underlying: str = "",
    instrument_type: str = "",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return NSE F&O instruments with optional filters."""
    return _query_instruments(
        exchange_code="NFO",
        search=search,
        underlying=underlying,
        instrument_type=instrument_type,
        limit=limit,
        conn=conn,
    )


def get_indices(
    conn: sqlite3.Connection | None = None,
    search: str = "",
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Return index instruments (INDEX + AMXIDX types) across all exchanges."""
    with _maybe_open(conn) as c:
        rows = c.execute(
            """
            SELECT i.token, i.symbol, i.name, e.exchange_code, t.type_code,
                   i.expiry, i.strike_price, i.lot_size, i.tick_size,
                   i.freeze_qty, i.is_cas_enabled
            FROM instruments i
            JOIN exchanges e        ON i.exchange_id = e.exchange_id
            JOIN instrument_types t ON i.type_id    = t.type_id
            WHERE t.type_code IN ('INDEX', 'AMXIDX')
              AND (:search = '' OR i.name LIKE :search_pat OR i.symbol LIKE :search_pat)
            ORDER BY i.name
            LIMIT :limit
            """,
            {
                "search": search,
                "search_pat": f"%{search}%",
                "limit": limit,
            },
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_fo_underlying_names(
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Return distinct underlying stock names that have F&O contracts."""
    with _maybe_open(conn) as c:
        rows = c.execute(
            """
            SELECT DISTINCT i.name
            FROM instruments i
            JOIN exchanges e        ON i.exchange_id = e.exchange_id
            JOIN instrument_types t ON i.type_id    = t.type_id
            WHERE e.exchange_code = 'NFO'
              AND t.type_code IN ('FUTSTK', 'OPTSTK')
            ORDER BY i.name
            """
        ).fetchall()
    return [row[0] for row in rows]


def get_sync_history(limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent sync log entries."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT sync_id, started_at, completed_at, record_count, status, message
            FROM sync_log
            ORDER BY sync_id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            "sync_id": r[0],
            "started_at": r[1],
            "completed_at": r[2],
            "record_count": r[3],
            "status": r[4],
            "message": r[5],
        }
        for r in rows
    ]


def get_instrument_counts() -> dict[str, int]:
    """Return instrument counts grouped by exchange segment."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT e.exchange_code, COUNT(*)
            FROM instruments i
            JOIN exchanges e ON i.exchange_id = e.exchange_id
            GROUP BY e.exchange_code
            ORDER BY e.exchange_code
            """
        ).fetchall()
    return {row[0]: row[1] for row in rows}


def get_last_sync() -> dict[str, Any] | None:
    """Return the most recent sync log entry, or None if never synced."""
    history = get_sync_history(limit=1)
    return history[0] if history else None


# ── Internal helpers ──────────────────────────────────────────────────────────


@contextmanager
def _maybe_open(conn: sqlite3.Connection | None) -> Generator[sqlite3.Connection, None, None]:
    """Use the provided connection or open a new one."""
    if conn is not None:
        yield conn
        return
    with get_connection() as c:
        yield c


def _query_instruments(
    exchange_code: str,
    search: str = "",
    underlying: str = "",
    instrument_type: str = "",
    limit: int = 500,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Generic instrument query with optional filters."""
    with _maybe_open(conn) as c:
        query = """
            SELECT i.token, i.symbol, i.name, e.exchange_code, t.type_code,
                   i.expiry, i.strike_price, i.lot_size, i.tick_size,
                   i.freeze_qty, i.is_cas_enabled
            FROM instruments i
            JOIN exchanges e        ON i.exchange_id = e.exchange_id
            JOIN instrument_types t ON i.type_id    = t.type_id
            WHERE e.exchange_code = :exchange_code
              AND (:search = '' OR i.name LIKE :search_pat OR i.symbol LIKE :search_pat)
              AND (:underlying = '' OR i.name = :underlying)
              AND (:instrument_type = '' OR t.type_code = :instrument_type)
            ORDER BY i.name, i.expiry, i.strike_price
            LIMIT :limit
        """
        rows = c.execute(
            query,
            {
                "exchange_code": exchange_code,
                "search": search,
                "search_pat": f"%{search}%",
                "underlying": underlying,
                "instrument_type": instrument_type,
                "limit": limit,
            },
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    """Map a query result tuple to a dict."""
    return {
        "token": row[0],
        "symbol": row[1],
        "name": row[2],
        "exchange": row[3],
        "type": row[4],
        "expiry": row[5],
        "strike_price": row[6],
        "lot_size": row[7],
        "tick_size": row[8],
        "freeze_qty": row[9],
        "is_cas_enabled": bool(row[10]),
    }
