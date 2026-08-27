"""AngelOne SmartAPI — master instrument data fetcher.

Downloads the full instrument master script from AngelOne's public CDN
and returns structured data for storage in the local database.

The master script is a public JSON endpoint that does **not** require
authentication.  It contains every tradable instrument across all
exchanges (NSE, BSE, NFO, MCX, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Exchange segments of interest
NSE_CASH = "NSE"  # NSE cash / equity segment
NFO = "NFO"  # NSE F&O segment

# Instrument type codes
INDEX = "INDEX"  # Cash-market indices
AMXIDX = "AMXIDX"  # Extended-matrix indices (thematic, strategy, etc.)
FUTIDX = "FUTIDX"  # Index futures
OPTIDX = "OPTIDX"  # Index options
FUTSTK = "FUTSTK"  # Stock futures
OPTSTK = "OPTSTK"  # Stock options

# All F&O-relevant instrument types
FO_INSTRUMENT_TYPES = frozenset({FUTIDX, OPTIDX, FUTSTK, OPTSTK})

# All index-type instrument types
INDEX_INSTRUMENT_TYPES = frozenset({INDEX, AMXIDX})

# Request timeout in seconds
_REQUEST_TIMEOUT = 60


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InstrumentRecord:
    """A single instrument row from the master script."""

    token: str
    symbol: str
    name: str
    expiry: str
    strike_price: float
    lot_size: int
    instrument_type: str
    exchange_seg: str
    tick_size: float
    freeze_qty: int
    is_cas_enabled: bool


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_all_instruments() -> list[InstrumentRecord]:
    """Download and parse the full master instrument list from AngelOne.

    Returns:
        A list of ``InstrumentRecord`` instances — one per row in the
        master script JSON.

    Raises:
        requests.HTTPError: If the CDN returns a non-200 status.
        ValueError: If the response body is not a valid JSON list.
    """
    logger.info("Fetching master instrument list from AngelOne CDN …")
    response = requests.get(settings.smartapi_master_url, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()

    raw: Any = response.json()
    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON list, got {type(raw).__name__}")

    records = [_parse_record(row) for row in raw]
    logger.info("Fetched %d instruments from AngelOne.", len(records))
    return records


def filter_nse_cash(records: list[InstrumentRecord]) -> list[InstrumentRecord]:
    """Return only NSE cash-segment instruments (equities + indices)."""
    return [r for r in records if r.exchange_seg == NSE_CASH]


def filter_fo_instruments(records: list[InstrumentRecord]) -> list[InstrumentRecord]:
    """Return only NSE F&O instruments (stock/index futures & options)."""
    return [
        r
        for r in records
        if r.exchange_seg == NFO and r.instrument_type in FO_INSTRUMENT_TYPES
    ]


def filter_indices(records: list[InstrumentRecord]) -> list[InstrumentRecord]:
    """Return only index instruments across all exchange segments."""
    return [r for r in records if r.instrument_type in INDEX_INSTRUMENT_TYPES]


def get_fo_stock_names(fo_records: list[InstrumentRecord]) -> set[str]:
    """Extract the distinct underlying stock names from F&O records."""
    return {r.name for r in fo_records if r.instrument_type in FO_INSTRUMENT_TYPES}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _parse_record(row: dict[str, Any]) -> InstrumentRecord:
    """Convert a raw JSON dict into a typed ``InstrumentRecord``."""
    return InstrumentRecord(
        token=str(row.get("token", "")),
        symbol=str(row.get("symbol", "")),
        name=str(row.get("name", "")),
        expiry=str(row.get("expiry", "")),
        strike_price=float(row.get("strike", 0)),
        lot_size=int(row.get("lotsize", 0)),
        instrument_type=str(row.get("instrumenttype", "")),
        exchange_seg=str(row.get("exch_seg", "")),
        tick_size=float(row.get("tick_size", 0)),
        freeze_qty=int(float(row.get("freeze_qty", 0))),
        is_cas_enabled=bool(row.get("is_cas_enabled", False)),
    )
