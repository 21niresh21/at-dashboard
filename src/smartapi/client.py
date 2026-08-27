"""AngelOne SmartAPI client for market data.

Provides authenticated access to AngelOne's SmartAPI for fetching
historical OHLCV candle data.
"""

from __future__ import annotations

# Patch logzero for Python 3.11+ compatibility (must run before SmartApi import)
import src._logzero_compat  # noqa: F401

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from SmartApi import SmartConnect

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# Module-level client instance (lazy-initialized)
_client: SmartConnect | None = None


def _get_client() -> SmartConnect:
    """Return the singleton SmartConnect client, initializing if needed."""
    global _client  # noqa: PLW0603
    if _client is not None:
        return _client

    if not settings.smartapi_api_key:
        raise RuntimeError(
            "SmartAPI API key not configured. Set SMARTAPI_API_KEY in .env"
        )

    _client = SmartConnect(api_key=settings.smartapi_api_key)

    if settings.smartapi_client_id and settings.smartapi_password:
        try:
            totp = None
            if settings.smartapi_totp_secret:
                import pyotp

                totp = pyotp.TOTP(settings.smartapi_totp_secret).now()

            data = _client.generateSession(
                settings.smartapi_client_id,
                settings.smartapi_password,
                totp,
            )
            if data.get("status") and data.get("data", {}).get("jwtToken"):
                logger.info("SmartAPI session established for %s", settings.smartapi_client_id)
            else:
                logger.warning("SmartAPI login returned: %s", data.get("message", "unknown"))
        except Exception as exc:
            logger.warning("SmartAPI login failed: %s", exc)

    return _client


# ── Interval mapping ──────────────────────────────────────────────────────────

_INTERVAL_MAP = {
    "1d": "ONE_DAY",
    "1h": "ONE_HOUR",
    "15m": "FIFTEEN_MINUTE",
    "5m": "FIVE_MINUTE",
    "1m": "ONE_MINUTE",
}

_PERIOD_DAYS = {
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
}


def _map_interval(interval: str) -> str:
    """Map a common interval string to SmartAPI interval code."""
    return _INTERVAL_MAP.get(interval, "ONE_DAY")


def _map_period(period: str) -> tuple[str, str]:
    """Map a period string to (from_date, to_date) in SmartAPI format."""
    days = _PERIOD_DAYS.get(period, 90)
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    return from_date.strftime("%Y-%m-%d %H:%M"), to_date.strftime("%Y-%m-%d %H:%M")


# ── Public API ────────────────────────────────────────────────────────────────


def fetch_ohlcv(
    exchange: str,
    symbol_token: str,
    period: str = "3mo",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV candle data from AngelOne SmartAPI.

    Args:
        exchange: Exchange segment (``"NSE"``, ``"NFO"``, etc.).
        symbol_token: Numeric instrument token from the master script.
        period: Lookback period (``"1mo"``, ``"3mo"``, ``"6mo"``, ``"1y"``).
        interval: Candle interval (``"1d"``, ``"1h"``, ``"15m"``, ``"5m"``, ``"1m"``).

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
        Empty DataFrame if no data is available.
    """
    client = _get_client()
    smartapi_interval = _map_interval(interval)
    from_date, to_date = _map_period(period)

    param = {
        "exchange": exchange,
        "symboltoken": symbol_token,
        "interval": smartapi_interval,
        "fromdate": from_date,
        "todate": to_date,
    }

    try:
        response = client.getCandleData(param)
    except Exception as exc:
        logger.warning("SmartAPI candle fetch failed for %s/%s: %s", exchange, symbol_token, exc)
        return pd.DataFrame()

    if not response or not response.get("status") or not response.get("data"):
        logger.debug("No candle data for %s/%s", exchange, symbol_token)
        return pd.DataFrame()

    # SmartAPI returns data as list of lists: [timestamp, open, high, low, close, volume]
    raw_data = response["data"]
    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(
        raw_data,
        columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Close"], inplace=True)
    return df
