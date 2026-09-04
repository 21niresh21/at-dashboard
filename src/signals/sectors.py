"""Sector / industry mapping for F&O underlying stocks.

Provides a static mapping of Indian F&O stocks to their industry sectors.
Used by the scalper page to filter scan targets by industry.
"""

from __future__ import annotations

# ── Sector → Stocks mapping ───────────────────────────────────────────────────
# Covers the major F&O underlying stocks grouped by industry sector.
# Stock names must match the AngelOne instrument master (uppercase).

SECTOR_STOCKS: dict[str, list[str]] = {
    "Banking": [
        "HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "SBIN",
        "BANKBARODA", "INDUSINDBK", "PNB", "CANBK", "BANDHANBNK",
        "AUBANK", "FEDERALBNK", "IDFCFIRSTBANK",
    ],
    "IT": [
        "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTTS",
        "MPHASIS", "COFORGE", "PERSISTENT", "LTIM",
    ],
    "Auto": [
        "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT",
        "HEROMOTOCO", "TVSMOTOR", "BHARATFORG", "MRF", "BOSCHLTD",
        "ASHOKLEY",
    ],
    "Pharma": [
        "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
        "BIOCON", "LUPIN", "AUROPHARMA", "TORNTPHARM", "GLAND",
    ],
    "Metals & Mining": [
        "TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "VEDL",
        "NMDC", "NATIONALUM", "SAIL", "JINDALSTEL",
    ],
    "Energy & Power": [
        "RELIANCE", "ONGC", "NTPC", "POWERGRID", "BPCL", "IOC",
        "ADANIGREEN", "ADANIPORTS", "TATAPOWER",
    ],
    "FMCG": [
        "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR",
        "MARICO", "GODREJCP", "COLPAL", "MCDOWELL-N", "UNITDSPR",
    ],
    "Capital Goods & Infra": [
        "LT", "SIEMENS", "ABB", "HAVELLS", "BHEL", "BEL",
        "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE",
    ],
    "Financial Services": [
        "BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
        "ICICIPRULI", "CHOLAFIN", "M&MFIN", "SHRIRAMFIN",
    ],
    "Telecom & Media": [
        "BHARTIARTL", "IDEA", "ZEEL", "PVR", "SUNTV",
    ],
    "Chemicals": [
        "PIDILITIND", "SRF", "UPL", "AARTIIND", "DEEPAKNTR",
        "ATUL", "CLEAN", "NAVINFLUOR",
    ],
    "Cement": [
        "ULTRACEMCO", "GRASIM", "SHREECEM", "AMBUJACEM",
        "ACC", "DALKHIAST", "RAMCOCEM",
    ],
}


def get_sectors() -> list[str]:
    """Return all available sector names, sorted alphabetically."""
    return sorted(SECTOR_STOCKS.keys())


def get_stocks_by_sector(sector: str) -> list[str]:
    """Return stock names for a given sector.

    Args:
        sector: Sector name (must be one of ``get_sectors()``).

    Returns:
        List of uppercase stock names, or empty list if sector not found.
    """
    return list(SECTOR_STOCKS.get(sector, []))


def get_stock_sector(stock_name: str) -> str | None:
    """Return the sector for a given stock name, or None if unmapped."""
    upper = stock_name.upper()
    for sector, stocks in SECTOR_STOCKS.items():
        if upper in stocks:
            return sector
    return None
