"""Scalper page — fast-timeframe scalping scanner.

Scans F&O stocks on intraday candles (1m / 5m) and computes
VWAP reversion, Bollinger Band squeeze / breakout, and ATR-based
volatility signals for scalping setups.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from src.components.ui import page_header
from src.database import models as db
from src.logging_config import get_logger
from src.signals.technical import (
    compute_atr,
    compute_bollinger_bands,
    compute_vwap,
)
from src.smartapi.client import fetch_ohlcv

logger = get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_VWAP_INTERVAL = "5m"
_DEFAULT_BB_PERIOD = 20
_DEFAULT_BB_STD = 2.0
_DEFAULT_BB_SQUEEZE_PCT = 10.0  # bandwidth percentile below which = squeeze
_DEFAULT_ATR_PERIOD = 14
_DEFAULT_ATR_MIN = 0.5  # minimum ATR as % of price to be worth scalping

# Session state keys
_KEY_SCALP_INTERVAL = "scalp_interval"
_KEY_VWAP_SIGNALS = "vwap_signals"
_KEY_BB_SIGNALS = "bb_signals"
_KEY_ATR_SIGNALS = "atr_signals"
_KEY_SCANNING = "scalp_scanning"


# ── Public entry point ────────────────────────────────────────────────────────


def render() -> None:
    """Render the scalper page."""
    page_header(
        title="Scalper",
        description=(
            "Fast-timeframe scalping scanner. Uses intraday candles (1m / 5m) "
            "to find VWAP reversion, Bollinger Band, and volatility setups."
        ),
    )

    # Global timeframe selector
    _render_timeframe_selector()

    tab_vwap, tab_bb, tab_atr = st.tabs(
        ["VWAP Reversion", "Bollinger Bands", "ATR Volatility"]
    )

    with tab_vwap:
        _render_vwap_tab()
    with tab_bb:
        _render_bb_tab()
    with tab_atr:
        _render_atr_tab()


# ── Timeframe selector ────────────────────────────────────────────────────────


def _render_timeframe_selector() -> None:
    """Global candle interval selector shared across all tabs."""
    if _KEY_SCALP_INTERVAL not in st.session_state:
        st.session_state[_KEY_SCALP_INTERVAL] = _DEFAULT_VWAP_INTERVAL

    col1, col2 = st.columns([1, 3])
    with col1:
        interval = st.selectbox(
            "Candle Interval",
            options=["1m", "5m"],
            index=["1m", "5m"].index(st.session_state[_KEY_SCALP_INTERVAL]),
            key="scalp_interval_select",
            help="Intraday candle interval for scalping scans.",
        )
    st.session_state[_KEY_SCALP_INTERVAL] = interval


# ── VWAP Reversion Tab ───────────────────────────────────────────────────────


def _render_vwap_tab() -> None:
    """VWAP reversion scalping tab."""
    st.subheader("VWAP Reversion")
    st.markdown(
        "Scans for stocks where price is near VWAP — a key intraday "
        "support / resistance level. Signals indicate potential bounces "
        "or rejections around VWAP."
    )

    st.divider()
    _render_scan_controls(strategy="vwap")

    signals = st.session_state.get(_KEY_VWAP_SIGNALS, [])
    if signals:
        _render_vwap_results(signals)


def _render_vwap_results(signals: list[dict]) -> None:
    """Display VWAP scan results."""
    st.subheader("VWAP Scan Results")

    buy = [s for s in signals if s["signal"] == "BUY"]
    sell = [s for s in signals if s["signal"] == "SELL"]
    neutral = [s for s in signals if s["signal"] == "NEUTRAL"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Scanned", len(signals))
    with col2:
        st.metric("Near VWAP (BUY)", len(buy))
    with col3:
        st.metric("Above VWAP (SELL)", len(sell))
    with col4:
        st.metric("Neutral", len(neutral))
    st.divider()

    df = pd.DataFrame(signals)
    display_cols = [
        "symbol", "close", "vwap", "distance_pct", "signal", "reason",
    ]
    st.dataframe(
        df[display_cols], use_container_width=True, hide_index=True,
    )

    if buy:
        st.markdown(f"### 🟢 Near VWAP — Potential Bounce ({len(buy)})")
        buy_df = pd.DataFrame(buy)
        st.dataframe(
            buy_df[display_cols], use_container_width=True, hide_index=True,
        )

    if sell:
        st.markdown(f"### 🔴 Above VWAP — Potential Rejection ({len(sell)})")
        sell_df = pd.DataFrame(sell)
        st.dataframe(
            sell_df[display_cols], use_container_width=True, hide_index=True,
        )


# ── Bollinger Bands Tab ──────────────────────────────────────────────────────


def _render_bb_tab() -> None:
    """Bollinger Band squeeze / breakout scalping tab."""
    st.subheader("Bollinger Bands")
    st.markdown(
        "Detects Bollinger Band squeezes (low volatility) and breakouts "
        "(price touching or piercing a band). Squeezes often precede "
        "sharp moves — a favourite scalping setup."
    )

    # Config
    col1, col2, col3 = st.columns(3)
    with col1:
        bb_period = st.number_input(
            "BB Period", min_value=5, max_value=100, value=_DEFAULT_BB_PERIOD,
            step=1, key="bb_cfg_period",
            help="Moving average lookback period. Default: 20",
        )
    with col2:
        bb_std = st.number_input(
            "Std Dev Multiplier", min_value=0.5, max_value=4.0,
            value=_DEFAULT_BB_STD, step=0.5, key="bb_cfg_std",
            help="Number of standard deviations for band width. Default: 2.0",
        )
    with col3:
        squeeze_pct = st.number_input(
            "Squeeze Threshold (%)", min_value=1.0, max_value=30.0,
            value=_DEFAULT_BB_SQUEEZE_PCT, step=1.0, key="bb_cfg_squeeze",
            help="Bandwidth percentile below which = squeeze. Default: 10%",
        )

    st.divider()
    _render_scan_controls(
        strategy="bb",
        extra_config={"period": int(bb_period), "std": float(bb_std), "squeeze_pct": float(squeeze_pct)},
    )

    signals = st.session_state.get(_KEY_BB_SIGNALS, [])
    if signals:
        _render_bb_results(signals)


def _render_bb_results(signals: list[dict]) -> None:
    """Display Bollinger Band scan results."""
    st.subheader("Bollinger Band Scan Results")

    squeeze = [s for s in signals if s["signal"] == "SQUEEZE"]
    upper_touch = [s for s in signals if s["signal"] == "UPPER_TOUCH"]
    lower_touch = [s for s in signals if s["signal"] == "LOWER_TOUCH"]
    inside = [s for s in signals if s["signal"] == "INSIDE"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Scanned", len(signals))
    with col2:
        st.metric("Squeeze", len(squeeze))
    with col3:
        st.metric("Upper Touch", len(upper_touch))
    with col4:
        st.metric("Lower Touch", len(lower_touch))
    with col5:
        st.metric("Inside Bands", len(inside))
    st.divider()

    df = pd.DataFrame(signals)
    display_cols = [
        "symbol", "close", "bb_upper", "bb_lower", "bb_width_pct",
        "signal", "reason",
    ]
    st.dataframe(
        df[display_cols], use_container_width=True, hide_index=True,
    )

    if squeeze:
        st.markdown(f"### 🔵 Squeeze — Low Volatility ({len(squeeze)})")
        sq_df = pd.DataFrame(squeeze)
        st.dataframe(
            sq_df[display_cols], use_container_width=True, hide_index=True,
        )

    if upper_touch:
        st.markdown(f"### 🔴 Upper Band Touch ({len(upper_touch)})")
        ut_df = pd.DataFrame(upper_touch)
        st.dataframe(
            ut_df[display_cols], use_container_width=True, hide_index=True,
        )

    if lower_touch:
        st.markdown(f"### 🟢 Lower Band Touch ({len(lower_touch)})")
        lt_df = pd.DataFrame(lower_touch)
        st.dataframe(
            lt_df[display_cols], use_container_width=True, hide_index=True,
        )


# ── ATR Volatility Tab ───────────────────────────────────────────────────────


def _render_atr_tab() -> None:
    """ATR-based volatility filter and ranking tab."""
    st.subheader("ATR Volatility Filter")
    st.markdown(
        "Ranks stocks by Average True Range (ATR) as a percentage of price. "
        "High ATR% means the stock moves more intraday — better for scalping. "
        "Low ATR% stocks are filtered out as not worth the spread."
    )

    # Config
    col1, col2 = st.columns(2)
    with col1:
        atr_period = st.number_input(
            "ATR Period", min_value=5, max_value=100, value=_DEFAULT_ATR_PERIOD,
            step=1, key="atr_cfg_period",
            help="ATR lookback period. Default: 14",
        )
    with col2:
        atr_min_pct = st.number_input(
            "Min ATR % of Price", min_value=0.0, max_value=10.0,
            value=_DEFAULT_ATR_MIN, step=0.1, key="atr_cfg_min",
            help="Minimum ATR as %% of price to be worth scalping. Default: 0.5%",
        )

    st.divider()
    _render_scan_controls(
        strategy="atr",
        extra_config={"period": int(atr_period), "min_pct": float(atr_min_pct)},
    )

    signals = st.session_state.get(_KEY_ATR_SIGNALS, [])
    if signals:
        _render_atr_results(signals, float(atr_min_pct))


def _render_atr_results(signals: list[dict], min_pct: float) -> None:
    """Display ATR volatility scan results."""
    st.subheader("ATR Volatility Results")

    tradeable = [s for s in signals if s["atr_pct"] >= min_pct]
    illiquid = [s for s in signals if s["atr_pct"] < min_pct]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Scanned", len(signals))
    with col2:
        st.metric("Scalable (≥ min ATR%)", len(tradeable))
    with col3:
        st.metric("Filtered Out", len(illiquid))
    st.divider()

    if tradeable:
        st.markdown(f"### 🟢 Scalable Stocks — ATR% ≥ {min_pct}% ({len(tradeable)})")
        df = pd.DataFrame(tradeable)
        display_cols = [
            "symbol", "close", "atr", "atr_pct", "rank",
        ]
        st.dataframe(
            df[display_cols], use_container_width=True, hide_index=True,
        )
    else:
        st.info("No stocks met the minimum ATR% threshold. Try lowering it.")

    if illiquid:
        with st.expander(f"Filtered Out ({len(illiquid)})", expanded=False):
            illiq_df = pd.DataFrame(illiquid)
            display_cols = ["symbol", "close", "atr", "atr_pct"]
            st.dataframe(
                illiq_df[display_cols], use_container_width=True, hide_index=True,
            )


# ── Shared scan controls ─────────────────────────────────────────────────────


def _render_scan_controls(
    strategy: str,
    extra_config: dict | None = None,
) -> None:
    """Render scan target selection and execute scan for the given strategy."""
    st.subheader("Scan Targets")

    fo_names = db.get_fo_underlying_names()
    if not fo_names:
        st.warning("No F&O stocks in database. Run a sync from the Screener page first.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        target = st.selectbox(
            "Select stocks to scan",
            options=["All F&O Stocks", "Top 20 by Liquidity"],
            key=f"scalp_{strategy}_target",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scanning = st.session_state.get(_KEY_SCANNING, False)
        label = f"Run {strategy.upper()} Scan"
        scan_clicked = st.button(
            label, type="primary", use_container_width=True, disabled=scanning,
        )

    if scan_clicked:
        _execute_scan(strategy, target, fo_names, extra_config or {})


# ── Scan execution ────────────────────────────────────────────────────────────


def _execute_scan(
    strategy: str,
    target: str,
    fo_names: list[str],
    extra_config: dict,
) -> None:
    """Execute a scalping scan for the given strategy."""
    st.session_state[_KEY_SCANNING] = True

    instruments = _get_fo_instruments_for_scan(fo_names, target)
    total = len(instruments)
    interval = st.session_state[_KEY_SCALP_INTERVAL]
    progress = st.progress(0, text="Starting scan …")

    scan_fn = {
        "vwap": _scan_vwap_single,
        "bb": _scan_bb_single,
        "atr": _scan_atr_single,
    }[strategy]

    signals_key = {
        "vwap": _KEY_VWAP_SIGNALS,
        "bb": _KEY_BB_SIGNALS,
        "atr": _KEY_ATR_SIGNALS,
    }[strategy]

    try:
        results: list[dict] = []
        for idx, inst in enumerate(instruments, start=1):
            progress.progress(
                idx / total,
                text=f"Scanning {inst['symbol']} ({idx}/{total}) …",
            )
            sig = scan_fn(inst, interval, extra_config)
            if sig is not None:
                results.append(sig)

        progress.progress(1.0, text="Scan complete!")

        # Post-process ATR results: sort by ATR% descending and assign ranks
        if strategy == "atr":
            results.sort(key=lambda s: s["atr_pct"], reverse=True)
            for rank, sig in enumerate(results, start=1):
                sig["rank"] = rank

        st.session_state[signals_key] = results
        st.success(f"{strategy.upper()} scan complete: {len(results)} signals generated.")

    except Exception as exc:
        progress.progress(1.0, text="Scan failed.")
        st.error(f"{strategy.upper()} scan failed: {exc}")
        logger.exception("%s scalp scan failed.", strategy.upper())

    finally:
        st.session_state[_KEY_SCANNING] = False


# ── Individual scan functions ─────────────────────────────────────────────────


def _scan_vwap_single(
    inst: dict,
    interval: str,
    _config: dict,
) -> dict | None:
    """Scan a single instrument for VWAP reversion signal."""
    df = fetch_ohlcv(
        exchange=inst.get("exchange", "NSE"),
        symbol_token=inst.get("token", ""),
        period="1mo",
        interval=interval,
    )
    if df.empty or len(df) < 5:
        return None

    vwap_series = compute_vwap(df["High"], df["Low"], df["Close"], df["Volume"])

    latest_close = float(df["Close"].iloc[-1])
    latest_vwap = float(vwap_series.iloc[-1])
    distance_pct = ((latest_close - latest_vwap) / latest_vwap) * 100

    # Classify signal
    if abs(distance_pct) <= 0.3:
        signal = "BUY"
        reason = f"Price near VWAP ({distance_pct:+.2f}%) — potential bounce"
    elif distance_pct > 0.3:
        signal = "SELL"
        reason = f"Price above VWAP ({distance_pct:+.2f}%) — potential rejection"
    else:
        signal = "NEUTRAL"
        reason = f"Price below VWAP ({distance_pct:+.2f}%)"

    return {
        "symbol": inst["symbol"],
        "close": round(latest_close, 2),
        "vwap": round(latest_vwap, 2),
        "distance_pct": round(distance_pct, 2),
        "signal": signal,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def _scan_bb_single(
    inst: dict,
    interval: str,
    config: dict,
) -> dict | None:
    """Scan a single instrument for Bollinger Band signal."""
    df = fetch_ohlcv(
        exchange=inst.get("exchange", "NSE"),
        symbol_token=inst.get("token", ""),
        period="1mo",
        interval=interval,
    )
    period = config.get("period", _DEFAULT_BB_PERIOD)
    num_std = config.get("std", _DEFAULT_BB_STD)
    squeeze_pct = config.get("squeeze_pct", _DEFAULT_BB_SQUEEZE_PCT)

    if df.empty or len(df) < period:
        return None

    upper, middle, lower = compute_bollinger_bands(
        df["Close"], period=period, num_std=num_std,
    )

    latest_close = float(df["Close"].iloc[-1])
    latest_upper = float(upper.iloc[-1])
    latest_lower = float(lower.iloc[-1])
    latest_middle = float(middle.iloc[-1])

    # Bandwidth as percentage of middle band
    bb_width = latest_upper - latest_lower
    bb_width_pct = (bb_width / latest_middle) * 100 if latest_middle else 0

    # Compute bandwidth percentile over the lookback window
    bandwidth_series = (upper - lower) / middle * 100
    bandwidth_pctile = float(
        (bandwidth_series < bb_width_pct).sum() / len(bandwidth_series) * 100
    )

    # Classify signal
    if bandwidth_pctile <= squeeze_pct:
        signal = "SQUEEZE"
        reason = f"BB squeeze: bandwidth {bb_width_pct:.2f}% at {bandwidth_pctile:.0f}th percentile"
    elif latest_close >= latest_upper:
        signal = "UPPER_TOUCH"
        reason = f"Price ({latest_close:.2f}) touching upper band ({latest_upper:.2f})"
    elif latest_close <= latest_lower:
        signal = "LOWER_TOUCH"
        reason = f"Price ({latest_close:.2f}) touching lower band ({latest_lower:.2f})"
    else:
        signal = "INSIDE"
        reason = f"Price inside bands, width {bb_width_pct:.2f}%"

    return {
        "symbol": inst["symbol"],
        "close": round(latest_close, 2),
        "bb_upper": round(latest_upper, 2),
        "bb_middle": round(latest_middle, 2),
        "bb_lower": round(latest_lower, 2),
        "bb_width_pct": round(bb_width_pct, 2),
        "signal": signal,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def _scan_atr_single(
    inst: dict,
    interval: str,
    config: dict,
) -> dict | None:
    """Scan a single instrument for ATR volatility signal."""
    df = fetch_ohlcv(
        exchange=inst.get("exchange", "NSE"),
        symbol_token=inst.get("token", ""),
        period="1mo",
        interval=interval,
    )
    period = config.get("period", _DEFAULT_ATR_PERIOD)

    if df.empty or len(df) < period + 1:
        return None

    atr_series = compute_atr(df["High"], df["Low"], df["Close"], period=period)

    latest_atr = float(atr_series.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    atr_pct = (latest_atr / latest_close) * 100 if latest_close else 0

    return {
        "symbol": inst["symbol"],
        "close": round(latest_close, 2),
        "atr": round(latest_atr, 4),
        "atr_pct": round(atr_pct, 2),
        "rank": 0,  # placeholder — filled after sorting
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_fo_instruments_for_scan(fo_names: list[str], target: str) -> list[dict]:
    """Get instrument dicts for F&O underlying stocks."""
    real_names = [
        n for n in fo_names
        if not any(n.startswith(str(i)) and "NSETEST" in n for i in range(10))
    ]

    instruments: list[dict] = []
    for name in real_names:
        records = db.get_nse_instruments(search=name, limit=5)
        for r in records:
            if r["name"].upper() == name.upper():
                instruments.append(r)
                break

    if target == "Top 20 by Liquidity":
        instruments = instruments[:20]

    return instruments
