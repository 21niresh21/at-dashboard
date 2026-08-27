"""Confluence page — multi-signal confluence trading signal scanner.

Combines RSI, EMA crossover, MACD, and Volume signals to produce
higher-confidence trading recommendations.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import page_header
from src.database import models as db
from src.logging_config import get_logger
from src.signals.engine import scan_confluence_signals
from src.signals.models import (
    ConfluenceSignal,
    EMAConfig,
    MACDConfig,
    RSIConfig,
    SignalAction,
    VolumeConfig,
)

logger = get_logger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────────────

_DEFAULT_RSI_PERIOD = 14
_DEFAULT_OVERSOLD = 30.0
_DEFAULT_OVERBOUGHT = 70.0

_DEFAULT_EMA_FAST = 9
_DEFAULT_EMA_SLOW = 21

_DEFAULT_MACD_FAST = 12
_DEFAULT_MACD_SLOW = 26
_DEFAULT_MACD_SIGNAL = 9

_DEFAULT_VOLUME_PERIOD = 20
_DEFAULT_VOLUME_SURGE = 2.0

# Session state keys
_KEY_CONFIGS = "confluence_configs"
_KEY_SIGNALS = "confluence_signals"
_KEY_SCANNING = "confluence_scanning"


# ── Public entry point ────────────────────────────────────────────────────────


def render() -> None:
    """Render the confluence page."""
    page_header(
        title="Multi-Signal Confluence",
        description="Combine RSI, EMA crossover, MACD, and Volume signals for higher-confidence trading recommendations.",
    )

    # Initialize session state
    if _KEY_CONFIGS not in st.session_state:
        st.session_state[_KEY_CONFIGS] = {
            "rsi": RSIConfig(
                period=_DEFAULT_RSI_PERIOD,
                oversold=_DEFAULT_OVERSOLD,
                overbought=_DEFAULT_OVERBOUGHT,
            ),
            "ema": EMAConfig(
                fast_period=_DEFAULT_EMA_FAST,
                slow_period=_DEFAULT_EMA_SLOW,
            ),
            "macd": MACDConfig(
                fast_period=_DEFAULT_MACD_FAST,
                slow_period=_DEFAULT_MACD_SLOW,
                signal_period=_DEFAULT_MACD_SIGNAL,
            ),
            "volume": VolumeConfig(
                period=_DEFAULT_VOLUME_PERIOD,
                surge_threshold=_DEFAULT_VOLUME_SURGE,
            ),
        }

    # Configuration panel
    _render_config_panel()

    st.divider()

    # Scan controls
    _render_scan_controls()

    # Results
    signals = st.session_state.get(_KEY_SIGNALS, [])
    if signals:
        _render_signals(signals)


# ── Configuration panel ───────────────────────────────────────────────────────


def _render_config_panel() -> None:
    """Render configuration panels for all indicators."""
    st.subheader("Indicator Configuration")

    configs = st.session_state[_KEY_CONFIGS]

    # RSI Config
    with st.expander("RSI Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            rsi_period = st.number_input(
                "RSI Period", min_value=2, max_value=100, value=configs["rsi"].period,
                step=1, key="conf_rsi_period",
            )
        with col2:
            rsi_oversold = st.number_input(
                "Oversold Threshold", min_value=5.0, max_value=45.0, value=configs["rsi"].oversold,
                step=1.0, key="conf_rsi_oversold",
            )
        with col3:
            rsi_overbought = st.number_input(
                "Overbought Threshold", min_value=55.0, max_value=95.0, value=configs["rsi"].overbought,
                step=1.0, key="conf_rsi_overbought",
            )

    # EMA Config
    with st.expander("EMA Crossover Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            ema_fast = st.number_input(
                "Fast EMA Period", min_value=2, max_value=100, value=configs["ema"].fast_period,
                step=1, key="conf_ema_fast",
            )
        with col2:
            ema_slow = st.number_input(
                "Slow EMA Period", min_value=5, max_value=200, value=configs["ema"].slow_period,
                step=1, key="conf_ema_slow",
            )

    # MACD Config
    with st.expander("MACD Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            macd_fast = st.number_input(
                "Fast EMA Period", min_value=2, max_value=100, value=configs["macd"].fast_period,
                step=1, key="conf_macd_fast",
            )
        with col2:
            macd_slow = st.number_input(
                "Slow EMA Period", min_value=5, max_value=200, value=configs["macd"].slow_period,
                step=1, key="conf_macd_slow",
            )
        with col3:
            macd_signal = st.number_input(
                "Signal Period", min_value=2, max_value=100, value=configs["macd"].signal_period,
                step=1, key="conf_macd_signal",
            )

    # Volume Config
    with st.expander("Volume Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            vol_period = st.number_input(
                "Volume Period", min_value=5, max_value=100, value=configs["volume"].period,
                step=1, key="conf_vol_period",
            )
        with col2:
            vol_surge = st.number_input(
                "Surge Threshold (x avg)", min_value=1.0, max_value=10.0, value=configs["volume"].surge_threshold,
                step=0.5, key="conf_vol_surge",
            )

    # Reset button
    if st.button("Reset All to Defaults", key="conf_reset"):
        st.session_state[_KEY_CONFIGS] = {
            "rsi": RSIConfig(),
            "ema": EMAConfig(),
            "macd": MACDConfig(),
            "volume": VolumeConfig(),
        }
        st.rerun()

    # Update configs
    st.session_state[_KEY_CONFIGS] = {
        "rsi": RSIConfig(period=int(rsi_period), oversold=float(rsi_oversold), overbought=float(rsi_overbought)),
        "ema": EMAConfig(fast_period=int(ema_fast), slow_period=int(ema_slow)),
        "macd": MACDConfig(fast_period=int(macd_fast), slow_period=int(macd_slow), signal_period=int(macd_signal)),
        "volume": VolumeConfig(period=int(vol_period), surge_threshold=float(vol_surge)),
    }


# ── Scan controls ─────────────────────────────────────────────────────────────


def _render_scan_controls() -> None:
    """Render scan target selection and execute button."""
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
            key="conf_scan_target",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scanning = st.session_state.get(_KEY_SCANNING, False)
        scan_clicked = st.button(
            "Run Confluence Scan",
            type="primary",
            use_container_width=True,
            disabled=scanning,
        )

    if scan_clicked:
        _execute_scan(target, fo_names)


# ── Scan execution ────────────────────────────────────────────────────────────


def _execute_scan(target: str, fo_names: list[str]) -> None:
    """Execute the confluence scan."""
    st.session_state[_KEY_SCANNING] = True

    configs = st.session_state[_KEY_CONFIGS]
    instruments = _get_fo_instruments_for_scan(fo_names, target)
    total = len(instruments)

    st.info(
        f"Scanning {total} instruments with confluence of "
        f"RSI({configs['rsi'].period}), EMA({configs['ema'].fast_period}/{configs['ema'].slow_period}), "
        f"MACD({configs['macd'].fast_period}/{configs['macd'].slow_period}/{configs['macd'].signal_period}), "
        f"Volume({configs['volume'].period}) …"
    )

    progress = st.progress(0, text="Starting scan …")

    try:
        signals = scan_confluence_signals(
            instruments,
            rsi_config=configs["rsi"],
            ema_config=configs["ema"],
            macd_config=configs["macd"],
            volume_config=configs["volume"],
        )

        progress.progress(1.0, text="Scan complete!")
        st.session_state[_KEY_SIGNALS] = signals
        st.success(f"Confluence scan complete: {len(signals)} signals generated.")

    except Exception as exc:
        progress.progress(1.0, text="Scan failed.")
        st.error(f"Confluence scan failed: {exc}")
        logger.exception("Confluence scan failed.")

    finally:
        st.session_state[_KEY_SCANNING] = False


# ── Results display ───────────────────────────────────────────────────────────


def _render_signals(signals: list[ConfluenceSignal]) -> None:
    """Display confluence scan results."""
    st.subheader("Confluence Scan Results")

    # Summary metrics
    buy_signals = [s for s in signals if s.overall_signal == SignalAction.BUY]
    sell_signals = [s for s in signals if s.overall_signal == SignalAction.SELL]
    hold_signals = [s for s in signals if s.overall_signal == SignalAction.HOLD]
    high_confidence = [s for s in signals if s.confidence >= 3]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", len(signals))
    with col2:
        st.metric("BUY Signals", len(buy_signals))
    with col3:
        st.metric("SELL Signals", len(sell_signals))
    with col4:
        st.metric("High Confidence (≥3)", len(high_confidence))

    st.divider()

    # Filter controls
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        signal_filter = st.multiselect(
            "Filter by Overall Signal",
            options=[SignalAction.BUY.value, SignalAction.SELL.value, SignalAction.HOLD.value],
            default=[SignalAction.BUY.value, SignalAction.SELL.value, SignalAction.HOLD.value],
            key="conf_signal_filter",
        )

    with filter_col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            min_value=0,
            max_value=4,
            value=0,
            step=1,
            key="conf_min_confidence",
        )

    # Apply filters
    filtered = [
        s for s in signals
        if s.overall_signal.value in signal_filter and s.confidence >= min_confidence
    ]

    if not filtered:
        st.info("No signals match your filters.")
        return

    # Display as DataFrame
    df = pd.DataFrame([s.to_dict() for s in filtered])
    st.dataframe(df, use_container_width=True, hide_index=True)

    # High-confidence BUY signals
    high_conf_buy = [s for s in buy_signals if s.confidence >= 3]
    if high_conf_buy:
        st.markdown(f"###  High-Confidence BUY Signals ({len(high_conf_buy)})")
        buy_df = pd.DataFrame([s.to_dict() for s in high_conf_buy])
        st.dataframe(buy_df, use_container_width=True, hide_index=True)

    # High-confidence SELL signals
    high_conf_sell = [s for s in sell_signals if s.confidence >= 3]
    if high_conf_sell:
        st.markdown(f"### 🔴 High-Confidence SELL Signals ({len(high_conf_sell)})")
        sell_df = pd.DataFrame([s.to_dict() for s in high_conf_sell])
        st.dataframe(sell_df, use_container_width=True, hide_index=True)


# ── Shared helpers ────────────────────────────────────────────────────────────


def _get_fo_instruments_for_scan(fo_names: list[str], target: str) -> list[dict]:
    """Get instrument dicts for F&O underlying stocks."""
    # Filter out test symbols
    real_names = [n for n in fo_names if not any(n.startswith(str(i)) and "NSETEST" in n for i in range(10))]

    instruments = []
    for name in real_names:
        records = db.get_nse_instruments(search=name, limit=5)
        for r in records:
            if r["name"].upper() == name.upper():
                instruments.append(r)
                break

    if target == "Top 20 by Liquidity":
        instruments = instruments[:20]

    return instruments
