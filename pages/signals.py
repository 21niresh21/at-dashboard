"""Signals page — RSI, EMA, MACD and Volume trading signal scanners.

Scans F&O stocks from the local database, computes technical indicators,
and displays BUY / SELL / HOLD signals with configurable parameters.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.components.ui import page_header
from src.database import models as db
from src.logging_config import get_logger
from src.signals.models import (
    EMAConfig,
    EMACrossoverSignal,
    MACDConfig,
    MACDSignal,
    RSIConfig,
    RSISignal,
    SignalAction,
    SignalStrength,
    VolumeConfig,
    VolumeSignal,
)

logger = get_logger(__name__)

# ─ Defaults ──────────────────────────────────────────────────────────────────

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
_KEY_RSI_CONFIG = "rsi_config"
_KEY_EMA_CONFIG = "ema_config"
_KEY_MACD_CONFIG = "macd_config"
_KEY_VOLUME_CONFIG = "volume_config"

_KEY_RSI_SIGNALS = "rsi_signals"
_KEY_EMA_SIGNALS = "ema_signals"
_KEY_MACD_SIGNALS = "macd_signals"
_KEY_VOLUME_SIGNALS = "volume_signals"
_KEY_SCANNING = "scanning"


# ── Public entry point ────────────────────────────────────────────────────────


def render() -> None:
    """Render the signals page."""
    page_header(
        title="Signals",
        description=(
            "Trading signals for F&O stocks. Configure parameters and scan "
            "to generate actionable signals."
        ),
    )

    # Initialize session state
    if _KEY_RSI_CONFIG not in st.session_state:
        st.session_state[_KEY_RSI_CONFIG] = RSIConfig(
            period=_DEFAULT_RSI_PERIOD,
            oversold=_DEFAULT_OVERSOLD,
            overbought=_DEFAULT_OVERBOUGHT,
        )
    if _KEY_EMA_CONFIG not in st.session_state:
        st.session_state[_KEY_EMA_CONFIG] = EMAConfig(
            fast_period=_DEFAULT_EMA_FAST,
            slow_period=_DEFAULT_EMA_SLOW,
        )
    if _KEY_MACD_CONFIG not in st.session_state:
        st.session_state[_KEY_MACD_CONFIG] = MACDConfig(
            fast_period=_DEFAULT_MACD_FAST,
            slow_period=_DEFAULT_MACD_SLOW,
            signal_period=_DEFAULT_MACD_SIGNAL,
        )
    if _KEY_VOLUME_CONFIG not in st.session_state:
        st.session_state[_KEY_VOLUME_CONFIG] = VolumeConfig(
            period=_DEFAULT_VOLUME_PERIOD,
            surge_threshold=_DEFAULT_VOLUME_SURGE,
        )

    tab_rsi, tab_ema, tab_macd, tab_vol = st.tabs(
        ["RSI Strategy", "EMA Crossover", "MACD", "Volume"]
    )

    with tab_rsi:
        _render_rsi_tab()
    with tab_ema:
        _render_ema_tab()
    with tab_macd:
        _render_macd_tab()
    with tab_vol:
        _render_volume_tab()


# ── RSI Tab ──────────────────────────────────────────────────────────────────


def _render_rsi_tab() -> None:
    """RSI strategy tab."""
    st.subheader("RSI Configuration")

    config: RSIConfig = st.session_state[_KEY_RSI_CONFIG]

    col1, col2, col3 = st.columns(3)
    with col1:
        period = st.number_input(
            "RSI Period", min_value=2, max_value=100, value=config.period,
            step=1, help="Lookback period for RSI calculation. Default: 14",
            key="rsi_cfg_period",
        )
    with col2:
        oversold = st.number_input(
            "Oversold Threshold", min_value=5.0, max_value=45.0,
            value=config.oversold, step=1.0,
            help="RSI below this triggers a BUY signal. Default: 30",
            key="rsi_cfg_oversold",
        )
    with col3:
        overbought = st.number_input(
            "Overbought Threshold", min_value=55.0, max_value=95.0,
            value=config.overbought, step=1.0,
            help="RSI above this triggers a SELL signal. Default: 70",
            key="rsi_cfg_overbought",
        )

    if st.button("Reset to Defaults", key="rsi_reset"):
        st.session_state[_KEY_RSI_CONFIG] = RSIConfig()
        st.rerun()

    st.session_state[_KEY_RSI_CONFIG] = RSIConfig(
        period=int(period), oversold=float(oversold), overbought=float(overbought),
    )

    st.divider()
    _render_scan_controls(strategy="rsi")

    signals = st.session_state.get(_KEY_RSI_SIGNALS, [])
    if signals:
        _render_rsi_signals(signals)


def _scan_rsi_single(inst: dict, config: RSIConfig) -> RSISignal | None:
    """Scan a single instrument for RSI signal."""
    from src.signals.engine import compute_rsi_signal

    return compute_rsi_signal(
        symbol=inst["symbol"],
        name=inst.get("name", inst["symbol"]),
        exchange=inst.get("exchange", "NSE"),
        config=config,
        token=inst.get("token", ""),
    )


def _render_rsi_signals(signals: list[RSISignal]) -> None:
    """Display RSI scan results."""
    st.subheader("RSI Scan Results")
    _render_signal_summary(signals)

    df = pd.DataFrame([s.to_dict() for s in signals])
    st.dataframe(df, use_container_width=True, hide_index=True)

    buy = [s for s in signals if s.signal == SignalAction.BUY]
    sell = [s for s in signals if s.signal == SignalAction.SELL]

    if buy:
        st.markdown(f"### 🟢 BUY Signals — Oversold ({len(buy)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in buy]),
            use_container_width=True, hide_index=True,
        )
    if sell:
        st.markdown(f"### 🔴 SELL Signals — Overbought ({len(sell)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in sell]),
            use_container_width=True, hide_index=True,
        )


# ── EMA Crossover Tab ────────────────────────────────────────────────────────


def _render_ema_tab() -> None:
    """EMA crossover strategy tab."""
    st.subheader("EMA Crossover Configuration")

    config: EMAConfig = st.session_state[_KEY_EMA_CONFIG]

    col1, col2 = st.columns(2)
    with col1:
        fast = st.number_input(
            "Fast EMA Period", min_value=2, max_value=100, value=config.fast_period,
            step=1, help="Fast EMA lookback period. Default: 9", key="ema_cfg_fast",
        )
    with col2:
        slow = st.number_input(
            "Slow EMA Period", min_value=5, max_value=200, value=config.slow_period,
            step=1, help="Slow EMA lookback period. Default: 21", key="ema_cfg_slow",
        )

    if st.button("Reset to Defaults", key="ema_reset"):
        st.session_state[_KEY_EMA_CONFIG] = EMAConfig()
        st.rerun()

    st.session_state[_KEY_EMA_CONFIG] = EMAConfig(
        fast_period=int(fast), slow_period=int(slow),
    )

    st.divider()
    _render_scan_controls(strategy="ema")

    signals = st.session_state.get(_KEY_EMA_SIGNALS, [])
    if signals:
        _render_ema_signals(signals)


def _scan_ema_single(inst: dict, config: EMAConfig) -> EMACrossoverSignal | None:
    """Scan a single instrument for EMA crossover signal."""
    from src.signals.engine import compute_ema_crossover_signal

    return compute_ema_crossover_signal(
        symbol=inst["symbol"],
        name=inst.get("name", inst["symbol"]),
        exchange=inst.get("exchange", "NSE"),
        config=config,
        token=inst.get("token", ""),
    )


def _render_ema_signals(signals: list[EMACrossoverSignal]) -> None:
    """Display EMA crossover scan results."""
    st.subheader("EMA Crossover Scan Results")
    _render_signal_summary(signals)

    df = pd.DataFrame([s.to_dict() for s in signals])
    st.dataframe(df, use_container_width=True, hide_index=True)

    buy = [s for s in signals if s.signal == SignalAction.BUY]
    sell = [s for s in signals if s.signal == SignalAction.SELL]

    if buy:
        st.markdown(f"### 🟢 BUY Signals — Golden Cross ({len(buy)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in buy]),
            use_container_width=True, hide_index=True,
        )
    if sell:
        st.markdown(f"### 🔴 SELL Signals — Death Cross ({len(sell)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in sell]),
            use_container_width=True, hide_index=True,
        )


# ── MACD Tab ─────────────────────────────────────────────────────────────────


def _render_macd_tab() -> None:
    """MACD strategy tab."""
    st.subheader("MACD Configuration")

    config: MACDConfig = st.session_state[_KEY_MACD_CONFIG]

    col1, col2, col3 = st.columns(3)
    with col1:
        fast = st.number_input(
            "Fast EMA Period", min_value=2, max_value=100, value=config.fast_period,
            step=1, help="Fast EMA period for MACD. Default: 12", key="macd_cfg_fast",
        )
    with col2:
        slow = st.number_input(
            "Slow EMA Period", min_value=5, max_value=200, value=config.slow_period,
            step=1, help="Slow EMA period for MACD. Default: 26", key="macd_cfg_slow",
        )
    with col3:
        signal = st.number_input(
            "Signal Period", min_value=2, max_value=100, value=config.signal_period,
            step=1, help="Signal line EMA period. Default: 9", key="macd_cfg_signal",
        )

    if st.button("Reset to Defaults", key="macd_reset"):
        st.session_state[_KEY_MACD_CONFIG] = MACDConfig()
        st.rerun()

    st.session_state[_KEY_MACD_CONFIG] = MACDConfig(
        fast_period=int(fast), slow_period=int(slow), signal_period=int(signal),
    )

    st.divider()
    _render_scan_controls(strategy="macd")

    signals = st.session_state.get(_KEY_MACD_SIGNALS, [])
    if signals:
        _render_macd_signals(signals)


def _scan_macd_single(inst: dict, config: MACDConfig) -> MACDSignal | None:
    """Scan a single instrument for MACD signal."""
    from src.signals.engine import compute_macd_signal

    return compute_macd_signal(
        symbol=inst["symbol"],
        name=inst.get("name", inst["symbol"]),
        exchange=inst.get("exchange", "NSE"),
        config=config,
        token=inst.get("token", ""),
    )


def _render_macd_signals(signals: list[MACDSignal]) -> None:
    """Display MACD scan results."""
    st.subheader("MACD Scan Results")
    _render_signal_summary(signals)

    df = pd.DataFrame([s.to_dict() for s in signals])
    st.dataframe(df, use_container_width=True, hide_index=True)

    buy = [s for s in signals if s.signal == SignalAction.BUY]
    sell = [s for s in signals if s.signal == SignalAction.SELL]

    if buy:
        st.markdown(f"### 🟢 BUY Signals — Bullish Crossover ({len(buy)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in buy]),
            use_container_width=True, hide_index=True,
        )
    if sell:
        st.markdown(f"### 🔴 SELL Signals — Bearish Crossover ({len(sell)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in sell]),
            use_container_width=True, hide_index=True,
        )


# ── Volume Tab ───────────────────────────────────────────────────────────────


def _render_volume_tab() -> None:
    """Volume analysis tab."""
    st.subheader("Volume Configuration")

    config: VolumeConfig = st.session_state[_KEY_VOLUME_CONFIG]

    col1, col2 = st.columns(2)
    with col1:
        period = st.number_input(
            "Volume Average Period", min_value=5, max_value=100,
            value=config.period, step=1,
            help="Lookback period for average volume. Default: 20",
            key="vol_cfg_period",
        )
    with col2:
        surge = st.number_input(
            "Surge Threshold (× avg)", min_value=1.0, max_value=10.0,
            value=config.surge_threshold, step=0.5,
            help="Volume ratio above this = surge. Default: 2.0×",
            key="vol_cfg_surge",
        )

    if st.button("Reset to Defaults", key="vol_reset"):
        st.session_state[_KEY_VOLUME_CONFIG] = VolumeConfig()
        st.rerun()

    st.session_state[_KEY_VOLUME_CONFIG] = VolumeConfig(
        period=int(period), surge_threshold=float(surge),
    )

    st.divider()
    _render_scan_controls(strategy="volume")

    signals = st.session_state.get(_KEY_VOLUME_SIGNALS, [])
    if signals:
        _render_volume_signals(signals)


def _scan_volume_single(inst: dict, config: VolumeConfig) -> VolumeSignal | None:
    """Scan a single instrument for volume signal."""
    from src.signals.engine import compute_volume_signal

    return compute_volume_signal(
        symbol=inst["symbol"],
        name=inst.get("name", inst["symbol"]),
        exchange=inst.get("exchange", "NSE"),
        config=config,
        token=inst.get("token", ""),
    )


def _render_volume_signals(signals: list[VolumeSignal]) -> None:
    """Display Volume scan results."""
    st.subheader("Volume Scan Results")
    _render_signal_summary(signals)

    df = pd.DataFrame([s.to_dict() for s in signals])
    st.dataframe(df, use_container_width=True, hide_index=True)

    buy = [s for s in signals if s.signal == SignalAction.BUY]
    sell = [s for s in signals if s.signal == SignalAction.SELL]

    if buy:
        st.markdown(f"### 🟢 BUY Signals — Volume Breakout ({len(buy)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in buy]),
            use_container_width=True, hide_index=True,
        )
    if sell:
        st.markdown(f"### 🔴 SELL Signals — Volume Breakdown ({len(sell)})")
        st.dataframe(
            pd.DataFrame([s.to_dict() for s in sell]),
            use_container_width=True, hide_index=True,
        )


# ── Shared Helpers ───────────────────────────────────────────────────────────


def _render_scan_controls(strategy: str) -> None:
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
            key=f"{strategy}_scan_target",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        scanning = st.session_state.get(_KEY_SCANNING, False)
        label = f"Run {strategy.upper()} Scan"
        scan_clicked = st.button(
            label, type="primary", use_container_width=True, disabled=scanning,
        )

    if scan_clicked:
        _execute_scan(strategy, target, fo_names)


def _execute_scan(strategy: str, target: str, fo_names: list[str]) -> None:
    """Execute a scan for the given strategy."""
    st.session_state[_KEY_SCANNING] = True

    instruments = _get_fo_instruments_for_scan(fo_names, target)
    total = len(instruments)
    progress = st.progress(0, text="Starting scan …")

    scan_fn = {
        "rsi": _scan_rsi_single,
        "ema": _scan_ema_single,
        "macd": _scan_macd_single,
        "volume": _scan_volume_single,
    }[strategy]

    config_key = {
        "rsi": _KEY_RSI_CONFIG,
        "ema": _KEY_EMA_CONFIG,
        "macd": _KEY_MACD_CONFIG,
        "volume": _KEY_VOLUME_CONFIG,
    }[strategy]

    signals_key = {
        "rsi": _KEY_RSI_SIGNALS,
        "ema": _KEY_EMA_SIGNALS,
        "macd": _KEY_MACD_SIGNALS,
        "volume": _KEY_VOLUME_SIGNALS,
    }[strategy]

    config = st.session_state[config_key]

    try:
        results: list = []
        for idx, inst in enumerate(instruments, start=1):
            progress.progress(
                idx / total,
                text=f"Scanning {inst['symbol']} ({idx}/{total}) …",
            )
            sig = scan_fn(inst, config)
            if sig is not None:
                results.append(sig)

        progress.progress(1.0, text="Scan complete!")
        st.session_state[signals_key] = results
        st.success(f"{strategy.upper()} scan complete: {len(results)} signals generated.")

    except Exception as exc:
        progress.progress(1.0, text="Scan failed.")
        st.error(f"{strategy.upper()} scan failed: {exc}")
        logger.exception("%s scan failed.", strategy.upper())

    finally:
        st.session_state[_KEY_SCANNING] = False


def _render_signal_summary(signals: list) -> None:
    """Render BUY / SELL / HOLD metric cards for any signal list."""
    buy = sum(1 for s in signals if s.signal == SignalAction.BUY)
    sell = sum(1 for s in signals if s.signal == SignalAction.SELL)
    hold = sum(1 for s in signals if s.signal == SignalAction.HOLD)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Signals", len(signals))
    with col2:
        st.metric("BUY", buy)
    with col3:
        st.metric("SELL", sell)
    with col4:
        st.metric("HOLD", hold)
    st.divider()


def _get_fo_instruments_for_scan(fo_names: list[str], target: str) -> list[dict]:
    """Get instrument dicts for F&O underlying stocks.

    Filters out test/dummy symbols and returns only real stocks.
    """
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
