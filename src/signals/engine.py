"""Signal engine — generates trading signals from market data.

Fetches historical price data via AngelOne SmartAPI, computes technical
indicators, and produces actionable signals based on configurable rules.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from src.logging_config import get_logger
from src.signals.models import (
    ConfluenceSignal,
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
from src.signals.technical import (
    compute_ema,
    compute_macd,
    compute_rsi,
    compute_volume_analysis,
    detect_ema_crossover,
    detect_macd_crossover,
    detect_volume_surge,
)
from src.smartapi.client import fetch_ohlcv as _api_fetch_ohlcv

logger = get_logger(__name__)

# ── Data fetching ─────────────────────────────────────────────────────────────


def _fetch_ohlcv(
    exchange: str,
    symbol_token: str,
    period: str = "3mo",
) -> pd.DataFrame:
    """Download OHLCV data from AngelOne SmartAPI.

    Args:
        exchange: Exchange segment (``"NSE"``, ``"NFO"``, etc.).
        symbol_token: Numeric instrument token from the master script.
        period: Lookback period (default ``"3mo"``).

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume.
    """
    df = _api_fetch_ohlcv(exchange=exchange, symbol_token=symbol_token, period=period)
    if df.empty:
        logger.warning("No data returned for token %s on %s", symbol_token, exchange)
    return df


# ── Signal generation ────────────────────────────────────────────────────────


def compute_rsi_signal(
    symbol: str,
    name: str,
    exchange: str,
    config: RSIConfig,
    token: str = "",
) -> RSISignal | None:
    """Compute an RSI-based signal for a single instrument.

    Args:
        symbol: Instrument symbol (e.g. "RELIANCE").
        name: Instrument display name.
        exchange: Exchange segment (NSE / NFO).
        config: RSI configuration (period, thresholds).
        token: AngelOne instrument token for SmartAPI data fetch.

    Returns:
        An ``RSISignal`` if data is available, else ``None``.
    """
    df = _fetch_ohlcv(exchange, token)

    if df.empty or len(df) < config.period:
        logger.debug("Insufficient data for %s (token=%s)", symbol, token)
        return None

    rsi_series = compute_rsi(df["Close"], period=config.period)
    latest_rsi = float(rsi_series.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)

    action, strength, reason = _classify_rsi(latest_rsi, config)

    return RSISignal(
        symbol=symbol,
        name=name,
        exchange=exchange,
        rsi_value=round(latest_rsi, 2),
        signal=action,
        strength=strength,
        timestamp=now,
        close_price=round(latest_close, 2),
        reason=reason,
    )


def scan_rsi_signals(
    instruments: Sequence[dict],
    config: RSIConfig,
    max_concurrent: int = 5,
) -> list[RSISignal]:
    """Scan a list of instruments and return RSI signals.

    Args:
        instruments: List of instrument dicts (from the database).
        config: RSI configuration.
        max_concurrent: Not used yet (reserved for async future).

    Returns:
        List of ``RSISignal`` objects, sorted by RSI value ascending.
    """
    signals: list[RSISignal] = []
    total = len(instruments)

    for idx, inst in enumerate(instruments, start=1):
        symbol = inst["symbol"]
        name = inst.get("name", symbol)
        exchange = inst.get("exchange", "NSE")
        token = inst.get("token", "")

        if (idx % 20) == 0 or idx == total:
            logger.info("Scanning RSI: %d/%d (%s)", idx, total, symbol)

        sig = compute_rsi_signal(symbol, name, exchange, config, token=token)
        if sig is not None:
            signals.append(sig)

    # Sort: strongest buy signals first (lowest RSI), then strongest sell (highest RSI)
    signals.sort(key=lambda s: s.rsi_value)
    return signals


# ── Classification logic ──────────────────────────────────────────────────────


def _classify_rsi(
    rsi: float,
    config: RSIConfig,
) -> tuple[SignalAction, SignalStrength, str]:
    """Map an RSI value to a signal action, strength, and reason string."""
    if rsi <= config.oversold:
        # Deep oversold → strong buy
        depth = config.oversold - rsi
        if depth >= 10:
            return SignalAction.BUY, SignalStrength.STRONG, f"RSI {rsi:.1f} deeply oversold (<{config.oversold})"
        return SignalAction.BUY, SignalStrength.MODERATE, f"RSI {rsi:.1f} oversold (<{config.oversold})"

    if rsi >= config.overbought:
        # Deep overbought → strong sell
        depth = rsi - config.overbought
        if depth >= 10:
            return SignalAction.SELL, SignalStrength.STRONG, f"RSI {rsi:.1f} deeply overbought (>{config.overbought})"
        return SignalAction.SELL, SignalStrength.MODERATE, f"RSI {rsi:.1f} overbought (>{config.overbought})"

    # Neutral zone
    mid = (config.oversold + config.overbought) / 2
    if abs(rsi - mid) < 10:
        return SignalAction.HOLD, SignalStrength.WEAK, f"RSI {rsi:.1f} neutral zone"
    return SignalAction.HOLD, SignalStrength.MODERATE, f"RSI {rsi:.1f} trending toward {'overbought' if rsi > mid else 'oversold'}"


# ── EMA Crossover Signal Generation ─────────────────────────────────────────


def compute_ema_crossover_signal(
    symbol: str,
    name: str,
    exchange: str,
    config: EMAConfig,
    token: str = "",
) -> EMACrossoverSignal | None:
    """Compute an EMA crossover signal for a single instrument.

    Args:
        symbol: Instrument symbol (e.g. "RELIANCE").
        name: Instrument display name.
        exchange: Exchange segment (NSE / NFO).
        config: EMA configuration (fast_period, slow_period).
        token: AngelOne instrument token for SmartAPI data fetch.

    Returns:
        An ``EMACrossoverSignal`` if data is available, else ``None``.
    """
    df = _fetch_ohlcv(exchange, token)

    if df.empty or len(df) < config.slow_period:
        logger.debug("Insufficient data for %s (token=%s)", symbol, token)
        return None

    # Compute EMAs
    fast_ema_series = compute_ema(df["Close"], period=config.fast_period)
    slow_ema_series = compute_ema(df["Close"], period=config.slow_period)

    # Detect crossover
    crossover = detect_ema_crossover(fast_ema_series, slow_ema_series)

    latest_fast_ema = float(fast_ema_series.iloc[-1])
    latest_slow_ema = float(slow_ema_series.iloc[-1])
    latest_crossover = float(crossover.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)

    action, strength, reason = _classify_ema_crossover(
        latest_crossover, latest_fast_ema, latest_slow_ema, config
    )

    return EMACrossoverSignal(
        symbol=symbol,
        name=name,
        exchange=exchange,
        fast_ema=round(latest_fast_ema, 2),
        slow_ema=round(latest_slow_ema, 2),
        signal=action,
        strength=strength,
        timestamp=now,
        close_price=round(latest_close, 2),
        reason=reason,
    )


def scan_ema_crossover_signals(
    instruments: Sequence[dict],
    config: EMAConfig,
    max_concurrent: int = 5,
) -> list[EMACrossoverSignal]:
    """Scan a list of instruments and return EMA crossover signals.

    Args:
        instruments: List of instrument dicts (from the database).
        config: EMA configuration.
        max_concurrent: Not used yet (reserved for async future).

    Returns:
        List of ``EMACrossoverSignal`` objects, sorted by signal priority.
    """
    signals: list[EMACrossoverSignal] = []
    total = len(instruments)

    for idx, inst in enumerate(instruments, start=1):
        symbol = inst["symbol"]
        name = inst.get("name", symbol)
        exchange = inst.get("exchange", "NSE")
        token = inst.get("token", "")

        if (idx % 20) == 0 or idx == total:
            logger.info("Scanning EMA crossover: %d/%d (%s)", idx, total, symbol)

        sig = compute_ema_crossover_signal(symbol, name, exchange, config, token=token)
        if sig is not None:
            signals.append(sig)

    # Sort: BUY signals first (strongest), then SELL, then HOLD
    priority = {SignalAction.BUY: 0, SignalAction.SELL: 1, SignalAction.HOLD: 2}
    signals.sort(key=lambda s: (priority.get(s.signal, 3), s.symbol))
    return signals


# ── EMA Crossover Classification Logic ───────────────────────────────────────


def _classify_ema_crossover(
    crossover: float,
    fast_ema: float,
    slow_ema: float,
    config: EMAConfig,
) -> tuple[SignalAction, SignalStrength, str]:
    """Map an EMA crossover event to a signal action, strength, and reason string.

    Args:
        crossover: Crossover indicator (1.0 = bullish, -1.0 = bearish, 0.0 = none).
        fast_ema: Current fast EMA value.
        slow_ema: Current slow EMA value.
        config: EMA configuration.

    Returns:
        Tuple of (action, strength, reason).
    """
    if crossover > 0:
        # Bullish crossover: fast EMA crossed above slow EMA
        distance_pct = ((fast_ema - slow_ema) / slow_ema) * 100
        if distance_pct > 2.0:
            return (
                SignalAction.BUY,
                SignalStrength.STRONG,
                f"Golden cross: EMA{config.fast_period} ({fast_ema:.2f}) crossed above EMA{config.slow_period} ({slow_ema:.2f}), gap {distance_pct:.1f}%",
            )
        return (
            SignalAction.BUY,
            SignalStrength.MODERATE,
            f"Golden cross: EMA{config.fast_period} ({fast_ema:.2f}) crossed above EMA{config.slow_period} ({slow_ema:.2f})",
        )

    if crossover < 0:
        # Bearish crossover: fast EMA crossed below slow EMA
        distance_pct = ((slow_ema - fast_ema) / slow_ema) * 100
        if distance_pct > 2.0:
            return (
                SignalAction.SELL,
                SignalStrength.STRONG,
                f"Death cross: EMA{config.fast_period} ({fast_ema:.2f}) crossed below EMA{config.slow_period} ({slow_ema:.2f}), gap {distance_pct:.1f}%",
            )
        return (
            SignalAction.SELL,
            SignalStrength.MODERATE,
            f"Death cross: EMA{config.fast_period} ({fast_ema:.2f}) crossed below EMA{config.slow_period} ({slow_ema:.2f})",
        )

    # No crossover - check trend direction
    if fast_ema > slow_ema:
        distance_pct = ((fast_ema - slow_ema) / slow_ema) * 100
        return (
            SignalAction.HOLD,
            SignalStrength.MODERATE if distance_pct > 1.0 else SignalStrength.WEAK,
            f"Bullish trend: EMA{config.fast_period} ({fast_ema:.2f}) above EMA{config.slow_period} ({slow_ema:.2f}), gap {distance_pct:.1f}%",
        )
    else:
        distance_pct = ((slow_ema - fast_ema) / slow_ema) * 100
        return (
            SignalAction.HOLD,
            SignalStrength.MODERATE if distance_pct > 1.0 else SignalStrength.WEAK,
            f"Bearish trend: EMA{config.fast_period} ({fast_ema:.2f}) below EMA{config.slow_period} ({slow_ema:.2f}), gap {distance_pct:.1f}%",
        )


# ── MACD Signal Generation ──────────────────────────────────────────────────


def compute_macd_signal(
    symbol: str,
    name: str,
    exchange: str,
    config: MACDConfig,
    token: str = "",
) -> MACDSignal | None:
    """Compute a MACD-based signal for a single instrument."""
    df = _fetch_ohlcv(exchange, token)

    if df.empty or len(df) < config.slow_period + config.signal_period:
        logger.debug("Insufficient data for %s (token=%s)", symbol, token)
        return None

    macd_line, signal_line, histogram = compute_macd(
        df["Close"],
        fast_period=config.fast_period,
        slow_period=config.slow_period,
        signal_period=config.signal_period,
    )

    crossover = detect_macd_crossover(macd_line, signal_line)

    latest_macd = float(macd_line.iloc[-1])
    latest_signal = float(signal_line.iloc[-1])
    latest_histogram = float(histogram.iloc[-1])
    latest_crossover = float(crossover.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)

    action, strength, reason = _classify_macd(
        latest_crossover, latest_macd, latest_signal, latest_histogram, config
    )

    return MACDSignal(
        symbol=symbol,
        name=name,
        exchange=exchange,
        macd_line=round(latest_macd, 2),
        signal_line=round(latest_signal, 2),
        histogram=round(latest_histogram, 2),
        signal=action,
        strength=strength,
        timestamp=now,
        close_price=round(latest_close, 2),
        reason=reason,
    )


def scan_macd_signals(
    instruments: Sequence[dict],
    config: MACDConfig,
) -> list[MACDSignal]:
    """Scan a list of instruments and return MACD signals."""
    signals: list[MACDSignal] = []
    total = len(instruments)

    for idx, inst in enumerate(instruments, start=1):
        symbol = inst["symbol"]
        name = inst.get("name", symbol)
        exchange = inst.get("exchange", "NSE")
        token = inst.get("token", "")

        if (idx % 20) == 0 or idx == total:
            logger.info("Scanning MACD: %d/%d (%s)", idx, total, symbol)

        sig = compute_macd_signal(symbol, name, exchange, config, token=token)
        if sig is not None:
            signals.append(sig)

    priority = {SignalAction.BUY: 0, SignalAction.SELL: 1, SignalAction.HOLD: 2}
    signals.sort(key=lambda s: (priority.get(s.signal, 3), s.symbol))
    return signals


def _classify_macd(
    crossover: float,
    macd: float,
    signal: float,
    histogram: float,
    config: MACDConfig,
) -> tuple[SignalAction, SignalStrength, str]:
    """Map MACD values to a signal action, strength, and reason."""
    if crossover > 0:
        # Bullish crossover
        if histogram > 0:
            return (
                SignalAction.BUY,
                SignalStrength.STRONG,
                f"MACD bullish crossover: MACD ({macd:.2f}) crossed above signal ({signal:.2f}), histogram {histogram:.2f}",
            )
        return (
            SignalAction.BUY,
            SignalStrength.MODERATE,
            f"MACD bullish crossover: MACD ({macd:.2f}) crossed above signal ({signal:.2f})",
        )

    if crossover < 0:
        # Bearish crossover
        if histogram < 0:
            return (
                SignalAction.SELL,
                SignalStrength.STRONG,
                f"MACD bearish crossover: MACD ({macd:.2f}) crossed below signal ({signal:.2f}), histogram {histogram:.2f}",
            )
        return (
            SignalAction.SELL,
            SignalStrength.MODERATE,
            f"MACD bearish crossover: MACD ({macd:.2f}) crossed below signal ({signal:.2f})",
        )

    # No crossover - check trend
    if macd > signal:
        return (
            SignalAction.HOLD,
            SignalStrength.MODERATE if histogram > 0 else SignalStrength.WEAK,
            f"MACD bullish: MACD ({macd:.2f}) above signal ({signal:.2f}), histogram {histogram:.2f}",
        )
    return (
        SignalAction.HOLD,
        SignalStrength.MODERATE if histogram < 0 else SignalStrength.WEAK,
        f"MACD bearish: MACD ({macd:.2f}) below signal ({signal:.2f}), histogram {histogram:.2f}",
    )


# ── Volume Signal Generation ───────────────────────────────────────────────


def compute_volume_signal(
    symbol: str,
    name: str,
    exchange: str,
    config: VolumeConfig,
    token: str = "",
) -> VolumeSignal | None:
    """Compute a volume-based signal for a single instrument."""
    df = _fetch_ohlcv(exchange, token)

    if df.empty or len(df) < config.period:
        logger.debug("Insufficient data for %s (token=%s)", symbol, token)
        return None

    volume_ratio, volume_trend = compute_volume_analysis(df["Volume"], period=config.period)
    surge = detect_volume_surge(df["Volume"], period=config.period, threshold=config.surge_threshold)

    latest_ratio = float(volume_ratio.iloc[-1])
    latest_trend = float(volume_trend.iloc[-1])
    latest_surge = float(surge.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)

    action, strength, reason = _classify_volume(
        latest_ratio, latest_trend, latest_surge, config
    )

    return VolumeSignal(
        symbol=symbol,
        name=name,
        exchange=exchange,
        volume_ratio=round(latest_ratio, 2),
        volume_trend=latest_trend,
        signal=action,
        strength=strength,
        timestamp=now,
        close_price=round(latest_close, 2),
        reason=reason,
    )


def scan_volume_signals(
    instruments: Sequence[dict],
    config: VolumeConfig,
) -> list[VolumeSignal]:
    """Scan a list of instruments and return volume signals."""
    signals: list[VolumeSignal] = []
    total = len(instruments)

    for idx, inst in enumerate(instruments, start=1):
        symbol = inst["symbol"]
        name = inst.get("name", symbol)
        exchange = inst.get("exchange", "NSE")
        token = inst.get("token", "")

        if (idx % 20) == 0 or idx == total:
            logger.info("Scanning Volume: %d/%d (%s)", idx, total, symbol)

        sig = compute_volume_signal(symbol, name, exchange, config, token=token)
        if sig is not None:
            signals.append(sig)

    priority = {SignalAction.BUY: 0, SignalAction.SELL: 1, SignalAction.HOLD: 2}
    signals.sort(key=lambda s: (priority.get(s.signal, 3), s.symbol))
    return signals


def _classify_volume(
    ratio: float,
    trend: float,
    surge: float,
    config: VolumeConfig,
) -> tuple[SignalAction, SignalStrength, str]:
    """Map volume metrics to a signal action, strength, and reason."""
    if surge > 0:
        # Volume surge detected
        if trend > 0:
            return (
                SignalAction.BUY,
                SignalStrength.STRONG,
                f"Volume surge ({ratio:.1f}x avg) with increasing volume — potential breakout",
            )
        return (
            SignalAction.SELL,
            SignalStrength.STRONG,
            f"Volume surge ({ratio:.1f}x avg) with decreasing volume — potential breakdown",
        )

    if ratio > 1.5:
        # Above average volume
        if trend > 0:
            return (
                SignalAction.BUY,
                SignalStrength.MODERATE,
                f"Above-average volume ({ratio:.1f}x) with increasing trend",
            )
        return (
            SignalAction.HOLD,
            SignalStrength.MODERATE,
            f"Above-average volume ({ratio:.1f}x) but decreasing trend",
        )

    if ratio < 0.5:
        # Below average volume
        return (
            SignalAction.HOLD,
            SignalStrength.WEAK,
            f"Below-average volume ({ratio:.1f}x) — low conviction",
        )

    # Normal volume
    if trend > 0:
        return (
            SignalAction.HOLD,
            SignalStrength.WEAK,
            f"Normal volume ({ratio:.1f}x avg) with slight increase",
        )
    return (
        SignalAction.HOLD,
        SignalStrength.WEAK,
        f"Normal volume ({ratio:.1f}x avg) with slight decrease",
    )


# ── Confluence Signal Generation ────────────────────────────────────────────


def compute_confluence_signal(
    symbol: str,
    name: str,
    exchange: str,
    rsi_config: RSIConfig,
    ema_config: EMAConfig,
    macd_config: MACDConfig,
    volume_config: VolumeConfig,
    token: str = "",
) -> ConfluenceSignal | None:
    """Compute a multi-signal confluence signal for a single instrument.

    Combines RSI, EMA crossover, MACD, and Volume signals to produce
    a higher-confidence trading recommendation.
    """
    # Fetch data once
    df = _fetch_ohlcv(exchange, token)
    if df.empty:
        return None

    # Compute individual signals
    rsi_sig = _compute_rsi_from_df(df, symbol, name, exchange, rsi_config)
    ema_sig = _compute_ema_from_df(df, symbol, name, exchange, ema_config)
    macd_sig = _compute_macd_from_df(df, symbol, name, exchange, macd_config)
    vol_sig = _compute_volume_from_df(df, symbol, name, exchange, volume_config)

    # Extract actions
    rsi_action = rsi_sig.signal if rsi_sig else None
    ema_action = ema_sig.signal if ema_sig else None
    macd_action = macd_sig.signal if macd_sig else None
    vol_action = vol_sig.signal if vol_sig else None

    # Count agreements
    actions = [a for a in [rsi_action, ema_action, macd_action, vol_action] if a is not None]
    buy_count = actions.count(SignalAction.BUY)
    sell_count = actions.count(SignalAction.SELL)
    total = len(actions)

    # Determine overall signal
    if buy_count >= 3:
        overall = SignalAction.BUY
        confidence = buy_count
        reason = f"Strong BUY confluence: {buy_count}/{total} indicators agree (RSI={rsi_action}, EMA={ema_action}, MACD={macd_action}, Vol={vol_action})"
    elif sell_count >= 3:
        overall = SignalAction.SELL
        confidence = sell_count
        reason = f"Strong SELL confluence: {sell_count}/{total} indicators agree (RSI={rsi_action}, EMA={ema_action}, MACD={macd_action}, Vol={vol_action})"
    elif buy_count >= 2:
        overall = SignalAction.BUY
        confidence = buy_count
        reason = f"Moderate BUY confluence: {buy_count}/{total} indicators agree"
    elif sell_count >= 2:
        overall = SignalAction.SELL
        confidence = sell_count
        reason = f"Moderate SELL confluence: {sell_count}/{total} indicators agree"
    else:
        overall = SignalAction.HOLD
        confidence = max(buy_count, sell_count)
        reason = f"Mixed signals: BUY={buy_count}, SELL={sell_count}, HOLD={total - buy_count - sell_count}"

    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)

    return ConfluenceSignal(
        symbol=symbol,
        name=name,
        exchange=exchange,
        rsi_signal=rsi_action,
        ema_signal=ema_action,
        macd_signal=macd_action,
        volume_signal=vol_action,
        overall_signal=overall,
        confidence=confidence,
        timestamp=now,
        close_price=round(latest_close, 2),
        reason=reason,
    )


def _compute_rsi_from_df(df: pd.DataFrame, symbol: str, name: str, exchange: str, config: RSIConfig) -> RSISignal | None:
    """Compute RSI signal from existing DataFrame."""
    if len(df) < config.period:
        return None
    rsi_series = compute_rsi(df["Close"], period=config.period)
    latest_rsi = float(rsi_series.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    action, strength, reason = _classify_rsi(latest_rsi, config)
    return RSISignal(
        symbol=symbol, name=name, exchange=exchange,
        rsi_value=round(latest_rsi, 2), signal=action, strength=strength,
        timestamp=now, close_price=round(latest_close, 2), reason=reason,
    )


def _compute_ema_from_df(df: pd.DataFrame, symbol: str, name: str, exchange: str, config: EMAConfig) -> EMACrossoverSignal | None:
    """Compute EMA crossover signal from existing DataFrame."""
    if len(df) < config.slow_period:
        return None
    fast_ema_series = compute_ema(df["Close"], period=config.fast_period)
    slow_ema_series = compute_ema(df["Close"], period=config.slow_period)
    crossover = detect_ema_crossover(fast_ema_series, slow_ema_series)
    latest_fast = float(fast_ema_series.iloc[-1])
    latest_slow = float(slow_ema_series.iloc[-1])
    latest_cross = float(crossover.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    action, strength, reason = _classify_ema_crossover(latest_cross, latest_fast, latest_slow, config)
    return EMACrossoverSignal(
        symbol=symbol, name=name, exchange=exchange,
        fast_ema=round(latest_fast, 2), slow_ema=round(latest_slow, 2),
        signal=action, strength=strength, timestamp=now,
        close_price=round(latest_close, 2), reason=reason,
    )


def _compute_macd_from_df(df: pd.DataFrame, symbol: str, name: str, exchange: str, config: MACDConfig) -> MACDSignal | None:
    """Compute MACD signal from existing DataFrame."""
    if len(df) < config.slow_period + config.signal_period:
        return None
    macd_line, signal_line, histogram = compute_macd(
        df["Close"], fast_period=config.fast_period,
        slow_period=config.slow_period, signal_period=config.signal_period,
    )
    crossover = detect_macd_crossover(macd_line, signal_line)
    latest_macd = float(macd_line.iloc[-1])
    latest_signal = float(signal_line.iloc[-1])
    latest_hist = float(histogram.iloc[-1])
    latest_cross = float(crossover.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    action, strength, reason = _classify_macd(latest_cross, latest_macd, latest_signal, latest_hist, config)
    return MACDSignal(
        symbol=symbol, name=name, exchange=exchange,
        macd_line=round(latest_macd, 2), signal_line=round(latest_signal, 2),
        histogram=round(latest_hist, 2), signal=action, strength=strength,
        timestamp=now, close_price=round(latest_close, 2), reason=reason,
    )


def _compute_volume_from_df(df: pd.DataFrame, symbol: str, name: str, exchange: str, config: VolumeConfig) -> VolumeSignal | None:
    """Compute Volume signal from existing DataFrame."""
    if len(df) < config.period:
        return None
    volume_ratio, volume_trend = compute_volume_analysis(df["Volume"], period=config.period)
    surge = detect_volume_surge(df["Volume"], period=config.period, threshold=config.surge_threshold)
    latest_ratio = float(volume_ratio.iloc[-1])
    latest_trend = float(volume_trend.iloc[-1])
    latest_surge = float(surge.iloc[-1])
    latest_close = float(df["Close"].iloc[-1])
    now = datetime.now(timezone.utc)
    action, strength, reason = _classify_volume(latest_ratio, latest_trend, latest_surge, config)
    return VolumeSignal(
        symbol=symbol, name=name, exchange=exchange,
        volume_ratio=round(latest_ratio, 2), volume_trend=latest_trend,
        signal=action, strength=strength, timestamp=now,
        close_price=round(latest_close, 2), reason=reason,
    )


def scan_confluence_signals(
    instruments: Sequence[dict],
    rsi_config: RSIConfig,
    ema_config: EMAConfig,
    macd_config: MACDConfig,
    volume_config: VolumeConfig,
) -> list[ConfluenceSignal]:
    """Scan a list of instruments and return confluence signals."""
    signals: list[ConfluenceSignal] = []
    total = len(instruments)

    for idx, inst in enumerate(instruments, start=1):
        symbol = inst["symbol"]
        name = inst.get("name", symbol)
        exchange = inst.get("exchange", "NSE")
        token = inst.get("token", "")

        if (idx % 20) == 0 or idx == total:
            logger.info("Scanning Confluence: %d/%d (%s)", idx, total, symbol)

        sig = compute_confluence_signal(
            symbol, name, exchange,
            rsi_config, ema_config, macd_config, volume_config,
            token=token,
        )
        if sig is not None:
            signals.append(sig)

    # Sort by confidence (highest first), then by overall signal priority
    priority = {SignalAction.BUY: 0, SignalAction.SELL: 1, SignalAction.HOLD: 2}
    signals.sort(key=lambda s: (-s.confidence, priority.get(s.overall_signal, 3), s.symbol))
    return signals
