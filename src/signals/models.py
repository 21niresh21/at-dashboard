"""Signal data models and configuration.

Defines the dataclasses and enums used throughout the signal engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class SignalAction(str, Enum):
    """Trading action recommended by a signal."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalStrength(str, Enum):
    """Confidence level of a signal."""

    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"


class SignalType(str, Enum):
    """Type of trading signal."""

    RSI = "RSI"
    EMA_CROSSOVER = "EMA_CROSSOVER"
    MACD = "MACD"
    VOLUME = "VOLUME"


@dataclass(frozen=True)
class RSIConfig:
    """Configuration for RSI-based signal generation.

    Attributes:
        period: RSI lookback period (default 14).
        oversold: RSI threshold below which a BUY signal is generated (default 30).
        overbought: RSI threshold above which a SELL signal is generated (default 70).
    """

    period: int = 14
    oversold: float = 30.0
    overbought: float = 70.0


@dataclass(frozen=True)
class EMAConfig:
    """Configuration for EMA crossover signal generation.

    Attributes:
        fast_period: Fast EMA period (default 9).
        slow_period: Slow EMA period (default 21).
    """

    fast_period: int = 9
    slow_period: int = 21


@dataclass(frozen=True)
class RSISignal:
    """A single RSI-based trading signal.

    Attributes:
        symbol: Instrument symbol (e.g. "RELIANCE").
        name: Instrument display name.
        exchange: Exchange segment (NSE / NFO).
        rsi_value: Current RSI value (0–100).
        signal: Recommended action (BUY / SELL / HOLD).
        strength: Signal confidence (STRONG / MODERATE / WEAK).
        timestamp: When the signal was generated (UTC).
        close_price: Latest closing price.
        reason: Human-readable explanation of the signal.
    """

    symbol: str
    name: str
    exchange: str
    rsi_value: float
    signal: SignalAction
    strength: SignalStrength
    timestamp: datetime
    close_price: float
    reason: str
    signal_type: SignalType = SignalType.RSI

    def to_dict(self) -> dict:
        """Convert to a flat dict for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "rsi": self.rsi_value,
            "signal": self.signal.value,
            "strength": self.strength.value,
            "close": self.close_price,
            "reason": self.reason,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
            "type": self.signal_type.value,
        }


@dataclass(frozen=True)
class EMACrossoverSignal:
    """A single EMA crossover trading signal.

    Attributes:
        symbol: Instrument symbol (e.g. "RELIANCE").
        name: Instrument display name.
        exchange: Exchange segment (NSE / NFO).
        fast_ema: Current fast EMA value.
        slow_ema: Current slow EMA value.
        signal: Recommended action (BUY / SELL / HOLD).
        strength: Signal confidence (STRONG / MODERATE / WEAK).
        timestamp: When the signal was generated (UTC).
        close_price: Latest closing price.
        reason: Human-readable explanation of the signal.
    """

    symbol: str
    name: str
    exchange: str
    fast_ema: float
    slow_ema: float
    signal: SignalAction
    strength: SignalStrength
    timestamp: datetime
    close_price: float
    reason: str
    signal_type: SignalType = SignalType.EMA_CROSSOVER

    def to_dict(self) -> dict:
        """Convert to a flat dict for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "fast_ema": round(self.fast_ema, 2),
            "slow_ema": round(self.slow_ema, 2),
            "signal": self.signal.value,
            "strength": self.strength.value,
            "close": self.close_price,
            "reason": self.reason,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
            "type": self.signal_type.value,
        }


@dataclass(frozen=True)
class MACDConfig:
    """Configuration for MACD signal generation.

    Attributes:
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).
    """

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9


@dataclass(frozen=True)
class VolumeConfig:
    """Configuration for volume-based signal generation.

    Attributes:
        period: Lookback period for average volume (default 20).
        surge_threshold: Volume ratio threshold for surge detection (default 2.0).
    """

    period: int = 20
    surge_threshold: float = 2.0


@dataclass(frozen=True)
class MACDSignal:
    """A single MACD-based trading signal.

    Attributes:
        symbol: Instrument symbol (e.g. "RELIANCE").
        name: Instrument display name.
        exchange: Exchange segment (NSE / NFO).
        macd_line: Current MACD line value.
        signal_line: Current signal line value.
        histogram: Current histogram value (MACD - signal).
        signal: Recommended action (BUY / SELL / HOLD).
        strength: Signal confidence (STRONG / MODERATE / WEAK).
        timestamp: When the signal was generated (UTC).
        close_price: Latest closing price.
        reason: Human-readable explanation of the signal.
    """

    symbol: str
    name: str
    exchange: str
    macd_line: float
    signal_line: float
    histogram: float
    signal: SignalAction
    strength: SignalStrength
    timestamp: datetime
    close_price: float
    reason: str
    signal_type: SignalType = SignalType.MACD

    def to_dict(self) -> dict:
        """Convert to a flat dict for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "macd": round(self.macd_line, 2),
            "signal": round(self.signal_line, 2),
            "histogram": round(self.histogram, 2),
            "action": self.signal.value,
            "strength": self.strength.value,
            "close": self.close_price,
            "reason": self.reason,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
            "type": self.signal_type.value,
        }


@dataclass(frozen=True)
class VolumeSignal:
    """A single volume-based trading signal.

    Attributes:
        symbol: Instrument symbol (e.g. "RELIANCE").
        name: Instrument display name.
        exchange: Exchange segment (NSE / NFO).
        volume_ratio: Current volume / average volume ratio.
        volume_trend: Volume trend direction (1.0 = increasing, -1.0 = decreasing).
        signal: Recommended action (BUY / SELL / HOLD).
        strength: Signal confidence (STRONG / MODERATE / WEAK).
        timestamp: When the signal was generated (UTC).
        close_price: Latest closing price.
        reason: Human-readable explanation of the signal.
    """

    symbol: str
    name: str
    exchange: str
    volume_ratio: float
    volume_trend: float
    signal: SignalAction
    strength: SignalStrength
    timestamp: datetime
    close_price: float
    reason: str
    signal_type: SignalType = SignalType.VOLUME

    def to_dict(self) -> dict:
        """Convert to a flat dict for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "volume_ratio": round(self.volume_ratio, 2),
            "volume_trend": self.volume_trend,
            "action": self.signal.value,
            "strength": self.strength.value,
            "close": self.close_price,
            "reason": self.reason,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
            "type": self.signal_type.value,
        }


@dataclass(frozen=True)
class ConfluenceSignal:
    """Multi-signal confluence trading signal.

    Combines signals from multiple indicators to produce a higher-confidence
    trading recommendation.

    Attributes:
        symbol: Instrument symbol.
        name: Instrument display name.
        exchange: Exchange segment.
        rsi_signal: RSI signal action (or None if not computed).
        ema_signal: EMA crossover signal action (or None).
        macd_signal: MACD signal action (or None).
        volume_signal: Volume signal action (or None).
        overall_signal: Combined recommendation (BUY / SELL / HOLD).
        confidence: Number of indicators agreeing (0-4).
        timestamp: When the signal was generated (UTC).
        close_price: Latest closing price.
        reason: Human-readable explanation.
    """

    symbol: str
    name: str
    exchange: str
    rsi_signal: SignalAction | None
    ema_signal: SignalAction | None
    macd_signal: SignalAction | None
    volume_signal: SignalAction | None
    overall_signal: SignalAction
    confidence: int
    timestamp: datetime
    close_price: float
    reason: str

    def to_dict(self) -> dict:
        """Convert to a flat dict for DataFrame construction."""
        return {
            "symbol": self.symbol,
            "name": self.name,
            "exchange": self.exchange,
            "rsi": self.rsi_signal.value if self.rsi_signal else "N/A",
            "ema": self.ema_signal.value if self.ema_signal else "N/A",
            "macd": self.macd_signal.value if self.macd_signal else "N/A",
            "volume": self.volume_signal.value if self.volume_signal else "N/A",
            "overall": self.overall_signal.value,
            "confidence": self.confidence,
            "close": self.close_price,
            "reason": self.reason,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M UTC"),
        }
