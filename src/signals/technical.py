"""Technical indicator calculations.

Pure functions that operate on pandas DataFrames.
No external state — easy to test and reuse.
"""

from __future__ import annotations

import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a price series.

    Uses the standard Wilder smoothing method (EMA-based).

    Args:
        series: Closing price series.
        period: RSI lookback period (default 14).

    Returns:
        A pandas Series of RSI values (0–100).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Compute Exponential Moving Average (EMA).

    Args:
        series: Price series (typically closing prices).
        period: EMA lookback period.

    Returns:
        A pandas Series of EMA values.
    """
    return series.ewm(span=period, adjust=False).mean()


def detect_ema_crossover(
    fast_ema: pd.Series,
    slow_ema: pd.Series,
) -> pd.Series:
    """Detect EMA crossover events.

    Args:
        fast_ema: Fast EMA series.
        slow_ema: Slow EMA series.

    Returns:
        A pandas Series with values:
        - 1.0: Bullish crossover (fast crossed above slow)
        - -1.0: Bearish crossover (fast crossed below slow)
        - 0.0: No crossover
    """
    # Current position: is fast above slow?
    fast_above = fast_ema > slow_ema
    # Previous position (fill NaN with False for first row)
    fast_above_prev = fast_above.shift(1).fillna(False)

    # Crossover detection
    bullish = fast_above & ~fast_above_prev  # Fast crossed above slow
    bearish = ~fast_above & fast_above_prev  # Fast crossed below slow

    crossover = pd.Series(0.0, index=fast_ema.index)
    crossover[bullish] = 1.0
    crossover[bearish] = -1.0

    return crossover


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD (Moving Average Convergence Divergence).

    Args:
        series: Closing price series.
        fast_period: Fast EMA period (default 12).
        slow_period: Slow EMA period (default 26).
        signal_period: Signal line EMA period (default 9).

    Returns:
        Tuple of (macd_line, signal_line, histogram).
    """
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def detect_macd_crossover(
    macd_line: pd.Series,
    signal_line: pd.Series,
) -> pd.Series:
    """Detect MACD crossover events.

    Args:
        macd_line: MACD line series.
        signal_line: Signal line series.

    Returns:
        A pandas Series with values:
        - 1.0: Bullish crossover (MACD crossed above signal)
        - -1.0: Bearish crossover (MACD crossed below signal)
        - 0.0: No crossover
    """
    macd_above = macd_line > signal_line
    macd_above_prev = macd_above.shift(1).fillna(False)

    bullish = macd_above & ~macd_above_prev
    bearish = ~macd_above & macd_above_prev

    crossover = pd.Series(0.0, index=macd_line.index)
    crossover[bullish] = 1.0
    crossover[bearish] = -1.0

    return crossover


def compute_volume_analysis(
    volume: pd.Series,
    period: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Compute volume analysis indicators.

    Args:
        volume: Volume series.
        period: Lookback period for average volume (default 20).

    Returns:
        Tuple of (volume_ratio, volume_trend).
        - volume_ratio: Current volume / average volume (>1 = above average)
        - volume_trend: 1.0 if volume increasing, -1.0 if decreasing, 0.0 if flat
    """
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    volume_ratio = volume / avg_volume

    # Volume trend: compare current volume to previous volume
    volume_change = volume.diff()
    volume_trend = pd.Series(0.0, index=volume.index)
    volume_trend[volume_change > 0] = 1.0
    volume_trend[volume_change < 0] = -1.0

    return volume_ratio, volume_trend


def detect_volume_surge(
    volume: pd.Series,
    period: int = 20,
    threshold: float = 2.0,
) -> pd.Series:
    """Detect volume surges (unusually high volume).

    Args:
        volume: Volume series.
        period: Lookback period for average volume (default 20).
        threshold: Volume ratio threshold for surge detection (default 2.0).

    Returns:
        A pandas Series with values:
        - 1.0: Volume surge detected (volume > threshold * average)
        - 0.0: Normal volume
    """
    avg_volume = volume.rolling(window=period, min_periods=period).mean()
    volume_ratio = volume / avg_volume

    surge = pd.Series(0.0, index=volume.index)
    surge[volume_ratio > threshold] = 1.0

    return surge


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Compute Average True Range (ATR).

    True Range is the greatest of:
      - Current high minus current low
      - |Current high minus previous close|
      - |Current low  minus previous close|

    ATR is the EMA-smoothed True Range (Wilder smoothing).

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        period: ATR lookback period (default 14).

    Returns:
        A pandas Series of ATR values.
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder smoothing: alpha = 1 / period
    atr = true_range.ewm(alpha=1 / period, min_periods=period).mean()
    return atr


def compute_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Compute Volume Weighted Average Price (VWAP).

    VWAP = cumulative(typical_price * volume) / cumulative(volume)

    Typical price = (high + low + close) / 3.

    Args:
        high: High price series.
        low: Low price series.
        close: Closing price series.
        volume: Volume series.

    Returns:
        A pandas Series of VWAP values.
    """
    typical_price = (high + low + close) / 3.0
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()

    vwap = cumulative_tp_vol / cumulative_vol
    return vwap


def compute_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands.

    Middle band = SMA(period)
    Upper band  = SMA + num_std * std
    Lower band  = SMA - num_std * std

    Args:
        series: Price series (typically closing prices).
        period: Moving average lookback period (default 20).
        num_std: Number of standard deviations for band width (default 2.0).

    Returns:
        Tuple of (upper_band, middle_band, lower_band).
    """
    middle_band = series.rolling(window=period, min_periods=period).mean()
    rolling_std = series.rolling(window=period, min_periods=period).std()

    upper_band = middle_band + num_std * rolling_std
    lower_band = middle_band - num_std * rolling_std

    return upper_band, middle_band, lower_band
