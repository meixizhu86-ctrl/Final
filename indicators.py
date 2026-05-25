import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def bbands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    dif = ema(close, fast_period) - ema(close, slow_period)
    dea = dif.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    hist = dif - dea
    return dif, dea, hist


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    rsv_period: int = 9,
    k_period: int = 3,
    d_period: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    lowest_low = low.rolling(window=rsv_period, min_periods=rsv_period).min()
    highest_high = high.rolling(window=rsv_period, min_periods=rsv_period).max()
    denominator = (highest_high - lowest_low).replace(0, np.nan)
    rsv = ((close - lowest_low) / denominator * 100).fillna(50)
    k = rsv.ewm(alpha=1 / k_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def crossed_above(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) <= right.shift(1)) & (left > right)


def crossed_below(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) >= right.shift(1)) & (left < right)
