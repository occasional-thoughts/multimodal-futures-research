"""Technical feature engine (Phase 7). Covers all five categories the plan asks for
-- trend, momentum, volatility, volume, price structure -- deliberately kept to ~22
features rather than "150 indicators because you can" (Phase 7's own warning). Every
feature uses only trailing (backward-looking) windows: nothing here reads a value that
wouldn't actually have been known at the decision cutoff (see trading_frequency.md).
"""

import numpy as np
import pandas as pd

TECH_COLUMNS = [
    # Trend
    "sma_10", "sma_50", "ema_12", "ema_26", "ma_slope_10",
    # Momentum
    "return_1d", "rsi_14", "macd", "macd_signal", "roc_10",
    # Volatility
    "realized_vol_20", "atr_14", "bb_width", "vol_change_10",
    # Volume
    "volume_change_1d", "relative_volume_20",
    # Price structure
    "gap", "high_low_range", "dist_from_high_20", "dist_from_low_20",
    "breakout_up", "breakout_down",
    # Trend-indicator momentum signals (Baz et al. 2015; adopted by Lim, Zohren &
    # Roberts 2019 "Enhancing Time-Series Momentum Strategies Using Deep Neural
    # Networks" -- see PHASE_TRACKER.md's Model 6 section). Deliberately different
    # from the plain `macd`/`macd_signal` pair above: those are raw MACD values on
    # one (12, 26) window pair; these are volatility-normalized trend-strength
    # SCORES at three different timescales, doubly normalized so they're comparable
    # in magnitude to each other and stationary across regimes (a raw MACD value on
    # a $180 crude contract isn't comparable to one on a $2,000 gold contract; a
    # normalized score is).
    "macd_trend_8_24", "macd_trend_16_48", "macd_trend_32_96",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return ema_fast, ema_slow, macd, macd_signal


def _bollinger_width(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper, lower = mid + n_std * std, mid - n_std * std
    return (upper - lower) / mid


def _macd_trend_signal(close: pd.Series, short: int, long: int) -> pd.Series:
    """Baz, Granger, Harvey, Le Roux & Rattray (2015) "Dissecting Investment
    Strategies in the Cross Section and Time Series" trend-indicator formula, as used
    by Lim/Zohren/Roberts (2019) and Wood/Zohren/Roberts's Momentum Transformer for
    exactly this asset class. Two normalization passes, not one: the raw EWMA
    difference is first scaled by 63-day (~1 quarter) realized price volatility to
    make it comparable across assets of very different price levels, then that
    already-normalized series is *itself* re-normalized by its own 252-day (~1 year)
    rolling std so the SCORE has a roughly consistent scale across time regimes too
    (a quiet-market trend score and a crisis-period trend score are put on the same
    footing). The final exp(-y^2/4)/0.89 response curve compresses extreme z-scores
    (a trend signal of z=6 isn't 3x more informative than z=2) while the /0.89
    divisor keeps the compressed output's variance close to 1, matching the
    original paper exactly (not an arbitrary constant).
    """
    macd = close.ewm(span=short, adjust=False).mean() - close.ewm(span=long, adjust=False).mean()
    q = macd / close.rolling(63).std()
    y = q / q.rolling(252).std()
    return y * np.exp(-(y**2) / 4) / 0.89


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range -- unlike a plain high-low range, true range also accounts
    for gaps from the prior close, which a simple daily range misses."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(window).mean()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: open, high, low, close, volume, return (from prices.py).
    All trailing windows only -- no look-ahead leakage."""
    out = pd.DataFrame(index=df.index)
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]

    # --- Trend ---
    sma_10 = c.rolling(10).mean()
    sma_50 = c.rolling(50).mean()
    ema_fast, ema_slow, macd, macd_signal = _macd(c)
    out["sma_10"] = sma_10 / c - 1
    out["sma_50"] = sma_50 / c - 1
    out["ema_12"] = ema_fast / c - 1
    out["ema_26"] = ema_slow / c - 1
    out["ma_slope_10"] = sma_10.diff(5) / sma_10.shift(5)  # slope of the MA itself, not price

    # --- Momentum ---
    out["return_1d"] = df["return"]
    out["rsi_14"] = _rsi(c, 14)
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["roc_10"] = c.pct_change(10)  # rate of change over 10 days, distinct from MACD

    # --- Volatility ---
    realized_vol_20 = df["return"].rolling(20).std()
    out["realized_vol_20"] = realized_vol_20
    out["atr_14"] = _atr(h, l, c, 14) / c
    out["bb_width"] = _bollinger_width(c, 20)
    out["vol_change_10"] = realized_vol_20 / realized_vol_20.shift(10) - 1

    # --- Volume ---
    # A handful of real trading days (holiday-adjacent low-activity sessions, e.g.
    # July 4th) report zero volume -- pct_change() from a zero baseline produces
    # +/-inf, not NaN, which silently slips past a plain .dropna() check (a real bug
    # caught by testing on live data: it broke sklearn's StandardScaler downstream in
    # Phase 15, not visible from a NaN-only check in Phase 7). Treat zero volume as
    # missing before the pct_change so the derived feature is correctly NaN instead.
    v_safe = v.replace(0, np.nan)
    out["volume_change_1d"] = v_safe.pct_change()
    out["relative_volume_20"] = v_safe / v_safe.rolling(20).mean()

    # --- Price structure ---
    out["gap"] = o / c.shift(1) - 1  # today's open vs yesterday's close
    out["high_low_range"] = (h - l) / c
    rolling_high_20 = h.shift(1).rolling(20).max()  # excludes today -- can't compare
    rolling_low_20 = l.shift(1).rolling(20).min()   # today's own high/low to itself
    out["dist_from_high_20"] = (c - rolling_high_20) / c
    out["dist_from_low_20"] = (c - rolling_low_20) / c
    out["breakout_up"] = (c > rolling_high_20).astype(float)
    out["breakout_down"] = (c < rolling_low_20).astype(float)

    # --- Trend-indicator momentum signals (see _macd_trend_signal docstring) ---
    out["macd_trend_8_24"] = _macd_trend_signal(c, 8, 24)
    out["macd_trend_16_48"] = _macd_trend_signal(c, 16, 48)
    out["macd_trend_32_96"] = _macd_trend_signal(c, 32, 96)

    return out
