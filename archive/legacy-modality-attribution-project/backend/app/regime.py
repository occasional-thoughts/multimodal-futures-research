import pandas as pd

from app.config import EMBARGO_DAYS, VOL_WINDOW


def label_regimes(returns: pd.Series, window: int = VOL_WINDOW, embargo: int = EMBARGO_DAYS) -> pd.Series:
    """Calm/volatile split on trailing realized volatility, median-thresholded.

    Rows within `embargo` trading days of the calm/volatile boundary are marked
    "embargo" (excluded from evaluation) so a rolling-window volatility estimate
    never mixes information from across the boundary it's used to define.
    """
    realized_vol = returns.rolling(window).std()
    median_vol = realized_vol.median()
    regime = pd.Series(index=returns.index, dtype=object)
    regime[realized_vol <= median_vol] = "calm"
    regime[realized_vol > median_vol] = "volatile"

    changes = regime.ne(regime.shift()).cumsum()
    for _, group in regime.groupby(changes):
        idx = group.index
        embargo_idx = idx[:embargo].union(idx[-embargo:])
        regime.loc[embargo_idx] = "embargo"

    return regime
