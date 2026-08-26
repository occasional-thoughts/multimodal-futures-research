"""Target construction (Phase 14) -- implements the output quantities defined in
problem_statement.md as actual computable columns.

Important and deliberate contrast with every feature module built so far (technical.py,
macro.py, fundamentals.py, news_pipeline.py, timing.py): those must NEVER look ahead of
the decision cutoff. Targets are the opposite by construction -- a supervised-learning
label has to be a forward-looking outcome, or there's nothing to predict. The risk this
creates is accidentally letting a target column leak into the feature set; keeping
targets in their own module, never imported by anything in features/ or data/, is the
guard against that.

Fix (research-motivated horizon extension): the original version hardcoded column
names to "5d" regardless of the `horizon` argument actually passed in -- calling
compute_targets(horizon=20) would silently produce a column still named
"target_return_5d" that actually held a 20-day return. Caught before it caused
confusion, not after. Column names are now genuinely parameterized by horizon.
"""

import numpy as np
import pandas as pd

HORIZON_DAYS = 5  # kept as the original default horizon


def target_columns(horizon: int) -> list[str]:
    return ["target_return_1d", f"target_return_{horizon}d", f"target_direction_{horizon}d", f"target_volatility_{horizon}d"]


TARGET_COLUMNS = target_columns(HORIZON_DAYS)  # backward-compatible default (5-day), used by every script built before the horizon-extension research


def compute_targets(price_df: pd.DataFrame, horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """price_df must have a 'close' column and a 'return' column (daily return),
    as produced by app.data.prices.fetch_price_history. Every row's targets describe
    what happens AFTER that row's date -- by construction, the last `horizon` rows of
    any price series will have NaN targets (there's no future data left to compute
    them from), which is correct and expected, not a bug to silently fill.

    target_return_1d is always a fixed 1-day return regardless of `horizon` (the
    secondary/diagnostic output from problem_statement.md); the other three columns
    are parameterized by `horizon` and named accordingly.
    """
    close = price_df["close"]
    out = pd.DataFrame(index=price_df.index)
    return_col, direction_col, vol_col = f"target_return_{horizon}d", f"target_direction_{horizon}d", f"target_volatility_{horizon}d"

    # r_{t+1} = (P_{t+1} - P_t) / P_t -- fixed 1-day, independent of `horizon`
    out["target_return_1d"] = close.shift(-1) / close - 1

    # r_{t,t+horizon} = (P_{t+horizon} - P_t) / P_t
    out[return_col] = close.shift(-horizon) / close - 1

    # Realized ground-truth direction -- what a probability-of-positive-return
    # model is trained/evaluated against.
    out[direction_col] = (out[return_col] > 0).astype(float)
    out.loc[out[return_col].isna(), direction_col] = np.nan

    # Realized forward volatility over the same horizon -- the label for the
    # "expected volatility" output; std of the `horizon` forward daily returns.
    forward_returns = pd.concat([price_df["return"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    out[vol_col] = forward_returns.std(axis=1)

    return out
