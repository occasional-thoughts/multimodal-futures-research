"""Model 8: regularized gradient-boosted trees (XGBoost) on the full REAL feature set
(technical incl. Baz et al. trend indicators, macro, and CFTC COT incl. a new
positioning-extremity transform) -- deliberately NO news, so this model carries zero
risk of the leak found and confirmed in Models 6 and 7.

Grounded in real literature, not a fallback guess: gradient-boosted trees are
repeatedly shown to outperform deep learning specifically on small-sample tabular
data -- Shwartz-Ziv & Armon (2021), "Tabular data: Deep learning is not all you need",
found XGBoost beat deep models on 8 of 11 tabular benchmark datasets, and this remains
close to a literature consensus for exactly this data regime (a few thousand rows,
heterogeneous engineered features) that every GRU/co-attention model in this project
(1-7) has been applied to. Worth a properly-built, honestly-evaluated try, not
previously done at this scale: Phase 15's original XGBoost baseline predates the
10-year data extension, the Baz trend-indicator features, and COT entirely.

Also adds a real, literature-motivated feature this project had access to but never
computed: `cot_commercial_percentile_3y` / `cot_speculator_percentile_3y` (the
"COT Index" -- trailing 3-year percentile rank of net positioning, not the raw level).
Practitioner and academic sources flag positioning EXTREMES specifically (not the
level itself) as informative, with commercial hedgers at extremes correctly signaling
direction in some studies on the order of 70% of the time -- see `app/data/cot.py`.

Trained PER-MARKET (not pooled with an asset embedding like the neural models) --
each market's macro feature set has a different column count (see macro.py/
fundamentals.py), and a tree model has no natural equivalent to an NN's per-asset
embedder for handling that; per-market GBDTs are also simply the standard way this
kind of baseline is built (Phase 15's original baselines were per-market too).

Regularization follows the fix already established in Phase 15 (train_baselines.py):
max_depth=3, subsample/colsample<1, reg_alpha/reg_lambda, early_stopping_rounds on a
real validation split -- applied here too, not assumed still sufficient at 10y without
checking. `scale_pos_weight` handles class imbalance, matching this project's
established balanced-accuracy discipline (training_utils.py) -- checkpoint selection
and reporting both use balanced accuracy, not raw, for the same reason documented
there: a majority-class collapse must not be able to hide behind a flattering number.

Run from backend/: python scripts/train_model8_gbdt.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from xgboost import XGBClassifier

from app.config import HISTORY_PERIOD
from app.data.cot import COT_COLUMNS, fetch_cot_features
from app.data.fundamentals import CL_COLUMNS, GC_COLUMNS, fetch_cl_fundamentals, fetch_gc_fundamentals
from app.data.macro import ZN_MACRO_COLUMNS, fetch_zn_macro_features
from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.targets import compute_targets

TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
ASSETS = ["ZN=F", "CL=F", "GC=F"]

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS),
}

_market_data_cache: dict = {}


def build_market_data(ticker: str) -> tuple[pd.DataFrame, list]:
    if ticker in _market_data_cache:
        return _market_data_cache[ticker]
    macro_fetcher, macro_columns = _MACRO_FETCHERS[ticker]
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)
    tech_df = compute_indicators(price_df)
    macro_df = macro_fetcher(price_df.index)
    cot_df = fetch_cot_features(ticker, price_df.index)
    targets_df = compute_targets(price_df)  # default 5-day horizon
    full = pd.concat([tech_df, macro_df, cot_df, targets_df], axis=1).dropna()
    feature_columns = TECH_COLUMNS + macro_columns + COT_COLUMNS
    _market_data_cache[ticker] = (full, feature_columns)
    return full, feature_columns


def train_gbdt_market(ticker: str, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_end_frac=1.0, verbose=True):
    full, feature_columns = build_market_data(ticker)
    n = len(full)
    train_end, val_end, test_end = int(n * train_frac), int(n * (train_frac + val_frac)), int(n * test_end_frac)
    train, val, test = full.iloc[:train_end], full.iloc[train_end:val_end], full.iloc[val_end:test_end]

    X_train, y_train = train[feature_columns], train["target_direction_5d"]
    X_val, y_val = val[feature_columns], val["target_direction_5d"]
    X_test, y_test = test[feature_columns], test["target_direction_5d"]

    pos = y_train.sum()
    scale_pos_weight = (len(y_train) - pos) / max(pos, 1.0)  # same class-imbalance fix as every NN model here

    model = XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0, scale_pos_weight=scale_pos_weight,
        eval_metric="logloss", early_stopping_rounds=20, random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    balanced_acc = balanced_accuracy_score(y_test, pred)
    pred_dist = dict(zip(*np.unique(pred, return_counts=True)))
    if verbose:
        print(f"{ticker}: acc={acc:.3f} balanced_acc={balanced_acc:.3f} pred_dist={pred_dist} ({len(y_test)} test rows)")
    return {"acc": acc, "balanced_acc": balanced_acc}


if __name__ == "__main__":
    for ticker in ASSETS:
        train_gbdt_market(ticker)
