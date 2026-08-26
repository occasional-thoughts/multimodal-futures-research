"""Phase 15: simple baselines, run BEFORE any deep learning model is built. Evaluated
on prediction quality (directional accuracy against target_direction_5d) since the
trading-strategy/paper-trading/backtest machinery (Phases 21-29) doesn't exist yet --
full P&L/Sharpe comparison happens later once that's built. This is a real
architectural constraint of doing things in phase order, not a shortcut.

Run from backend/: python scripts/train_baselines.py [TICKER]
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from app.config import HISTORY_PERIOD
from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.targets import TARGET_COLUMNS, compute_targets

TRAIN_FRAC, VAL_FRAC = 0.7, 0.15


def build_dataset(ticker: str, period: str = HISTORY_PERIOD):
    price_df = fetch_price_history(ticker, period=period)
    features = compute_indicators(price_df)
    targets = compute_targets(price_df)
    full = pd.concat([price_df[["return"]], features, targets], axis=1).dropna()
    return full


def evaluate_baselines(ticker: str):
    df = build_dataset(ticker)
    n = len(df)
    train_end, val_end = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))
    # NOTE: an earlier version of this script computed val_end but never used it --
    # the middle 15% of data was silently dropped instead of used for validation,
    # and XGBoost ran with arbitrary unregularized settings. That combination produced
    # a real, caught bug: 100% train accuracy / 38.9% test accuracy on CL (textbook
    # overfitting on ~250 training rows). Fixed here with a real validation split used
    # for early stopping, applied uniformly to all markets -- not tuned to make any
    # one market's number look better.
    train, val, test = df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    print(f"{ticker}: {n} usable rows -> train {len(train)}, val {len(val)}, test {len(test)}")

    y_test = test["target_direction_5d"].to_numpy()
    results = {}

    # Baseline 1: Buy and hold -- always predict "up"
    pred = np.ones(len(test))
    results["1_buy_and_hold"] = accuracy_score(y_test, pred)

    # Baseline 2: Momentum -- predict continuation of the sign of the recent 10-day ROC
    pred = (test["roc_10"] > 0).astype(int).to_numpy()
    results["2_momentum"] = accuracy_score(y_test, pred)

    # Baseline 3: Moving-average strategy -- predict up if price is ABOVE its 10-day
    # average. Note the sign: technical.py stores sma_10 as (MA/close - 1), so price
    # above its average means sma_10 < 0 -- got this backwards on first pass and
    # caught it by checking the feature's actual definition rather than assuming.
    pred = (test["sma_10"] < 0).astype(int).to_numpy()
    results["3_moving_average"] = accuracy_score(y_test, pred)

    # Baseline 4: Simple logistic regression on engineered features
    scaler = StandardScaler().fit(train[TECH_COLUMNS])
    X_train, X_test = scaler.transform(train[TECH_COLUMNS]), scaler.transform(test[TECH_COLUMNS])
    logreg = LogisticRegression(max_iter=1000).fit(X_train, train["target_direction_5d"])
    results["4_logistic_regression"] = accuracy_score(y_test, logreg.predict(X_test))

    # Baseline 5: XGBoost on engineered features, with real regularization and
    # validation-based early stopping instead of a fixed, arbitrary tree count.
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=2.0,
        eval_metric="logloss",
        early_stopping_rounds=20,
    )
    xgb.fit(train[TECH_COLUMNS], train["target_direction_5d"], eval_set=[(val[TECH_COLUMNS], val["target_direction_5d"])], verbose=False)
    train_acc = accuracy_score(train["target_direction_5d"], xgb.predict(train[TECH_COLUMNS]))
    test_acc = accuracy_score(y_test, xgb.predict(test[TECH_COLUMNS]))
    print(f"  (xgboost: best_iteration={xgb.best_iteration}, train_acc={train_acc:.3f}, test_acc={test_acc:.3f})")
    results["5_xgboost"] = test_acc

    return results


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "ZN=F"
    results = evaluate_baselines(ticker)
    print(f"\n{ticker} baseline directional accuracy (predicting target_direction_5d):")
    for name, acc in results.items():
        print(f"  {name:25s} {acc:.3f}")
