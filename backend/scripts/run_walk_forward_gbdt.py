"""Walk-forward evaluation of Model 8 (regularized XGBoost on tech+macro+COT, no news
-- see train_model8_gbdt.py's module docstring). Same 3-fold expanding-window
methodology as every other walk-forward runner in this project -- a different model
family doesn't get a pass on the scrutiny that caught real problems in every one so far.

Run from backend/: python scripts/run_walk_forward_gbdt.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from scripts.train_model8_gbdt import ASSETS, train_gbdt_market

FOLDS = [
    {"train_frac": 0.5, "val_frac": 0.10, "test_end_frac": 0.65},
    {"train_frac": 0.65, "val_frac": 0.10, "test_end_frac": 0.80},
    {"train_frac": 0.80, "val_frac": 0.10, "test_end_frac": 0.95},
]

if __name__ == "__main__":
    fold_results = {t: [] for t in ASSETS}
    for i, fold in enumerate(FOLDS):
        print(f"\n=== Fold {i + 1}/{len(FOLDS)} ===")
        for ticker in ASSETS:
            r = train_gbdt_market(ticker, verbose=True, **fold)
            fold_results[ticker].append(r["balanced_acc"])

    print("\n=== Walk-forward summary (Model 8: XGBoost tech+macro+COT, balanced accuracy) ===")
    print(f"{'Market':<8}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for ticker in ASSETS:
        accs = fold_results[ticker]
        print(f"{ticker:<8}{accs[0]:<10.3f}{accs[1]:<10.3f}{accs[2]:<10.3f}{np.mean(accs):<10.3f}{np.std(accs):<10.3f}")
