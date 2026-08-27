"""Walk-forward evaluation of Model 9 (Deep Momentum Network pooled across 13
futures markets -- see train_model9_dmn_broad.py's module docstring). Same 3-fold
expanding-window methodology as every other walk-forward runner in this project.

Run from backend/: python scripts/run_walk_forward_dmn_broad.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from scripts.train_model9_dmn_broad import train_dmn_broad

FOLDS = [
    {"train_frac": 0.5, "val_frac": 0.10, "test_end_frac": 0.65},
    {"train_frac": 0.65, "val_frac": 0.10, "test_end_frac": 0.80},
    {"train_frac": 0.80, "val_frac": 0.10, "test_end_frac": 0.95},
]

if __name__ == "__main__":
    fold_results = []
    for i, fold in enumerate(FOLDS):
        print(f"\n=== Fold {i + 1}/{len(FOLDS)} ===")
        result = train_dmn_broad(seed=42, verbose=False, **fold)
        print(f"  Portfolio Sharpe: DMN={result['dmn_sharpe']:.3f}  buy&hold={result['bh_sharpe']:.3f}  classical_trend={result['classical_sharpe']:.3f}")
        fold_results.append(result)

    print("\n=== Walk-forward summary (Model 9: broad-universe DMN, portfolio Sharpe) ===")
    print(f"{'Strategy':<18}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for key, label in [("dmn_sharpe", "DMN (learned)"), ("bh_sharpe", "Buy & hold"), ("classical_sharpe", "Classical trend")]:
        vals = [fr[key] for fr in fold_results]
        print(f"{label:<18}{vals[0]:<10.3f}{vals[1]:<10.3f}{vals[2]:<10.3f}{np.mean(vals):<10.3f}{np.std(vals):<10.3f}")
