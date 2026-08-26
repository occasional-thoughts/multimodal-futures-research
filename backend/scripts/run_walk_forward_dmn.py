"""Walk-forward evaluation of Model 6 (Deep Momentum Network, Sharpe-optimized
position sizing -- see train_model6_dmn.py's module docstring for the full research
grounding and the news-feature-leak fix). Same 3-fold expanding-window methodology as
Phase 26's run_walk_forward.py -- a Sharpe-ratio reframing doesn't get a pass on the
scrutiny that caught real problems in every earlier model.

Run from backend/: python scripts/run_walk_forward_dmn.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from scripts.train_model6_dmn import ASSETS, train_dmn

FOLDS = [
    {"train_frac": 0.5, "val_frac": 0.10, "test_end_frac": 0.65},
    {"train_frac": 0.65, "val_frac": 0.10, "test_end_frac": 0.80},
    {"train_frac": 0.80, "val_frac": 0.10, "test_end_frac": 0.95},
]

if __name__ == "__main__":
    fold_results = []
    for i, fold in enumerate(FOLDS):
        print(f"\n=== Fold {i + 1}/{len(FOLDS)} ===")
        result = train_dmn(seed=42, verbose=False, **fold)
        print(
            f"  Portfolio Sharpe: DMN={result['dmn_sharpe']:.3f}  "
            f"buy&hold={result['bh_sharpe']:.3f}  classical_trend={result['classical_sharpe']:.3f}  "
            + "  ".join(f"{t}={result['per_market_dmn_sharpe'][t]:.3f}" for t in ASSETS)
        )
        fold_results.append(result)

    print("\n=== Walk-forward summary (Model 6: Deep Momentum Network, portfolio Sharpe) ===")
    print(f"{'Strategy':<18}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for key, label in [("dmn_sharpe", "DMN (learned)"), ("bh_sharpe", "Buy & hold"), ("classical_sharpe", "Classical trend")]:
        vals = [fr[key] for fr in fold_results]
        print(f"{label:<18}{vals[0]:<10.3f}{vals[1]:<10.3f}{vals[2]:<10.3f}{np.mean(vals):<10.3f}{np.std(vals):<10.3f}")

    print(f"\n{'Market (DMN)':<18}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for ticker in ASSETS:
        vals = [fr["per_market_dmn_sharpe"][ticker] for fr in fold_results]
        print(f"{ticker:<18}{vals[0]:<10.3f}{vals[1]:<10.3f}{vals[2]:<10.3f}{np.mean(vals):<10.3f}{np.std(vals):<10.3f}")
