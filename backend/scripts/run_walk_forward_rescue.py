"""Walk-forward evaluation of the "rescue" model (20-day horizon + CFTC COT data),
using the exact same 3-fold expanding-window methodology as Phase 26's
run_walk_forward.py -- a single split isn't trustworthy on its own (that's the whole
lesson of Phase 26), so the rescue attempt gets the same scrutiny as everything else,
not a pass on the methodology that caught the original problem.

Run from backend/: python scripts/run_walk_forward_rescue.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from scripts.train_model5_rescue import ASSETS, train_joint

FOLDS = [
    {"train_frac": 0.5, "val_frac": 0.10, "test_end_frac": 0.65},
    {"train_frac": 0.65, "val_frac": 0.10, "test_end_frac": 0.80},
    {"train_frac": 0.80, "val_frac": 0.10, "test_end_frac": 0.95},
]

if __name__ == "__main__":
    fold_results = []
    for i, fold in enumerate(FOLDS):
        print(f"\n=== Fold {i + 1}/{len(FOLDS)} ===")
        result = train_joint(seed=42, verbose=False, **fold)
        print(f"  overall raw={result['overall_acc']:.3f} balanced={result['overall_balanced_acc']:.3f}  " + "  ".join(f"{t}={result['per_market_balanced_acc'][t]:.3f}" for t in ASSETS))
        fold_results.append(result["per_market_balanced_acc"])

    print("\n=== Walk-forward summary (rescue model: 20d horizon + COT, balanced accuracy) ===")
    print(f"{'Market':<8}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for ticker in ASSETS:
        accs = [fr[ticker] for fr in fold_results]
        print(f"{ticker:<8}{accs[0]:<10.3f}{accs[1]:<10.3f}{accs[2]:<10.3f}{np.mean(accs):<10.3f}{np.std(accs):<10.3f}")
