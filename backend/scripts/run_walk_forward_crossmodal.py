"""Walk-forward evaluation of Model 7 (real semantic-embedding news representation +
STONK-style cross-modal attention + MSGCA-style gate -- see train_model7_crossmodal.py's
module docstring). Same 3-fold expanding-window methodology as Phase 26's
run_walk_forward.py and everything since -- a literature-grounded architecture change
doesn't get a pass on the scrutiny that caught real problems in every earlier model,
including this project's own single-split result for Model 6 that didn't survive
walk-forward.

Run from backend/: python scripts/run_walk_forward_crossmodal.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from scripts.train_model7_crossmodal import ASSETS, train_crossmodal

FOLDS = [
    {"train_frac": 0.5, "val_frac": 0.10, "test_end_frac": 0.65},
    {"train_frac": 0.65, "val_frac": 0.10, "test_end_frac": 0.80},
    {"train_frac": 0.80, "val_frac": 0.10, "test_end_frac": 0.95},
]

if __name__ == "__main__":
    fold_results = []
    for i, fold in enumerate(FOLDS):
        print(f"\n=== Fold {i + 1}/{len(FOLDS)} ===")
        result = train_crossmodal(seed=42, verbose=False, **fold)
        print(f"  overall raw={result['overall_acc']:.3f} balanced={result['overall_balanced_acc']:.3f}  " + "  ".join(f"{t}={result['per_market_balanced_acc'][t]:.3f}" for t in ASSETS))
        fold_results.append(result["per_market_balanced_acc"])

    print("\n=== Walk-forward summary (Model 7: cross-modal semantic-embedding fusion, balanced accuracy) ===")
    print(f"{'Market':<8}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for ticker in ASSETS:
        accs = [fr[ticker] for fr in fold_results]
        print(f"{ticker:<8}{accs[0]:<10.3f}{accs[1]:<10.3f}{accs[2]:<10.3f}{np.mean(accs):<10.3f}{np.std(accs):<10.3f}")
