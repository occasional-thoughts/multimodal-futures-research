"""Phase 26: walk-forward backtesting. Retrains and re-evaluates the joint model
across 3 expanding-window folds instead of trusting the single static split every
prior phase has used -- directly tests whether ZN's apparent edge (and CL's apparent
weakness) holds up across different time periods, or was a property of one particular
test window. "The model can only learn from the past" -- each fold's train window
only extends forward, never backward, and never touches that fold's own test period.

Run from backend/: python scripts/run_walk_forward.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from scripts.train_model4_joint import ASSETS, train_joint

# 3 expanding-window folds: train grows each time, test window slides forward.
# Kept to 3 (not more) given ~2 years of data per market -- more folds would leave
# each test window too thin to mean anything.
FOLDS = [
    {"train_frac": 0.5, "val_frac": 0.10, "test_end_frac": 0.65},
    {"train_frac": 0.65, "val_frac": 0.10, "test_end_frac": 0.80},
    {"train_frac": 0.80, "val_frac": 0.10, "test_end_frac": 0.95},
]

if __name__ == "__main__":
    fold_results = []
    for i, fold in enumerate(FOLDS):
        print(f"\n=== Fold {i + 1}/{len(FOLDS)}: train=[0:{fold['train_frac']}] val=[...{fold['train_frac'] + fold['val_frac']:.2f}] test=[...{fold['test_end_frac']}] ===")
        result = train_joint(seed=42, verbose=False, **fold)
        # Balanced accuracy is the headline number now, not raw accuracy -- raw
        # accuracy is exactly what let the class-imbalance collapse hide behind a
        # falsely-impressive number in the original 10-year re-run (see
        # app/models/training_utils.py's module docstring). Raw is still printed
        # alongside it for reference, never as the number that gets trusted alone.
        print(f"  overall raw={result['overall_acc']:.3f} balanced={result['overall_balanced_acc']:.3f}  " + "  ".join(f"{t}={result['per_market_balanced_acc'][t]:.3f}" for t in ASSETS))
        fold_results.append(result["per_market_balanced_acc"])

    print("\n=== Walk-forward summary across all folds (balanced accuracy) ===")
    print(f"{'Market':<8}{'Fold 1':<10}{'Fold 2':<10}{'Fold 3':<10}{'Mean':<10}{'Std':<10}")
    for ticker in ASSETS:
        accs = [fr[ticker] for fr in fold_results]
        print(f"{ticker:<8}{accs[0]:<10.3f}{accs[1]:<10.3f}{accs[2]:<10.3f}{np.mean(accs):<10.3f}{np.std(accs):<10.3f}")
