"""Phase 21 demo: run the model-agnostic strategy layer (app/strategy.py) on top of
the Phase 20 joint model's predictions. Thresholds calibrated on validation data only,
then frozen and applied to test -- the test set never influences threshold choice.

Run from backend/: python scripts/run_strategy_demo.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from app.strategy import calibrate_thresholds, compute_score
from scripts.train_model4_joint import ASSET_IDX, ASSETS, train_joint

if __name__ == "__main__":
    result = train_joint()

    val_return = result["val_pred"]["return_5d"].detach().numpy()
    val_vol = result["val_pred"]["volatility_5d"].detach().numpy()
    val_score = compute_score(val_return, val_vol)
    val_realized = result["y_val"]["target_return_5d"].numpy()

    thresholds = calibrate_thresholds(val_score, val_realized)
    print(f"\nCalibrated on validation only: buy_threshold={thresholds.buy_threshold:.3f}, sell_threshold={thresholds.sell_threshold:.3f}")

    test_return = result["test_pred"]["return_5d"].detach().numpy()
    test_vol = result["test_pred"]["volatility_5d"].detach().numpy()
    test_score = compute_score(test_return, test_vol)
    test_realized = result["y_test"]["target_return_5d"].numpy()
    test_signal = thresholds.signal(test_score)

    print("\nTest-set signal distribution and outcomes (thresholds never saw this data):")
    aid_test = result["aid_test"].numpy()
    for ticker in ASSETS:
        mask = aid_test == ASSET_IDX[ticker]
        sig, realized = test_signal[mask], test_realized[mask]
        counts = {s: int((sig == s).sum()) for s in ["BUY", "HOLD", "SELL"]}
        # Strategy P&L: BUY profits from a positive move, SELL (short) profits from a
        # negative one -- sign-flipped for SELL, unlike the raw price return, so this
        # number is directly "did the position make money," not just "what did price do."
        buy_pnl = realized[sig == "BUY"].mean() if counts["BUY"] else float("nan")
        sell_pnl = (-realized[sig == "SELL"]).mean() if counts["SELL"] else float("nan")
        print(f"  {ticker}: {counts}  avg strategy P&L per position -- BUY={buy_pnl:.4f}  SELL={sell_pnl:.4f}")
