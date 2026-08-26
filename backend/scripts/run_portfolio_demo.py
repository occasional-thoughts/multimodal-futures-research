"""Phase 22 demo: walk the Portfolio state machine day-by-day over the real test
period, using the Phase 20 joint model's predictions and Phase 21's calibrated
strategy signals. This is the first point in the plan where "what do we currently
hold" actually matters, not just a one-shot signal per row.

Run from backend/: python scripts/run_portfolio_demo.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np

from app.portfolio import Portfolio
from app.strategy import calibrate_thresholds, compute_score
from scripts.train_model4_joint import ASSET_IDX, ASSETS, train_joint

if __name__ == "__main__":
    result = train_joint()

    val_score = compute_score(result["val_pred"]["return_5d"].detach().numpy(), result["val_pred"]["volatility_5d"].detach().numpy())
    thresholds = calibrate_thresholds(val_score, result["y_val"]["target_return_5d"].numpy())
    print(f"\nCalibrated thresholds (validation only): buy={thresholds.buy_threshold:.3f} sell={thresholds.sell_threshold:.3f}")

    test_score = compute_score(result["test_pred"]["return_5d"].detach().numpy(), result["test_pred"]["volatility_5d"].detach().numpy())
    test_signal = thresholds.signal(test_score)
    aid_test = result["aid_test"].numpy()
    dates_test, prices_test, day_idx_test = result["dates_test"], result["prices_test"], result["day_idx_test"]

    portfolio = Portfolio(assets=ASSETS)

    for ticker in ASSETS:
        mask = aid_test == ASSET_IDX[ticker]
        order = np.argsort(day_idx_test[mask])  # walk this market's own test rows in chronological order
        for day_idx, price, signal, date in zip(
            day_idx_test[mask][order], prices_test[mask][order], test_signal[mask][order], dates_test[mask][order]
        ):
            portfolio.step(int(day_idx), ticker, float(price), signal)

    print(f"\nOpen positions at end of test period: {portfolio.open_positions_summary()}")
    print(f"\nClosed trades: {len(portfolio.closed_trades)}")
    for ticker in ASSETS:
        trades = [t for t in portfolio.closed_trades if t["asset"] == ticker]
        if not trades:
            print(f"  {ticker}: no closed trades")
            continue
        pnls = [t["pnl"] for t in trades]
        print(f"  {ticker}: {len(trades)} trades, avg P&L={np.mean(pnls):.4f}, win rate={np.mean([p > 0 for p in pnls]):.2f}")
