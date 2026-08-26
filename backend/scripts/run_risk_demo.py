"""Phase 23 demo: position sizing + portfolio-level risk limits, walked across all 3
markets in true calendar-date order (not each market's own row index, which isn't
comparable across markets) -- this is what makes the daily loss circuit breaker and
the concentration limit meaningful: they need to see everything happening on the same
real day, not day 100 of ZN's series next to an unrelated day 100 of CL's.

Run from backend/: python scripts/run_risk_demo.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from app.portfolio import Portfolio
from app.risk import MAX_CONTRACTS_PER_ASSET, size_position
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

    # One row per (date, asset, price, signal, score, day_idx), sorted by real date --
    # the actual synchronization key across markets.
    rows = pd.DataFrame(
        {
            "date": dates_test,
            "asset": [ASSETS[a] for a in aid_test],
            "price": prices_test,
            "signal": test_signal,
            "score": test_score,
            "day_idx": day_idx_test,
        }
    ).sort_values("date")

    portfolio = Portfolio(assets=ASSETS)

    for date, day_rows in rows.groupby("date"):
        portfolio.start_day(date)
        for _, row in day_rows.iterrows():
            size = size_position(row["score"], thresholds.buy_threshold, thresholds.sell_threshold, MAX_CONTRACTS_PER_ASSET)
            portfolio.step(int(row["day_idx"]), row["asset"], row["price"], row["signal"], size)

    print(f"\nOpen positions at end of test period (direction x size): {portfolio.open_positions_summary()}")
    print(f"Closed trades: {len(portfolio.closed_trades)}")

    for ticker in ASSETS:
        trades = [t for t in portfolio.closed_trades if t["asset"] == ticker]
        if not trades:
            print(f"  {ticker}: no closed trades")
            continue
        pnls = [t["pnl_pct"] for t in trades]
        sizes = [t["size"] for t in trades]
        weighted_pnl = np.average(pnls, weights=sizes)
        print(f"  {ticker}: {len(trades)} trades, avg size={np.mean(sizes):.1f} contracts, size-weighted avg P&L={weighted_pnl:.4f}, win rate={np.mean([p > 0 for p in pnls]):.2f}")
