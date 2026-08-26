"""Phase 24 demo: convert Phase 23's percentage-based trade P&L into real dollar P&L,
accounting for contract specs, slippage, and commission -- and check margin usage
against the portfolio's actual position sizes. Reuses the exact same simulation as
Phase 23 (no changes to trading rules, per the plan's ablation-fairness requirement)
and only adds the dollar-conversion layer on top.

Run from backend/: python scripts/run_futures_mechanics_demo.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from app.futures_mechanics import get_specs, trade_dollar_pnl
from app.portfolio import Portfolio
from app.risk import MAX_CONTRACTS_PER_ASSET, size_position
from app.strategy import calibrate_thresholds, compute_score
from scripts.train_model4_joint import ASSET_IDX, ASSETS, train_joint

if __name__ == "__main__":
    result = train_joint()

    val_score = compute_score(result["val_pred"]["return_5d"].detach().numpy(), result["val_pred"]["volatility_5d"].detach().numpy())
    thresholds = calibrate_thresholds(val_score, result["y_val"]["target_return_5d"].numpy())

    test_score = compute_score(result["test_pred"]["return_5d"].detach().numpy(), result["test_pred"]["volatility_5d"].detach().numpy())
    test_signal = thresholds.signal(test_score)
    aid_test = result["aid_test"].numpy()
    dates_test, prices_test, day_idx_test = result["dates_test"], result["prices_test"], result["day_idx_test"]

    rows = pd.DataFrame(
        {"date": dates_test, "asset": [ASSETS[a] for a in aid_test], "price": prices_test,
         "signal": test_signal, "score": test_score, "day_idx": day_idx_test}
    ).sort_values("date")

    portfolio = Portfolio(assets=ASSETS)
    for date, day_rows in rows.groupby("date"):
        portfolio.start_day(date)
        for _, row in day_rows.iterrows():
            size = size_position(row["score"], thresholds.buy_threshold, thresholds.sell_threshold, MAX_CONTRACTS_PER_ASSET)
            portfolio.step(int(row["day_idx"]), row["asset"], row["price"], row["signal"], size)

    print(f"\n{'Asset':<8}{'Trades':<8}{'Gross P&L':<14}{'Commission':<14}{'Net P&L':<14}{'Peak margin used':<18}")
    grand_gross, grand_commission, grand_net = 0.0, 0.0, 0.0
    for ticker in ASSETS:
        trades = [t for t in portfolio.closed_trades if t["asset"] == ticker]
        if not trades:
            print(f"{ticker:<8}{'0':<8}")
            continue
        gross = commission = net = 0.0
        peak_margin = 0.0
        for t in trades:
            r = trade_dollar_pnl(ticker, t["direction"], t["size"], t["entry_price"], t["exit_price"])
            gross += r["gross_pnl"]
            commission += r["commission"]
            net += r["net_pnl"]
            peak_margin = max(peak_margin, r["margin_required"])
        grand_gross, grand_commission, grand_net = grand_gross + gross, grand_commission + commission, grand_net + net
        print(f"{ticker:<8}{len(trades):<8}${gross:<13,.2f}${commission:<13,.2f}${net:<13,.2f}${peak_margin:<17,.2f}")

    print(f"\n{'TOTAL':<8}{len(portfolio.closed_trades):<8}${grand_gross:<13,.2f}${grand_commission:<13,.2f}${grand_net:<13,.2f}")
    if grand_gross:
        print(f"\nCommission consumed {grand_commission / abs(grand_gross) * 100:.1f}% of gross P&L magnitude across all trades")
    total_margin_if_all_max_size = sum(get_specs(a).maintenance_margin * MAX_CONTRACTS_PER_ASSET for a in ASSETS)
    print(f"Worst-case simultaneous margin requirement (all 3 assets at max size at once): ${total_margin_if_all_max_size:,.2f}")
