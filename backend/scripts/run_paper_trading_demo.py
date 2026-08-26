"""Phase 25 demo: the full paper-trading engine, producing a real equity curve
(starting_cash + realized + unrealized P&L, day by day) over the test period --
not just a list of closed trades. This equity curve is what Phase 26 (backtest) and
Phase 29 (Sharpe/drawdown) consume next.

Run from backend/: python scripts/run_paper_trading_demo.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from app.paper_trading import PaperTradingEngine
from app.strategy import calibrate_thresholds, compute_score
from scripts.train_model4_joint import ASSETS, train_joint

if __name__ == "__main__":
    result = train_joint()

    val_score = compute_score(result["val_pred"]["return_5d"].detach().numpy(), result["val_pred"]["volatility_5d"].detach().numpy())
    thresholds = calibrate_thresholds(val_score, result["y_val"]["target_return_5d"].numpy())

    test_score = compute_score(result["test_pred"]["return_5d"].detach().numpy(), result["test_pred"]["volatility_5d"].detach().numpy())
    test_signal = thresholds.signal(test_score)
    aid_test = result["aid_test"].numpy()

    rows = pd.DataFrame(
        {"date": result["dates_test"], "asset": [ASSETS[a] for a in aid_test], "price": result["prices_test"],
         "signal": test_signal, "score": test_score, "day_idx": result["day_idx_test"]}
    )

    engine = PaperTradingEngine(assets=ASSETS, thresholds=thresholds, starting_cash=100_000.0)
    equity_curve = engine.run(rows)

    print(f"\nStarting cash: ${engine.starting_cash:,.2f}")
    print(f"Final equity:  ${equity_curve['equity'].iloc[-1]:,.2f}")
    print(f"Total return:  {(equity_curve['equity'].iloc[-1] / engine.starting_cash - 1) * 100:.2f}%")
    print(f"Peak equity:   ${equity_curve['equity'].max():,.2f}")
    print(f"Trough equity: ${equity_curve['equity'].min():,.2f}")
    print(f"Total closed trades: {len(engine.portfolio.closed_trades)}")
    print(f"\nEquity curve (first 3 / last 3 days):")
    print(pd.concat([equity_curve.head(3), equity_curve.tail(3)]).to_string(index=False))
