"""Paper-trading engine (Phase 25). Formalizes the plan's own 9-step daily loop on
top of everything built in Phases 20-24 (predictions, strategy signals, risk-sized
positions, portfolio state, real dollar mechanics):

  1. Load available information   -- upstream (features/targets, already point-in-time safe)
  2. Generate predictions          -- upstream (the trained joint model)
  3. Generate signals              -- app.strategy
  4. Check existing positions      -- app.portfolio.Portfolio
  5. Apply risk rules              -- app.risk
  6. Execute simulated trades      -- app.portfolio.Portfolio.step
  7. Update portfolio              -- this class
  8. Mark positions to market      -- this class (the piece that was still missing)
  9. Record P&L                    -- self.equity_curve

This directly builds the "develop and execute new strategies in simulated markets"
capability the internship description names -- and Phase 26 (backtesting) / Phase 29
(Sharpe, drawdown, etc.) both consume this class's `equity_curve`, not just a list of
closed trades, since risk-adjusted metrics need a full account-value time series.
"""

import pandas as pd

from app.futures_mechanics import mark_to_market_pnl, trade_dollar_pnl
from app.portfolio import Portfolio
from app.risk import MAX_CONTRACTS_PER_ASSET, size_position
from app.strategy import StrategyThresholds


class PaperTradingEngine:
    def __init__(self, assets: list[str], thresholds: StrategyThresholds, starting_cash: float = 100_000.0):
        self.portfolio = Portfolio(assets=assets)
        self.thresholds = thresholds
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.equity_curve: list[dict] = []

    def run(self, rows: pd.DataFrame) -> pd.DataFrame:
        """`rows` must have columns: date, asset, price, signal, score, day_idx --
        exactly what Phase 21-24's demos already build. Processes every real calendar
        date in order, one full day (all assets active that day) per iteration."""
        for date, day_rows in rows.sort_values("date").groupby("date"):
            self.portfolio.start_day(date)

            current_equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.starting_cash
            for _, row in day_rows.iterrows():
                size = size_position(row["score"], self.thresholds.buy_threshold, self.thresholds.sell_threshold, MAX_CONTRACTS_PER_ASSET)
                trades_before = len(self.portfolio.closed_trades)
                self.portfolio.step(int(row["day_idx"]), row["asset"], row["price"], row["signal"], size, current_equity)
                if len(self.portfolio.closed_trades) > trades_before:
                    t = self.portfolio.closed_trades[-1]
                    r = trade_dollar_pnl(t["asset"], t["direction"], t["size"], t["entry_price"], t["exit_price"])
                    self.cash += r["net_pnl"]
                    t["net_pnl"] = r["net_pnl"]  # attach real dollar P&L onto the trade record itself

            # Mark every still-open position to today's price for this asset.
            unrealized = 0.0
            for asset, pos in self.portfolio.positions.items():
                if pos is None:
                    continue
                today_price = day_rows.loc[day_rows["asset"] == asset, "price"]
                if not today_price.empty:
                    unrealized += mark_to_market_pnl(asset, pos.direction, pos.size, pos.entry_price, today_price.iloc[0])

            equity = self.cash + unrealized
            self.equity_curve.append({"date": date, "cash": self.cash, "unrealized_pnl": unrealized, "equity": equity})

        return pd.DataFrame(self.equity_curve)
