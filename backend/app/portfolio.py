"""Portfolio state (Phase 22), extended with position sizing + risk limits (Phase
23), and a margin-aware concentration check (Phase 25 fix -- see risk.py's docstring
for why raw contract-count concentration wasn't actually solvency-safe). Tracks what's
currently held per asset so the system knows how to respond to a new signal given an
existing position -- directly the question the plan raises: "If it asked me to buy
something yesterday, what happens today?" Answer: a position stays open until it hits
the 5-day holding period fixed in trading_frequency.md, then closes and marks its
P&L; only once flat does a new BUY/SELL signal open a new position. No pyramiding.
"""

from dataclasses import dataclass

from app.futures_mechanics import get_specs
from app.risk import daily_loss_breaker_tripped, margin_limit_ok

HOLDING_PERIOD_DAYS = 5  # matches trading_frequency.md's v1 default trading horizon


@dataclass
class Position:
    asset: str
    direction: int  # +1 long, -1 short
    size: int  # contracts
    entry_day_idx: int
    entry_price: float


class Portfolio:
    def __init__(self, assets: list[str], holding_period: int = HOLDING_PERIOD_DAYS):
        self.positions: dict[str, Position | None] = {a: None for a in assets}
        self.holding_period = holding_period
        self.closed_trades: list[dict] = []
        self._today_realized_pnl_pct = 0.0  # reset via start_day(), used by the daily loss breaker
        self._current_day = None

    def start_day(self, day_idx) -> None:
        """Call once before processing any asset for a new calendar day -- resets the
        daily-loss circuit breaker's running total."""
        if day_idx != self._current_day:
            self._current_day = day_idx
            self._today_realized_pnl_pct = 0.0

    def total_margin_used(self) -> float:
        return sum(get_specs(p.asset).maintenance_margin * p.size for p in self.positions.values() if p is not None)

    def step(self, day_idx: int, asset: str, price: float, signal: str, size: int, account_equity: float) -> None:
        """Called once per asset per trading day, in order (call start_day(day_idx)
        once per day before the per-asset calls, so the loss breaker resets correctly).
        `size` is the confidence-scaled contract count from app.risk.size_position --
        0 means the signal didn't clear either threshold, equivalent to HOLD.
        `account_equity` is needed for the margin-utilization check -- this ties
        position sizing to actual solvency, not just an arbitrary contract cap."""
        pos = self.positions[asset]

        if pos is not None and (day_idx - pos.entry_day_idx) >= self.holding_period:
            pnl_pct = pos.direction * (price - pos.entry_price) / pos.entry_price
            self.closed_trades.append(
                {"asset": asset, "direction": pos.direction, "size": pos.size, "entry_day": pos.entry_day_idx,
                 "exit_day": day_idx, "entry_price": pos.entry_price, "exit_price": price, "pnl_pct": pnl_pct}
            )
            self._today_realized_pnl_pct += pnl_pct * pos.size
            self.positions[asset] = None
            pos = None

        if pos is None and signal in ("BUY", "SELL") and size > 0:
            if daily_loss_breaker_tripped(self._today_realized_pnl_pct):
                return  # circuit breaker: no new risk today, even if a signal fired
            if not margin_limit_ok(asset, size, self.total_margin_used(), account_equity):
                return  # would breach the margin-utilization limit -- real dollar check, not just contract count
            direction = 1 if signal == "BUY" else -1
            self.positions[asset] = Position(asset, direction, size, day_idx, price)
        # if pos is None and signal == "HOLD" (or size == 0): stay flat, nothing to do
        # if pos is not None and still within holding period: position rides regardless
        #   of today's signal -- no early-exit rule in v1

    def open_positions_summary(self) -> dict:
        return {a: (p.direction * p.size if p else 0) for a, p in self.positions.items()}
