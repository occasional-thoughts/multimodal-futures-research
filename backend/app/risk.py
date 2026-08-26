"""Position sizing and risk limits (Phase 23). Deliberately simple, clearly-defined
rules -- not an institutional-grade risk engine, per the plan's own scoping note.

Position size scales with how far past the calibrated threshold a score is (more
confidence -> larger size), capped at MAX_CONTRACTS_PER_ASSET. Two portfolio-level
limits are enforced on top: a margin-based concentration limit (see fix note below)
and MAX_DAILY_LOSS_PCT (a circuit breaker -- no new positions open on a day where
realized P&L already breached this).

Fix (Phase 25): the original concentration limit capped raw CONTRACT COUNT (max 6
total across all assets) -- caught during Phase 25's paper-trading run that this
doesn't actually protect against exceeding available capital, because contracts
aren't interchangeable: one GC contract requires ~12.5x the margin of one ZN contract
(see futures_mechanics.py). 6 contracts concentrated in GC could require far more
margin than 6 contracts of ZN. Replaced with a check against actual DOLLAR MARGIN
required vs. a fraction of account equity, which is the thing that actually protects
solvency.
"""

import numpy as np

from app.futures_mechanics import get_specs

MAX_CONTRACTS_PER_ASSET = 3
MAX_DAILY_LOSS_PCT = -0.03  # circuit breaker: stop opening new positions this day if breached
MAX_MARGIN_UTILIZATION = 0.5  # never commit more than 50% of account equity to margin at once


def size_position(score: float, buy_threshold: float, sell_threshold: float, max_contracts: int = MAX_CONTRACTS_PER_ASSET) -> int:
    """Contracts to trade, scaled by confidence past the relevant threshold. Returns 0
    if score doesn't clear either threshold (i.e. HOLD)."""
    if score > buy_threshold and buy_threshold != 0:
        confidence = (score - buy_threshold) / abs(buy_threshold)
    elif score < sell_threshold and sell_threshold != 0:
        confidence = (sell_threshold - score) / abs(sell_threshold)
    else:
        return 0
    size = 1 + max(confidence, 0)
    return int(np.clip(round(size), 1, max_contracts))


def margin_limit_ok(asset: str, additional_contracts: int, current_margin_used: float, account_equity: float, max_utilization: float = MAX_MARGIN_UTILIZATION) -> bool:
    """Would adding this position push total margin usage past max_utilization of
    current account equity? Checks real dollar margin, not contract count, so it
    actually reflects solvency risk across assets with very different notional sizes."""
    additional_margin = get_specs(asset).maintenance_margin * additional_contracts
    return (current_margin_used + additional_margin) <= max_utilization * account_equity


def daily_loss_breaker_tripped(today_realized_pnl_pct: float, threshold: float = MAX_DAILY_LOSS_PCT) -> bool:
    return today_realized_pnl_pct <= threshold
