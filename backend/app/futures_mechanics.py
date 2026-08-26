"""Futures-specific trading mechanics (Phase 24). Converts the percentage P&L the
Portfolio has been tracking into real dollar P&L per contract, accounting for slippage
and commission -- price moving by X% doesn't directly tell you dollars made/lost
without knowing the contract's actual specs.

Tick sizes/values sourced from CME/NYMEX/COMEX contract specs (market_driver_map.md,
Phase 1). Maintenance margins are real, current, broker-published figures (AMP
Futures, since we hold positions overnight for the 5-day horizon -- day-trading
margins are a different, lower figure that doesn't apply here) -- not the bare
exchange minimum, which requires a JS-rendered CME page that wasn't scrapeable from
this environment; flagged as an approximation from a real source, not fabricated.
"""

from dataclasses import dataclass

_TICK_VALUE = {"ZN=F": 15.625, "CL=F": 10.00, "GC=F": 10.00}
_TICK_SIZE = {"ZN=F": 1 / 64, "CL=F": 0.01, "GC=F": 0.10}  # in price units (points for ZN, $ for CL/GC)
_DOLLAR_PER_PRICE_UNIT = {"ZN=F": 1000.0, "CL=F": 1000.0, "GC=F": 100.0}  # $ change per 1.0 move in quoted price
_MAINTENANCE_MARGIN = {"ZN=F": 2062.00, "CL=F": 9165.00, "GC=F": 25743.00}  # broker-published, per contract

SLIPPAGE_TICKS = 2  # execution assumed this many ticks worse than the theoretical price, each way
COMMISSION_PER_CONTRACT_PER_SIDE = 2.25  # typical retail/prop futures commission


@dataclass
class ContractSpecs:
    tick_value: float
    tick_size: float
    dollar_per_price_unit: float
    maintenance_margin: float


def get_specs(asset: str) -> ContractSpecs:
    return ContractSpecs(_TICK_VALUE[asset], _TICK_SIZE[asset], _DOLLAR_PER_PRICE_UNIT[asset], _MAINTENANCE_MARGIN[asset])


def apply_slippage(price: float, asset: str, direction: int, entering: bool) -> float:
    """Execution price after slippage -- always assumed to move AGAINST the position:
    entering a long (or exiting a short) fills higher than theoretical; entering a
    short (or exiting a long) fills lower. `entering=True` for the entry fill,
    False for the exit fill."""
    tick = _TICK_SIZE[asset]
    adverse_sign = direction if entering else -direction
    return price + adverse_sign * SLIPPAGE_TICKS * tick


def mark_to_market_pnl(asset: str, direction: int, size: int, entry_price: float, current_price: float) -> float:
    """Unrealized P&L on a STILL-OPEN position, marked at today's price. No slippage
    or commission -- those only apply when a trade actually executes (entry/exit),
    not to a paper mark of a position that hasn't been closed yet."""
    specs = get_specs(asset)
    return direction * (current_price - entry_price) * specs.dollar_per_price_unit * size


def trade_dollar_pnl(asset: str, direction: int, size: int, entry_price: float, exit_price: float) -> dict:
    """Real dollar P&L for one closed trade, after slippage (applied to both fills)
    and commission (charged per contract, per side -- so twice per round-trip trade)."""
    specs = get_specs(asset)
    filled_entry = apply_slippage(entry_price, asset, direction, entering=True)
    filled_exit = apply_slippage(exit_price, asset, direction, entering=False)

    gross_pnl = direction * (filled_exit - filled_entry) * specs.dollar_per_price_unit * size
    commission = COMMISSION_PER_CONTRACT_PER_SIDE * size * 2  # entry + exit
    net_pnl = gross_pnl - commission

    return {
        "gross_pnl": gross_pnl, "commission": commission, "net_pnl": net_pnl,
        "filled_entry": filled_entry, "filled_exit": filled_exit,
        "margin_required": specs.maintenance_margin * size,
    }
