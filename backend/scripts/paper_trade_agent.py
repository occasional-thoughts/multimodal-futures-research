"""Forward paper-trading harness for the multi-agent trading council (app/agent/).

WHY FORWARD, NOT BACKTEST -- the constraint that defines this script, established
empirically rather than assumed:

The council reasons over real current news. Free news sources (keyless Yahoo) return
only articles published in roughly the last two days -- verified directly. Any
"backtest" on historical dates would therefore run the news, sentiment and debate
agents over an empty news set while appearing to work, producing a number that
looks like a result but tests only the quantitative path. Feeding them TODAY's
news against a PAST date instead would be catastrophic look-ahead bias. Neither is
acceptable, so the council runs forward: one real trading day at a time, decision
logged the moment it is made, before the outcome exists.

That is slower than a backtest (n grows by one per market per trading day) but every
observation is genuinely out-of-sample -- which is more than most published work in
this area can claim. The 2026 audit of LLM-trading research (arXiv:2605.19337)
screened 77 studies and found only 2 of 19 that met minimum standards had extractable
time-consistent train/test splits, only 1 modelled transaction costs, and none were
fully reproducible.

Markets: CL=F and GC=F. ZN=F is excluded -- Yahoo returns zero news articles for it
(verified), so the news half of the council would have no input and any decision it
produced would be a fabricated third data point.

Everything runs locally through Ollama: free, no API key, no data leaves the machine.

Run from backend/:
    python scripts/paper_trade_agent.py
    python scripts/paper_trade_agent.py --markets CL=F --model qwen3:14b
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")

MARKETS = ["CL=F", "GC=F"]
LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "paper_trades" / "agent_decisions.jsonl"


def price_sanity_check(decision_text: str, actual_price: float | None, tolerance: float = 0.12, facts: dict | None = None) -> dict:
    """Flag decisions whose reasoning cites prices far from the real one.

    Written after the off-the-shelf TradingAgents framework failed exactly this way
    on its first live run, and would have been logged as a clean data point without
    it: its technical analyst broke down (emitting a pandas tutorial instead of
    analysing the CSV it was handed), so the downstream agents had no price anchor
    and invented one -- proposing "selling CL=F at 95 with a stop at 85" while crude
    traded at 82.28. The output was confident, well-structured, cited real news, and
    was numerically fictional.

    Deliberately checks the FARTHEST cited price, not the nearest. A first version
    checked whether *any* cited price was close and it passed the very failure it was
    written to catch -- $85 is only 3.3% from $82.28, so the row looked fine while the
    fictional $95 entry went unnoticed. Coherent same-day levels should all sit near
    the current price, so one wildly-off level is the signal that grounding was lost.

    A SECOND bug, also caught by running it rather than trusting it: the bare-number
    regex swept up every figure in the brief's plausible numeric band, so a perfectly
    grounded crude decision was flagged because it quoted RSI 60.7, 42% volatility and
    the 118.06 dollar index -- none of which are prices. A guard that discards valid
    decisions corrupts the study as surely as one that admits invalid ones, so bare
    numbers now require price-like context and any figure matching a known non-price
    metric from the brief is excluded outright. `facts` (from brief.py) supplies those
    known values.

    A decision citing no price is not penalised (returns None): absence of a price
    claim is not evidence of a wrong one.
    """
    import re

    if actual_price is None or not decision_text:
        return {"price_anchor_ok": None, "cited_prices": []}

    # Values the brief legitimately reports that are NOT prices. Rounded to 1dp so
    # "RSI 60.9" in the brief matches "RSI is 60.9" (or 60.7 after the model's own
    # restatement) in the decision text without demanding exact equality.
    non_prices = set()
    if facts:
        for key, val in facts.items():
            if not isinstance(val, (int, float)) or key in ("close", "actual_price"):
                continue
            for variant in (val, val * 100):
                if abs(variant) >= 1:
                    non_prices.add(round(float(variant), 1))

    def is_known_non_price(v: float) -> bool:
        return any(abs(v - np) <= 0.6 for np in non_prices)

    candidates = []
    # $-prefixed figures are unambiguous price claims.
    for m in re.finditer(r"\$\s?(\d{1,6}(?:,\d{3})*(?:\.\d+)?)", decision_text):
        candidates.append(float(m.group(1).replace(",", "")))

    # Bare numbers only count when they sit in explicit price context.
    lo, hi = actual_price * 0.3, actual_price * 3.0
    price_context = re.compile(
        # KEY_LEVEL first: it is the structured field the risk manager is required to
        # emit, so it is the single most important number to validate. An earlier
        # version omitted it and silently skipped validation on a real GC=F decision
        # whose only price claim WAS its KEY_LEVEL -- the guard reported "no price
        # cited" on a decision that cited exactly one, the one that mattered.
        r"(?:key[_ ]?level|price|level|support|resistance|target|stop|entry|exit|"
        r"breakout|breakdown|range|band|bracket\w*|between|toward\w*|above|below|near|"
        r"trading at|sell(?:ing)? at|buy(?:ing)? at)\D{0,24}?"
        r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{2,6}(?:\.\d+)?)",
        re.IGNORECASE,
    )
    for m in price_context.finditer(decision_text):
        v = float(m.group(1).replace(",", ""))
        if lo <= v <= hi and not is_known_non_price(v):
            candidates.append(v)

    candidates = [c for c in candidates if not is_known_non_price(c)]
    if not candidates:
        return {"price_anchor_ok": None, "cited_prices": []}

    max_dev = max(abs(c - actual_price) / actual_price for c in candidates)
    return {
        "price_anchor_ok": bool(max_dev <= tolerance),
        "max_price_deviation": round(max_dev, 4),
        "cited_prices": sorted(set(candidates))[:12],
        "actual_price": actual_price,
    }


def run_once(markets: list[str], trade_date: str, log_path: Path, model: str, verbose: bool) -> list[dict]:
    from app.agent.brief import build_brief
    from app.agent.council import TradingCouncil, to_dict

    council = TradingCouncil(model=model)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for ticker in markets:
        if verbose:
            print(f"\n=== {ticker} ({trade_date}) ===")
        row = {
            "logged_at_utc": datetime.now(timezone.utc).isoformat(),
            "trade_date": trade_date,
            "ticker": ticker,
            "model": model,
            "engine": "council",
        }
        try:
            brief = build_brief(ticker)
            row["decision_price"] = brief["price"]
            row["n_headlines"] = brief["facts"].get("n_headlines", 0)
            # Snapshot the quantitative facts the decision was made from, so the
            # decision can be audited later against exactly what the agent saw.
            row["facts"] = brief["facts"]

            result = council.run(brief, verbose=verbose)
            row.update({
                "action": result.action,
                "confidence": result.confidence,
                "decision_raw": result.reasoning,
                "council": to_dict(result),
                "status": "ok",
            })
            row.update(price_sanity_check(result.reasoning, brief["price"], facts=brief["facts"]))
        except Exception as e:
            row["status"] = "error"
            row["error"] = f"{type(e).__name__}: {e}"
            if verbose:
                traceback.print_exc()

        with log_path.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        rows.append(row)

        if row["status"] == "ok":
            anchor = row.get("price_anchor_ok")
            flag = "" if anchor is not False else f"  [!] PRICE ANCHOR FAILED cited={row.get('cited_prices')}"
            print(f"-> {ticker}: {row['action']} ({row['confidence']} confidence) @ {row['decision_price']:,.2f}{flag}")
        else:
            print(f"-> {ticker}: ERROR {row.get('error')}")

    return rows


def main():
    p = argparse.ArgumentParser(description="Forward paper-trade with the local multi-agent trading council.")
    p.add_argument("--markets", nargs="*", default=MARKETS)
    p.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                   help="Trade date (defaults to today; do NOT backdate -- see module docstring)")
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--log-path", default=str(LOG_PATH))
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    rows = run_once(args.markets, args.date, Path(args.log_path), args.model, verbose=not args.quiet)
    ok = sum(1 for r in rows if r["status"] == "ok")
    grounded = sum(1 for r in rows if r.get("price_anchor_ok") is not False and r["status"] == "ok")
    print(f"\nLogged {len(rows)} rows ({ok} ok, {grounded} price-grounded) -> {args.log_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
