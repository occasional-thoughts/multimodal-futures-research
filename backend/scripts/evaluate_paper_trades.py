"""Evaluate accumulated forward paper-trading decisions from paper_trade_agent.py.

Joins each logged decision to the price that actually materialized afterward, and
compares the agent against the same honest baseline used everywhere else in this
project (buy-and-hold), computed from the identical logged dates so neither side
gets a different sample.

Deliberately refuses to report a Sharpe ratio or any headline performance number
until there are enough independent observations for it to mean anything. This is
the direct lesson of this project's own history: Phase 26 found a single static
split had manufactured an apparent edge that vanished under walk-forward, the
20-day rescue experiment turned out to have ~6 truly independent observations
hiding behind 167 overlapping rows, and Model 7's strong-looking result was traced
to a data leak. A 5-day forward log would be the same mistake in a new costume, so
the threshold is enforced in code rather than left to judgment in the moment.

Run from backend/:  python scripts/evaluate_paper_trades.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "paper_trades" / "agent_decisions.jsonl"

# Below this many independent observations per market, print the log but refuse to
# report performance statistics. 20 is still small -- it is a floor for "worth
# looking at", NOT a threshold for "statistically established". See module docstring.
MIN_OBS_FOR_STATS = 20

_ACTIONS = {"BUY": 1.0, "LONG": 1.0, "SELL": -1.0, "SHORT": -1.0, "HOLD": 0.0}


def parse_action(row) -> float | None:
    """Map a logged decision to a position in {-1, 0, +1}.

    Prefers the explicit `action` field the council emits (parsed at decision time
    from a structured ACTION: line) over re-parsing free text after the fact --
    text-scraping a verdict out of multi-paragraph reasoning that mentions every
    action word is exactly how a subtly wrong label gets into an evaluation.
    Falls back to text parsing only for rows predating that field.

    Returns None when no action can be determined, rather than defaulting to HOLD:
    silently converting an unparseable decision into a real 'flat' position would
    inject fabricated observations into the results.
    """
    action = row.get("action") if hasattr(row, "get") else None
    if isinstance(action, str) and action.upper() in _ACTIONS:
        return _ACTIONS[action.upper()]

    decision_raw = row.get("decision_raw") if hasattr(row, "get") else None
    if not decision_raw:
        return None
    text = str(decision_raw).upper()
    # Check the end first: these agents conclude with their verdict, and the
    # reasoning above it often mentions every action word along the way.
    for chunk in (text[-200:], text):
        found = [(chunk.rfind(k), v) for k, v in _ACTIONS.items() if k in chunk]
        if found:
            return max(found)[1]
    return None


def load_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"No log yet at {path}. Run paper_trade_agent.py first (it appends one row per market per run).")
        return pd.DataFrame()
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a partially-written line from an interrupted run; skip, don't crash
    return pd.DataFrame(rows)


def realized_forward_returns(ticker: str, dates: list[str], horizon_days: int) -> dict:
    """Forward return from each decision date's close to `horizon_days` trading days
    later. Dates still inside the horizon return NaN -- they haven't resolved yet and
    must not be counted."""
    try:
        hist = yf.Ticker(ticker).history(period="2y")
    except Exception as e:
        print(f"  ! could not fetch history for {ticker}: {e}")
        return {}
    if not len(hist):
        return {}
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()
    closes = hist["Close"]

    out = {}
    for d in dates:
        ts = pd.Timestamp(d).normalize()
        idx = closes.index.searchsorted(ts)
        if idx >= len(closes):
            continue
        fwd = idx + horizon_days
        out[d] = float(closes.iloc[fwd] / closes.iloc[idx] - 1) if fwd < len(closes) else np.nan
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log-path", default=str(LOG_PATH))
    p.add_argument("--horizon", type=int, default=5, help="Trading days to hold each decision")
    args = p.parse_args()

    df = load_log(Path(args.log_path))
    if df.empty:
        return 1

    ok = df[df["status"] == "ok"].copy()
    print(f"Log: {len(df)} rows total, {len(ok)} successful agent runs, {len(df) - len(ok)} errors")

    # Rows from the abandoned off-the-shelf TradingAgents engine are excluded rather
    # than deleted: that engine's technical analyst failed and its decisions cited
    # hallucinated price levels (see paper_trade_agent.py:price_sanity_check), so its
    # decisions are not comparable to the council's. Keeping them in the log
    # preserves the record; mixing them into results would corrupt it.
    if "engine" in ok.columns:
        legacy = ok["engine"].isna().sum()
        if legacy:
            print(f"  ~ {legacy} row(s) from the superseded TradingAgents engine excluded (not comparable; retained in log)")
        ok = ok[ok["engine"] == "council"]
    elif len(ok):
        print("  ~ all rows predate the council engine; nothing comparable to evaluate")
        ok = ok.iloc[0:0]

    if ok.empty:
        print("No successful council runs to evaluate yet.")
        return 1

    # Drop decisions whose reasoning cited prices far from the real one -- see
    # paper_trade_agent.py:price_sanity_check for the real first-run failure this
    # guards against (agent argued "sell at 95, stop 85" while crude was at 82.28).
    # Rows predating the check have no column/value; those are kept (None), since
    # absence of a check is not evidence of failure -- but the count is reported.
    if "price_anchor_ok" in ok.columns:
        bad = (ok["price_anchor_ok"] == False).sum()  # noqa: E712 -- None must not match
        if bad:
            print(f"  ! {bad} decision(s) excluded: cited prices inconsistent with the real decision-time price")
        ok = ok[ok["price_anchor_ok"] != False]  # noqa: E712
        unchecked = ok["price_anchor_ok"].isna().sum()
        if unchecked:
            print(f"  ~ {unchecked} decision(s) had no price-anchor check (logged before the check existed, or cited no price)")

    if ok.empty:
        print("\nNo decisions remain after quality filtering.")
        return 1

    ok["position"] = ok.apply(parse_action, axis=1)
    unparsed = ok["position"].isna().sum()
    if unparsed:
        print(f"  ! {unparsed} decisions could not be parsed into an action (excluded, not defaulted to HOLD)")
    ok = ok.dropna(subset=["position"])

    print(f"\n{'Market':<8}{'Decisions':<12}{'Resolved':<12}{'Action mix'}")
    per_market = {}
    for ticker, grp in ok.groupby("ticker"):
        dates = sorted(grp["trade_date"].unique())
        fwd = realized_forward_returns(ticker, dates, args.horizon)
        grp = grp.assign(fwd_return=grp["trade_date"].map(fwd))
        resolved = grp.dropna(subset=["fwd_return"])
        mix = dict(grp["position"].value_counts())
        print(f"{ticker:<8}{len(grp):<12}{len(resolved):<12}{mix}")
        per_market[ticker] = resolved

    total_resolved = min((len(v) for v in per_market.values()), default=0)
    if total_resolved < MIN_OBS_FOR_STATS:
        print(
            f"\nNot reporting performance statistics yet: the thinnest market has {total_resolved} "
            f"resolved observation(s), below the {MIN_OBS_FOR_STATS} floor.\n"
            "This is deliberate (see module docstring) -- this project has three separate documented\n"
            "cases where a small sample produced a confident-looking number that later evaporated.\n"
            "Keep running paper_trade_agent.py daily; re-run this script as the log grows."
        )
        return 0

    print(f"\n{'Market':<8}{'Agent mean':<14}{'Buy&hold mean':<16}{'Agent hit-rate':<16}{'N'}")
    for ticker, resolved in per_market.items():
        agent = (resolved["position"] * resolved["fwd_return"]).to_numpy()
        bh = resolved["fwd_return"].to_numpy()
        traded = agent[resolved["position"].to_numpy() != 0]
        hit = float((traded > 0).mean()) if len(traded) else float("nan")
        print(f"{ticker:<8}{np.mean(agent):<14.5f}{np.mean(bh):<16.5f}{hit:<16.3f}{len(resolved)}")
    print(
        "\nNote: means are per-decision forward returns, not annualized, and exclude transaction\n"
        "costs. With N this small these are descriptive only -- not evidence of an edge."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
