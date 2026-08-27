"""Point-in-time market brief: assembles every real data source this project has
into one clean, LLM-readable text block.

This exists because of a concrete, diagnosed failure. Running the off-the-shelf
TradingAgents framework (arXiv:2412.20138) with a local 8B model, the technical
analyst was handed a raw price CSV and responded with a pandas tutorial ("to
correctly parse your data ... use pd.read_csv") instead of an analysis. With no
real price anchor reaching the downstream agents, the bull/bear/risk debate then
invented one: it argued "selling CL=F at 95 with a stop at 85" while crude was
actually trading at 82.28, and produced a confident, well-structured, and
entirely ungrounded "Sell".

The root cause was NOT model size -- it was handing a language model a table and
expecting arithmetic. The fix is to do the numerical work in verified Python
first (this project already has a 25-feature indicator engine, real FRED/EIA
macro data, and CFTC positioning, all point-in-time correct) and hand the model
finished numbers in prose. That plays to what an LLM is actually good at
(weighing conflicting evidence, reasoning about causes) and removes what it is
bad at (parsing CSVs, computing indicators, remembering the current price).

Every number here comes from the same modules the ML models used, so the agent
and the models are reasoning over an identical information set -- which makes the
comparison between them meaningful rather than apples-to-oranges.

Point-in-time discipline is inherited, not re-implemented: technical.py uses only
trailing windows, macro.py/fundamentals.py/cot.py each model their real
publication lag. Nothing here reads a value that wouldn't have been knowable at
the decision cutoff.
"""

from __future__ import annotations

import pandas as pd
import requests

from app.data.cot import COT_COLUMNS, fetch_cot_features
from app.data.fundamentals import CL_COLUMNS, GC_COLUMNS, fetch_cl_fundamentals, fetch_gc_fundamentals
from app.data.macro import ZN_MACRO_COLUMNS, fetch_zn_macro_features
from app.data.prices import fetch_price_history
from app.features.technical import compute_indicators

_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

MARKET_NAMES = {
    "CL=F": "WTI Crude Oil futures",
    "GC=F": "Gold futures",
    "ZN=F": "10-Year US Treasury Note futures",
}

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS),
}


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def fetch_headlines(ticker: str, limit: int = 12) -> list[dict]:
    """Real current headlines. Returns [] on failure rather than raising -- a news
    outage should degrade the brief, not kill the run (and the brief says so
    explicitly, so the agent knows it is reasoning without news rather than
    silently assuming there was none)."""
    try:
        resp = requests.get(_SEARCH_URL, params={"q": ticker, "newsCount": limit}, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        out = []
        for it in resp.json().get("news", []):
            content = it.get("content", it)
            title = content.get("title") or it.get("title")
            if not title:
                continue
            pub = it.get("providerPublishTime")
            out.append({
                "title": title,
                "publisher": (content.get("provider") or {}).get("displayName", "unknown"),
                "published": str(pd.to_datetime(pub, unit="s").date()) if pub else "",
            })
        return out
    except Exception:
        return []


def build_brief(ticker: str, period: str = "3y") -> dict:
    # 3y, not 1y: the Baz et al. trend scores normalize by a trailing 252-day std,
    # so a 1-year fetch leaves them NaN on the most recent row (caught by running
    # this and seeing "+nan" in the brief, not by assuming it worked).
    """Returns {'text': <LLM-readable brief>, 'price': float, 'facts': {...}}.

    `facts` carries the same numbers in machine-readable form so downstream code
    can verify the agent's claims against reality (see paper_trade_agent.py's
    price-anchor guard) instead of taking the model's word for them.
    """
    name = MARKET_NAMES.get(ticker, ticker)
    price_df = fetch_price_history(ticker, period=period)
    tech = compute_indicators(price_df)
    latest_date = price_df.index[-1]
    close = float(price_df["close"].iloc[-1])
    t = tech.iloc[-1]

    def ret_over(n: int) -> float | None:
        if len(price_df) > n:
            return float(price_df["close"].iloc[-1] / price_df["close"].iloc[-1 - n] - 1)
        return None

    facts = {
        "ticker": ticker,
        "as_of": str(latest_date.date()),
        "close": close,
        "return_1d": ret_over(1),
        "return_5d": ret_over(5),
        "return_20d": ret_over(20),
        "rsi_14": float(t["rsi_14"]),
        "macd": float(t["macd"]),
        "realized_vol_20_annualized": float(t["realized_vol_20"]) * (252 ** 0.5),
        "atr_14_pct_of_price": float(t["atr_14"]),
        "dist_from_20d_high": float(t["dist_from_high_20"]),
        "dist_from_20d_low": float(t["dist_from_low_20"]),
        "sma_10_vs_price": float(t["sma_10"]),
        "sma_50_vs_price": float(t["sma_50"]),
        "trend_score_short": float(t["macd_trend_8_24"]),
        "trend_score_med": float(t["macd_trend_16_48"]),
        "trend_score_long": float(t["macd_trend_32_96"]),
    }

    lines = [
        f"# MARKET BRIEF — {name} ({ticker})",
        f"Data as of close {facts['as_of']}. All figures computed from real market data; "
        f"no figure below is estimated or recalled from memory.",
        "",
        "## PRICE (this is the ONLY correct current price — use it for every level you quote)",
        f"- Last close: **{close:,.2f}**",
        f"- Return 1d: {_pct(facts['return_1d']) if facts['return_1d'] is not None else 'n/a'}"
        f" | 5d: {_pct(facts['return_5d']) if facts['return_5d'] is not None else 'n/a'}"
        f" | 20d: {_pct(facts['return_20d']) if facts['return_20d'] is not None else 'n/a'}",
        "",
        "## TREND & MOMENTUM",
        f"- RSI(14): {facts['rsi_14']:.1f}  (>70 overbought, <30 oversold)",
        f"- MACD: {facts['macd']:+.3f}",
        f"- Price vs 10-day avg: {_pct(-facts['sma_10_vs_price'])} | vs 50-day avg: {_pct(-facts['sma_50_vs_price'])}",
        f"- Normalized trend scores (range about -1 to +1; + = uptrend): "
        f"short {facts['trend_score_short']:+.2f}, medium {facts['trend_score_med']:+.2f}, long {facts['trend_score_long']:+.2f}",
        "",
        "## VOLATILITY & RANGE",
        f"- Annualized realized vol (20d): {facts['realized_vol_20_annualized'] * 100:.1f}%",
        f"- ATR(14) as % of price: {facts['atr_14_pct_of_price'] * 100:.2f}%",
        f"- Distance from 20-day high: {_pct(facts['dist_from_20d_high'])} | from 20-day low: {_pct(facts['dist_from_20d_low'])}",
        f"- A 1-ATR move from here is roughly {close * facts['atr_14_pct_of_price']:,.2f} "
        f"(i.e. {close * (1 - facts['atr_14_pct_of_price']):,.2f} to {close * (1 + facts['atr_14_pct_of_price']):,.2f}).",
    ]

    # --- Macro / fundamentals (real FRED / EIA / World Bank series, per market) ---
    try:
        fetcher, cols = _MACRO_FETCHERS[ticker]
        macro = fetcher(price_df.index)
        mrow = macro.iloc[-1]
        lines += ["", "## MACRO / FUNDAMENTALS (real data, publication-lag adjusted)"]
        for c in cols:
            v = mrow.get(c)
            if pd.notna(v):
                lines.append(f"- {c}: {float(v):,.4f}")
                facts[c] = float(v)
    except Exception as e:
        lines += ["", f"## MACRO / FUNDAMENTALS\n- unavailable this run ({type(e).__name__})"]

    # --- CFTC positioning ---
    try:
        cot = fetch_cot_features(ticker, price_df.index)
        crow = cot.iloc[-1]
        lines += ["", "## CFTC POSITIONING (Commitment of Traders, 3-day release lag applied)"]
        for c in COT_COLUMNS:
            v = crow.get(c)
            if pd.notna(v):
                facts[c] = float(v)
        if "cot_speculator_net_pct_oi" in facts:
            lines.append(f"- Speculators net long: {facts['cot_speculator_net_pct_oi'] * 100:+.1f}% of open interest")
        if "cot_commercial_net_pct_oi" in facts:
            lines.append(f"- Commercial hedgers net: {facts['cot_commercial_net_pct_oi'] * 100:+.1f}% of open interest")
        if "cot_speculator_percentile_3y" in facts:
            lines.append(
                f"- Speculator positioning is at the {facts['cot_speculator_percentile_3y']:.0f}th percentile of its 3-year range "
                "(above 90 = crowded long, historically prone to sharp unwinds; below 10 = crowded short)"
            )
        if "cot_commercial_percentile_3y" in facts:
            lines.append(f"- Commercial hedger positioning: percentile {facts['cot_commercial_percentile_3y']:.0f} of its 3-year range")
    except Exception as e:
        lines += ["", f"## CFTC POSITIONING\n- unavailable this run ({type(e).__name__})"]

    # --- Real news ---
    headlines = fetch_headlines(ticker)
    lines += ["", "## RECENT HEADLINES (real, current)"]
    if headlines:
        for h in headlines:
            lines.append(f"- [{h['published']}] {h['title']} ({h['publisher']})")
    else:
        lines.append("- No headlines retrieved this run. Reason about the quantitative evidence only; "
                     "do NOT invent news to justify a view.")
    facts["n_headlines"] = len(headlines)

    return {"text": "\n".join(lines), "price": close, "facts": facts, "headlines": headlines}


if __name__ == "__main__":
    import sys

    b = build_brief(sys.argv[1] if len(sys.argv) > 1 else "CL=F")
    print(b["text"])
