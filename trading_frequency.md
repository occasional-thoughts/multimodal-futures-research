# Trading Frequency & Decision Timing

Phase 4 deliverable. Short by design (this phase is scoped at half a day) — but the
decision-cutoff definition here is load-bearing: [problem_statement.md](problem_statement.md)
already deferred the exact timing to this document, and Phase 27's look-ahead-bias
checklist will audit every feature against it.

## Frequency: daily, one decision per trading day per market

v1 makes exactly one decision per trading day per market (ZN, CL, GC) — not intraday.
Reasoning: ZN/CL/GC trade nearly 24 hours a day via Globex, and intraday modeling would
add a second axis of complexity (execution latency, tick-level noise, microstructure
effects) before the core multimodal question (does macro/news add value over technical
alone) has even been tested once. Intraday is a plausible later extension, not a v1
requirement.

## The decision cutoff: precisely, not loosely

**Decision cutoff for day $t$ = the daily close used in our OHLCV data for day $t$.**

This is a deliberate, stated simplification: our data source (direct Yahoo Finance
fetch, reused from the prior project — see `backend/app/data/prices.py`) provides one
OHLCV bar per trading day. The "close" in that bar is our operational definition of
"end of day $t$" for every part of this project — technical indicators, macro-data
alignment, and news-cutoff filtering (Phase 13) all key off this same timestamp. We are
not attempting to reconstruct the exact official settlement window (which varies
slightly by exchange/product); we're fixing one consistent, data-source-driven
definition and applying it uniformly, which is what actually matters for avoiding
look-ahead bias — internal consistency, not perfect real-world precision.

## Execution: next valid open, not same-day close

A subtle and common backtesting mistake is generating a signal from day $t$'s close and
then assuming you could also have *traded* at that exact same close — in reality there's
no time between "observing the close" and "the close already happened" to act on it.

**v1 convention: decide using $\mathcal{I}_t$ (information through day $t$'s close) →
execute the resulting trade at day $t{+}1$'s open.**

```
End of day t (decision cutoff)
      ↓
Collect information available by cutoff (I_t)
      ↓
Generate signal (Phase 16-21)
      ↓
Execute at day t+1's open   <-- one full bar of delay, deliberately
      ↓
Hold for the defined horizon
```

This is the more conservative of the two common conventions and is the one used
throughout the rest of the project.

## Holding period: fixed 5-day horizon (v1 default)

[problem_statement.md](problem_statement.md) defines both a 1-day and a 5-day return
target. For v1, **the 5-day return is the primary trading horizon** — a position opened
at day $t{+}1$'s open is closed at day $t{+}6$'s open (5 trading days later), with no
early stop-loss/take-profit exit yet. Reasoning: daily-bar futures returns are noisy at
a 1-day horizon relative to transaction costs, and a 5-day hold gives the multimodal
signal (especially macro and news, which tend to matter over days not single sessions)
room to actually matter. The 1-day return target is retained as a secondary/diagnostic
output, not the v1 trading horizon.

**Explicitly deferred to later phases**: dynamic exit rules, stop-loss/take-profit
levels, and position-sizing logic all belong to Phase 21 (strategy layer) and Phase 23
(position sizing + risk limits) — this phase only fixes the *default* holding period so
the rest of the pipeline (features, targets, backtest) has one unambiguous horizon to
build against.
