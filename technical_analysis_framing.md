# What Technical Analysis Is Actually Trying to Capture

Phase 8 deliverable. The point of this phase isn't new code — it's language. Every
indicator in `technical.py` can be described two ways: as a rigid rule ("RSI below 30
means buy") or as a hypothesis about market behavior that gets *tested*, not assumed.
The second is what an interviewer at a systematic trading firm wants to hear. This
document reframes all 22 features from Phase 7 that way, so the reframing is ready
before it's needed out loud.

**The pattern, stated once**: never say "X means Y." Say "X measures Z, which I
hypothesize is informative for direction/risk — and I test whether that holds in this
market/regime rather than assuming it."

## Trend

- **`sma_10`, `sma_50`** (price relative to its 10/50-day average) — Not: "price above
  the average is bullish." Instead: these measure how far current price has drifted from
  its recent and longer-run mean — a hypothesis that persistent drift (trend) or
  reversion toward the mean may both be present, and which one dominates is itself
  something to test per market and regime, not assumed.
- **`ema_12`, `ema_26`** — Same idea as the SMAs, but weighted toward recent prices —
  measures a *faster-reacting* notion of trend, useful for comparing how much a signal
  changes when recency is weighted more heavily.
- **`ma_slope_10`** — Not "the trend is up." Measures whether the trend *itself* is
  accelerating or decelerating — a second-derivative view, distinct from just knowing
  price is above or below an average.

## Momentum

- **`return_1d`** — The most basic momentum signal: did price just move, and which way.
  Included raw (not just via derived indicators) so the model can learn its own
  transformation rather than only seeing pre-processed versions of the same information.
- **`rsi_14`** — Not "below 30 means buy." Measures recent momentum and potential
  overextension (how lopsided recent gains vs. losses have been). Whether an overextended
  reading predicts reversal, continuation, or nothing at all is a market- and
  regime-dependent question — this is the plan's own worked example, and it generalizes
  to every other indicator here.
- **`macd`, `macd_signal`** — Measures whether short-term momentum is accelerating
  relative to longer-term momentum, and a smoothed version of that same signal used to
  catch when it's turning. Not a crossover "buy/sell" rule — a measure of momentum
  divergence to test for predictive value.
- **`roc_10`** — A simpler, unsmoothed momentum measure (10-day % change) — included
  alongside MACD deliberately, since a smoothed and unsmoothed momentum measure can
  disagree, and that disagreement is itself potentially informative.

## Volatility

- **`realized_vol_20`** — How much the price has actually been moving lately. Not a
  directional signal at all — a *risk* measure, feeding directly into the expected-
  volatility output from `problem_statement.md`.
- **`atr_14`** — Similar to realized volatility but accounts for gaps (the true range
  includes the distance from yesterday's close, not just today's high-low spread) —
  captures a slightly different notion of "how much is this market actually moving"
  than close-to-close volatility does.
- **`bb_width`** — How wide the statistically "normal" price band currently is —
  another volatility proxy, useful because it can diverge from realized volatility
  during regime transitions (band width reacts to recent extremes, not just average
  moves).
- **`vol_change_10`** — Not the level of volatility, but whether volatility itself is
  rising or falling — a regime-change signal (this connects directly to Phase 33's
  calm/volatile regime analysis).

## Volume

- **`volume_change_1d`, `relative_volume_20`** — Not "high volume confirms the move."
  Measures whether trading activity is unusually high or low relative to recent norms —
  a hypothesis that unusual activity may indicate unusual information reaching the
  market (news, positioning shifts, forced liquidation), which is testable, not assumed.

## Price structure

- **`gap`** — The jump between yesterday's close and today's open — measures
  information that arrived while the market was closed (overnight news, other markets
  moving) rather than information reflected during the trading session itself.
- **`high_low_range`** — How wide today's actual trading range was — a same-day
  volatility/uncertainty measure, distinct from the multi-day volatility measures above.
- **`dist_from_high_20`, `dist_from_low_20`** — Not "near a high means keep buying."
  Measures proximity to recent extremes — a hypothesis that behavior *near* a recent
  extreme (breakout continuation vs. exhaustion/reversal) differs from behavior in the
  middle of a range, which market microstructure research finds evidence for and against
  in different contexts — again, testable, not assumed.
- **`breakout_up`, `breakout_down`** — Explicit binary flags for closing beyond the
  recent 20-day high/low. Included as their own features (not just left implicit in the
  distance measures above) because a model can weight "did a breakout literally happen"
  differently from "how far below a breakout are we" — they carry different information.

## The one-sentence version for an interview

*"I don't treat any of these as trading rules — each one is a hypothesis about what
kind of market behavior might be informative, and the model (and the ablation study in
Phase 30) is what actually tests whether, and under what conditions, that hypothesis
holds."*
