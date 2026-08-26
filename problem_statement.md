# Problem Statement

Phase 2 deliverable. This defines *exactly* what the model predicts, before any data
collection or modeling begins — everything downstream (features, targets, evaluation,
the ablation study) has to trace back to this definition without drift.

## What we are NOT doing

We are not predicting **"will ZN/CL/GC go up or down tomorrow."** That framing — binary,
vague about the horizon, silent about confidence — is what the prior version of this
project used, and it's too weak a target for a system meant to drive real trading
decisions. It also collapses two genuinely different questions (*direction* and
*magnitude*) into one bit, and says nothing about how *uncertain* the prediction is.

## What we ARE doing

**Given all information available at a specific decision time, estimate the probability
and magnitude of a favorable price movement in a given futures contract over a defined
future horizon, along with the expected uncertainty of that movement.**

Formally, at decision time $t$ (end of trading day $t$, using only information with a
timestamp at or before the decision cutoff — see "Information set" below), for each
market $m \in \{\text{ZN}, \text{CL}, \text{GC}\}$, the model estimates:

$$r_{t+1}^{(m)} = \frac{P_{t+1}^{(m)} - P_t^{(m)}}{P_t^{(m)}} \qquad \text{(1-day return)}$$

$$r_{t,t+5}^{(m)} = \frac{P_{t+5}^{(m)} - P_t^{(m)}}{P_t^{(m)}} \qquad \text{(5-day return)}$$

where $P_t^{(m)}$ is the continuous-futures closing price for market $m$ at day $t$
(continuous-contract construction is Phase 6 — until then, treat $P$ as the front-month
close).

## The four quantities the model must output, per market, per day

1. **Expected 1-day return** — $\mathbb{E}[r_{t+1}^{(m)} \mid \mathcal{I}_t]$
2. **Expected 5-day return** — $\mathbb{E}[r_{t,t+5}^{(m)} \mid \mathcal{I}_t]$
3. **Probability of a positive 5-day return** — $P(r_{t,t+5}^{(m)} > 0 \mid \mathcal{I}_t)$
4. **Expected volatility over the horizon** — an estimate of the dispersion/uncertainty
   around quantity (2), not just its central estimate

Example output shape (illustrative, not real numbers yet):

```
ZN
  Expected 1D return:              +0.14%
  Expected 5D return:              +0.61%
  Probability of positive 5D return: 67%
  Expected volatility (5D):         0.48%
```

**Why four numbers and not one classification label**: a model that says "67% probability
of a positive 5-day return, but expected volatility is high" is telling the strategy layer
something a binary up/down label can't — that the trade may be directionally right on
average but too risky to size confidently. The trading-strategy layer (Phase 21, not this
phase) is what turns these four numbers into a BUY/HOLD/SELL decision — this phase's job
stops at producing an honest, well-calibrated estimate of return and risk, not at deciding
what to do about it.

## The information set $\mathcal{I}_t$

$\mathcal{I}_t$ is **everything the model is allowed to see when making its estimate at
decision time $t$** — and by construction, nothing with a timestamp after that cutoff:

$$\mathcal{I}_t = \{\, x : \text{timestamp}(x) \le \text{decision\_cutoff}(t) \,\}$$

covering three sources, each built out fully in later phases:
- **Technical** — price/volume history and derived indicators, through day $t$'s close (Phase 7)
- **Macro** — economic series and event data, using only the vintage/value that was
  actually published by the cutoff, not later-revised figures (Phase 5.2, 5.3, 9)
- **News** — headlines/articles with a **publication timestamp**, not just a publication
  date, at or before the cutoff (Phase 13)

This constraint — $\text{Information timestamp} \le \text{Decision timestamp}$ — is the
project's single most important discipline and is what everything in Phase 27's
look-ahead-bias checklist exists to enforce. It's stated here, at the problem-definition
stage, because every feature and every target built afterward has to satisfy it by
construction, not by a later patch.

The exact decision-cutoff time of day (e.g., "at daily close" vs. some fixed clock time)
is nailed down precisely in **Phase 4** — this phase assumes a daily decision frequency
(one estimate per trading day per market), consistent with the plan's own recommendation
to start there before considering intraday horizons.

## Scope boundary for this phase

This phase defines the **prediction problem** only — not the trading strategy, not
position sizing, not the paper-trading mechanics. Those are Phases 21–25. Keeping this
boundary explicit matters: if the eventual strategy performs badly, this problem
statement is what lets us tell whether the *prediction* was bad or the *strategy built on
top of a reasonable prediction* was bad — conflating the two would make the ablation
study (Phase 30) uninterpretable.

## How this changes the ablation study framing

Section 30 of the plan (the ablation study) compares Models 1–4 by **holding this problem
statement fixed** and varying only the information set $\mathcal{I}_t$ is built from
(technical only → + macro → + news → + co-attention fusion). Every model produces the
same four output quantities, evaluated against the same targets, over the same test
period, under the same trading rules — the problem statement is the one thing that must
**not** change between them.
