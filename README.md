# Multimodal Futures Strategy Research & Paper-Trading Platform

Central research question: **Can a multimodal model combining technical patterns,
macroeconomic fundamentals, and financial news improve futures trading decisions
compared with progressively simpler information sets?**

Universe: **ZN** (10-Year Treasury Note futures), **CL** (Crude Oil futures), **GC**
(Gold futures).

## Start here

- **[PHASE_TRACKER.md](PHASE_TRACKER.md)** — the full 37-phase plan and current progress. This is the source of truth for what's done and what's next.
- **[market_driver_map.md](market_driver_map.md)** — futures mechanics + what actually drives ZN, CL, and GC (Phase 1 deliverable).
- **[problem_statement.md](problem_statement.md)** — the exact prediction problem: 4 output quantities per market per day, the information-set/point-in-time discipline, and the scope boundary vs. the trading-strategy layer (Phase 2 deliverable).
- **[trading_frequency.md](trading_frequency.md)** — daily decisions, the exact decision-cutoff/execution-delay convention, and the default 5-day holding period (Phase 4 deliverable).
- **[data_architecture.md](data_architecture.md)** — schemas and concrete sources for the four data categories (market, macro, economic-event/surprise, news), including real FRED/EIA series IDs and two honestly-flagged sourcing gaps (Phase 5 deliverable).
- **[continuous_contracts.md](continuous_contracts.md)** — the continuous-futures roll/adjustment methodology, and an empirically-verified gap (individual dated contracts aren't available from our free data source) documented rather than hidden (Phase 6 deliverable).

## Current status

Phases 1-26 are done. **Phase 26 (walk-forward backtesting) overturned the "ZN is the
standout market" conclusion from Phases 15-25** — the single static split used
throughout the ablation happened to land on each market's most flattering fold.
`HISTORY_PERIOD` has since been extended to 10y (~2,511 rows/market) to give later
experiments more statistical power.

**A class-imbalance collapse bug was found and fixed on the 10-year re-run**: all
three markets, in both the standard 5-day model and the 20-day+COT "rescue" model
(below), had collapsed to predicting one direction 85-100% of the time — undetected by
the original validation-accuracy safeguard because walk-forward validation and test
periods are often directionally correlated. Fixed with class-weighted loss + balanced
accuracy for both checkpoint selection and reporting (see PHASE_TRACKER.md for the
full diagnosis). **Honest result after the fix: the standard 5-day model shows no
real directional edge on ZN, CL, or GC** — balanced accuracy sits within a few points
of chance (0.45-0.59) across all markets and folds, a confident null result, not a
disappointing one to hide.

The research-motivated "rescue" attempt (20-day horizon per Moskowitz et al.'s
time-series momentum literature + CFTC Commitment of Traders positioning data,
`backend/scripts/train_model5_rescue.py`) remains **inconclusive rather than
negative**: even at 10 years of history, non-overlapping 20-day evaluation windows
leave only 6 independent test observations per market per fold — too few for any
number to mean anything, confirmed and not just assumed (see PHASE_TRACKER.md).
More history or pooled-fold significance testing is the honest next step, not a
verdict on the horizon idea either way.

**Model 6 (new): a Deep Momentum Network**, reimplementing real published research
rather than iterating further on direction classification — [Lim, Zohren & Roberts
(2019)](https://arxiv.org/pdf/1904.04912) and [Wood, Giegerich, Roberts & Zohren
(2021)](https://github.com/kieranjwood/trading-momentum-transformer): output a
continuous position size trained by directly optimizing a differentiable Sharpe
ratio, so >50% directional accuracy is no longer required for a positive result.
Building it caught a real data-leakage bug (the synthetic placeholder news feature
directly encoded the 1-day return this model trades — first run reported an
impossible Sharpe of 11.6; fixed by dropping that feature stream). **Honest
walk-forward result after the fix: the learned model underperforms both buy-and-hold
and a simple hand-built trend rule in every fold** (mean Sharpe −0.96 vs. 0.84 and
0.67) — a clean negative result, reported as one, not reached for a better cut of it.
See PHASE_TRACKER.md for the full writeup.

**Model 7 (new): real semantic-embedding news + cross-modal fusion**, built after
checking this project's own architecture diagram against the code and finding a real
gap — every prior model discarded FinBERT's semantic embedding and used only its
3-class sentiment probabilities. Reimplements
[STONK](https://arxiv.org/abs/2508.13327) (numeric market features as the attention
query, text embeddings as key/value) plus an MSGCA-style gate. **Honest result: the
apparent improvement doesn't survive an ablation.** CL's balanced accuracy jumped
from 0.492 to 0.626 with the richer embedding — but zeroing out the entire news
input collapsed *every* market to exactly ~0.500, proving the gain was the model
exploiting the (already-known, diluted) news-feature leak more effectively, not a
real architecture benefit. Re-checked Model 4's original result the same way — it
held up fine (0.474→0.492 with news removed, no collapse), so this is specific to
Model 7's richer representation, not a retroactive problem for everything else. See
PHASE_TRACKER.md for the full ablation writeup.

**Model 8 (new): regularized XGBoost, tech+macro+COT, zero news exposure** —
deliberately avoids the leak problem entirely rather than patching around it again.
Grounded in real literature (Shwartz-Ziv & Armon 2021: gradient-boosted trees
repeatedly beat deep learning on small tabular data, exactly this project's regime),
plus a new "COT Index" positioning-extremity feature the raw COT data never had.
**Honest result: no meaningful improvement** — ZN 0.494, CL 0.495, GC 0.523, all
within noise of Model 4's 0.474/0.492/0.486. This is now the 4th independently-built,
materially different method (classification, Sharpe-regression, cross-modal fusion,
gradient-boosted trees) that finds no exploitable 5-day-horizon edge on ZN/CL/GC —
a real, convergent research finding, not a failure to find the right trick. See
PHASE_TRACKER.md for the full writeup. Everything else (the P&L-based
ablation, regime/failure analysis, dashboard, report) is queued.

Real, verified data sources now wired in: Yahoo Finance (prices + real current
per-ticker news), FRED (macro, no API key needed via the public CSV endpoint), EIA
(crude inventories/production, no API key needed via the legacy `.xls` endpoint).

**Open gap**: no free, historically-deep macro-news archive confirmed working yet
(GDELT unreachable from this environment; Yahoo's search only works for ticker
queries, not topic keywords) — historical training data still uses a clearly-labeled
placeholder. See Phase 11 in the tracker.

## What's reused from the prior version of this project

This project started as a different, narrower study (explainability-robustness testing
on stock/news fusion). That version is preserved in **[archive/legacy-modality-attribution-project/](archive/legacy-modality-attribution-project/)**,
not deleted — a few pieces of it carry forward:

- `backend/app/data/prices.py` — direct Yahoo Finance fetch (handles `=F` futures
  tickers correctly; `yfinance`'s own session hangs in this environment).
- `backend/app/features/technical.py` — RSI, MACD, moving averages, Bollinger Bands,
  realized volatility — a head start on Phase 7's technical feature engine.
- `backend/app/features/sentiment.py` — FinBERT sentiment scoring, a head start on
  Phase 11's news pipeline.

See the archive's own README for what else in there is worth referencing (notably a
real experiment showing a Temporal Fusion Transformer underperforming XGBoost on this
kind of small-sample financial data — relevant when choosing Phase 16's model
architecture).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` reflects the prior version's dependencies for now — expect this to
grow as later phases add macro data sources, sequence models, and the backtesting/paper-trading engine.
