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
Extending `HISTORY_PERIOD` from 2y to 5y confirmed this was real small-sample
instability. A research-motivated "rescue" attempt (20-day horizon per Moskowitz et
al.'s time-series momentum literature + CFTC Commitment of Traders positioning data,
`backend/scripts/train_model5_rescue.py`) surfaced a second overlapping-window
autocorrelation problem, since fixed by evaluating on non-overlapping windows — which
revealed the true effective sample size at a 20-day horizon is only ~3 independent
observations per market per fold, too few to draw any conclusion yet. More history or
more pooled folds is the honest next step, not a verdict on the horizon idea either
way. See PHASE_TRACKER.md for the full breakdown. Everything else (the P&L-based
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
