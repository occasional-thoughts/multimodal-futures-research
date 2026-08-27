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
`HISTORY_PERIOD` has since been extended to 10y (~2,511 rows/market), and a real
class-imbalance collapse bug (all markets predicting one direction 85-100% of the
time, undetected by the original validation-accuracy safeguard) was found and fixed
with class-weighted loss + balanced accuracy — full diagnosis in PHASE_TRACKER.md.

**Six independently-built, materially different methods have now been tried on the
core research question** ("does technical+macro+news+COT information predict
short-horizon futures direction"), each checked before being trusted (ablations,
prediction-distribution inspection, leak tracing) rather than reported at face value:

| Model | Approach | Honest result |
|---|---|---|
| 4 | 5-day direction classification | Near-chance (balanced acc. 0.47-0.49) |
| 5 | 20-day classification + COT | Inconclusive — too few independent test windows even at 10y |
| 6 | Sharpe-optimized position sizing, 3 markets | Negative — lost to buy-and-hold and a classical trend rule every fold |
| 7 | Real semantic-embedding + cross-modal fusion ([STONK](https://arxiv.org/abs/2508.13327)) | Apparent gain, **ablation-confirmed to be a data-leak artifact**, not real |
| 8 | Regularized XGBoost, tech+macro+COT, no news | Near-chance, matching Model 4 — rules out "the deep architecture is the problem" |
| 9 | Sharpe-optimized position sizing, **13 markets** | Genuine partial improvement (mean Sharpe -0.96 → -0.47), still short of buy-and-hold |

**This convergence is itself the finding.** Four structurally different model
families, applied to real leak-checked features, land in the same place: no
exploitable directional edge at a 5-day horizon on ZN/CL/GC from technical, macro,
and CFTC positioning data alone. The one lever that measurably moved a result —
pooling a broader futures universe for the Sharpe-regression framing — is consistent
with, not contradicting, everything else found (real diversification benefit, not a
predictive edge). A genuinely interesting side-finding along the way: Model 7's
richer semantic embedding could exploit the known-diluted synthetic-news leak more
effectively than a shallower sentiment score could, even at a horizon previously
judged "safe" — see PHASE_TRACKER.md for the ablation that caught it.

**Model 10 (live): a multi-agent LLM trading council.** Seven local LLM agents —
technical, macro, positioning and news analysts feeding a bull-vs-bear debate and a
risk manager — reasoning over the *same* information set as Models 4-9, so the
comparison is honest. Architecture follows
[TradingAgents](https://arxiv.org/abs/2412.20138); runs entirely on local Ollama
(free, no API key, no data leaves the machine).

Running the off-the-shelf framework first produced an instructive failure: its
technical analyst was handed a raw CSV and emitted *a pandas tutorial*, leaving the
debate with no price anchor — so it argued *"sell at 95, stop at 85"* while crude
traded at **82.28**. The fix was architectural, not a bigger model: do all numerical
work in verified Python and hand the model finished figures in prose. The rebuilt
council's first decision cited every figure correctly (RSI 60.9, MACD +0.799, ATR
range 79.41-85.49, CFTC 25th percentile) and correctly declined to trade on
mid-range positioning.

**This is a forward study, not a backtest** — verified that free news is only ~2 days
deep, which makes historical backtesting of any news-driven agent *vacuous* (the news
agents would reason over an empty set while appearing to work). Decisions are logged
and committed to git daily *before outcomes are known*. The evaluator refuses to
report performance below 20 resolved observations per market, enforced in code.

Full per-model writeups, citations (Lim/Zohren/Roberts 2019, Wood/Zohren/Roberts's
Momentum Transformer, Moskowitz et al. 2012, Baz et al. 2015, STONK, MSGCA,
Shwartz-Ziv & Armon 2021, arXiv:2607.00475), and the walk-forward tables behind every
number above are in PHASE_TRACKER.md. Everything else (the P&L-based ablation,
regime/failure analysis, dashboard, final report) is queued.

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
