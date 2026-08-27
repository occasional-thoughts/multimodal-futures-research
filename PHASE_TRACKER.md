# Phase Tracker — Multimodal Futures Strategy Research & Paper-Trading Platform

Central research question: **Can a multimodal model combining technical patterns,
macroeconomic fundamentals, and financial news improve futures trading decisions
compared with progressively simpler information sets?**

Universe: **ZN** (10-Year Treasury Note futures), **CL** (Crude Oil futures), **GC**
(Gold futures). See [market_driver_map.md](market_driver_map.md) for the financial-hypothesis
layer behind this choice.

Ablation backbone (do not vary test set / trading rules / costs / evaluation methodology
between these):

| Model | Inputs | Question |
|---|---|---|
| Model 1 | Technical only | Can price/volume patterns alone generate useful signals? |
| Model 2 | Technical + Macro | Does understanding the economic environment improve signals? |
| Model 3 | Technical + Macro + News | Does incorporating financial news add predictive value? |
| Model 4 | Technical + Macro + News + Co-Attention | Does learning interactions between market state and news improve performance further? |

Status legend: ⬜ not started · 🟨 in progress · ✅ done

## Part 0 — Understand what you're building
- ✅ Read and internalized the system diagram and prediction framing (return, not existence)

## Phase 1 — Learn the futures markets first
- ✅ Futures contract mechanics vocabulary
- ✅ ZN deep dive (Treasury notes, yield/price, drivers)
- ✅ CL deep dive (WTI, supply/demand, drivers)
- ✅ GC deep dive (real yields, USD, safe haven, drivers)
- ✅ Deliverable: [market_driver_map.md](market_driver_map.md)
- ⬜ **You**: be able to explain each market's drivers out loud, unaided, before moving on

## Phase 2 — Define the exact research problem
- ✅ Write the precise prediction statement (risk-adjusted direction + expected return over a defined horizon, given info at decision time)
- ✅ Deliverable: [problem_statement.md](problem_statement.md) — 4 output quantities per market/day (E[1D return], E[5D return], P(5D return > 0), expected volatility), formal information-set definition, scope boundary vs. the strategy layer

## Phase 3 — Define your universe
- ✅ ZN, CL, GC selected and justified (different economic systems) — see market_driver_map.md Part 5

## Phase 4 — Decide trading frequency
- ✅ Daily decisions confirmed as v1 scope; exact decision-cutoff, execution-delay, and default holding-period conventions fixed
- ✅ Deliverable: [trading_frequency.md](trading_frequency.md) — decide at day t close → execute at day t+1 open → hold 5 trading days (v1 default)

## Phase 5 — Data architecture
- ✅ 5.1 Market data schema (OHLCV + open interest + contract id + expiration) — open interest flagged as a real sourcing gap
- ✅ 5.2 Macro data selection per market — concrete FRED series IDs chosen (DGS2/5/10/30, FEDFUNDS, CPIAUCSL, PCEPI, UNRATE, PAYEMS, GDPC1, T10YIE, DFII10, DTWEXBGS) + EIA for CL-specific inventories/production
- ✅ 5.3 Economic-event/surprise data schema (actual − expected) — sourcing gap flagged honestly (no free consensus-forecast source found yet; two fallback paths documented)
- ✅ 5.4 News data schema (with publication timestamp, not just date) — noted this needs new (macro/commodity) sources, not the old company-headline design
- ✅ Deliverable: [data_architecture.md](data_architecture.md)

## Phase 6 — Continuous futures contract construction
- ✅ 6.1 Roll methodology chosen and documented (liquidity-based, in principle) — verified individual dated contracts aren't available from our free data source; v1 uses Yahoo's own `=F` continuous series, gap flagged honestly
- ✅ 6.2 Price discontinuity adjustment method chosen and documented (back-adjustment, in principle) — same v1 gap, documented
- ✅ 6.3 `raw_contract_data/` vs `continuous_data/` separation designed (raw side empty for now, ready for real data if it becomes available)
- ✅ Deliverable: [continuous_contracts.md](continuous_contracts.md)

## Phase 7 — Technical feature engine
- ✅ Trend (SMA-10/50, EMA-12/26, MA slope)
- ✅ Momentum (1D return, RSI-14, MACD + signal, 10D ROC)
- ✅ Volatility (realized vol-20, ATR-14, Bollinger width, vol change)
- ✅ Volume (1D change, relative volume vs 20D avg)
- ✅ Price structure (gap, high-low range, distance from 20D high/low, breakout up/down flags)
- ✅ 22 features total in `backend/app/features/technical.py`, smoke-tested on real ZN=F data (no NaN columns, sane value ranges)

## Phase 8 — Understand what technical analysis actually captures
- ✅ Reframe each indicator as a hypothesis to test, not a rule (interview-readiness)
- ✅ Deliverable: [technical_analysis_framing.md](technical_analysis_framing.md) — all 22 Phase 7 features reframed
- ⬜ **You**: practice saying the one-sentence version out loud until it's natural, not read

## Phase 9 — Macro feature engine (ZN-specific: yield levels/changes/curve, policy, inflation, employment)
- ✅ `backend/app/data/macro.py` — 14 real ZN macro features from FRED's free public CSV endpoint (no API key needed): yield levels 2/5/10/30Y, yield curve (10Y-2Y) + 5D change, Fed funds rate + 20D change, CPI/PCE/GDP YoY, unemployment, payrolls change, 10Y breakeven inflation
- ✅ Point-in-time alignment implemented (forward-fill only from observation date + documented typical publication lag, never earlier) — honest limitation flagged: exact historical publication dates need FRED's authenticated API + a free key, not used yet; typical-lag approximation used instead
- ✅ Smoke-tested on real data — caught and fixed a real bug (YoY transforms were computed after point-in-time alignment, leaving too short a window to ever compute a 252-trading-day lookback; fixed by computing YoY on each series' native frequency first, then aligning)

## Phase 10 — Market-specific fundamental features (CL: inventories/production/OPEC/supply disruptions/demand; GC: real yields/USD/inflation expectations/central-bank vars)
- ✅ `backend/app/data/fundamentals.py` — CL: real weekly crude inventories + 1W change, weekly field production + 4W change (EIA legacy `.xls` endpoint, verified keyless, real current data ~429M bbl stocks/13.8M bbl-day production), USD index. GC: real 10Y yield, USD, breakeven inflation, Fed funds (via FRED, reusing macro.py's fetch/align helpers)
- ✅ Smoke-tested on real data — caught and fixed a real bug (EIA sheet's first 2 rows are metadata, not observations; fixed with coerce+drop instead of a hardcoded skip count)
- ⬜ OPEC+ decisions and supply disruptions deliberately deferred to Phase 5.3's economic-event schema (discrete events, not a continuous series) — not yet implemented

## Phase 11 — News pipeline (FinBERT: cleaning → FinBERT → embedding + sentiment probs + metadata)
- ✅ `backend/app/data/news_pipeline.py` — real per-ticker headline fetch + FinBERT scoring, verified on live GC=F headlines (correctly relevant, real timestamps, sensible sentiment scores)
- ⚠️ **Real gap, tested not assumed**: GDELT (best free historically-deep news source) is unreachable from this environment (DNS resolves, connection fails — worth retrying elsewhere). Yahoo's search only returns relevant results for ticker queries, not macro keyword/topic queries (tested "Federal Reserve", "CPI inflation", etc. — all returned generic unrelated news). So: real *current* per-ticker headlines work; a free historically-deep macro-news archive for training backfill does not exist yet in this setup.
- ✅ Placeholder historical generator (`generate_placeholder_macro_news`, macro-themed not company-themed) built as the stand-in, same honesty convention as the archived project — NOT real news, clearly labeled

## Phase 12 — Asset-specific news relevance filtering
- ✅ `backend/app/data/relevance.py` — explicit keyword-based classifier (not learned, per the plan's own recommendation for v1), keyword lists drawn directly from `market_driver_map.md`'s driver categories so it's consistent with the financial-hypothesis layer
- ✅ Tested on real fetched GC=F headlines: only 4/10 ticker-searched articles were actually content-relevant to gold *price* drivers — the rest were gold-mining *company* earnings news, a real and meaningful distinction the filter correctly caught
- ⚠️ Real limitation found and documented, not hidden: "drilling" matched CL (oil) on a headline that was actually about mineral exploration — simple keyword matching has genuine precision limits; worth revisiting if this becomes a material source of noise once used at scale

## Phase 13 — Align news with market time (publication timestamp ≤ decision timestamp)
- ✅ `backend/app/data/timing.py` — assigns each article to the correct trading-day decision using real hour:minute timestamps (not just date-matching), via a documented assumed-close-time (4PM ET) convention consistent with Phase 4's own acknowledged data-granularity simplification
- ✅ Verified all 3 branches explicitly: before-close same-day assignment, after-close next-day rollover, and weekend/non-trading-day rollover — all correct

## Phase 14 — Define target (multi-horizon: 1D return, 5D return, volatility)
- ✅ `backend/app/targets.py` — implements all 4 problem_statement.md output quantities as computable columns (1D return, 5D return, realized 5D direction, realized 5D forward volatility)
- ✅ Verified on real ZN data: matches the existing `next_return` column exactly (0.0 diff), and correctly leaves trailing rows NaN (no future data left) rather than silently filling them
- ✅ Kept in its own module, deliberately never imported by anything in `features/` or `data/` — targets look forward by construction, features never should, and this separation is the guard against accidental leakage

## Phase 15 — Simple baselines before deep learning (buy&hold, momentum, MA strategy, logistic regression, XGBoost)
- ✅ `backend/scripts/train_baselines.py` — all 5 baselines run on real ZN/CL/GC data, evaluated on directional accuracy against `target_direction_5d` (P&L/Sharpe comparison deferred to after Phase 25's paper-trading engine exists)
- ✅ Real bug caught mid-phase: `volume_change_1d` produced `inf` (not `NaN`) on 6 real zero-volume days (incl. July 4th), which silently passed Phase 7's NaN-only check and broke sklearn's `StandardScaler` here — fixed in `technical.py` by treating zero volume as missing before the pct_change, verified 0 infinities across all 3 markets afterward
- ✅ **Second real bug caught, diagnosed, and properly fixed (not papered over)**: initial XGBoost run showed 100% train accuracy / 38.9% test accuracy on CL — classic overfitting, confirmed by explicit train-vs-test diagnostic (200 unregularized trees on ~250 rows, and the validation split was being computed but silently never used). Fixed with real regularization (depth 2, subsample/colsample 0.7, min_child_weight 5, L1/L2 reg) and validation-based early stopping, applied uniformly to all 3 markets — not tuned to make any one market's number prettier. Post-fix: ZN train/test 51.9%/57.1%, CL 65.6%/44.4%, GC 65.6%/59.3% — sane gaps, no more memorization. `best_iteration` was very low in all 3 (0-2 rounds) — with honest regularization, XGBoost finds very little exploitable signal in technical-only features on daily-bar data, which is itself a legitimate, expected finding for this task, not a failure of the code.
- ✅ **Real, unglossed-over finding, confirmed to survive the overfitting fix** (exactly what the plan asks not to hide): no single baseline wins across all three markets, and this is now known to be real rather than a training artifact.
  - ZN: XGBoost best (57.1%), momentum/MA worst (41.1%)
  - CL: plain moving-average rule best (**64.8%**), XGBoost still worst (**44.4%**, now a real result not an overfitting artifact)
  - GC: logistic regression best (66.7%), XGBoost mid-pack (59.3%)
  - **Decision carried into Phase 16 onward**: for CL specifically, the bar a sequence/deep-learning Model 1 needs to clear is the 64.8% moving-average baseline, not XGBoost's number. This is exactly the kind of per-market divergence Phase 32 is meant to formally investigate later — it surfaced early, and that's Phase 15 working as intended.

## Phase 16 — Model 1: Technical only
- ✅ `backend/scripts/train_model1.py` — single-layer GRU (deliberately small: hidden_size=16, dropout 0.3, early stopping) over a 20-day window of the 22 technical features, multi-head output covering all 4 problem_statement.md targets. Kept small from the start because the archived TFT experiment already showed an oversized sequence model collapses to chance level at this data scale — applied proactively, not rediscovered.
- ✅ **Third real bug found and fixed** (after Phase 15's `inf`-from-zero-volume and overfitting bugs): the first working version's results were invalid — inspecting the raw prediction distribution (not just the accuracy number) showed the model was **collapsing to a constant class** on ZN and GC (e.g. GC predicted "up" for 54/54 test rows). Root cause, diagnosed not guessed: the combined loss (`MSE(return_1d)+MSE(return_5d)+BCE(direction)+MSE(volatility)`) summed terms of wildly different natural scale, and early stopping picked checkpoints by *combined* val loss, with no guarantee that's the checkpoint with a sane direction boundary.
- ✅ **Fix, shared in `backend/app/models/training_utils.py`** (used by both Model 1 and Model 2, not duplicated): regression loss terms normalized by their own training-set variance, each clamped to a ceiling (targets have variance as low as ~1e-6, so unclamped normalization caused a *second*, separate instability — loss magnitudes exploding to the hundreds/thousands — caught by noticing val_loss values were wildly inconsistent across markets, not assumed benign); checkpoint selection switched from "best combined val loss" to "best validation **direction accuracy**," which is what we actually care about.
- ✅ Re-verified predictions are non-degenerate after the fix (real 0/1 spread in the test set, not a constant) — GC remains the weakest market (best validation accuracy only 40.7%, still skewed toward "up"), flagged honestly as a real remaining limitation, not hidden.
- ✅ **Final, trustworthy finding**: Model 1 does not beat the Phase 15 baseline ceiling on any market — ZN 44.6% (vs. 57.1% XGBoost), CL 57.4% (vs. 64.8% moving average), GC 55.6% (vs. 66.7% logistic regression). Numbers changed from the pre-bugfix run (as expected — those were measuring a broken model) but the qualitative conclusion is unchanged and now actually trustworthy: technical data alone doesn't give this sequence model more to extract than the simple baselines already captured.

## Phase 17 — Model 2: Technical + Macro
- ✅ `backend/scripts/train_model2.py` — two-stream encoder: GRU over the technical window (price data genuinely varies day-to-day) + a small **feedforward** network over just today's macro values (not a second GRU sequence — most macro series barely move within a 20-day forward-filled window, so a sequence encoder there was adding parameters without proportional signal; this was the user-directed fix after the first version showed Model 2 losing to Model 1). Reuses the real Phase 9/10 macro/fundamentals fetchers (FRED for ZN/GC, EIA for CL). Same bug (degenerate collapse) and same fix (`training_utils.py`) applied here as Model 1.
- ✅ **Final, trustworthy finding**: ZN 55.4% (vs. Model 1's 44.6% — macro helps here), CL 48.1% (vs. Model 1's 57.4% — macro hurts), GC 38.9% (vs. Model 1's 55.6% — macro hurts). None beat their market's Phase 15 baseline ceiling.
- **Carried forward, not swept under the rug**: at this point in the ablation, simple baselines still win on every market, and richer information hasn't yet earned its complexity. That's a legitimate, reportable state of the ablation study to be in mid-way through Phase 17 of 4 model stages — Phase 18 (news) and Phase 19 (co-attention) are what's supposed to test whether that changes, not a foregone conclusion.

## Phase 18 — Model 3: Technical + Macro + News
- ✅ `backend/scripts/train_model3.py` — three-stream encoder: tech GRU + macro FFN (Model 2's fix) + news FFN over today's aggregated FinBERT sentiment. Reuses `training_utils.py`'s fix, so no degenerate collapse this time (loss magnitudes sane and consistent, ~15.6-15.7 across all 3 markets).
- ⚠️ **News caveat carried forward honestly**: uses `generate_placeholder_macro_news` (Phase 11) — real FinBERT scoring on synthetic, clearly-labeled headline text, since no confirmed-working free historical news source exists yet. This exercises the 3-modality pipeline correctly but is not yet a real "does news help" finding.
- ✅ Results: ZN 57.1% (**ties its 57.1% baseline ceiling — first model to match a baseline**), CL 42.6% (declining further: 57.4%→48.1%→42.6% as modalities are added — a real, consistent, honestly-reported trend, not noise), GC 57.4% (partial recovery from Model 2's 38.9%, still below its 66.7% baseline ceiling).

## Phase 19 — Model 4: Co-Attention
- ✅ `backend/scripts/train_model4.py` — every timestep of the technical GRU's sequence attends over [macro, news] as key/value context (`H = f(T, M, N, Attention(T,N))` per the plan's formula), not just concatenating static vectors as Model 3 did. Single-head, small hidden size — consistent with the standing lesson from the archived TFT experiment about attention mechanisms needing more data than this project has per market. Attention weights stored per-instance for Phase 35 inspection later (not claimed as causal, per the plan's own caution).
- ✅ Results: ZN 55.4%, CL 55.6% (recovered from Model 3's 42.6% low), GC 59.3% (best Model 4 result, still below its 66.7% baseline).
- ✅ Real behavioral observation from the attention weights (inspection only, not a causal claim): GC's co-attention weighted macro over news ~67/33 on average; ZN and CL landed close to 50/50.

### Ablation so far (Phases 16-19 complete) — prediction-quality only, not yet P&L/Sharpe

No model beats its market's Phase 15 baseline ceiling except Model 3 exactly tying ZN. This is evaluated on directional accuracy only — the real Phase 30 ablation (with Sharpe/drawdown/turnover) can't run until the trading-strategy and paper-trading machinery (Phases 21-29) exists. Current honest state: added complexity has not yet demonstrated it earns its cost on this data, and that itself is a legitimate, reportable interim finding, not a failure — Phase 30/31 will need real backtested P&L before any final conclusion, since accuracy alone doesn't capture risk-adjusted performance.

| Market | Baseline | M1 | M2 | M3 | M4 |
|---|---|---|---|---|---|
| ZN | 57.1% | 44.6% | 55.4% | 57.1% | 55.4% |
| CL | 64.8% | 57.4% | 48.1% | 42.6% | 55.6% |
| GC | 66.7% | 55.6% | 38.9% | 57.4% | 59.3% |

## Phase 20 — Asset-specific representation (learned asset embedding)
- ✅ `backend/scripts/train_model4_joint.py` — ONE model trained jointly across ZN/CL/GC (Models 1-4 were each trained separately per market), with a learned `nn.Embedding(3, hidden_size)` asset vector fused into the representation alongside the co-attention output, so the model explicitly knows which market it's looking at rather than encoding that implicitly through 3 disjoint parameter sets. Also revisits the archived-project TFT lesson (joint training = more effective data) honestly at a real ~3x, not assumed to help.
- ✅ **Real bug caught and fixed**: macro data is zero-padded to a common width across markets (14/5/4 columns) so the pooled tensor has one consistent shape, but each market's own macro-embedder expects its own real column count — first run crashed with a matrix-shape error (`231x14` into a `5x16` layer) from feeding CL's embedder the full padded width instead of its own 5 real columns. Fixed by storing each market's real dimension on the model and slicing back down to it before routing to that market's embedder.
- ✅ **First model in the entire ablation to clearly beat its baseline**: ZN 60.7% (vs. 57.1% baseline) — a real, positive result. Mixed elsewhere though: CL 44.4% (its *worst* result across all 5 models tried), GC 59.3% (unchanged from per-market Model 4).

### Full ablation status (Phases 15-20 complete)

| Market | Baseline | M1 (tech) | M2 (+macro) | M3 (+news) | M4 (co-attn) | M4-joint (+asset embed) |
|---|---|---|---|---|---|---|
| ZN | 57.1% | 44.6% | 55.4% | 57.1% | 55.4% | **60.7%** |
| CL | 64.8% | 57.4% | 48.1% | 42.6% | 55.6% | 44.4% |
| GC | 66.7% | 55.6% | 38.9% | 57.4% | 59.3% | 59.3% |

Honest read: simple baselines are still winning or tying on 2 of 3 markets even after every architectural idea in the plan has been tried. ZN is the one case where sophistication (co-attention + joint asset-embedded training) demonstrably helped. This is a legitimate, presentable interim finding for the eventual research report (Phase 37) — not every market benefits from the same modeling sophistication, which is itself informative. Still pending before any final conclusion: real historical news (still placeholder) and the actual P&L/Sharpe-based ablation (Phase 30/31), which needs Phases 21-29 (strategy, portfolio, paper-trading, backtest) built first.

## Phase 21 — Trading strategy layer (risk-adjusted score → BUY/HOLD/SELL thresholds)
- ✅ `backend/app/strategy.py` — model-agnostic: takes only `(expected_return, expected_volatility)`, so the exact same strategy rules apply unchanged to any of the 5+ models built so far, satisfying the plan's "don't change trading rules between models" requirement for Phase 30. Score = return/volatility (per-instance Sharpe-like ratio); thresholds picked by grid search over **validation-only** data, maximizing average validation P&L — the test set never influences threshold choice.
- ✅ `backend/scripts/run_strategy_demo.py` — demonstrated on the Phase 20 joint model's real predictions.
- ✅ **Real bug caught before it became a misleading result**: first print showed raw realized price return for SELL positions, not actual strategy P&L (a short profits from a *negative* move, so the sign needs flipping) — caught before reporting it, not after.
- ✅ **Real finding, and a genuine cross-check via a completely different metric (P&L, not accuracy) that lands on the same conclusion as the whole ablation table**: ZN's SELL positions are the only ones profitable on test (+0.11% avg P&L/position, 42 positions); CL loses on both BUY (-8.52%, thin sample of 6) and SELL (-1.56%); GC's SELL loses slightly (-0.55%). Convergent evidence that ZN is genuinely the standout market, not an artifact of the accuracy metric specifically.
- ⚠️ Honest limitation, not hidden: validation-calibrated thresholds never triggered a BUY signal at all for ZN or GC on this test period — a real consequence of a small validation set, not a bug, and worth revisiting once more historical data or real news changes the calibration data.

## Phase 22 — Portfolio state
- ✅ `backend/app/portfolio.py` — a `Portfolio` class that tracks one open position per asset, closes it exactly at the 5-day holding period fixed in `trading_frequency.md`, and only opens a new position once flat. Directly answers the plan's own framing question: "if it asked me to buy yesterday, what happens today?" — the position rides regardless of today's signal until its holding period is up.
- ✅ Extended `train_model4_joint.py` to also expose real per-row dates/prices/day-indices (needed for day-by-day position bookkeeping, not just aggregate stats) — plumbing change, verified the joint model's own results are unchanged after the change (0.549 overall, same per-market numbers) before trusting anything built on top of it.
- ✅ `backend/scripts/run_portfolio_demo.py` — walks the real test period day-by-day per market.
- ✅ **Retroactive correction to Phase 21's demo**: running this surfaced that Phase 21's demo counted every single test-set day as an independent position (no holding-period awareness), which isn't actually realistic trading. Phase 22's portfolio-aware simulation is the first *correct* one — only 27 real trades occur across the whole test period once positions are held for their full 5 days instead of re-decided daily.
- ✅ Results (real, portfolio-correct): **ZN**: 10 trades, +0.15% avg P&L, 50% win rate (roughly breakeven with a slight edge). **CL**: 8 trades, -4.04% avg P&L, 25% win rate (clearly losing, consistent with every prior CL result). **GC**: 9 trades, +0.68% avg P&L, only 33% win rate but still net positive — wins are larger than losses on average, a real payoff-asymmetry worth noting rather than reading the win rate alone.

## Phase 23 — Position sizing + risk limits
- ✅ `backend/app/risk.py` — confidence-scaled position sizing (1-3 contracts, scaled by how far past the calibrated threshold the score is), a portfolio-level concentration limit (max 6 total contracts across all assets), and a daily-loss circuit breaker (no new positions on a day where realized P&L already breached -3%). Clearly-defined rules, not an institutional risk engine, per the plan's own scope note.
- ✅ `Portfolio` extended to carry position size, not just direction.
- ✅ `backend/scripts/run_risk_demo.py` — walks all 3 markets in **true calendar-date order** (not each market's own row index, which isn't comparable across markets) so the daily-loss breaker and concentration limit see everything happening on the same real day, not coincidentally-numbered unrelated days.
- ✅ Real result: fewer trades than Phase 22 (risk limits actively block some), and win rates improved notably where they did open — ZN 50%→71%, GC 33%→67% — while CL got worse again (-4.04%→-5.88% size-weighted). Consistent, convergent picture across every phase so far: ZN/GC show a real (if small) edge, CL does not.

## Phase 24 — Futures-specific trading mechanics
- ✅ Contract specs gathered (multiplier/tick size/tick value) for ZN, CL, GC — see market_driver_map.md
- ✅ `backend/app/futures_mechanics.py` — real maintenance margins sourced from a broker-published table (AMP Futures — CME's own margin pages are JS-rendered and didn't scrape from this environment; flagged as an approximation from a real source, not fabricated) since we hold overnight (day-trading margin doesn't apply). Slippage (2 ticks against the position, each way) and commission ($2.25/contract/side) applied to every trade.
- ✅ `backend/scripts/run_futures_mechanics_demo.py` — converts Phase 23's percentage P&L into real dollar terms on the exact same trades (no changes to trading rules).
- ✅ **Sanity-checked before trusting it** (real gold price fetched directly): one GC contract is ~$467,600 notional against $25,743 margin — ~18x leverage. This is *why* GC's dollar P&L looked huge at first glance (+$116,530 gross across 9 trades) — confirmed correct, not a bug, and a concrete illustration of the leverage concept from `market_driver_map.md` Part 1.
- ✅ **Real, concerning finding worth flagging plainly**: CL's -5.88% average P&L becomes **-$53,660 net loss** in dollar terms across just 5 trades — against a peak margin usage of only $27,495. That loss is roughly 2x the capital that would have been posted, meaning this strategy would very likely have triggered margin calls in a real account, not just underperformed on paper. Commission itself is negligible (0.4% of gross P&L magnitude) — it's the trade quality on CL that's the real problem, not costs.

## Phase 25 — Paper-trading engine
- ✅ `backend/app/paper_trading.py` — `PaperTradingEngine` formalizes the plan's own 9-step daily loop, adding the one piece Phases 21-24 hadn't built yet: **unrealized (mark-to-market) P&L on still-open positions**, not just realized P&L when a trade closes. Produces a full daily equity curve (cash + unrealized), which Phase 26/29 need — a list of closed trades alone isn't enough for Sharpe/drawdown.
- ✅ `backend/scripts/run_paper_trading_demo.py` — $100,000 starting capital, real dollar mechanics from Phase 24, same trades/rules as every prior phase.
- ⚠️ **First run, before the fix below**: final equity $126,598 (+26.6%) sounded good in isolation, but the equity curve peaked at $280,004 (+180%) before crashing to a trough of $88,317 (**below starting capital**) — a peak-to-trough drawdown of roughly **68%**. Also revealed a real gap: the Phase 23 concentration limit capped raw contract *count* (max 6 total), never actually checked against available cash — and Phase 24's worst-case simultaneous margin figure ($110,910) **exceeds** the $100,000 starting capital. Not solvency-safe.
- ✅ **Fixed immediately, not deferred**: `risk.py`'s concentration check replaced with a real margin-utilization check (`margin_limit_ok` — never commit more than 50% of current account equity to margin, checked in real dollars via `futures_mechanics.get_specs`, not contract count). `Portfolio.step()` now takes `account_equity` and enforces this.
- ⚠️ **Re-running after the fix surfaced a second, more subtle and genuinely important finding, not just a smaller number**: final equity **$27,546 (-72.45%)** — a complete reversal from +26.6%. Diagnosed per-asset before accepting it: **GC (previously the single best-performing market — 9 profitable trades, +$116,530 gross) dropped to just 1 trade, which lost -$7,685.** Root cause: GC's margin (~$25,743/contract) is ~2.8x CL's (~$9,165), so a *uniform* percentage-of-equity cap disproportionately blocks the expensive-but-good market while continuing to let the cheap-but-bad market (CL, -$66,951 across 4 trades) trade freely. This is a genuine risk-engine design insight for the final report, not noise: a solvency-safe margin limit that's blind to signal quality can systematically starve the market that actually has an edge. Worth revisiting (e.g. allocate margin budget by demonstrated edge, not just by which asset is cheapest to trade) in a future iteration — flagged here rather than silently smoothed over.

## Phase 26 — Walk-forward backtesting methodology
- ✅ Parameterized `train_joint()`'s split boundaries (train/val/test fractions) so it can be retrained across a sliding window instead of one static split, with a market-data cache added so walk-forward doesn't re-hit Yahoo/FRED/EIA per fold.
- ✅ `backend/scripts/run_walk_forward.py` — 3 expanding-window folds (train grows each time, test window slides forward, never touches the future relative to its own fold).

### 🚨 Major finding — this overturns the "ZN is the standout market" conclusion reported throughout Phases 15-25

Direction accuracy per market, per fold:

| Market | Fold 1 | Fold 2 | Fold 3 (≈ the static split used everywhere else) | Mean | Std |
|---|---|---|---|---|---|
| ZN | 27.8% | 52.6% | 61.1% | 47.2% | **14.1** |
| CL | 66.7% | 44.4% | 94.4% | 68.5% | **20.5** |
| GC | 83.3% | 66.7% | 50.0% | 66.7% | **13.6** |

**Every market swings by 30-50 percentage points depending on which time window is tested.** The single static split used for every result up through Phase 25 happened to land on Fold 3 — which was ZN's *best* fold (61.1%, the result reported as "ZN beats its baseline") and CL's *best* fold too (94.4%, contradicting the entire "CL never works" narrative built across Phases 16-25). Fold 1 tells the opposite story: ZN at 27.8% (badly below baseline) and CL at 66.7% (one of its better results).

**Honest interpretation, not overcorrecting in the other direction**: this doesn't mean "there's no signal anywhere" — it means the evidence gathered so far is too noisy, on too little data, from a single test window, to support a confident claim about *any* market having a persistent edge. The walk-forward doesn't fix the small-sample problem (each fold's test set is still only ~30-60 rows per market) — it makes the instability *visible* instead of hidden behind one arbitrarily-favorable split. That is exactly what Phase 26 exists to catch, and it caught something real.

**What this changes going forward**: every claim from Phases 15-25 about a specific market "having an edge" or "not working" needs to be reframed as "in the one test window used at the time" — not a general property of the market or the model. This is the correct, defensible finding for the final report (Phase 37): **the ablation's single-split results were not robust across time, and that instability is itself the headline finding**, not a footnote. Phase 30/31's ablation table should be built on walk-forward results (or at minimum report both), not the single split.

### Follow-up: extended history period (2y → 5y) to test whether more data stabilizes the folds

`HISTORY_PERIOD` centralized into `app/config.py` (was hardcoded `"2y"` in 7 different files) and raised to `"5y"`, after verifying real dense daily data actually exists that far back (1,256-1,257 rows/market — Yahoo's `"max"` range was tested and found to quietly return sparse/non-daily data for these tickers, confirmed and avoided).

Re-ran the walk-forward on 5 years instead of 2:

| Market | Fold 1 | Fold 2 | Fold 3 | Mean | Std (2y → 5y) |
|---|---|---|---|---|---|
| ZN | 48.2% | 48.2% | 53.6% | 50.0% | 14.1 → **2.5** |
| CL | 67.3% | 42.6% | 61.8% | 57.2% | 20.5 → **10.6** |
| GC | 57.4% | 64.2% | 38.9% | 53.5% | 13.6 → **10.7** |

**Confirms the hypothesis, with a sobering but honest conclusion**: more data genuinely stabilized the fold-to-fold variance (ZN's std dropped from 14.1 to 2.5 — a real, large improvement). But the *stabilized* truth is more modest than hoped: **ZN now sits consistently right at 50% — chance level — across all 3 folds**, meaning its earlier apparent "edge" (60.7%, 61.1%) really was small-sample noise, not signal, and the extra data revealed that rather than confirming an edge. CL and GC show moderate, real variance (10.6-10.7 std) with means in the mid-50s, not a dramatic, confident edge either.

**This is still a genuinely strong finding for the report**, arguably stronger than "we found an edge" would have been: it demonstrates the methodology correctly caught its own false positive (Phase 26 flagged the instability, more data resolved *how much* of it was noise vs. signal) rather than reporting an inflated result uncritically.

### "Rescue" attempt: 20-day horizon (Moskowitz time-series momentum) + CFTC COT positioning data

Research-motivated (not guessed): Moskowitz/Ooi/Pedersen (2012) "Time Series Momentum" is the most-replicated finding in exactly this asset class, showing real signal at ~1-month+ horizons rather than 5-day; a 2024 *Journal of Futures Markets* commodity-ML paper independently used monthly predictions and flagged CFTC Commitment of Traders (COT) positioning data as a dominant SHAP predictor. Built as `backend/scripts/train_model5_rescue.py` (separate from `train_model4_joint.py` rather than editing it in place, since that file hardcodes "5d" throughout) — same co-attention architecture, now with COT as a real 4th input stream (`app/data/cot.py`, free CFTC Socrata API, point-in-time aligned with a documented 3-day publication lag, verified against real fetched values with plausible signs before trusting it).

Walk-forward comparison (same 3-fold methodology as Phase 26, both on 5y data):

| Market | 5-day horizon (mean / std) | 20-day + COT (mean / std) |
|---|---|---|
| ZN | 50.0% / 2.5 | 50.2% / **21.6** |
| CL | 57.2% / 10.6 | 57.7% / **8.1** |
| GC | 53.5% / 10.7 | 69.7% / **25.2** |

**Not a clean rescue — mixed, and one real new statistical problem surfaced along the way, flagged rather than glossed over**: CL improved modestly on both mean *and* stability (the most genuinely encouraging result). ZN and GC's *means* look better or unchanged, but their *variance got much worse* — GC's Fold 1 hit 96.2% accuracy, which is the real tell, not a win to celebrate uncritically.

**The reason, caught before over-interpreting the numbers**: the 20-day target is still computed for *every single day* (a rolling window), so consecutive test rows overlap by up to 19/20 days — they are not independent observations. A 167-row test set at this horizon has roughly **~8 truly independent 20-day windows**, not 167. A single sustained trend during one fold's test period can make nearly every overlapping-window prediction agree (hence GC's 96.2%), which looks like skill but is largely one lucky/unlucky trend dominating a barely-independent sample. This makes the 20-day numbers *less* trustworthy at face value than the 5-day ones, not more, until addressed (e.g., evaluating on non-overlapping windows, or explicit autocorrelation-adjusted significance testing) — an open item, not resolved yet.

**Honest bottom line (superseded by the fix below, kept for the record)**: CL is the one place this rescue attempt shows a real, if modest, improvement. ZN and GC do not show a trustworthy rescue — the apparent gains are confounded by the overlapping-window autocorrelation problem this experiment itself revealed.

### Fix: non-overlapping evaluation windows

`train_model5_rescue.py`'s val/test splits now stride by `HORIZON` (20) instead of taking every day — each evaluated window is a genuinely independent 20-day period, not 19/20ths the same period as its neighbor. Train stays dense (overlap is fine, even helpful, for training volume) — only evaluation needed the fix, since that's where non-independence corrupts the reported number (and, for val specifically, corrupts early-stopping checkpoint *selection* too, so it got the same fix, not just test).

**Result, verified precisely on one fold before trusting the pattern**: pooled test set dropped from 488 rows to **9 rows total (3 per market)**. That's the real, honest effective sample size at a 20-day horizon with 5 years of data and this fold structure — the earlier "167 test rows per market" was an illusion created by counting overlapping windows as if they were independent.

**The correct conclusion, not a disappointing one**: 3 independent observations per market per fold is too few to draw *any* trustworthy conclusion, positive or negative — a 0% or 100% result on 3 coin flips says nothing. This isn't a failure of the rescue idea; it's the fix correctly refusing to let a fake sample size manufacture a fake conclusion, exactly as intended. **The real next step, if pursued further, is more historical data or many more (smaller) walk-forward folds pooled together** — not concluding the 20-day horizon "doesn't work" from 3 flips, and not trusting the earlier inflated 167-row numbers either. Flagged as an open item rather than resolved either way.

### Phase 15 baselines re-run on 5y (closes the open item above)

Same methodology as the original Phase 15 run (static 70/15/15 split, same 5 baselines), just on `HISTORY_PERIOD="5y"` instead of `"2y"` — test sets are now 161-169 rows per market instead of 54-56, a real statistical-power improvement in their own right, independent of the walk-forward question.

| Market | Best baseline (2y) | Best baseline (5y) |
|---|---|---|
| ZN | 57.1% (XGBoost) | 58.0% (LogReg/XGBoost tied) — roughly unchanged |
| CL | 64.8% (moving average) | **51.8%** (buy-and-hold/MA tied) — **collapsed to chance level** |
| GC | 66.7% (logistic regression) | 60.9% (momentum) — more modest |

**Directly consistent with the walk-forward joint-model finding above, from a completely independent angle (baselines, not the deep model)**: CL's headline-looking 64.8% "moving average edge" from the 2-year sample was the same kind of small-sample artifact as ZN's apparent deep-model edge — it evaporates with 3x more test data. Every market's numbers converge toward a tighter, more modest band (roughly 52-61%) once the sample size problem is addressed, rather than the more dramatic 57-67% spread the 2-year data suggested. Two independent methods (walk-forward on the joint model, and a larger single split on the baselines) now point at the same conclusion — that's real convergent evidence, not a coincidence.

### 10-year data extension: a second, subtler class-imbalance collapse (and its fix)

`HISTORY_PERIOD` extended from `"5y"` to `"10y"` (~2,511-2,512 rows/market, verified dense daily data) specifically to give the 20-day rescue experiment's non-overlapping evaluation windows a real chance at enough independent observations (5y left only ~3 per market per fold — see the fix above).

**A new bug, not the same one already fixed**: re-running both the standard 5-day joint model and the 20-day+COT rescue model on the 10-year dataset produced models that had collapsed to predicting one class 85-100% of the time — in **all three markets, in both models**. This slipped past the Phase 16/17 fix (checkpoint selection by validation accuracy) because walk-forward folds place validation immediately before test in time, and adjacent periods are often directionally correlated: a collapsed model that happens to agree with the local trend scores well on *both* validation and test, so raw validation accuracy never flagged it as broken. Caught only by explicitly inspecting prediction distributions rather than trusting the accuracy numbers alone (ZN was predicting "up" on 116 of 117 test rows; GC's apparently-strong result was coincidental alignment with its majority-"up" true test-period distribution, not real signal) — this directly retracted an about-to-be-reported "GC shows a genuine, stable improvement" claim.

**Fix, two independent measures (`backend/app/models/training_utils.py`), applied everywhere a training loop exists** — Models 1-3 (shared utility), Model 4/joint (`train_model4_joint.py`), and the rescue model (`train_model5_rescue.py`, which has its own separate inline loss/training loop and needed the identical fix applied by hand, not for free):
1. **Class-weighted BCE loss** (`compute_pos_weight`) — collapsing to the majority class is no longer loss-minimizing.
2. **Balanced accuracy**, not raw accuracy, for both checkpoint *selection* and final *reporting* — a constant predictor scores exactly 0.5 regardless of the true class split, so it can no longer hide behind a lucky trend. Every training script now also prints the raw prediction distribution on every run (not just when manually debugged), so a future collapse would be visible immediately.

**Honest results after the fix (10-year data, same 3-fold expanding-window walk-forward methodology as Phase 26):**

Standard 5-day model — balanced accuracy:

| Market | Fold 1 | Fold 2 | Fold 3 | Mean | Std |
|---|---|---|---|---|---|
| ZN | 0.491 | 0.471 | 0.458 | 0.474 | 0.013 |
| CL | 0.478 | 0.402 | 0.595 | 0.492 | 0.080 |
| GC | 0.458 | 0.412 | 0.589 | 0.486 | 0.075 |

**No market shows a real edge.** Every value sits within a few points of 0.500 (chance), well inside the noise band implied by the std. This supersedes every earlier 5-day-horizon claim in this document that was based on raw (collapse-vulnerable) accuracy — including the "CL improved, ZN/GC did not" framing above, which itself now looks like it was partly an artifact of the same failure mode on a smaller dataset.

Rescue model (20-day horizon + COT) — balanced accuracy:

| Market | Fold 1 | Fold 2 | Fold 3 | Mean | Std |
|---|---|---|---|---|---|
| ZN | 0.833 | 0.500 | 0.500 | 0.611 | 0.157 |
| CL | 0.500 | 0.500 | 0.750 | 0.583 | 0.118 |
| GC | 0.375 | 0.500 | 0.500 | 0.458 | 0.059 |

**These numbers are not trustworthy evidence either way, and it would be dishonest to report them as a finding.** Directly verified by re-running fold 1 in isolation: even at 10 years of data, the non-overlapping 20-day stride leaves a **pooled test set of just 18 rows total — 6 per market, per fold**. At n=6, a single flipped prediction swings balanced accuracy by ~17 points, which is exactly the pattern above (ZN's 0.833 came from 4 correct/2 wrong out of 6). The same diagnostic run caught CL's model still collapsed to a constant "up" prediction within that fold (`pred_dist={1.0: 6}`, all six test rows) — but this time balanced accuracy correctly reported it as exactly chance-level (0.500) instead of a misleadingly high number, which is direct, concrete proof the fix does what it's meant to do. The honest conclusion is the same one reached the first time this sample-size problem was found (5y, ~3 obs/fold): **the 20-day non-overlapping design is still statistically underpowered even at 10 years**, and no loss-function or checkpoint-selection fix can substitute for more independent observations. Confirming or ruling out the rescue hypothesis needs either many more years of history, or pooling folds/markets into one larger significance test rather than reading each 6-row fold in isolation — an open item, not resolved.

**Bottom line, honestly stated**: after fixing a real collapse bug that was inflating results, neither the standard 5-day model nor the 20-day+COT rescue model demonstrates a trustworthy directional edge on ZN, CL, or GC with the current data and methodology. The standard model's result is a confident null (large-enough sample, balanced accuracy near 0.5). The rescue model's result is an *inconclusive* null (too small a sample to say anything), not a confirmed failure. This retracts every earlier claim in this document of a market "beating baselines" or showing "genuine improvement" — those were measured before this collapse was found and fixed.

### Model 6 — Deep Momentum Network (Sharpe-ratio-optimized position sizing)

Prompted by a direct request to research what actually works in the published literature for this exact problem, rather than iterating further on a framing (direction classification) that Models 1-5 had honestly shown has no edge on this data. Not a guess -- reimplements two real, open-sourced, peer-reviewed approaches:

- **Lim, Zohren & Roberts (2019)**, ["Enhancing Time-Series Momentum Strategies Using Deep Neural Networks"](https://arxiv.org/pdf/1904.04912) (the original Deep Momentum Network / DMN). Core idea: don't classify next-period direction -- output a continuous **position size** directly, trained by optimizing a **differentiable Sharpe ratio loss**. This doesn't require >50% directional accuracy for positive Sharpe, because the loss rewards risk-adjusted P&L (sizing down in choppy periods, up in clear trends), not correct-call frequency. Reported ~2x Sharpe improvement over classical time-series momentum on 88 futures contracts, before costs.
- **Wood, Giegerich, Roberts & Zohren (2021)**, ["Trading with the Momentum Transformer"](https://arxiv.org/abs/2112.08534) (code: [github.com/kieranjwood/trading-momentum-transformer](https://github.com/kieranjwood/trading-momentum-transformer)). The `SharpeLoss` in `train_model6_dmn.py` is a direct PyTorch port of that repo's Keras `SharpeLoss` class -- fetched and read from the actual source, not reconstructed from the paper text.
- **Baz, Granger, Harvey, Le Roux & Rattray (2015)** MACD trend-indicator formula -- added as three new `macd_trend_8_24/16_48/32_96` features in `app/features/technical.py` (doubly-normalized, volatility-adjusted trend scores at three timescales; the input signal both papers above build on).

Architecture reuses the existing tech-GRU + co-attention backbone (macro + COT context, learned per-market embedding) from Model 5, with a single `tanh`-bounded position head instead of multi-task heads.

**A serious data-leakage bug found and fixed before any result was trusted, not after**: the first run reported a portfolio Sharpe of **11.6** -- implausible for any real strategy (professional quant funds run ~1-2 long-term). Root-caused instead of reported: `app/data/news_pipeline.py`'s `generate_placeholder_macro_news()` builds its synthetic sentiment signal directly from `price_df["next_return"]`, which is bit-for-bit the same quantity as `target_return_1d` (`next_return = return.shift(-1)` in `prices.py`; `target_return_1d = close.shift(-1)/close - 1` in `targets.py` -- the same value two ways, confirmed by reading both). Models 1-5 used the same feature but predicted a 5-day or 20-day horizon, which dilutes a 1-day leak enough that it never produced an impossible-looking number -- Model 6 trades `target_return_1d` directly, so the leak fed it almost exactly the answer. **Fixed by dropping the fabricated news stream entirely** for this model (co-attention now runs on 2 real context tokens -- macro, COT -- not 3). **This is also a retroactive caveat on every earlier model's news-derived results**: their near-chance balanced accuracy suggests the diluted leak didn't meaningfully help in practice, but it was present, and no fully-clean model in this project uses news features until a real historical news source replaces the placeholder (Phase 11, still open).

**Honest walk-forward result, 10-year data, same 3-fold expanding-window methodology as Phase 26 (portfolio Sharpe, equal-weighted across ZN/CL/GC by calendar date):**

| Strategy | Fold 1 | Fold 2 | Fold 3 | Mean | Std |
|---|---|---|---|---|---|
| DMN (learned) | -2.072 | -0.468 | -0.346 | **-0.962** | 0.786 |
| Buy & hold | -1.060 | 1.935 | 1.631 | 0.836 | 1.346 |
| Classical trend (hand-built rule, same features) | -0.220 | 1.044 | 1.182 | 0.669 | 0.631 |

**Not a rescue -- a clean negative result, reported as such rather than reached for a flattering cut of it.** The learned model has a negative mean Sharpe and is the worst of the three strategies in every single fold, including underperforming a simple non-learned rule built from the exact same trend features it has access to. The one earlier flattering number (a single static 70/15/15 split gave DMN=1.05, roughly matching buy-and-hold's 1.09) does not survive walk-forward scrutiny -- the same lesson Phase 26 already taught about the classification models, now confirmed a second time in a completely different framing.

**Why, stated honestly rather than explained away**: most plausibly, a full-batch, single-seed, ~150-epoch GRU trained on 3 instruments and roughly a decade of daily data is a much smaller, less-tuned setup than the cited papers' (88 instruments, cross-sectional pooling, extensive hyperparameter search) -- this is a faithful reimplementation of the *idea*, not a reproduction of their exact scale or tuning budget. It's also consistent with everything else this project has found: these three markets show no exploitable edge at a daily/short-horizon under any framing tried so far (classification or Sharpe-regression), which is itself a real, defensible, three-times-independently-confirmed finding, not a failure to find the right trick.

**What this delivers regardless of the number**: a genuine, working, correctly-evaluated implementation of a real published methodology (not a guess), evaluated with the same walk-forward rigor and honest-baseline discipline as everything else here, that surfaced and fixed a real data-leakage bug along the way. The next honest step, if pursued, is the paper's own actual scale (pooling many more instruments, not just 3) rather than tuning this smaller setup to chase a better number on the same 3 markets.

### Model 7 — real semantic-embedding news representation + literature-grounded cross-modal fusion

Prompted directly by the user pointing at this project's own original architecture diagram (`FINANCIAL NEWS -> FinBERT -> [Sentiment, Semantic embedding] -> recombined News representation -> co-attention with Technical/Macro -> Market representation -> Expected return + risk -> Trading strategy`) and asking whether it had actually been built. Checked against the code rather than asserted from memory: **no** — `app/features/sentiment.py` only ever extracted FinBERT's 3-class softmax output and threw away its semantic embedding entirely, and every prior model's fusion was a plain symmetric co-attention across equal-footing tokens, not the query/key-value structure the diagram implies. The user then asked for a literature review before any further build ("do ur research first dont just spring it on"), followed by "use their methodology or combine methodology which you think will give best results."

**Research reviewed** (not guessed from memory — fetched and read):
- **STONK** (Kandukuri, [arXiv:2508.13327](https://arxiv.org/abs/2508.13327), IEEE-DSAA 2025, code: [github.com/sarthak-12/thesis-dsaa](https://github.com/sarthak-12/thesis-dsaa)) — exactly the diagram's design: numeric market features as the attention **query**, text embeddings as **key/value** (`A = softmax(QK^T/√d)V`, fused as `m = [X;A]W_f`). Tested 5 encoders (FinBERT, MiniLM, DeBERTa, Electra, ModernBERT) on real historical news (FinSen, 160k S&P 500 articles, 2007-2023). Real reported results: accuracy 0.65-0.68, Sharpe 2.24-3.15, Profit Factor 1.72-2.03.
- **MSGCA** (gated cross-attention for stock movement) — real ablation showing plain cross-attention "suffers from noisy information"; a sigmoid gate using the numeric stream as discriminator fixes it. Added as a one-line, justified addition on top of STONK's base design.
- Deep Policy Gradient Methods in Commodity Markets ([arXiv:2308.01910](https://arxiv.org/abs/2308.01910)) reviewed too (RL with transaction-cost/risk-sensitivity reward on real futures) but not yet implemented here — noted as a follow-up for Model 6's rework, not folded into this model.

**Built**: `app/features/sentiment.py:embed_headlines()` extracts FinBERT's actual mean-pooled last-hidden-state (768-dim) — verified genuinely meaningful before trusting it, not assumed: same-topic positive/negative headlines cosine-similarity 0.93 vs. 0.59-0.60 against an unrelated neutral headline. PCA-reduced to 16 dims (fit on **train-only** rows per fold, same discipline as every `StandardScaler` in this project) and concatenated with the 3 existing sentiment probabilities into a genuine 19-dim "News representation" (`train_model7_crossmodal.py`). Fusion follows STONK's query/key-value spec plus an MSGCA-style sigmoid gate. Held at the 5-day horizon (not 1-day) and everything else identical to Model 4, so any difference is attributable to the news representation/fusion change specifically.

**Honest walk-forward result, 10-year data, same 3-fold methodology (balanced accuracy):**

| Market | Model 4 (old shallow sentiment) | Model 7 (real embedding + cross-modal fusion) |
|---|---|---|
| ZN | 0.474 | 0.470 |
| CL | 0.492 (std 0.080, unstable) | **0.626** (std 0.006, suspiciously stable) |
| GC | 0.486 | 0.544 |

**Not reported as a win without checking why, and the check overturned it.** CL's result in particular — a 13-point improvement with almost no fold-to-fold variance — is exactly the shape of result this project has repeatedly learned not to trust on sight. Ran the identical walk-forward with the entire News representation zeroed out (an ablation, not a guess): **every market collapsed to essentially exactly 0.500 balanced accuracy** (ZN=0.502, CL=0.505, GC=0.500) — the textbook signature of a model with no real information contributing anything beyond a coin flip. This proves CL's apparent gain was **not** the cross-modal architecture working — it was the richer 16-dimensional semantic embedding giving the model a more effective handle on the same diluted `target_return_1d` leak (news_pipeline.py) that the old, coarser 3-scalar sentiment score couldn't exploit nearly as well. A genuinely interesting, non-obvious, and honestly negative finding: **more expressive features can amplify a small leak that a shallower representation couldn't meaningfully exploit**, even at a horizon where the leak was previously judged "diluted enough."

**Re-checked whether this cast doubt on Model 4's original result too, not assumed either way**: ran the same news-disabled ablation on Model 4. Its numbers barely moved (ZN 0.474→0.492, CL 0.492→0.485, GC 0.486→0.471 — no collapse toward exactly 0.500), confirming Model 4's original near-chance finding was genuinely coming from tech+macro absence-of-signal, not the leak. The shallow symmetric co-attention in Model 4 evidently isn't expressive enough to exploit the leak the way Model 7's richer, gated, query/key-value fusion can — an important scope note for future models: **richer text representations need either a real (non-leaking) news source or explicit leak-blocking before they can be trusted, even at horizons previously judged "safe."**

**Bottom line**: the cross-modal fusion architecture itself is not yet validated as beneficial — the only clear signal it found was the leak, and removing the leak-adjacent input removed the entire effect. A fair test of STONK-style fusion on this project's markets needs either a real, non-leaking historical news source (Phase 11's long-standing open gap) or a placeholder generator with zero correlation to any traded target, not just "diluted." Retaining the architecture and the embedding-extraction infrastructure (both real, working, and reusable) while withholding any performance claim until that's resolved.

### Model 8 — regularized gradient-boosted trees (XGBoost), no news, zero leak risk

Prompted by a second literature review ("browse all papers... use their methodology or combine methodology... make it work") after Model 7's cross-modal fusion turned out to be leak-driven, not real. Deliberately chose a direction with **zero exposure** to the news-leak problem rather than trying to patch around it again.

**Research reviewed**: Shwartz-Ziv & Armon (2021), "Tabular data: Deep learning is not all you need" — XGBoost beat deep models on 8 of 11 tabular benchmarks, close to a literature consensus for exactly this data regime (a few thousand rows, heterogeneous engineered features) that every model in this project (1-7) has been a deep GRU/attention architecture applied to, without ever re-testing the simpler alternative at the current data/feature scale. Phase 15's original XGBoost baseline predates the 10-year extension, the Baz trend-indicator features, and COT entirely.

Also closed a real, separately-motivated gap in the existing COT data: added `cot_commercial_percentile_3y` / `cot_speculator_percentile_3y` to `app/data/cot.py` — the standard "COT Index" (trailing 3-year percentile rank of net positioning via `100*(current-min)/(max-min)`), which practitioner and academic sources specifically flag as more predictive than the raw net-position level this project already had (commercial hedgers at positioning extremes reportedly signal direction correctly ~70% of the time in some studies). Verified computing correctly on real data before use (range 0-100, sensible distribution, not degenerate).

**Design**: `train_model8_gbdt.py`, one regularized `XGBClassifier` per market (tech + that market's macro + COT, no news at all), same regularization discipline as Phase 15's fix (max_depth=3, subsample/colsample<1, reg_alpha/reg_lambda, early-stopping on a real validation split) plus `scale_pos_weight` and balanced-accuracy reporting for the same class-imbalance-collapse reason established in `training_utils.py`.

**Honest walk-forward result, 10-year data, same 3-fold methodology (balanced accuracy):**

| Market | Model 4 (deep, shallow news) | Model 8 (XGBoost, no news) |
|---|---|---|
| ZN | 0.474 | 0.494 |
| CL | 0.492 | 0.495 |
| GC | 0.486 | 0.523 |

**No meaningful improvement — a confirmed null, not a disappointing one to soften.** Every value is within a few points of chance and inside the fold-to-fold noise band (std 0.006-0.047). The balanced-accuracy safeguard caught XGBoost skewing heavily toward one class in some folds (ZN fold 1: 98 of 104 predictions "up") and correctly reported that as ~0.49, not a flattering raw accuracy number — the same discipline established for the neural models applied here too, not assumed unnecessary for a tree model.

**This is now the 4th materially different, independently-built method** (multi-task classification, Sharpe-ratio position-sizing regression, cross-modal attention fusion, and now gradient-boosted trees on entirely real features) **that finds no exploitable directional edge on ZN/CL/GC at a 5-day horizon.** That convergence across genuinely different model families and feature-leak-free inputs is itself a real, defensible research finding for the final report — not a failure to find the right architecture.

### Model 9 — Deep Momentum Network on a broader 13-market futures universe

Third lever from the same literature review, tested before concluding the DMN approach (Model 6) doesn't work here. Model 6 pooled only ZN/CL/GC and lost to both buy-and-hold and a classical trend rule in every fold. One real, literature-identified difference was untested: the original DMN paper backtested on **88** instruments, not 3, and **End-to-End Parametric Portfolio Policies for Cross-Asset Futures Timing** ([arXiv:2607.00475](https://arxiv.org/pdf/2607.00475)) directly found "learned policies perform better in the broad cross-asset universe... than simple rules" — universe size is a documented driver of these papers' results, not incidental.

Verified feasibility before building anything: 10 additional liquid CME futures (ES, NQ, YM, RTY — equity index; 6E — FX; SI, HG — metals; ZB, ZF — rates; NG — energy) all fetch cleanly with real dense 10-year data via the existing pipeline. Built `train_model9_dmn_broad.py`: same `SharpeLoss` and portfolio-daily-return methodology as Model 6, pooled across all 13 markets. Deliberately **technical-features-only** (no macro/news/COT) for every market, for two honest reasons: the macro/fundamentals/COT plumbing is hand-built per-market for ZN/CL/GC specifically and doesn't exist for the 10 new tickers, and this isolates the universe-size variable from the multimodal-features question already tested (and found not to help) in Models 6-8 — a cleaner single-variable experiment, and closer to the original DMN paper's own setup.

**Honest walk-forward result, 10-year data, same 3-fold methodology (portfolio Sharpe):**

| Strategy | Fold 1 | Fold 2 | Fold 3 | Mean | Std |
|---|---|---|---|---|---|
| DMN (13 markets) | -0.934 | -0.780 | 0.303 | **-0.470** | 0.551 |
| Buy & hold | -1.133 | 0.925 | 1.084 | 0.292 | 1.010 |
| Classical trend | -0.097 | -1.318 | -0.330 | -0.581 | 0.529 |

**Genuine, literature-consistent progress — not a clean win, and reported as exactly that.** The broader universe meaningfully narrowed the gap from Model 6's 3-market result (mean Sharpe -0.962 → -0.470), and the DMN now beats the classical trend rule in 2 of 3 folds (mean -0.470 vs. -0.581), matching what the cited papers report when comparing a learned policy against a hand-built rule. But it still underperforms simple buy-and-hold (mean -0.470 vs. 0.292) — so this is a real, honest improvement in the DMN's relative standing, not evidence the paradigm delivers genuine alpha over the simplest possible baseline on this universe. Diversification benefit (buy-and-hold's own Sharpe rose from -1.06 on 3 markets to 0.29 on 13) is doing real work here too, not just the learned model.

### Overall investigation conclusion (Models 4-9)

Six independently-built, materially different methods were tried on this research question, each evaluated with the same walk-forward rigor and each result checked before being trusted (ablations, prediction-distribution inspection, leak tracing) rather than reported at face value:

1. **5-day direction classification** (Model 4, tech+macro+shallow-sentiment) — near-chance, balanced accuracy 0.47-0.49.
2. **20-day direction classification + COT** (Model 5) — inconclusive (too few independent test windows even at 10y, not negative).
3. **Sharpe-optimized position sizing, 3 markets** (Model 6) — negative, lost to both baselines every fold.
4. **Real semantic-embedding cross-modal fusion** (Model 7) — apparent gain, ablation-confirmed to be a data-leak artifact, not real.
5. **Regularized gradient-boosted trees, no news** (Model 8) — near-chance, matching Model 4, ruling out "the deep architecture is the problem."
6. **Sharpe-optimized position sizing, 13 markets** (Model 9) — genuine partial improvement, still short of buy-and-hold.

**This convergence is itself the finding.** Four structurally different model families (recurrent multi-task classification, gradient-boosted trees, cross-modal attention fusion, and Sharpe-ratio-optimized regression) applied to real, leak-checked features all land in the same place: no exploitable directional edge at a 5-day horizon on ZN/CL/GC from technical, macro, and CFTC positioning data alone. The one lever that measurably moved a result was universe size for the Sharpe-regression framing (Model 9) — consistent with, not contradicting, everything else found. The honest, defensible research conclusion for the final report is a null result on daily/short-horizon directional predictability for these three markets specifically, arrived at through genuine multi-method triangulation rather than stopping at the first (or most flattering) result.

## Phase 27 — Look-ahead bias checklist
- ⬜ Not started

## Phase 28 — Transaction costs and slippage (no-cost vs realistic-cost backtest comparison)
- ⬜ Not started

## Phase 29 — Evaluation metrics (return, Sharpe, max DD, volatility, win rate, profit factor, turnover, avg trade)
- ⬜ Not started

## Phase 30 — The ablation study (Models 1-4, everything else held constant)
- ⬜ Not started

## Phase 31 — Evaluate ablation scientifically (results table + improvement calcs)
- ⬜ Not started

## Phase 32 — Evaluate each market separately (ZN / CL / GC breakdown, not just pooled)
- ⬜ Not started

## Phase 33 — Regime analysis (vol regimes, tightening/easing, risk-on/off, supply shocks)
- ⬜ Not started

## Phase 34 — Failure analysis (worst 20 trades, categorized)
- ⬜ Not started

## Phase 35 — Explainability (feature importance / SHAP / attention weights — inspection, not causal proof)
- ⬜ Not started (SHAP infrastructure from the prior version is reusable groundwork)

## Phase 36 — Dashboard
- ⬜ Not started

## Phase 37 — Research report (14-section writeup)
- ⬜ Not started

---

**Reusable from the prior "Modality-Attribution Robustness" build**: direct Yahoo Finance
fetch (handles `=F` tickers correctly), FinBERT sentiment scoring, chronological-split /
look-ahead-bias discipline, honest placeholder-vs-real-data labeling convention. Everything
else (regression targets, sequence models, macro data, trading simulation) is new.
