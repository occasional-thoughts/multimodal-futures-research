# Data Architecture

Phase 5 deliverable. This designs the four data categories and their schemas — and
identifies real, concrete sources for each — before any bulk collection starts in later
phases (9, 10, 11 for macro/news; 6 for continuous-contract construction).

## 5.1 Market data

One row per (contract, trading day). Reuses the existing direct-Yahoo-Finance fetch
(`backend/app/data/prices.py`), which already handles `=F` futures tickers correctly.

| Column | Type | Notes |
|---|---|---|
| `date` | date | trading day; see `trading_frequency.md` for the decision-cutoff convention |
| `contract` | string | e.g. `ZN=F` for now (front-month continuous via Yahoo); individual dated contracts (`ZNU26`, `ZNZ26`, ...) once Phase 6's roll logic is built |
| `open`, `high`, `low`, `close` | float | as provided by the source |
| `volume` | int | |
| `open_interest` | int | **not currently available** from the Yahoo endpoint we use — flagged as an open gap, see below |
| `expiration` | date | needed once individual dated contracts are pulled (Phase 6); not applicable to Yahoo's `=F` continuous series |
| `continuous_contract_id` | string | which continuous-series construction this row belongs to, once Phase 6 defines one (see Phase 6 note below) |

**Open gap**: open interest isn't in our current data source. CME/CBOT publish daily
open interest via their own data products (typically paid) or with a lag through free
channels; this is deferred rather than blocking Phase 5 — technical features (Phase 7)
don't require it, and it can be added later if a free source is found (CME's end-of-day
settlement files have some free tiers with a reporting delay).

**Relationship to Phase 6**: this schema already anticipates the raw-vs-continuous split
Phase 6 requires (`raw_contract_data/` for individual dated contracts, `continuous_data/`
for the stitched series) — Phase 5 just defines the columns; Phase 6 decides the roll
methodology and builds the continuous series from them.

## 5.2 Macro data

Market-specific series, sourced from **FRED (Federal Reserve Economic Data)** — free,
well-documented, versioned API (`api.stlouisfed.org`, requires a free API key). One row
per (series, date); series are naturally different frequencies (daily rates, monthly
CPI/employment, quarterly GDP) and get forward-filled to daily when joined against
market data in later phases (with the point-in-time discipline from `problem_statement.md`
strictly enforced — a monthly CPI print is only "known" from its actual release date
forward, never backdated to the reference month it describes).

| Column | Type | Notes |
|---|---|---|
| `series_id` | string | FRED series ID |
| `date` | date | the observation date FRED assigns |
| `value` | float | |
| `release_date` | date | **when this value actually became public** — not the same as `date` for monthly/quarterly series; this is what Phase 13's point-in-time join keys off |

**ZN (rates-driven) — series to pull:**

| Series | FRED ID |
|---|---|
| Fed funds effective rate | `FEDFUNDS` |
| 2-Year Treasury yield | `DGS2` |
| 5-Year Treasury yield | `DGS5` |
| 10-Year Treasury yield | `DGS10` |
| 30-Year Treasury yield | `DGS30` |
| CPI (headline) | `CPIAUCSL` |
| PCE price index | `PCEPI` |
| Unemployment rate | `UNRATE` |
| Nonfarm payrolls | `PAYEMS` |
| Real GDP | `GDPC1` |
| 10-Year breakeven inflation | `T10YIE` |

Derived: **yield curve spread** = `DGS10 - DGS2` (and its day-over-day change) — a
feature in its own right per Phase 9, not just a raw series.

**CL (commodity-driven) — series to pull:**

| Series | Source |
|---|---|
| Weekly crude inventories | EIA Weekly Petroleum Status Report — free API (`api.eia.gov`, free key) |
| U.S. crude production | EIA (same API) |
| USD strength | FRED `DTWEXBGS` (trade-weighted USD index) |
| Global growth proxy | FRED has OECD composite leading indicators; start simple, revisit if needed |

OPEC+ production decisions don't come as a clean time series — they're **discrete
events** (a meeting + a decision), which belongs in 5.3 (economic-event data) as an
event type, not a continuous macro series.

**GC (inflation/safe-haven-driven) — series to pull:**

| Series | FRED ID |
|---|---|
| Real 10-Year yield | `DFII10` (TIPS-implied real yield — this is the "real yields" driver identified in `market_driver_map.md`, more direct than computing DGS10 minus breakeven) |
| USD strength | `DTWEXBGS` (shared with CL) |
| 10-Year breakeven inflation | `T10YIE` (shared with ZN) |
| Fed funds rate | `FEDFUNDS` (shared with ZN) |

Note several series are intentionally shared across markets (USD, Fed funds, breakeven
inflation) — this is realistic, not a mistake: real markets share macro drivers, and
Phase 20's asset-embedding is partly what lets the model learn that the *same* macro
input matters differently for ZN vs. GC.

## 5.3 Economic-event (surprise) data

Distinct from 5.2: this is **discrete, scheduled releases** with an actual, an expected
(consensus), and a previous value — where the *gap between actual and expected* is
often more informative than the actual value alone (a strong CPI print that was already
expected moves markets far less than the same number as a surprise).

| Column | Type | Notes |
|---|---|---|
| `event_id` | string | |
| `event_date` | date | |
| `event_time` | timestamp | exact publication time, not just date — same discipline as news (5.4) |
| `event_type` | string | e.g. `CPI`, `NFP`, `FOMC_RATE_DECISION`, `OPEC_MEETING` |
| `actual` | float | |
| `expected` | float | consensus forecast |
| `previous` | float | |
| `relevant_markets` | list | which of ZN/CL/GC this event matters for (from `market_driver_map.md`'s primary/secondary driver lists) |

Derived (per Phase 5's own formulas):

$$\text{Surprise} = \text{Actual} - \text{Expected}$$
$$\text{Surprise}_{\text{relative}} = \frac{\text{Actual} - \text{Expected}}{\text{Historical volatility of that series}}$$

**Honest sourcing gap, flagged now rather than glossed over**: FRED and EIA give
*actual* published values, but neither publishes the *consensus/expected* figure — that
data point is normally a paid product (Bloomberg, Trading Economics, Econoday). Free
options exist (ForexFactory-derived economic calendars, available through a few
scraper-style community tools) but are less reliable/stable than FRED or EIA as a
long-term data dependency. **Two viable paths, to decide when this is actually built
(Phase 5.3 implementation, not this design doc):**
1. Use a free ForexFactory-derived calendar feed for `expected`/`actual`/`previous` on
   the small set of high-impact releases that matter here (CPI, NFP, FOMC decisions,
   PCE, GDP) — a manageable, low-volume need, not a full commercial calendar subscription.
2. If that proves unreliable, fall back to a **proxy surprise**: z-score the realized
   change in the FRED-published actual value against its own trailing history, as a
   stand-in for "how surprising was this release" — weaker than a true consensus-based
   surprise, but keeps the pipeline running on 100% free, stable sources. Document
   clearly as a proxy if used, the same honesty standard applied to the placeholder news
   signal in the archived prior project.

## 5.4 News data

One row per article. This needs to change from the prior project's design: the old
`news.py` was built for **company-style** headlines (Apple, JPMorgan, etc.) — this
project's universe is ZN/CL/GC, so the relevant news is macro/rates/energy/gold
coverage, not earnings headlines. Source selection happens in Phase 11; this phase only
fixes the schema.

| Column | Type | Notes |
|---|---|---|
| `article_id` | string | |
| `timestamp` | timestamp | **publication time, not just date** — this is what Phase 13's `Information timestamp ≤ Decision timestamp` rule checks against |
| `headline` | string | |
| `body` | string | may be empty for headline-only sources |
| `source` | string | |
| `asset_relevance` | list | which of ZN/CL/GC this article is tagged relevant to (Phase 12's relevance filtering) |
| `event_type` | string | optional link to a 5.3 event (e.g. an article *about* a CPI release) |

FinBERT scoring (`backend/app/features/sentiment.py`, reused from the prior project)
still applies at the per-article level once this table is populated — that part of the
pipeline doesn't change, only what fills the `headline`/`body` columns does.

## Summary: what's designed vs. what's collected

Phase 5 (this phase) fixes the **schema and source selection** for all four categories.
Actual bulk collection is deferred to the phases that consume it: 5.1 is already
implemented (`prices.py`) and gets finished by Phase 6 (continuous contracts); 5.2/5.3
get built out in Phases 9-10; 5.4 gets built out in Phases 11-13.
