# Continuous Futures Contract Methodology

Phase 6 deliverable. This is the phase the plan calls out as most likely to distinguish
this project from "downloaded a Yahoo Finance CSV" — so the honest version of that
distinction matters more than a version that pretends a limitation doesn't exist.

## The constraint, verified empirically before deciding anything

Individual dated futures contracts (e.g., a specific ZN contract expiring December
2025) are **not available through our current data source**. Tested directly:

```
ZNZ25.CBT   -> 404
ZNZ25       -> 404
ZNH26.CBT   -> 404
ZNH26       -> 404
ZN=F        -> OK (this is Yahoo's own pre-built continuous front-month series)
```

Confirmed against public documentation too: Yahoo's free data only exposes the
continuous `=F` symbol per root (e.g. `ZN=F`, `CL=F`, `GC=F`) — individual dated
contracts with month/year codes are order-entry symbols available through
broker/exchange data feeds, not free retail data APIs. This is a real, known
limitation of free data sources, confirmed rather than assumed.

## What this means for Steps 6.1 and 6.2

The plan asks for a chosen roll methodology (6.1) and a chosen price-discontinuity
adjustment (6.2), documented. Both are answered here in two parts: the methodology we'd
use **if** we had individual contract data, and the honest v1 reality given that we
don't (yet).

### 6.1 — Roll methodology

**Chosen approach, if/when individual contracts are available**: a **liquidity-based
roll** — roll from the front-month contract to the next when the next contract's volume
(or open interest, if available — see the Phase 5 gap on open interest) exceeds the
current front-month's, rather than a fixed-calendar roll. This tracks where the market's
actual trading activity has moved to, rather than an arbitrary fixed date that might roll
too early (while the front month is still the most liquid) or too late (after liquidity
has already shifted).

**v1 reality**: we don't have individual-contract volume data to compute this against.
**v1 decision: use Yahoo's `=F` continuous series as-is.** This means the roll
methodology is Yahoo's own — undocumented in detail, but understood to be a standard
front-month-volume-based roll, which is the same category of approach (liquidity-based)
we'd have chosen ourselves. This is a deliberate, documented simplification, not an
oversight — flagged explicitly here and again in the final report's Limitations section
(Phase 37).

### 6.2 — Price discontinuity adjustment

**Chosen approach, if/when individual contracts are available**: **back-adjustment** —
when rolling from one contract to the next, shift the entire historical price series by
the price gap at the roll date, so historical returns computed across a roll date reflect
the actual P&L a continuously-rolled position would have realized, rather than an
artificial jump caused only by switching contracts.

**v1 reality**: Yahoo's `=F` series is already a single continuous price series with
whatever adjustment (if any) Yahoo applies internally — undocumented, and not something
we can independently verify or replace without access to the underlying dated contracts.
This is the second half of the same v1 simplification as 6.1.

### 6.3 — Raw vs. continuous data separation

Implemented as designed, even though `raw_contract_data/` starts empty:

```
data/
  raw_contract_data/     # individual dated contracts -- empty for now, real gap
  continuous_data/       # Yahoo's =F continuous series -- what we actually use
```

Keeping this separation in place now (rather than skipping it because one side is empty)
means the moment individual-contract data becomes available — e.g. through a paid vendor,
a broker API, or CME's own historical data during the internship — the project has
somewhere correct to put it, and the roll methodology in 6.1/6.2 can be upgraded to a
real, independently-computed one without restructuring anything else.

## Why this is still a defensible position, not a shortcut being hidden

The plan itself frames Phase 6 as "one of the areas where you can really distinguish
yourself from someone who simply downloaded a Yahoo Finance CSV." The distinguishing
factor isn't *whether* you had access to individual contract data — it's whether you
**understood the problem, tested what was actually available, chose a methodology
deliberately, and documented the gap honestly** rather than not knowing continuous-
contract construction was a problem at all. Someone who downloaded a Yahoo CSV and used
it without ever knowing what "roll" or "back-adjustment" mean is in a different position
than someone who tested for dated-contract availability, confirmed the constraint,
designed the correct methodology anyway, and documented exactly where the v1 system
falls short of it. That difference is what this document is for.
