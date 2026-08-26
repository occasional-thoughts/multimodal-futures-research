# Market Driver Map — ZN, CL, GC

Phase 1 deliverable. This is the financial-hypothesis layer the rest of the project
sits on top of — read this until you can explain each market to an interviewer
**without mentioning the ML model at all.**

---

## Part 1 — Futures Contract Mechanics (the vocabulary)

A **futures contract** is not ownership of anything. It's a standardized, exchange-listed
agreement to buy or sell a specific quantity of an underlying asset at a specific price,
on or by a specific future date. Unlike a stock, a futures contract:

- **Expires.** It has a defined delivery/expiration month, after which that specific
  contract stops trading and either settles in cash or physical delivery.
- **Is symmetric.** For every long (buyer) there's a short (seller) — futures are a
  zero-sum contract between two counterparties, cleared through the exchange.
- **Is leveraged by construction.** You post a small fraction of the contract's notional
  value as collateral (margin), not the full value — so a small price move produces a
  much larger percentage move on your posted capital than the same move would on a stock.

**Core vocabulary:**

| Term | Meaning |
|---|---|
| Underlying asset | What the contract is ultimately based on (a Treasury note, a barrel of crude, an ounce of gold) |
| Expiration | The date the contract stops trading / settles |
| Settlement | How the contract is closed out — cash settlement (most financial futures) or physical delivery (common in commodities, though most traders roll or close before delivery) |
| Contract size | The standardized quantity one contract represents (e.g., 1,000 barrels for CL) |
| Tick size | The smallest allowed price increment |
| Tick value | The dollar amount one tick is worth, per contract |
| Margin | Collateral posted to hold a position — a fraction of notional value, not the full value |
| Leverage | The ratio of notional exposure to capital posted — margin is what *creates* leverage |
| Initial margin | Collateral required to **open** a position |
| Maintenance margin | The minimum equity that must be kept in the account to **hold** a position; falling below it triggers a margin call |
| Contango | Futures price **above** the expected future spot price (further-dated contracts cost more) — common when storage/financing costs dominate |
| Backwardation | Futures price **below** expected future spot — common when near-term supply is tight (e.g., oil during a supply shock) |
| Basis | Spot price minus futures price; converges toward zero as expiration approaches |
| Rolling contracts | Closing a position in the expiring (front-month) contract and opening the equivalent position in the next contract, to maintain continuous exposure |
| Continuous futures contract | A single stitched-together historical price series built by chaining successive front-month contracts — a modeling convenience, not something you can actually trade (see Phase 6) |

**The one-sentence test**: *"If I'm trading ZN, what exactly am I trading?"* — you're trading
a standardized, exchange-cleared, leveraged, expiring agreement referencing $100,000 face
value of a 10-Year U.S. Treasury Note — not a stock, not the bond itself, and not something
that exists forever.

---

## Part 2 — ZN: 10-Year U.S. Treasury Note Futures

**Contract specs** (CME): $100,000 face value of the underlying note; tick size = 1/2 of
1/32nd of a point; tick value = **$15.625**; a full 1-point move = $1,000/contract.

### The core relationship

A Treasury note pays a fixed **coupon** (interest) on a **face value**, maturing at a set
date. Its **yield** is the return an investor gets given what they paid for it — which
moves *inversely* to price, because a fixed coupon becomes relatively more or less
attractive as the price paid for it changes:

$$\text{Yield} \uparrow \Rightarrow \text{Bond Price} \downarrow \qquad \text{Yield} \downarrow \Rightarrow \text{Bond Price} \uparrow$$

**Duration** measures how sensitive a bond's price is to a change in yield (longer
maturity → generally higher duration → more price sensitivity per unit of yield change).
**Convexity** is the second-order correction to that — duration itself changes as yields
move, and convexity captures how much. The **yield curve** is the set of yields across
maturities (2Y, 5Y, 10Y, 30Y) — its *shape* (steep, flat, inverted) is itself a signal
economists watch, separate from the level of any single yield.

### What actually drives ZN

**Primary:**
- **Federal Reserve policy** — the Fed funds rate directly anchors the short end of the
  curve, and forward guidance / rate-path expectations move the whole curve, including
  the 10-year.
- **Inflation** (CPI, PCE) — higher/hotter inflation erodes the real value of a fixed
  coupon, pushing yields up (prices down); cooler inflation does the reverse.
  PCE (Personal Consumption Expenditures) is the Fed's own preferred inflation gauge, distinct from CPI.
- **Employment** (nonfarm payrolls, unemployment rate) — strong labor data signals a
  hot economy → higher rate-path expectations → yields up; weak data → the reverse.
- **Economic growth** (GDP) — stronger growth tends to push yields up (more borrowing
  demand, less need for stimulus); weaker growth pulls yields down.
- **Treasury supply** — the volume of new note/bond issuance affects the price needed
  to clear the market; heavy issuance can pressure prices down (yields up).

**Secondary:**
- **USD strength** — interacts with foreign demand for Treasuries.
- **Equity risk sentiment** — a "risk-off" flight to safety often bids Treasury prices up
  (yields down), independent of the macro data itself.
- **Geopolitical events** — same safe-haven dynamic as risk sentiment.

---

## Part 3 — CL: WTI Crude Oil Futures

**Contract specs** (NYMEX): 1,000 barrels per contract; tick size = $0.01/barrel; tick
value = **$10.00**; a $1.00/barrel move = $1,000/contract.

### The core relationship

CL prices the U.S. benchmark grade, **WTI (West Texas Intermediate)**, at Cushing,
Oklahoma delivery. Unlike ZN, there's no yield/price inverse relationship to learn — this
market is a more direct supply-and-demand commodity story, but the specific *mechanisms*
matter more than a memorized rule.

### What actually drives CL

**Primary:**
- **Supply** — U.S. shale production levels, OPEC+ production quotas and compliance,
  and unplanned supply disruptions (outages, sanctions, conflict) all shift how much oil
  physically reaches the market.
- **Inventories** — the weekly EIA (U.S. Energy Information Administration) crude stock
  report is one of the most closely watched data releases in the market; a larger-than-
  expected build is *usually* bearish (more supply sitting unused), a larger-than-expected
  draw is *usually* bullish — but the reaction depends on whether it was expected or a
  surprise, and on the broader context (see Phase 5.3's "surprise" concept).
- **OPEC+ decisions** — production cuts tend to be bullish (less supply), increases tend
  to be bearish — but the market reaction depends heavily on whether the decision was
  already priced in.
- **Demand indicators** — refinery utilization, driving-season seasonality, and broader
  industrial activity.

**Secondary:**
- **USD** — oil is priced in dollars globally, so a weaker dollar makes oil cheaper in
  other currencies (demand-supportive), and vice versa.
- **Geopolitical risk** — particularly anything touching major producing/transit regions.
- **Global growth** — a slowing global economy reduces expected future oil demand.

**A simplified mental model** (a starting hypothesis, not a rule the model should assume):
supply down → price up; demand up → price up; inventory builds → often bearish; OPEC+ cuts
→ potentially bullish. These relationships can and do break down — e.g., an inventory
build that was smaller than the market feared can still be bullish, because markets trade
on *surprise relative to expectation*, not the raw number.

---

## Part 4 — GC: COMEX Gold Futures

**Contract specs** (COMEX): 100 troy ounces of 0.995-fineness gold; tick size = $0.10/oz;
tick value = **$10.00**; a $1.00/oz move = $100/contract, a $10/oz move = $1,000/contract.

### The core relationship

Gold pays no coupon or dividend — its "cost" of holding is the **opportunity cost** of not
earning interest elsewhere, which is why **real yields** (nominal yield minus inflation
expectations) are its single most important driver, more so than nominal yields alone.

### What actually drives GC

**Primary:**
- **Real yields** — when real yields fall, the opportunity cost of holding
  non-yielding gold falls, so gold tends to rise; when real yields rise, gold tends to fall.
- **USD strength** — gold is dollar-priced globally, so a weaker dollar mechanically makes
  gold cheaper for holders of other currencies (demand-supportive), and a stronger dollar
  the reverse.
- **Monetary policy** — expectations of easier policy (rate cuts, balance-sheet expansion)
  tend to lower real yields and weaken the dollar, both gold-supportive.
- **Inflation expectations** — gold is a traditional (though imperfect and debated) hedge
  against inflation eroding currency value.

**Secondary:**
- **Geopolitical risk** — classic safe-haven demand during crises.
- **Central-bank demand** — central banks (notably in emerging markets) have been
  structural buyers in recent years, a slower-moving but real demand source distinct from
  day-to-day trading flows.

**A simplified mental model** (again, hypotheses to test): real yields down → gold often
up; USD down → gold often up; risk/geopolitical uncertainty up → gold may rise. The
"may" matters — gold's safe-haven behavior is less mechanically reliable than its real-yield
relationship, and can be dominated by other factors during any given episode.

---

## Part 5 — Summary Driver Map

| Market | Primary drivers | Secondary drivers | What it teaches you |
|---|---|---|---|
| **ZN** (10Y Treasury) | Fed policy, inflation (CPI/PCE), employment, economic growth, Treasury supply | USD, equity risk sentiment, geopolitical events | Interest rates / monetary policy |
| **CL** (Crude Oil) | Supply (shale, OPEC+), inventories (EIA), demand indicators | USD, geopolitical risk, global growth | Commodity supply & demand |
| **GC** (Gold) | Real yields, USD, monetary policy, inflation expectations | Geopolitical risk, safe-haven demand, central-bank purchases | Inflation / real yields / safe haven |

These three were chosen specifically because they expose the project to three **different
economic systems** — rates, commodities, and the inflation/safe-haven complex — rather than
three assets that would all move on the same handful of drivers. That's the actual reason
this is a stronger interview story than three similar large-cap tech stocks: each market
forces a genuinely different explanation of *why* it moves.

---

## A note on how to use this document

Everything above is a **hypothesis layer**, not a rule the model should be hard-coded to
believe. "OPEC+ cuts are bullish for oil" is a reasonable prior, not a law — the actual ML
model (starting in Phase 16) will be tested on whether these relationships show up as
statistically real, predictive signal in the data, not assumed. The value of this document
is that *you* now have a framework for explaining, in an interview, why each market moves,
independent of whatever the model ends up learning.
