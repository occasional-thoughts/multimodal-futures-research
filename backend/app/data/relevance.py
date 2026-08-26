"""Asset-specific news relevance filtering (Phase 12).

Explicit keyword-based classification, deliberately not a learned classifier -- the
plan is direct about this: "Eventually you could let the neural model learn relevance,
but for your first system, explicit relevance filtering is sensible." Keyword lists
are drawn directly from market_driver_map.md's driver categories, so this stays
consistent with the financial-hypothesis layer rather than being a separate,
disconnected guess at what matters.

A headline can be relevant to more than one market (e.g. "Fed signals more hikes" is
both ZN- and GC-relevant, matching their real shared driver -- Fed policy) -- relevance
is a set membership question per market, not a single best-match label.
"""

import re

# Keyword -> market. Matched case-insensitively as whole words/phrases against the
# headline text. Drawn from market_driver_map.md Part 2/3/4's primary + secondary
# drivers -- not a separate ad-hoc list.
_KEYWORDS = {
    "ZN": [
        "fed", "federal reserve", "fomc", "powell", "interest rate", "rate hike",
        "rate cut", "rate decision", "monetary policy", "inflation", "cpi", "pce",
        "employment", "unemployment", "payroll", "nonfarm", "jobs report", "gdp",
        "economic growth", "treasury", "treasuries", "yield", "bond market",
    ],
    "CL": [
        "opec", "opec+", "crude", "crude oil", "wti", "brent", "barrel", "barrels",
        "inventory", "inventories", "oil production", "shale", "refinery",
        "refineries", "drilling", "energy market", "petroleum", "gasoline", "eia",
    ],
    "GC": [
        "gold", "bullion", "safe haven", "safe-haven", "precious metal", "real yield",
        "real yields", "dollar", "usd", "central bank", "geopolitical", "inflation",
        "fed", "federal reserve",
    ],
}

_PATTERNS = {
    market: [re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE) for kw in kws]
    for market, kws in _KEYWORDS.items()
}


def classify_relevance(headline: str) -> list[str]:
    """Which of ZN/CL/GC this headline's content is actually relevant to -- may be
    zero, one, or multiple markets. Not the same question as "which ticker we
    searched under to find this article" (Phase 11's fetch-time tagging)."""
    if not headline:
        return []
    matches = []
    for market, patterns in _PATTERNS.items():
        if any(p.search(headline) for p in patterns):
            matches.append(market)
    return matches


def filter_relevant(articles, market: str):
    """Keep only articles whose content is actually relevant to `market`, regardless
    of which ticker they were originally fetched under."""
    mask = articles["headline"].apply(lambda h: market in classify_relevance(h))
    return articles[mask].reset_index(drop=True)
