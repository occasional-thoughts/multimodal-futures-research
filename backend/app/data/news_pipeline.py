"""News pipeline (Phase 11): cleaning -> FinBERT -> embedding + sentiment probabilities
+ metadata, stored per article, not just a daily-aggregated scalar.

Honest sourcing status, tested rather than assumed (see PHASE_TRACKER.md Phase 11):
- GDELT (a free, keyless, historically-deep news search engine, the strongest
  candidate for backfilling years of macro/commodity news) is unreachable from this
  environment -- DNS resolves but the connection itself fails. Worth retrying from a
  different network if this project moves off this machine.
- Yahoo's search endpoint (`fetch_recent_news`, reused from the prior project) returns
  real, correctly-relevant, real-timestamped articles when queried by an actual ticker
  symbol (verified for GC=F/CL=F/ZN=F) -- but is NOT a real keyword/topic search
  engine; querying it with macro phrases like "Federal Reserve" or "CPI inflation"
  returns generic unrelated market news, tested and confirmed unreliable for that use.
- Net result: we have a real, working source for *current* per-ticker headlines
  (good for the live view), but no free, keyless, historically-deep macro-news
  archive confirmed working yet. Historical backfill for training remains an open
  gap -- same category of gap as the old project's news corpus, now correctly
  re-scoped to macro/commodity themes instead of company headlines. See
  `generate_placeholder_macro_news` below for the stand-in used until it's resolved.
"""

import numpy as np
import pandas as pd
import requests

from app.features.sentiment import SENTIMENT_COLUMNS, score_headlines

_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

ARTICLE_COLUMNS = ["article_id", "timestamp", "headline", "source", "asset_relevance"] + SENTIMENT_COLUMNS


def fetch_recent_articles(ticker: str, asset_relevance: str, limit: int = 15) -> pd.DataFrame:
    """Real, current, correctly-relevant headlines for one ticker (Yahoo's search
    endpoint works well here -- confirmed unreliable only for topic/keyword queries,
    not ticker queries). Publication timestamp included, not just date."""
    resp = requests.get(_SEARCH_URL, params={"q": ticker, "newsCount": limit}, headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    items = resp.json().get("news", [])

    rows = []
    for it in items:
        content = it.get("content", it)
        title = content.get("title") or it.get("title")
        publish_time = it.get("providerPublishTime")
        if not title:
            continue
        rows.append(
            {
                "article_id": it.get("uuid", f"{ticker}-{publish_time}"),
                "timestamp": pd.to_datetime(publish_time, unit="s") if publish_time else pd.NaT,
                "headline": title,
                "source": (content.get("provider") or {}).get("displayName", "yahoo"),
                "asset_relevance": asset_relevance,
            }
        )
    return pd.DataFrame(rows, columns=ARTICLE_COLUMNS[:5])


def score_articles(articles: pd.DataFrame) -> pd.DataFrame:
    """Cleaning -> FinBERT -> sentiment probabilities, stored per-article (not
    collapsed to a daily aggregate here -- that aggregation is a modeling-time choice,
    made in Phase 13's alignment step, not baked in at ingestion)."""
    if articles.empty:
        return articles.assign(**{c: pd.Series(dtype=float) for c in SENTIMENT_COLUMNS})
    cleaned = articles["headline"].str.strip()
    probs = score_headlines(cleaned.tolist())
    out = articles.copy()
    out[SENTIMENT_COLUMNS] = probs
    return out


# --- Historical backfill placeholder (macro-themed), used only until a real
# historically-deep news source is confirmed working -- see module docstring. ---

_MACRO_TEMPLATES = {
    "ZN": {
        "positive": [
            "Treasury prices rise as yields fall on safe-haven demand",
            "Fed signals a more dovish rate path",
            "Treasuries rally on weaker-than-expected inflation data",
        ],
        "negative": [
            "Treasury prices fall as yields climb on rate-hike fears",
            "Fed signals a more hawkish rate path",
            "Treasuries slide on hotter-than-expected inflation data",
        ],
        "neutral": ["Traders await the next Fed decision", "Treasury market holds steady ahead of CPI"],
    },
    "CL": {
        "positive": ["Crude rallies on OPEC+ supply cut", "Oil gains on falling U.S. inventories"],
        "negative": ["Crude falls on unexpected inventory build", "Oil slides on weak demand outlook"],
        "neutral": ["Oil market awaits next EIA inventory report", "Traders watch OPEC+ meeting outcome"],
    },
    "GC": {
        "positive": ["Gold rallies as real yields fall", "Gold gains on safe-haven demand amid uncertainty"],
        "negative": ["Gold falls as real yields rise", "Gold slides as risk appetite improves"],
        "neutral": ["Gold holds steady ahead of Fed decision", "Traders await inflation data for gold direction"],
    },
}


def generate_placeholder_macro_news(dates: pd.DatetimeIndex, next_returns: pd.Series, market: str, seed: int = 0) -> pd.DataFrame:
    """NOT real news -- a synthetic, clearly-labeled stand-in for historical backfill,
    sentiment correlated with the actual next-period return plus noise, same honesty
    convention as the archived project's placeholder generator. Swap for a real
    historical source (a confirmed-working GDELT connection, a paid vendor, or a
    manually supplied corpus) before treating any downstream result as real.

    DATA-LEAKAGE WARNING, found the hard way (PHASE_TRACKER.md's Model 6 section):
    `next_returns` is virtually always `price_df["next_return"]`, which is bit-for-bit
    the same quantity as `target_return_1d` in app/targets.py. Any model that consumes
    the resulting SENTIMENT_COLUMNS features AND trades/scores against a 1-day-return
    label is training on a feature that directly encodes its own label -- this
    produced an impossible 11.6 Sharpe ratio in Model 6 before it was caught and
    fixed by dropping the news stream for that model entirely. A 5-day/20-day-horizon
    model dilutes this enough to not be obviously broken, but the leak is still
    present -- do not add a new model that uses these features against a 1-day target
    without either removing this feature or fixing the leak at its source.
    """
    templates = _MACRO_TEMPLATES[market]
    rng = np.random.default_rng(seed)
    rows = []
    for d, r in zip(dates, next_returns):
        noise = rng.normal(0, 1)
        z = 0.6 * np.sign(r) * min(abs(r) * 40, 1.0) + 0.4 * noise
        bucket = "positive" if z > 0.15 else "negative" if z < -0.15 else "neutral"
        headline = rng.choice(templates[bucket])
        rows.append({"article_id": f"{market}-{d.date()}", "timestamp": d, "headline": headline, "source": "placeholder", "asset_relevance": market})
    return pd.DataFrame(rows, columns=ARTICLE_COLUMNS[:5])
