"""Historical headline data for training.

The paper's design calls for the Kaggle financial-news corpus, date-aligned
to price history. Kaggle requires per-user authentication to download, so it
can't be fetched automatically here. Two paths are supported:

1. Real corpus (recommended for actual thesis results): download the Kaggle
   dataset yourself, and call `load_historical_news_csv(path)` with a CSV
   that has columns [date, ticker, headline].
2. Placeholder generator (`generate_placeholder_news`, used by default so the
   full pipeline is runnable and testable end-to-end right now): produces
   synthetic headline text whose sentiment is correlated with the stock's
   actual next-day return plus noise, so FinBERT still runs on real text and
   the fusion/SHAP/regime/consistency machinery all get exercised on signal
   that has a genuine (if synthetic) relationship to the label. This is NOT
   real news and must not be treated as a research result — swap in path (1)
   before drawing any conclusions for the write-up.
"""

import numpy as np
import pandas as pd

# Templates are asset-type-specific: "{company} shares climb" reads fine for a
# stock (equity) but not for gold or the S&P, which don't have "shares" or trade
# on company fundamentals -- commodity/index headlines instead reference macro
# drivers (safe-haven demand, supply, Fed/earnings-season sentiment).
_TEMPLATES = {
    "equity": {
        "positive": [
            "{company} shares climb after strong quarterly outlook",
            "Analysts raise price target on {company} citing robust demand",
            "{company} beats expectations, guidance revised upward",
            "Investors upbeat on {company} amid positive sector momentum",
            "{company} announces buyback program, stock reacts favorably",
        ],
        "negative": [
            "{company} shares slide on weaker-than-expected results",
            "Analysts cut price target on {company} amid demand concerns",
            "{company} misses estimates, guidance revised downward",
            "Investors cautious on {company} amid sector headwinds",
            "{company} faces scrutiny after disappointing outlook",
        ],
        "neutral": [
            "{company} holds annual investor meeting",
            "{company} announces routine executive appointment",
            "Market watches {company} ahead of upcoming earnings",
            "{company} maintains current guidance, no major changes",
        ],
    },
    "commodity": {
        "positive": [
            "{company} prices climb on strong demand outlook",
            "{company} rallies as investors seek safe haven",
            "{company} gains on tightening supply concerns",
            "{company} extends rally amid weaker dollar",
            "Traders turn bullish on {company} following inventory data",
        ],
        "negative": [
            "{company} prices slide on weak demand outlook",
            "{company} falls as safe-haven demand fades",
            "{company} drops on oversupply concerns",
            "{company} extends losses amid stronger dollar",
            "Traders turn bearish on {company} following inventory data",
        ],
        "neutral": [
            "{company} prices hold steady ahead of key data release",
            "Traders await fresh catalysts for {company}",
            "{company} trades in a narrow range",
            "Market watches {company} inventory report",
        ],
    },
    "index": {
        "positive": [
            "{company} rises on strong earnings season",
            "{company} climbs as investors cheer economic data",
            "{company} extends gains amid risk-on sentiment",
            "{company} rallies on hopes of favorable Fed policy",
            "{company} advances as growth outlook improves",
        ],
        "negative": [
            "{company} falls on weak earnings season",
            "{company} slides as investors digest soft economic data",
            "{company} extends losses amid risk-off sentiment",
            "{company} drops on fears of tighter Fed policy",
            "{company} declines as growth outlook weakens",
        ],
        "neutral": [
            "{company} trades flat ahead of key data release",
            "Investors await fresh catalysts for {company}",
            "{company} holds steady in quiet trading",
            "Market watches {company} ahead of Fed meeting",
        ],
    },
    # Fixed-income futures: headlines track the FUTURES PRICE direction (not
    # yield), since next_return is computed off the contract's own close price.
    # Bond prices rise when yields fall (safe-haven/dovish-Fed demand), and fall
    # when yields rise (rate-hike fears/strong growth data) -- opposite framing
    # from a plain "yields rose" headline, so this needs its own template set
    # rather than reusing "commodity".
    "rates": {
        "positive": [
            "{company} prices rise as yields fall on safe-haven demand",
            "{company} rallies as Fed signals a more dovish path",
            "{company} gains on weaker-than-expected inflation data",
            "{company} climbs as investors flee to safety amid risk-off sentiment",
            "Treasury futures advance on soft economic data",
        ],
        "negative": [
            "{company} prices fall as yields climb on rate-hike fears",
            "{company} slides as Fed signals a more hawkish path",
            "{company} drops on hotter-than-expected inflation data",
            "{company} declines as risk appetite improves and safe-haven demand fades",
            "Treasury futures retreat on strong economic data",
        ],
        "neutral": [
            "{company} trades flat ahead of Fed decision",
            "Traders await fresh catalysts for {company}",
            "{company} holds steady ahead of key inflation data",
            "Market watches {company} ahead of Treasury auction",
        ],
    },
}


def generate_placeholder_news(
    dates: pd.DatetimeIndex,
    next_returns: pd.Series,
    company: str,
    asset_type: str = "equity",
    signal_strength: float = 0.6,
    seed: int = 0,
) -> pd.DataFrame:
    """One synthetic headline per trading day, sentiment ~ correlated with next_return."""
    templates = _TEMPLATES[asset_type]
    rng = np.random.default_rng(seed)
    rows = []
    for d, r in zip(dates, next_returns):
        noise = rng.normal(0, 1)
        z = signal_strength * np.sign(r) * min(abs(r) * 40, 1.0) + (1 - signal_strength) * noise
        if z > 0.15:
            template = rng.choice(templates["positive"])
        elif z < -0.15:
            template = rng.choice(templates["negative"])
        else:
            template = rng.choice(templates["neutral"])
        rows.append({"date": d, "headline": template.format(company=company)})
    return pd.DataFrame(rows)


def load_historical_news_csv(path: str) -> pd.DataFrame:
    """Load a real date-aligned news corpus (columns: date, ticker, headline)."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "ticker", "headline"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required}, got {set(df.columns)}")
    return df
