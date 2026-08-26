"""Point-in-time news alignment (Phase 13). Enforces problem_statement.md's rule:

    Information timestamp <= Decision timestamp

for news specifically. Price/technical data is already day-granularity by
construction (Phase 7 uses only trailing daily bars), but news articles carry real
hour:minute publication timestamps -- this lets us do better than pure date-matching.

Convention (documented, not hidden): trading_frequency.md already accepts that our
data source only gives a daily close with no exact settlement time, and adopts "the
close in our OHLCV bar" as the operational decision cutoff. Here we extend that with
one further explicit assumption: **4:00 PM US Eastern** as the assumed close time,
the standard convention for a US-listed daily bar, used only to decide which trading
day an article's timestamp counts toward -- not claimed to be the literal official
settlement time for ZN/CL/GC specifically (which trade nearly 24 hours via Globex).
"""

import pandas as pd

ASSUMED_CLOSE_HOUR_ET = 16  # 4:00 PM US Eastern


def assign_decision_day(article_timestamp_utc: pd.Timestamp, trading_dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Which trading day's decision (see trading_frequency.md) this article is
    eligible to inform. An article published before the assumed close on trading day
    d is known as of day d's decision; published after, it only counts from the NEXT
    trading day. Returns None if it's after the last available trading day (can't be
    assigned yet)."""
    if pd.isna(article_timestamp_utc):
        return None
    ts_et = article_timestamp_utc.tz_localize("UTC").tz_convert("US/Eastern") if article_timestamp_utc.tzinfo is None else article_timestamp_utc.tz_convert("US/Eastern")
    published_date = ts_et.normalize().tz_localize(None)
    is_before_close = ts_et.hour < ASSUMED_CLOSE_HOUR_ET

    candidate_days = trading_dates[trading_dates >= published_date]
    if len(candidate_days) == 0:
        return None

    same_day_is_trading_day = published_date in trading_dates
    if same_day_is_trading_day and is_before_close:
        return published_date
    # published after close, or on a non-trading day (weekend/holiday) -- eligible
    # from the next available trading day onward, never earlier
    later_days = trading_dates[trading_dates > published_date]
    return later_days[0] if len(later_days) > 0 else None


def align_articles_to_decision_days(articles: pd.DataFrame, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Adds a `decision_day` column to a scored-articles DataFrame (from
    news_pipeline.py), dropping articles that can't yet be assigned to a trading day."""
    out = articles.copy()
    out["decision_day"] = out["timestamp"].apply(lambda ts: assign_decision_day(ts, trading_dates))
    return out.dropna(subset=["decision_day"]).reset_index(drop=True)
