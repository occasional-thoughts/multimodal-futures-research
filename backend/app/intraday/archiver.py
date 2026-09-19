"""Intraday bar archiver.

WHY THIS RUNS FIRST, BEFORE ANY MODELLING: free intraday data is a ROLLING
WINDOW, not a history. Verified limits on 2026-09-19:

    1m  -> last 5 days      5m  -> last 1 month
    15m -> last 1 month     1h  -> last 3 months

Every day this does not run, that day's 1-minute bars are permanently gone. No
amount of later work recovers them. A month of archiving buys a month of 1m
history that cannot be bought back afterwards at any price -- so this is the one
component where delay has a real, irreversible cost, and it is deliberately
built before the models that will eventually consume it.

Stores Parquet partitioned by ticker and interval, de-duplicated on timestamp so
re-running is always safe (overlapping fetches are the normal case, since each
pull re-covers days already held).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "intraday"

# (interval, fetch period) -- period is the max the free source will return.
INTERVALS = [("1m", "5d"), ("5m", "1mo"), ("15m", "1mo"), ("1h", "3mo")]
DEFAULT_TICKERS = ["CL=F", "GC=F"]


def _path(ticker: str, interval: str) -> Path:
    return DATA_DIR / f"{ticker.replace('=', '_')}_{interval}.parquet"


def fetch(ticker: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.Ticker(ticker).history(interval=interval, period=period)
    if not len(df):
        return pd.DataFrame()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index.name = "timestamp"
    return df


def archive(ticker: str, interval: str, period: str) -> dict:
    """Merge a fresh pull into the stored history. Returns what changed.

    Keeps the union of old and new, preferring NEW rows on conflict: a bar can be
    revised shortly after it closes, and the later fetch is the corrected one.
    """
    path = _path(ticker, interval)
    path.parent.mkdir(parents=True, exist_ok=True)

    new = fetch(ticker, interval, period)
    if new.empty:
        return {"ticker": ticker, "interval": interval, "added": 0, "total": 0, "status": "no data returned"}

    if path.exists():
        old = pd.read_parquet(path)
        before = len(old)
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        before = 0
        combined = new.sort_index()

    combined.to_parquet(path)
    return {
        "ticker": ticker,
        "interval": interval,
        "added": len(combined) - before,
        "total": len(combined),
        "span": f"{combined.index[0]:%Y-%m-%d} -> {combined.index[-1]:%Y-%m-%d}",
        "status": "ok",
    }


def archive_all(tickers: list[str] | None = None) -> list[dict]:
    out = []
    for t in tickers or DEFAULT_TICKERS:
        for interval, period in INTERVALS:
            try:
                out.append(archive(t, interval, period))
            except Exception as e:
                out.append({"ticker": t, "interval": interval, "status": f"ERROR {type(e).__name__}: {e}"})
    return out


def load(ticker: str, interval: str) -> pd.DataFrame:
    p = _path(ticker, interval)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


if __name__ == "__main__":
    import sys

    rows = archive_all(sys.argv[1:] or None)
    for r in rows:
        if r.get("status") == "ok":
            print(f"{r['ticker']:6} {r['interval']:>4}  +{r['added']:>5} new  total {r['total']:>6}  {r['span']}")
        else:
            print(f"{r['ticker']:6} {r['interval']:>4}  {r['status']}")
