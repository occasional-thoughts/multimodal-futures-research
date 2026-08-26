import pandas as pd
import requests

_CHART_URL = "https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Real daily OHLCV from Yahoo Finance.

    Hits the public chart API directly with `requests` instead of going through
    yfinance's `Ticker.history()`, whose curl_cffi-based cookie/crumb dance to
    fc.yahoo.com hangs in this environment even though the chart endpoint itself
    responds immediately to a plain request.
    """
    resp = requests.get(
        _CHART_URL.format(ticker=ticker),
        params={"range": period, "interval": "1d"},
        headers=_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()["chart"]["result"]
    if not result:
        raise ValueError(f"No price data returned for {ticker}")
    result = result[0]

    timestamps = pd.to_datetime(result["timestamp"], unit="s").normalize()
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        index=timestamps,
    ).dropna(subset=["close"])

    df["return"] = df["close"].pct_change()
    df["next_return"] = df["return"].shift(-1)
    df["label_up"] = (df["next_return"] > 0).astype(int)
    return df.dropna(subset=["return"])


_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


def fetch_recent_news(ticker: str, limit: int = 15) -> list[dict]:
    """Real recent headlines for the live demo, via Yahoo's search endpoint
    (same direct-`requests` approach as fetch_price_history -- yfinance's own
    `.news` property goes through the same hanging crumb dance).

    Only covers the last few days, so it is used for the demo's "today" view,
    not for historical training labels (see app/data/news.py).
    """
    resp = requests.get(
        _SEARCH_URL, params={"q": ticker, "newsCount": limit}, headers=_HEADERS, timeout=15
    )
    resp.raise_for_status()
    items = resp.json().get("news", [])
    return [{"title": it["title"]} for it in items if it.get("title")]
