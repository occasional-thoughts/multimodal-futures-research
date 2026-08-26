"""Macro feature engine (Phase 9, ZN-specific). Fetches real FRED series via the
public, no-API-key-required CSV endpoint, and aligns them to trading days respecting
the point-in-time discipline from problem_statement.md.

Honest limitation, flagged rather than hidden: FRED's free CSV endpoint gives the
*observation* date (e.g., "this CPI value describes August"), not the *publication*
date (CPI for August isn't actually released until mid-September). Getting exact
historical publication/vintage dates requires FRED's authenticated REST API with a
free API key (https://fred.stlouisfed.org/docs/api/api_key.html) via the
`realtime_start`/`realtime_end` parameters on `fred/series/observations` -- not used
here to avoid a hard dependency on credentials we don't have. Instead, RELEASE_LAG_DAYS
below encodes well-documented *typical* reporting lags per series (e.g. CPI ~13 days
after month-end), and every series is forward-filled only from observation_date +
that lag -- an approximation, not exact vintage data. Upgrading to the real thing later
is a small, isolated change (swap `_fetch_series` for the authenticated endpoint) once
an API key is available.
"""

import numpy as np
import pandas as pd
import requests

_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# Daily-frequency series: observation date ~= release date, no lag needed.
_DAILY_SERIES = {"DGS2", "DGS5", "DGS10", "DGS30", "FEDFUNDS", "T10YIE", "DFII10", "DTWEXBGS"}

# Approximate typical publication lag (calendar days after the observation's period-end)
# for series that are NOT same-day -- see module docstring for why these are estimates.
RELEASE_LAG_DAYS = {
    "CPIAUCSL": 13,   # BLS CPI: released ~mid-month for prior month
    "PCEPI": 30,      # BEA PCE: released ~end of month for prior month
    "UNRATE": 7,      # BLS Employment Situation: first Friday of following month
    "PAYEMS": 7,      # same release as UNRATE
    "GDPC1": 30,      # BEA advance GDP estimate: ~1 month after quarter-end
}

ZN_SERIES = ["DGS2", "DGS5", "DGS10", "DGS30", "FEDFUNDS", "CPIAUCSL", "PCEPI", "UNRATE", "PAYEMS", "GDPC1", "T10YIE"]

ZN_MACRO_COLUMNS = [
    "yield_2y", "yield_5y", "yield_10y", "yield_30y",
    "yield_curve_10y_2y", "yield_curve_chg_5d",
    "fed_funds_rate", "fed_funds_chg_20d",
    "cpi_yoy", "pce_yoy", "unemployment_rate", "payrolls_chg", "gdp_yoy",
    "breakeven_inflation_10y",
]

_cache: dict[str, pd.Series] = {}


def _fetch_series(series_id: str) -> pd.Series:
    """One FRED series as a date-indexed float Series, via the free public CSV
    endpoint (no API key required)."""
    if series_id in _cache:
        return _cache[series_id]
    resp = requests.get(_CSV_URL, params={"id": series_id}, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    series = df.set_index("date")["value"].dropna()
    _cache[series_id] = series
    return series


def _point_in_time_align(series: pd.Series, series_id: str, trading_dates: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill a macro series onto trading days, but only from the date it would
    ACTUALLY have been known (observation date + release lag), never earlier -- this
    is the point-in-time rule from problem_statement.md applied to macro data."""
    lag = pd.Timedelta(days=RELEASE_LAG_DAYS.get(series_id, 0))
    known_from = series.copy()
    known_from.index = known_from.index + lag
    # reindex onto trading days, forward-filling from the most recent *known* value
    combined_index = trading_dates.union(known_from.index).sort_values()
    aligned = known_from.reindex(combined_index).ffill()
    return aligned.reindex(trading_dates)


def fetch_zn_macro_features(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Real FRED data, point-in-time aligned to the given trading calendar.

    Important ordering: YoY/diff transforms are computed on each series' RAW,
    native-frequency history (e.g. CPI is monthly, 12-periods-back = 1 year) BEFORE
    point-in-time alignment -- not after. Computing pct_change(252) on a series
    already reindexed onto e.g. 1 year of trading days would need 252 trading days of
    *history within that window* to produce anything, which is never available for a
    short trading_dates span -- that was a real bug caught by testing on live data,
    not a hypothetical.
    """
    raw = {sid: _fetch_series(sid) for sid in ZN_SERIES}

    # Native-frequency transforms first (CPI/PCE/UNRATE/PAYEMS are monthly -> 12
    # periods back = YoY; GDP is quarterly -> 4 periods back = YoY).
    cpi_yoy_raw = raw["CPIAUCSL"].pct_change(12)
    pce_yoy_raw = raw["PCEPI"].pct_change(12)
    payrolls_chg_raw = raw["PAYEMS"].diff(1)  # month-over-month change in payrolls level
    gdp_yoy_raw = raw["GDPC1"].pct_change(4)

    aligned = {sid: _point_in_time_align(s, sid, trading_dates) for sid, s in raw.items()}
    cpi_yoy = _point_in_time_align(cpi_yoy_raw.dropna(), "CPIAUCSL", trading_dates)
    pce_yoy = _point_in_time_align(pce_yoy_raw.dropna(), "PCEPI", trading_dates)
    payrolls_chg = _point_in_time_align(payrolls_chg_raw.dropna(), "PAYEMS", trading_dates)
    gdp_yoy = _point_in_time_align(gdp_yoy_raw.dropna(), "GDPC1", trading_dates)

    out = pd.DataFrame(index=trading_dates)
    out["yield_2y"] = aligned["DGS2"]
    out["yield_5y"] = aligned["DGS5"]
    out["yield_10y"] = aligned["DGS10"]
    out["yield_30y"] = aligned["DGS30"]
    out["yield_curve_10y_2y"] = out["yield_10y"] - out["yield_2y"]
    out["yield_curve_chg_5d"] = out["yield_curve_10y_2y"].diff(5)
    out["fed_funds_rate"] = aligned["FEDFUNDS"]
    out["fed_funds_chg_20d"] = out["fed_funds_rate"].diff(20)
    out["cpi_yoy"] = cpi_yoy
    out["pce_yoy"] = pce_yoy
    out["unemployment_rate"] = aligned["UNRATE"]
    out["payrolls_chg"] = payrolls_chg
    out["gdp_yoy"] = gdp_yoy
    out["breakeven_inflation_10y"] = aligned["T10YIE"]

    return out.ffill()  # carry forward across non-trading gaps in the source series
