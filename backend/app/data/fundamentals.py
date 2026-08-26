"""Market-specific fundamental features (Phase 10): CL (crude oil) and GC (gold).

CL's inventory/production data comes from the EIA's legacy DNAV .xls download --
verified to work with no API key (the newer api.eia.gov v2 API returns 403 without
one; this older endpoint doesn't require it):
  https://www.eia.gov/dnav/pet/hist_xls/WCESTUS1w.xls  -- weekly crude ending stocks
  https://www.eia.gov/dnav/pet/hist_xls/WCRFPUS2w.xls  -- weekly field production

GC's fundamentals (real yields, USD, breakeven inflation) are already available via
FRED and are fetched by reusing macro.py's series helpers rather than duplicating
fetch logic.

OPEC+ production decisions are deliberately NOT modeled as a continuous series here --
per data_architecture.md, discrete scheduled decisions belong in the Phase 5.3
economic-event schema, not a macro time series.
"""

import pandas as pd
import requests

from app.data.macro import _fetch_series, _point_in_time_align

_EIA_XLS_URL = "https://www.eia.gov/dnav/pet/hist_xls/{series}w.xls"

# EIA's Weekly Petroleum Status Report: data "as of" Friday, released the following
# Wednesday -- ~5 calendar days lag, same point-in-time discipline as macro.py.
_EIA_RELEASE_LAG_DAYS = 5

CL_COLUMNS = ["crude_inventories", "crude_inventories_chg_1w", "crude_production", "crude_production_chg_4w", "usd_index"]
GC_COLUMNS = ["real_yield_10y", "usd_index", "breakeven_inflation_10y", "fed_funds_rate"]

_eia_cache: dict[str, pd.Series] = {}


def _fetch_eia_series(series_id: str) -> pd.Series:
    if series_id in _eia_cache:
        return _eia_cache[series_id]
    resp = requests.get(_EIA_XLS_URL.format(series=series_id), timeout=15)
    resp.raise_for_status()
    sheets = pd.read_excel(pd.io.common.BytesIO(resp.content), sheet_name=None)
    data_sheet = sheets[[k for k in sheets if k.startswith("Data")][0]]
    data_sheet.columns = ["date", "value"]
    # The sheet's first two rows are metadata ("Sourcekey", "Date"/column-label
    # rows), not observations -- coerce and drop rather than hardcode a row count,
    # since that's more robust to EIA changing the sheet layout slightly.
    data_sheet["date"] = pd.to_datetime(data_sheet["date"], errors="coerce")
    data_sheet["value"] = pd.to_numeric(data_sheet["value"], errors="coerce")
    series = data_sheet.dropna().set_index("date")["value"]
    _eia_cache[series_id] = series
    return series


def _point_in_time_align_eia(series: pd.Series, trading_dates: pd.DatetimeIndex) -> pd.Series:
    lag = pd.Timedelta(days=_EIA_RELEASE_LAG_DAYS)
    known_from = series.copy()
    known_from.index = known_from.index + lag
    combined_index = trading_dates.union(known_from.index).sort_values()
    return known_from.reindex(combined_index).ffill().reindex(trading_dates)


def fetch_cl_fundamentals(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    stocks_raw = _fetch_eia_series("WCESTUS1")
    production_raw = _fetch_eia_series("WCRFPUS2")

    stocks_chg_raw = stocks_raw.diff(1)  # week-over-week change, native weekly frequency
    production_chg_raw = production_raw.diff(4)  # ~1 month (4 weekly obs)

    usd = _point_in_time_align(_fetch_series("DTWEXBGS"), "DTWEXBGS", trading_dates)

    out = pd.DataFrame(index=trading_dates)
    out["crude_inventories"] = _point_in_time_align_eia(stocks_raw, trading_dates)
    out["crude_inventories_chg_1w"] = _point_in_time_align_eia(stocks_chg_raw.dropna(), trading_dates)
    out["crude_production"] = _point_in_time_align_eia(production_raw, trading_dates)
    out["crude_production_chg_4w"] = _point_in_time_align_eia(production_chg_raw.dropna(), trading_dates)
    out["usd_index"] = usd
    return out.ffill()


def fetch_gc_fundamentals(trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    real_yield = _point_in_time_align(_fetch_series("DFII10"), "DFII10", trading_dates)
    usd = _point_in_time_align(_fetch_series("DTWEXBGS"), "DTWEXBGS", trading_dates)
    breakeven = _point_in_time_align(_fetch_series("T10YIE"), "T10YIE", trading_dates)
    fed_funds = _point_in_time_align(_fetch_series("FEDFUNDS"), "FEDFUNDS", trading_dates)

    out = pd.DataFrame(index=trading_dates)
    out["real_yield_10y"] = real_yield
    out["usd_index"] = usd
    out["breakeven_inflation_10y"] = breakeven
    out["fed_funds_rate"] = fed_funds
    return out.ffill()
