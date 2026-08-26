"""CFTC Commitment of Traders (COT) positioning data -- a 4th information category
(who's long/short: speculators vs. commercial hedgers), distinct from technical,
macro, and news. Added after research (see PHASE_TRACKER.md) found this specifically
flagged as a dominant predictor for commodity futures in a 2024 peer-reviewed study
(Wang, Journal of Futures Markets), via SHAP feature-importance analysis.

Free, public-domain, no API key, via CFTC's Socrata-based public reporting API
(publicreporting.cftc.gov). Two different report formats depending on market:
- ZN uses the "Traders in Financial Futures" (TFF) report -- Treasury futures are
  classified there, not in the Legacy report at all (verified empirically: a Legacy
  API query for "TREASURY" returned zero rows; TFF returned the real contract).
- CL and GC use the "Legacy" report -- commercial vs. non-commercial classification.

Point-in-time discipline (matching macro.py/fundamentals.py): COT data is collected
"as of" Tuesday close and publicly released the following Friday at 3:30pm ET --
RELEASE_LAG_DAYS=3 encodes this documented schedule, same honesty convention as
macro.py's RELEASE_LAG_DAYS (a known typical lag, not authenticated vintage data).
"""

import pandas as pd
import requests

_BASE_URL = "https://publicreporting.cftc.gov/resource/{dataset}.json"
_LEGACY_DATASET = "6dca-aqww"
_TFF_DATASET = "gpe5-46if"

RELEASE_LAG_DAYS = 3  # Tuesday "as of" -> released the following Friday

COT_COLUMNS = ["cot_speculator_net_pct_oi", "cot_commercial_net_pct_oi", "cot_speculator_net_chg_1w"]

_MARKET_CONFIG = {
    "ZN=F": {"dataset": _TFF_DATASET, "code": "043602", "report_type": "tff"},
    "CL=F": {"dataset": _LEGACY_DATASET, "code": "067651", "report_type": "legacy"},
    "GC=F": {"dataset": _LEGACY_DATASET, "code": "088691", "report_type": "legacy"},
}

_cache: dict = {}


def _fetch_raw(ticker: str) -> pd.DataFrame:
    if ticker in _cache:
        return _cache[ticker]
    cfg = _MARKET_CONFIG[ticker]
    resp = requests.get(
        _BASE_URL.format(dataset=cfg["dataset"]),
        params={"$where": f"cftc_contract_market_code='{cfg['code']}'", "$order": "report_date_as_yyyy_mm_dd ASC", "$limit": 5000},
        timeout=20,
    )
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    _cache[ticker] = df
    return df


def _speculator_commercial_net_pct(df: pd.DataFrame, report_type: str) -> pd.DataFrame:
    oi = df["open_interest_all"].astype(float)
    if report_type == "legacy":
        spec_net = df["noncomm_positions_long_all"].astype(float) - df["noncomm_positions_short_all"].astype(float)
        comm_net = df["comm_positions_long_all"].astype(float) - df["comm_positions_short_all"].astype(float)
    else:  # tff -- "leveraged money" is the closest analog to speculative/hot-money positioning;
        # "asset manager" (real-money, longer-horizon institutional) is the closest analog to
        # commercial/hedging-style stable positioning, though the classifications aren't identical
        # across report types -- flagged here, not glossed over as if they were the same thing.
        spec_net = df["lev_money_positions_long"].astype(float) - df["lev_money_positions_short"].astype(float)
        comm_net = df["asset_mgr_positions_long"].astype(float) - df["asset_mgr_positions_short"].astype(float)

    out = pd.DataFrame(index=df["report_date"])
    out["cot_speculator_net_pct_oi"] = (spec_net / oi).to_numpy()
    out["cot_commercial_net_pct_oi"] = (comm_net / oi).to_numpy()
    out["cot_speculator_net_chg_1w"] = out["cot_speculator_net_pct_oi"].diff().to_numpy()
    return out


def fetch_cot_features(ticker: str, trading_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Point-in-time aligned onto `trading_dates`, same forward-fill-from-release-date
    pattern as macro.py/fundamentals.py."""
    cfg = _MARKET_CONFIG[ticker]
    raw = _fetch_raw(ticker)
    weekly = _speculator_commercial_net_pct(raw, cfg["report_type"])

    known_from = weekly.copy()
    known_from.index = known_from.index + pd.Timedelta(days=RELEASE_LAG_DAYS)
    combined_index = trading_dates.union(known_from.index).sort_values()
    aligned = known_from.reindex(combined_index).ffill().reindex(trading_dates)
    return aligned.ffill()
