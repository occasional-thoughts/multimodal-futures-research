import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import ARTIFACTS_DIR, TICKERS
from app.data.prices import fetch_recent_news
from app.explain.counterfactual import generate_counterfactual
from app.explain.shap_utils import top_features
from app.pipeline import load_bundle

app = FastAPI(title="Modality-Attribution Robustness API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache: dict = {}


def _get_bundle(ticker: str) -> dict:
    ticker = ticker.upper()
    if ticker not in _cache:
        path = ARTIFACTS_DIR / f"{ticker}.joblib"
        if not path.exists():
            raise HTTPException(404, f"No trained artifact for {ticker}. Run scripts/run_pipeline.py first.")
        _cache[ticker] = load_bundle(ticker)
    return _cache[ticker]


@app.get("/stocks")
def list_stocks():
    out = []
    for ticker in TICKERS:
        path = ARTIFACTS_DIR / f"{ticker}.joblib"
        if not path.exists():
            continue
        bundle = _get_bundle(ticker)
        test_dates = bundle["dates"][bundle["test_sl"]]
        out.append(
            {
                "ticker": ticker,
                "company_name": bundle["company_name"],
                "test_start": str(test_dates[0].date()),
                "test_end": str(test_dates[-1].date()),
            }
        )
    return out


@app.get("/dates/{ticker}")
def list_test_dates(ticker: str):
    bundle = _get_bundle(ticker)
    test_dates = bundle["dates"][bundle["test_sl"]]
    regimes = bundle["regime_labels"][bundle["test_sl"]]
    return [{"date": str(d.date()), "regime": r} for d, r in zip(test_dates, regimes)]


def _test_index_for_date(bundle: dict, date: str) -> int:
    test_dates = bundle["dates"][bundle["test_sl"]]
    matches = [i for i, d in enumerate(test_dates) if str(d.date()) == date]
    if not matches:
        raise HTTPException(404, f"{date} is not in the test period for this ticker")
    return matches[0]


@app.get("/predict/{ticker}/{date}")
def predict(ticker: str, date: str):
    bundle = _get_bundle(ticker)
    idx = _test_index_for_date(bundle, date)
    test_sl = bundle["test_sl"]
    regime = bundle["regime_labels"][test_sl][idx]

    results = {}
    for name, v in bundle["variants"].items():
        fo_test_matrix = v["fusion_output_all"].matrix[test_sl][idx : idx + 1]
        model = v["model"]
        pred = int(model.predict(fo_test_matrix)[0])
        proba = float(model.predict_proba(fo_test_matrix)[0][1])
        results[name] = {"prediction": "up" if pred == 1 else "down", "prob_up": proba, "accuracy": v["accuracy"]}

    return {"ticker": ticker, "date": date, "regime": regime, "predictions": results}


@app.get("/explain/{ticker}/{date}/{variant}")
def explain(ticker: str, date: str, variant: str):
    bundle = _get_bundle(ticker)
    idx = _test_index_for_date(bundle, date)
    v = bundle["variants"].get(variant)
    if v is None:
        raise HTTPException(404, f"Unknown fusion variant {variant}")

    shap_row = v["shap_values_test"][idx]
    news_share = float(v["news_share_test"][idx])

    return {
        "ticker": ticker,
        "date": date,
        "variant": variant,
        "news_share": news_share,
        "tech_share": 1 - news_share,
        "top_features": top_features(shap_row, v["feature_names"]),
    }


@app.get("/counterfactual/{ticker}/{date}/{variant}")
def counterfactual(ticker: str, date: str, variant: str):
    bundle = _get_bundle(ticker)
    idx = _test_index_for_date(bundle, date)
    v = bundle["variants"].get(variant)
    if v is None:
        raise HTTPException(404, f"Unknown fusion variant {variant}")

    test_sl = bundle["test_sl"]
    fo_all_matrix = v["fusion_output_all"].matrix
    row = fo_all_matrix[test_sl][idx]
    background = fo_all_matrix[bundle["train_sl"]]
    train_labels = bundle["labels"][bundle["train_sl"]]

    result = generate_counterfactual(v["model"], row, v["feature_names"], background, train_labels)
    return {"ticker": ticker, "date": date, "variant": variant, **result}


def _nan_to_none(obj):
    """Some (fusion-pair, regime) cells have too few matched test instances and
    fall back to float('nan') -- raw NaN isn't valid JSON, so it must be swapped
    for null before this can be serialized in a response."""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


@app.get("/consistency-report/{ticker}")
def consistency_report(ticker: str):
    bundle = _get_bundle(ticker)
    return _nan_to_none(bundle["consistency_report"])


@app.get("/live-news/{ticker}")
def live_news(ticker: str):
    return fetch_recent_news(ticker)
