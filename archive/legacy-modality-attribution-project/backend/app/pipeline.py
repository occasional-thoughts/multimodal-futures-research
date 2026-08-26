import numpy as np
import pandas as pd

from app.config import ARTIFACTS_DIR, HISTORY_PERIOD, TRAIN_FRAC, VAL_FRAC
from app.data.news import generate_placeholder_news
from app.data.prices import fetch_price_history
from app.explain.shap_utils import compute_shap, modality_attribution
from app.features.sentiment import SENTIMENT_COLUMNS, daily_sentiment_features
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.fusion.coattention import CoAttentionFusion
from app.fusion.concat import ConcatFusion
from app.fusion.weighted import WeightedFusion
from app.models.train import evaluate, train_xgb
from app.regime import label_regimes
from app.consistency import build_consistency_report

import joblib

FUSION_CLASSES = {"concat": ConcatFusion, "weighted": WeightedFusion, "coattention": CoAttentionFusion}


def _chronological_split(n: int):
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def run_ticker_pipeline(ticker: str, company_name: str | None = None, asset_type: str = "equity") -> dict:
    company_name = company_name or ticker
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)

    tech_df = compute_indicators(price_df)
    news_raw = generate_placeholder_news(price_df.index, price_df["next_return"], company_name, asset_type=asset_type)
    sentiment_df = daily_sentiment_features(news_raw)
    sentiment_df = sentiment_df.reindex(price_df.index).ffill().fillna(1 / 3)

    full = pd.concat([tech_df, sentiment_df], axis=1).join(price_df[["label_up", "return"]])
    full = full.dropna()

    tech = full[TECH_COLUMNS].to_numpy()
    news = full[SENTIMENT_COLUMNS].to_numpy()
    labels = full["label_up"].to_numpy()
    dates = full.index

    train_sl, val_sl, test_sl = _chronological_split(len(full))
    regime_labels = label_regimes(full["return"]).to_numpy()

    variants = {}
    news_share_by_variant = {}

    for name, cls in FUSION_CLASSES.items():
        fusion = cls().fit(tech[train_sl], news[train_sl], labels[train_sl], TECH_COLUMNS, SENTIMENT_COLUMNS)
        fo_train = fusion.transform(tech[train_sl], news[train_sl], TECH_COLUMNS, SENTIMENT_COLUMNS)
        fo_test = fusion.transform(tech[test_sl], news[test_sl], TECH_COLUMNS, SENTIMENT_COLUMNS)
        fo_all = fusion.transform(tech, news, TECH_COLUMNS, SENTIMENT_COLUMNS)

        model = train_xgb(fo_train.matrix, labels[train_sl])
        accuracy = evaluate(model, fo_test.matrix, labels[test_sl])

        shap_values_test = compute_shap(model, fo_test)
        tech_attr, news_attr, news_share = modality_attribution(shap_values_test, fo_test)

        variants[name] = {
            "fusion": fusion,
            "model": model,
            "accuracy": accuracy,
            "fusion_output_all": fo_all,
            "feature_names": fo_test.feature_names,
            "tech_group": fo_test.tech_group,
            "news_group": fo_test.news_group,
            "shap_values_test": shap_values_test,
            "news_share_test": news_share,
        }
        news_share_by_variant[name] = news_share

    test_regime_labels = regime_labels[test_sl]
    consistency_report = build_consistency_report(news_share_by_variant, test_regime_labels)

    bundle = {
        "ticker": ticker,
        "company_name": company_name,
        "dates": dates,
        "tech": tech,
        "news": news,
        "labels": labels,
        "regime_labels": regime_labels,
        "train_sl": train_sl,
        "val_sl": val_sl,
        "test_sl": test_sl,
        "variants": variants,
        "consistency_report": consistency_report,
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ARTIFACTS_DIR / f"{ticker}.joblib")
    return bundle


def load_bundle(ticker: str) -> dict:
    return joblib.load(ARTIFACTS_DIR / f"{ticker}.joblib")
