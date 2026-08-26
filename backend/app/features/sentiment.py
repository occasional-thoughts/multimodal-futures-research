import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SENTIMENT_COLUMNS = ["sent_positive", "sent_negative", "sent_neutral"]

_MODEL_NAME = "ProsusAI/finbert"
_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def score_headlines(headlines: list[str]) -> np.ndarray:
    """Returns an (n, 3) array of [positive, negative, neutral] probabilities per headline."""
    if not headlines:
        return np.zeros((0, 3))
    tokenizer, model = _load_model()
    with torch.no_grad():
        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=64)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).numpy()
    # FinBERT label order: positive, negative, neutral
    return probs


def daily_sentiment_features(news_df: pd.DataFrame) -> pd.DataFrame:
    """news_df: columns [date, headline] (possibly multiple headlines per date).

    Aggregates FinBERT class probabilities per day (mean across that day's headlines).
    """
    probs = score_headlines(news_df["headline"].tolist())
    scored = news_df[["date"]].copy()
    scored[SENTIMENT_COLUMNS] = probs
    daily = scored.groupby("date")[SENTIMENT_COLUMNS].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily
