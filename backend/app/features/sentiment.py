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


EMBED_DIM = 768  # FinBERT (bert-base) hidden size


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


def embed_headlines(headlines: list[str]) -> np.ndarray:
    """Returns an (n, 768) array of FinBERT SEMANTIC embeddings -- the actual encoder
    representation, not the 3-class sentiment probabilities `score_headlines` returns.

    Gap found and fixed here (see PHASE_TRACKER.md's Model 7 section, and
    github.com/sarthak-12/thesis-dsaa "STONK" (arXiv:2508.13327) for the design this
    follows): every prior model in this project discarded FinBERT's semantic content
    entirely and kept only its 3-way classifier softmax -- a real information loss
    that the original architecture diagram for this project always called for
    (Sentiment AND Semantic embedding as two separate branches recombined into one
    News representation), never previously built.

    Mean-pools the LAST hidden layer over real (non-padding) tokens, using the same
    forward pass already run for score_headlines rather than loading a second model
    -- `output_hidden_states=True` is the only change needed; the classification head
    on top is unused here, only the encoder body's own representation is kept.
    """
    if not headlines:
        return np.zeros((0, EMBED_DIM))
    tokenizer, model = _load_model()
    with torch.no_grad():
        inputs = tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=64)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (n, seq_len, 768)
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # (n, seq_len, 1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    return pooled.numpy()


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
