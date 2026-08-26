"""Proof-of-concept: Temporal Fusion Transformer on real numeric (price/technical)
data only -- no FinBERT, no fusion, no XAI yet. This is the "best deep learning
approach for numeric-only real-world stock prediction" step, evaluated standalone
before it gets wired into the multimodal pipeline.

TFT forecasts the continuous next-day return (not a raw up/down classification --
pytorch-forecasting's TFT is built around quantile regression), and direction
accuracy is derived from the sign of the predicted return, same as the label
definition used everywhere else in this project (label_up = next_return > 0).

Run from backend/: python scripts/train_tft.py [TICKER]
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from sklearn.metrics import accuracy_score

from app.config import ASSETS, HISTORY_PERIOD, TRAIN_FRAC, VAL_FRAC
from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators

MAX_ENCODER_LENGTH = 30  # 30-day lookback window
MAX_PREDICTION_LENGTH = 1  # predict 1 day ahead


def build_dataframe(ticker: str) -> pd.DataFrame:
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)
    tech_df = compute_indicators(price_df)
    full = tech_df.join(price_df[["next_return", "label_up"]]).dropna()
    full = full.reset_index(drop=True)
    full["time_idx"] = np.arange(len(full))  # sequential, gaps (weekends) collapsed
    full["ticker"] = ticker
    return full


def train_tft(ticker: str = "AAPL"):
    df = build_dataframe(ticker)
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    print(f"{ticker}: {n} usable days -> train {train_end}, val {val_end - train_end}, test {n - val_end}")

    training_cutoff = df["time_idx"].iloc[train_end]

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="next_return",
        group_ids=["ticker"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["ticker"],
        time_varying_unknown_reals=["next_return"] + TECH_COLUMNS,
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=False, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=32, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=32, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="adam",
    )
    print(f"TFT parameters: {sum(p.numel() for p in tft.parameters())}")

    trainer = pl.Trainer(
        max_epochs=40,
        accelerator="cpu",
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=6, mode="min")],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Evaluate directional accuracy on the held-out test range.
    # predict=False (not predict=True) is essential here: predict=True only yields
    # ONE prediction per group at the very end of the given slice, not a rolling
    # window across the whole test range -- using it silently collapsed the earlier
    # run's evaluation down to a near-empty sample, which is why it showed a bogus
    # 1.000 accuracy. return_index=True + return_y=True align every prediction to
    # its actual (group, time_idx, true value) instead of trusting positional slicing.
    test_df = df[df.time_idx > val_end - MAX_ENCODER_LENGTH].reset_index(drop=True)
    test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # predict() spins up its own internal Trainer, which defaults to auto-detecting
    # the accelerator (picked up MPS here) regardless of the training Trainer's
    # setting -- force CPU explicitly so the output tensors land somewhere .numpy()
    # can read directly.
    result = tft.predict(
        test_loader, mode="prediction", return_index=True, return_y=True, trainer_kwargs={"accelerator": "cpu"}
    )
    preds = result.output.cpu().numpy().flatten()
    actual = result.y[0].cpu().numpy().flatten()
    pred_index = result.index  # dataframe: ticker, time_idx per prediction

    print(f"{ticker}: {len(preds)} rolling test predictions (expected ~{n - val_end})")

    pred_direction = (preds > 0).astype(int)
    actual_direction = (actual > 0).astype(int)
    acc = accuracy_score(actual_direction, pred_direction)
    print(f"{ticker} TFT directional accuracy (sign of predicted next-day return): {acc:.3f}")

    # Sanity check: catch any remaining leakage/alignment bug by verifying the
    # actual values pulled from the model's own y really do match df's next_return
    # at the indices the model says it predicted.
    check = pred_index.copy()
    check["predicted_return"] = preds
    check["actual_return_from_model"] = actual
    merged = check.merge(df[["ticker", "time_idx", "next_return"]], on=["ticker", "time_idx"], how="left")
    mismatch = (merged["actual_return_from_model"] - merged["next_return"]).abs().max()
    print(f"Alignment sanity check (should be ~0): max diff = {mismatch:.6f}")

    return acc


def train_tft_joint(tickers: list[str]):
    """Same architecture, trained on all tickers pooled into one multi-series
    dataset (each ticker is its own group, sharing one set of model weights).
    Fixes the single-ticker version's real problem -- 449 rows is too little data
    for a ~22k-parameter model -- by multiplying the effective training set by
    the number of tickers, which is what TFT is actually designed to exploit
    (cross-learning shared patterns across series), not a workaround.
    """
    per_ticker_dfs = []
    for ticker in tickers:
        df = build_dataframe(ticker)
        n = len(df)
        train_end = int(n * TRAIN_FRAC)
        val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
        df["split_train_end"] = train_end
        df["split_val_end"] = val_end
        per_ticker_dfs.append(df)
        print(f"{ticker}: {n} usable days -> train {train_end}, val {val_end - train_end}, test {n - val_end}")

    full = pd.concat(per_ticker_dfs, ignore_index=True)

    train_df = pd.concat([g[g.time_idx <= g["split_train_end"].iloc[0]] for g in per_ticker_dfs], ignore_index=True)

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="next_return",
        group_ids=["ticker"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["ticker"],
        time_varying_unknown_reals=["next_return"] + TECH_COLUMNS,
        target_normalizer=GroupNormalizer(groups=["ticker"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, full, predict=False, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.15,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        optimizer="adam",
    )
    print(f"Joint TFT parameters: {sum(p.numel() for p in tft.parameters())}, training rows: {len(train_df)}")

    trainer = pl.Trainer(
        max_epochs=60,
        accelerator="cpu",
        enable_model_summary=False,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8, mode="min")],
        logger=False,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test slice per ticker: everything after that ticker's own val_end, plus
    # enough lookback before it for the encoder window.
    test_parts = []
    for g in per_ticker_dfs:
        val_end = g["split_val_end"].iloc[0]
        test_parts.append(g[g.time_idx > val_end - MAX_ENCODER_LENGTH])
    test_df = pd.concat(test_parts, ignore_index=True)

    test_dataset = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=True)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    result = tft.predict(
        test_loader, mode="prediction", return_index=True, return_y=True, trainer_kwargs={"accelerator": "cpu"}
    )
    preds = result.output.cpu().numpy().flatten()
    actual = result.y[0].cpu().numpy().flatten()
    pred_index = result.index.copy()
    pred_index["predicted_return"] = preds
    pred_index["actual_return_from_model"] = actual

    merged = pred_index.merge(full[["ticker", "time_idx", "next_return"]], on=["ticker", "time_idx"], how="left")
    mismatch = (merged["actual_return_from_model"] - merged["next_return"]).abs().max()
    print(f"Alignment sanity check (should be ~0): max diff = {mismatch:.6f}")

    merged["pred_direction"] = (merged["predicted_return"] > 0).astype(int)
    merged["actual_direction"] = (merged["next_return"] > 0).astype(int)

    overall_acc = accuracy_score(merged["actual_direction"], merged["pred_direction"])
    print(f"\nJoint TFT overall directional accuracy across {len(tickers)} tickers ({len(merged)} predictions): {overall_acc:.3f}\n")

    for ticker, group in merged.groupby("ticker"):
        acc = accuracy_score(group["actual_direction"], group["pred_direction"])
        print(f"  {ticker}: {acc:.3f} ({len(group)} predictions)")

    return overall_acc


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--joint":
        train_tft_joint([a["ticker"] for a in ASSETS])
    else:
        ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        train_tft(ticker)
