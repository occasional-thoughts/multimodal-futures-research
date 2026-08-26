"""Phase 19: Model 4 -- co-attention. Model 3 fused macro/news as static vectors
concatenated onto the technical GRU's final hidden state. This model instead lets
EVERY timestep of the technical sequence attend over macro and news as context (the
plan's own diagram: T as Query, [M, N] as Key/Value), so the model can learn, e.g.,
"news matters more on days where technical momentum is already turning" rather than
treating macro/news as a fixed additive offset regardless of technical state.

H = f(T, M, N, Attention(T, N))  -- per the plan's formula, with M included in the
attended context alongside N, not just concatenated separately.

Deliberately single-head, small hidden size -- consistent with every other model in
this ablation (Model 1-3), and with the standing lesson from the archived TFT
experiment: bigger attention mechanisms need more data than ~250-370 rows can support.

Run from backend/: python scripts/train_model4.py [TICKER]
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from app.data.fundamentals import CL_COLUMNS, GC_COLUMNS, fetch_cl_fundamentals, fetch_gc_fundamentals
from app.data.macro import ZN_MACRO_COLUMNS, fetch_zn_macro_features
from app.data.news_pipeline import generate_placeholder_macro_news, score_articles
from app.data.prices import fetch_price_history
from app.features.sentiment import SENTIMENT_COLUMNS
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.models.training_utils import compute_loss_norms, make_loss_fn, train_with_early_stopping
from app.targets import TARGET_COLUMNS, compute_targets

WINDOW = 20
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS, "ZN"),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS, "CL"),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS, "GC"),
}


class CoAttentionEncoder(nn.Module):
    def __init__(self, n_tech: int, n_macro: int, n_news: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.tech_gru = nn.GRU(n_tech, hidden_size, batch_first=True)
        self.macro_embed = nn.Sequential(nn.Linear(n_macro, hidden_size), nn.ReLU())
        self.news_embed = nn.Sequential(nn.Linear(n_news, hidden_size), nn.ReLU())

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**0.5

        self.drop = nn.Dropout(dropout)
        fused_size = hidden_size * 2  # final tech hidden state + attended context
        self.head_return_1d = nn.Linear(fused_size, 1)
        self.head_return_5d = nn.Linear(fused_size, 1)
        self.head_direction_5d = nn.Linear(fused_size, 1)
        self.head_volatility_5d = nn.Linear(fused_size, 1)
        self._last_attention_weights = None  # for Phase 35 inspection later

    def forward(self, x_tech, x_macro, x_news):
        tech_seq, h_tech = self.tech_gru(x_tech)  # tech_seq: (batch, window, hidden)
        macro_today, news_today = x_macro[:, -1, :], x_news[:, -1, :]
        macro_tok, news_tok = self.macro_embed(macro_today), self.news_embed(news_today)
        context = torch.stack([macro_tok, news_tok], dim=1)  # (batch, 2, hidden) -- [macro, news]

        q = self.query(tech_seq)  # (batch, window, hidden)
        k, v = self.key(context), self.value(context)  # (batch, 2, hidden)
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (batch, window, 2)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, v)  # (batch, window, hidden)
        self._last_attention_weights = attn_weights.detach()

        # Take the final timestep's tech hidden state + its attended context --
        # consistent with Model 1-3's convention of using the window-end representation
        # as the decision-day summary.
        fused = self.drop(torch.cat([tech_seq[:, -1, :], attended[:, -1, :]], dim=-1))
        return {
            "return_1d": self.head_return_1d(fused).squeeze(-1),
            "return_5d": self.head_return_5d(fused).squeeze(-1),
            "direction_5d_logit": self.head_direction_5d(fused).squeeze(-1),
            "volatility_5d": torch.nn.functional.softplus(self.head_volatility_5d(fused).squeeze(-1)),
        }


def build_sequences(tech, macro, news, targets, window):
    Xt, Xm, Xn, y = [], [], [], {c: [] for c in TARGET_COLUMNS}
    for i in range(window - 1, len(tech)):
        Xt.append(tech[i - window + 1 : i + 1])
        Xm.append(macro[i - window + 1 : i + 1])
        Xn.append(news[i - window + 1 : i + 1])
        for c in TARGET_COLUMNS:
            y[c].append(targets[c].iloc[i])
    return np.stack(Xt), np.stack(Xm), np.stack(Xn), {c: np.array(v, dtype=np.float32) for c, v in y.items()}


def train_model4(ticker: str, epochs: int = 150, patience: int = 15, seed: int = 42):
    torch.manual_seed(seed)
    macro_fetcher, macro_columns, market_code = _MACRO_FETCHERS[ticker]

    price_df = fetch_price_history(ticker, period="2y")
    tech_df = compute_indicators(price_df)
    macro_df = macro_fetcher(price_df.index)
    targets_df = compute_targets(price_df)

    articles = generate_placeholder_macro_news(price_df.index, price_df["next_return"], market_code)
    scored = score_articles(articles)
    news_df = scored.set_index("timestamp")[SENTIMENT_COLUMNS].reindex(price_df.index)

    full = pd.concat([tech_df, macro_df, news_df, targets_df], axis=1).dropna()
    n = len(full)
    train_end, val_end = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))

    tech_scaler = StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end])
    macro_scaler = StandardScaler().fit(full[macro_columns].iloc[:train_end])
    tech_scaled = tech_scaler.transform(full[TECH_COLUMNS])
    macro_scaled = macro_scaler.transform(full[macro_columns])
    news_scaled = full[SENTIMENT_COLUMNS].to_numpy()

    Xt, Xm, Xn, y = build_sequences(tech_scaled, macro_scaled, news_scaled, full[TARGET_COLUMNS], WINDOW)
    seq_train_end, seq_val_end = train_end - WINDOW + 1, val_end - WINDOW + 1

    def subset(sl):
        return (
            torch.tensor(Xt[sl], dtype=torch.float32),
            torch.tensor(Xm[sl], dtype=torch.float32),
            torch.tensor(Xn[sl], dtype=torch.float32),
            {k: torch.tensor(v[sl]) for k, v in y.items()},
        )

    Xt_train, Xm_train, Xn_train, y_train = subset(slice(0, seq_train_end))
    Xt_val, Xm_val, Xn_val, y_val = subset(slice(seq_train_end, seq_val_end))
    Xt_test, Xm_test, Xn_test, y_test = subset(slice(seq_val_end, None))
    print(f"{ticker}: {n} rows -> train={len(Xt_train)} val={len(Xt_val)} test={len(Xt_test)}")

    model = CoAttentionEncoder(n_tech=len(TECH_COLUMNS), n_macro=len(macro_columns), n_news=len(SENTIMENT_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    norms = compute_loss_norms(full.iloc[:train_end], TARGET_COLUMNS)
    compute_loss = make_loss_fn(norms)

    model = train_with_early_stopping(
        model, optimizer, compute_loss,
        (Xt_train, Xm_train, Xn_train), y_train,
        (Xt_val, Xm_val, Xn_val), y_val,
        epochs, patience,
    )
    model.eval()
    with torch.no_grad():
        train_pred = model(Xt_train, Xm_train, Xn_train)
        test_pred = model(Xt_test, Xm_test, Xn_test)  # computed last so _last_attention_weights reflects test

    train_direction = (torch.sigmoid(train_pred["direction_5d_logit"]) > 0.5).float().numpy()
    train_acc = accuracy_score(y_train["target_direction_5d"].numpy(), train_direction)
    test_direction = (torch.sigmoid(test_pred["direction_5d_logit"]) > 0.5).float().numpy()
    acc = accuracy_score(y_test["target_direction_5d"].numpy(), test_direction)

    avg_attn = model._last_attention_weights[:, -1, :].mean(dim=0)  # avg over test batch, final timestep
    print(f"{ticker} Model 4 (co-attention) train_acc={train_acc:.3f} test_acc={acc:.3f} avg_attn[macro,news]={avg_attn.tolist()}")
    return acc


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "ZN=F"
    train_model4(ticker)
