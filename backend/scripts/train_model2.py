"""Phase 17: Model 2 -- technical + macro. Two separate small encoders (one per
modality, per the plan's own diagram: market sequence -> temporal encoder, macro
sequence -> macro encoder, then fusion), concatenated before the same 4 prediction
heads used in Model 1. Everything else (window size, train/val/test split, early
stopping, evaluation) is held identical to Model 1 -- only the information set changes,
which is the entire point of the ablation study (Phase 30).

Run from backend/: python scripts/train_model2.py [TICKER]
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
from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.models.training_utils import compute_loss_norms, make_loss_fn, train_with_early_stopping
from app.targets import TARGET_COLUMNS, compute_targets

WINDOW = 20
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS),
}


class TwoStreamEncoder(nn.Module):
    """Technical stream: GRU over the full window (price data genuinely varies
    day-to-day, worth sequence-modeling). Macro stream: a small feedforward network
    over just TODAY's macro values, not a second GRU over the window.

    This replaces an earlier version that ran a full GRU over the macro window too --
    real evaluation showed that made results worse on 2/3 markets (see PHASE_TRACKER.md
    Phase 17), and the reason is structural: most macro series (FRED monthly/quarterly
    releases, EIA weekly data) barely change across a 20-day trading window once
    forward-filled, so a sequence encoder is mostly encoding near-duplicate constants
    across time steps -- extra learnable parameters with no proportional extra signal
    at ~230-370 training rows. A feedforward on the current value only is both a better
    match to how macro data actually behaves and a much smaller number of parameters."""

    def __init__(self, n_tech: int, n_macro: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.tech_gru = nn.GRU(n_tech, hidden_size, batch_first=True)
        self.macro_ffn = nn.Sequential(nn.Linear(n_macro, hidden_size), nn.ReLU(), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        fused_size = hidden_size * 2
        self.head_return_1d = nn.Linear(fused_size, 1)
        self.head_return_5d = nn.Linear(fused_size, 1)
        self.head_direction_5d = nn.Linear(fused_size, 1)
        self.head_volatility_5d = nn.Linear(fused_size, 1)

    def forward(self, x_tech, x_macro):
        _, h_tech = self.tech_gru(x_tech)
        macro_today = x_macro[:, -1, :]  # today's decision-day macro values only
        h_macro = self.macro_ffn(macro_today)
        h = self.drop(torch.cat([h_tech[-1], h_macro], dim=-1))
        return {
            "return_1d": self.head_return_1d(h).squeeze(-1),
            "return_5d": self.head_return_5d(h).squeeze(-1),
            "direction_5d_logit": self.head_direction_5d(h).squeeze(-1),
            "volatility_5d": torch.nn.functional.softplus(self.head_volatility_5d(h).squeeze(-1)),
        }


def build_sequences(tech: np.ndarray, macro: np.ndarray, targets: pd.DataFrame, window: int):
    X_tech, X_macro, y = [], [], {c: [] for c in TARGET_COLUMNS}
    for i in range(window - 1, len(tech)):
        X_tech.append(tech[i - window + 1 : i + 1])
        X_macro.append(macro[i - window + 1 : i + 1])
        for c in TARGET_COLUMNS:
            y[c].append(targets[c].iloc[i])
    return np.stack(X_tech), np.stack(X_macro), {c: np.array(v, dtype=np.float32) for c, v in y.items()}


def train_model2(ticker: str, epochs: int = 150, patience: int = 15, seed: int = 42):
    torch.manual_seed(seed)
    macro_fetcher, macro_columns = _MACRO_FETCHERS[ticker]

    price_df = fetch_price_history(ticker, period="2y")
    tech_df = compute_indicators(price_df)
    macro_df = macro_fetcher(price_df.index)
    targets_df = compute_targets(price_df)

    full = pd.concat([tech_df, macro_df, targets_df], axis=1).dropna()
    n = len(full)
    train_end, val_end = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))

    tech_scaler = StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end])
    macro_scaler = StandardScaler().fit(full[macro_columns].iloc[:train_end])
    tech_scaled = tech_scaler.transform(full[TECH_COLUMNS])
    macro_scaled = macro_scaler.transform(full[macro_columns])

    X_tech, X_macro, y = build_sequences(tech_scaled, macro_scaled, full[TARGET_COLUMNS], WINDOW)
    seq_train_end, seq_val_end = train_end - WINDOW + 1, val_end - WINDOW + 1

    def subset(sl):
        return (
            torch.tensor(X_tech[sl], dtype=torch.float32),
            torch.tensor(X_macro[sl], dtype=torch.float32),
            {k: torch.tensor(v[sl]) for k, v in y.items()},
        )

    Xt_train, Xm_train, y_train = subset(slice(0, seq_train_end))
    Xt_val, Xm_val, y_val = subset(slice(seq_train_end, seq_val_end))
    Xt_test, Xm_test, y_test = subset(slice(seq_val_end, None))
    print(f"{ticker}: {n} rows ({len(macro_columns)} macro cols) -> train={len(Xt_train)} val={len(Xt_val)} test={len(Xt_test)}")

    model = TwoStreamEncoder(n_tech=len(TECH_COLUMNS), n_macro=len(macro_columns))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    norms = compute_loss_norms(full.iloc[:train_end], TARGET_COLUMNS)
    compute_loss = make_loss_fn(norms)

    model = train_with_early_stopping(
        model, optimizer, compute_loss, (Xt_train, Xm_train), y_train, (Xt_val, Xm_val), y_val, epochs, patience
    )
    model.eval()
    with torch.no_grad():
        test_pred = model(Xt_test, Xm_test)

    with torch.no_grad():
        train_pred = model(Xt_train, Xm_train)
    train_direction = (torch.sigmoid(train_pred["direction_5d_logit"]) > 0.5).float().numpy()
    train_acc = accuracy_score(y_train["target_direction_5d"].numpy(), train_direction)

    test_direction = (torch.sigmoid(test_pred["direction_5d_logit"]) > 0.5).float().numpy()
    acc = accuracy_score(y_test["target_direction_5d"].numpy(), test_direction)
    print(f"{ticker} Model 2 (GRU, technical+macro) train_acc={train_acc:.3f} test_acc={acc:.3f}")
    return acc


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "ZN=F"
    train_model2(ticker)
