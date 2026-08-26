"""Phase 16: Model 1 -- technical data only. A temporal encoder over a rolling window
of the 22 technical features (Phase 7), producing all 4 problem_statement.md outputs
(E[1D return], E[5D return], P(5D>0), expected volatility) from one shared encoder.

Deliberately small: a single-layer GRU, not a Transformer. The archived prior project
(archive/legacy-modality-attribution-project/backend/scripts/train_tft.py) already
found that an oversized attention model collapses to chance level on ~300-450 rows of
daily futures data -- that lesson is applied here from the start (small hidden size,
real dropout, early stopping on a validation set) rather than rediscovered the hard way
a second time.

Purpose (per the plan): how much information can we extract from market behavior
alone? Evaluated against Phase 15's baseline ceiling per market, not in isolation.

Run from backend/: python scripts/train_model1.py [TICKER]
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

from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.models.training_utils import compute_loss_norms, make_loss_fn, train_with_early_stopping
from app.targets import TARGET_COLUMNS, compute_targets

WINDOW = 20  # trading days of lookback per sequence
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15


class GRUEncoder(nn.Module):
    """Small by design: 1 layer, small hidden size, dropout -- see module docstring."""

    def __init__(self, n_features: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, num_layers=1, batch_first=True, dropout=0.0)
        self.drop = nn.Dropout(dropout)
        self.head_return_1d = nn.Linear(hidden_size, 1)
        self.head_return_5d = nn.Linear(hidden_size, 1)
        self.head_direction_5d = nn.Linear(hidden_size, 1)
        self.head_volatility_5d = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, h_n = self.gru(x)
        h = self.drop(h_n[-1])
        return {
            "return_1d": self.head_return_1d(h).squeeze(-1),
            "return_5d": self.head_return_5d(h).squeeze(-1),
            "direction_5d_logit": self.head_direction_5d(h).squeeze(-1),
            "volatility_5d": torch.nn.functional.softplus(self.head_volatility_5d(h).squeeze(-1)),
        }


def build_sequences(features: np.ndarray, targets: pd.DataFrame, window: int):
    """Sequence i = features[i : i+window] -> predicts targets at row i+window-1
    (the day the window ends on, i.e. the decision day -- consistent with
    trading_frequency.md's decision-cutoff convention)."""
    X, y = [], {c: [] for c in TARGET_COLUMNS}
    for i in range(window - 1, len(features)):
        X.append(features[i - window + 1 : i + 1])
        for c in TARGET_COLUMNS:
            y[c].append(targets[c].iloc[i])
    X = np.stack(X)
    y = {c: np.array(v, dtype=np.float32) for c, v in y.items()}
    return X, y


def train_model1(ticker: str, epochs: int = 150, patience: int = 15, seed: int = 42):
    torch.manual_seed(seed)
    price_df = fetch_price_history(ticker, period="2y")
    features_df = compute_indicators(price_df)
    targets_df = compute_targets(price_df)

    full = pd.concat([features_df, targets_df], axis=1).dropna()
    n = len(full)
    train_end, val_end = int(n * TRAIN_FRAC), int(n * (TRAIN_FRAC + VAL_FRAC))

    scaler = StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end])
    scaled = scaler.transform(full[TECH_COLUMNS])

    X, y = build_sequences(scaled, full[TARGET_COLUMNS], WINDOW)
    # sequence i's label corresponds to original row i+WINDOW-1 -- shift split points
    seq_train_end, seq_val_end = train_end - WINDOW + 1, val_end - WINDOW + 1

    def subset(sl):
        return torch.tensor(X[sl], dtype=torch.float32), {k: torch.tensor(v[sl]) for k, v in y.items()}

    X_train, y_train = subset(slice(0, seq_train_end))
    X_val, y_val = subset(slice(seq_train_end, seq_val_end))
    X_test, y_test = subset(slice(seq_val_end, None))
    print(f"{ticker}: {n} rows -> sequences train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    model = GRUEncoder(n_features=len(TECH_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    norms = compute_loss_norms(full.iloc[:train_end], TARGET_COLUMNS)
    compute_loss = make_loss_fn(norms)

    model = train_with_early_stopping(
        model, optimizer, compute_loss, (X_train,), y_train, (X_val,), y_val, epochs, patience
    )
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)

    with torch.no_grad():
        train_pred = model(X_train)
    train_direction = (torch.sigmoid(train_pred["direction_5d_logit"]) > 0.5).float().numpy()
    train_acc = accuracy_score(y_train["target_direction_5d"].numpy(), train_direction)

    test_direction = (torch.sigmoid(test_pred["direction_5d_logit"]) > 0.5).float().numpy()
    acc = accuracy_score(y_test["target_direction_5d"].numpy(), test_direction)
    print(f"{ticker} Model 1 (GRU, technical-only) train_acc={train_acc:.3f} test_acc={acc:.3f}")
    return acc


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "ZN=F"
    train_model1(ticker)
