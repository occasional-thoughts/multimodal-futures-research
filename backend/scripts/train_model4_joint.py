"""Phase 20: asset-specific representation, added on top of Model 4's co-attention
architecture. Models 1-4 so far trained a completely separate model per market --
this trains ONE model jointly across ZN, CL, GC, with a learned per-asset embedding
so the model explicitly knows "this is ZN" rather than the model weights implicitly
encoding market identity through 3 unrelated sets of parameters.

Architectural note: macro column COUNTS differ per market (14 for ZN, 5 for CL, 4 for
GC -- see data_architecture.md/macro.py/fundamentals.py), so a fully shared macro
input layer isn't possible without inventing a fake unified macro schema. Kept
honest instead: each market keeps its own small macro embedder (mapping its own
column count to the same hidden_size), while the technical GRU, news embedder,
attention mechanism, asset embedding, and prediction heads are ALL shared across
markets. This also revisits the archived TFT lesson (joint multi-asset training
gave ~10x more effective training data) -- here it's a real ~3x (three markets
pooled instead of one), tested honestly rather than assumed to help.

Run from backend/: python scripts/train_model4_joint.py
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

from app.config import HISTORY_PERIOD
from app.data.fundamentals import CL_COLUMNS, GC_COLUMNS, fetch_cl_fundamentals, fetch_gc_fundamentals
from app.data.macro import ZN_MACRO_COLUMNS, fetch_zn_macro_features
from app.data.news_pipeline import generate_placeholder_macro_news, score_articles
from app.data.prices import fetch_price_history
from app.features.sentiment import SENTIMENT_COLUMNS
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.models.training_utils import compute_loss_norms, make_loss_fn
from app.targets import TARGET_COLUMNS, compute_targets

WINDOW = 20
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
ASSETS = ["ZN=F", "CL=F", "GC=F"]
ASSET_IDX = {a: i for i, a in enumerate(ASSETS)}

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS, "ZN"),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS, "CL"),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS, "GC"),
}


class JointCoAttentionEncoder(nn.Module):
    """Shared across all assets: tech_gru, news_embed, attention, asset_embed, heads.
    Per-asset: macro_embed (differing input dims -- see module docstring)."""

    def __init__(self, n_tech: int, macro_dims: dict, n_news: int, n_assets: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.macro_dims = macro_dims  # each market's REAL column count, before zero-padding to a common width
        self.tech_gru = nn.GRU(n_tech, hidden_size, batch_first=True)
        self.macro_embed = nn.ModuleDict(
            {asset: nn.Sequential(nn.Linear(dim, hidden_size), nn.ReLU()) for asset, dim in macro_dims.items()}
        )
        self.news_embed = nn.Sequential(nn.Linear(n_news, hidden_size), nn.ReLU())
        self.asset_embed = nn.Embedding(n_assets, hidden_size)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**0.5

        self.drop = nn.Dropout(dropout)
        fused_size = hidden_size * 3  # tech final state + attended context + asset embedding
        self.head_return_1d = nn.Linear(fused_size, 1)
        self.head_return_5d = nn.Linear(fused_size, 1)
        self.head_direction_5d = nn.Linear(fused_size, 1)
        self.head_volatility_5d = nn.Linear(fused_size, 1)

    def forward(self, x_tech, x_macro, x_news, asset_ids, asset_names):
        tech_seq, _ = self.tech_gru(x_tech)
        macro_today, news_today = x_macro[:, -1, :], x_news[:, -1, :]

        # Route each row through its own market's macro embedder (input dims differ,
        # so this can't be a single shared linear layer) -- but the OUTPUT feeds into
        # fully shared downstream layers.
        macro_tok = torch.zeros(x_tech.shape[0], tech_seq.shape[-1])
        for asset in asset_names:
            mask = asset_ids == ASSET_IDX[asset]
            if mask.any():
                # macro_today is padded to the max column count across markets --
                # slice back down to this market's own real columns before its embedder
                real_dim = self.macro_dims[asset]
                macro_tok[mask] = self.macro_embed[asset](macro_today[mask][:, :real_dim])

        news_tok = self.news_embed(news_today)
        context = torch.stack([macro_tok, news_tok], dim=1)

        q = self.query(tech_seq)
        k, v = self.key(context), self.value(context)
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        attended = torch.bmm(attn_weights, v)

        asset_vec = self.asset_embed(asset_ids)
        fused = self.drop(torch.cat([tech_seq[:, -1, :], attended[:, -1, :], asset_vec], dim=-1))
        return {
            "return_1d": self.head_return_1d(fused).squeeze(-1),
            "return_5d": self.head_return_5d(fused).squeeze(-1),
            "direction_5d_logit": self.head_direction_5d(fused).squeeze(-1),
            "volatility_5d": torch.nn.functional.softplus(self.head_volatility_5d(fused).squeeze(-1)),
        }


_market_data_cache: dict = {}  # avoids re-fetching Yahoo/FRED/EIA data across walk-forward folds (Phase 26)


def build_market_data(ticker: str):
    if ticker in _market_data_cache:
        return _market_data_cache[ticker]
    macro_fetcher, macro_columns, market_code = _MACRO_FETCHERS[ticker]
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)
    tech_df = compute_indicators(price_df)
    macro_df = macro_fetcher(price_df.index)
    targets_df = compute_targets(price_df)
    articles = generate_placeholder_macro_news(price_df.index, price_df["next_return"], market_code)
    news_df = score_articles(articles).set_index("timestamp")[SENTIMENT_COLUMNS].reindex(price_df.index)
    full = pd.concat([tech_df, macro_df, news_df, targets_df, price_df[["close"]]], axis=1).dropna()
    _market_data_cache[ticker] = (full, macro_columns)
    return full, macro_columns


def build_sequences_for_market(full: pd.DataFrame, macro_columns: list, tech_scaler, macro_scaler, ticker: str, window: int):
    tech_scaled = tech_scaler.transform(full[TECH_COLUMNS])
    macro_scaled = macro_scaler.transform(full[macro_columns])
    news_vals = full[SENTIMENT_COLUMNS].to_numpy()

    Xt, Xm, Xn, y, asset_ids, dates, prices, day_idx = [], [], [], {c: [] for c in TARGET_COLUMNS}, [], [], [], []
    for i in range(window - 1, len(full)):
        Xt.append(tech_scaled[i - window + 1 : i + 1])
        Xm.append(macro_scaled[i - window + 1 : i + 1])
        Xn.append(news_vals[i - window + 1 : i + 1])
        asset_ids.append(ASSET_IDX[ticker])
        dates.append(full.index[i])
        prices.append(full["close"].iloc[i])
        day_idx.append(i)  # position within this market's own full series -- used by Portfolio for holding-period math
        for c in TARGET_COLUMNS:
            y[c].append(full[c].iloc[i])
    return (
        np.stack(Xt), np.stack(Xm), np.stack(Xn), np.array(asset_ids),
        {c: np.array(v, dtype=np.float32) for c, v in y.items()},
        np.array(dates), np.array(prices, dtype=np.float64), np.array(day_idx),
    )


def train_joint(
    epochs: int = 150, patience: int = 15, seed: int = 42,
    train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC, test_end_frac: float = 1.0,
    verbose: bool = True,
):
    """train_frac/val_frac/test_end_frac (all as fractions of each market's own full
    series) let this be called with a SLIDING window -- the walk-forward mechanism
    for Phase 26, so the model can be retrained and re-evaluated on successive time
    periods instead of trusting a single static split."""
    torch.manual_seed(seed)
    per_market, macro_dims = {}, {}
    for ticker in ASSETS:
        full, macro_columns = build_market_data(ticker)
        n = len(full)
        train_end, val_end, test_end = int(n * train_frac), int(n * (train_frac + val_frac)), int(n * test_end_frac)
        per_market[ticker] = (full, macro_columns, train_end, val_end, test_end)
        macro_dims[ticker] = len(macro_columns)
        if verbose:
            print(f"{ticker}: {n} rows, {len(macro_columns)} macro cols, train=[0:{train_end}] val=[{train_end}:{val_end}] test=[{val_end}:{test_end}]")

    # Fit scalers per market on that market's own train slice (fair -- no cross-market leakage)
    tech_scalers = {t: StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end]) for t, (full, _, train_end, _, _) in per_market.items()}
    macro_scalers = {
        t: StandardScaler().fit(full[cols].iloc[:train_end]) for t, (full, cols, train_end, _, _) in per_market.items()
    }

    splits = {"train": [], "val": [], "test": []}
    for ticker, (full, macro_columns, train_end, val_end, test_end) in per_market.items():
        Xt, Xm, Xn, aid, y, dates, prices, day_idx = build_sequences_for_market(
            full, macro_columns, tech_scalers[ticker], macro_scalers[ticker], ticker, WINDOW
        )
        seq_train_end, seq_val_end, seq_test_end = train_end - WINDOW + 1, val_end - WINDOW + 1, test_end - WINDOW + 1
        for split_name, sl in [("train", slice(0, seq_train_end)), ("val", slice(seq_train_end, seq_val_end)), ("test", slice(seq_val_end, seq_test_end))]:
            splits[split_name].append((Xt[sl], Xm[sl], Xn[sl], aid[sl], {k: v[sl] for k, v in y.items()}, dates[sl], prices[sl], day_idx[sl]))

    def pool(split_name):
        parts = splits[split_name]
        Xt = np.concatenate([p[0] for p in parts])
        Xm_padded = np.zeros((len(Xt), WINDOW, max(macro_dims.values())))
        offset = 0
        for Xt_p, Xm_p, Xn_p, aid_p, y_p, dates_p, prices_p, day_idx_p in parts:
            Xm_padded[offset : offset + len(Xm_p), :, : Xm_p.shape[-1]] = Xm_p
            offset += len(Xm_p)
        Xn = np.concatenate([p[2] for p in parts])
        aid = np.concatenate([p[3] for p in parts])
        y = {c: np.concatenate([p[4][c] for p in parts]) for c in TARGET_COLUMNS}
        dates = np.concatenate([p[5] for p in parts])
        prices = np.concatenate([p[6] for p in parts])
        day_idx = np.concatenate([p[7] for p in parts])
        return (
            torch.tensor(Xt, dtype=torch.float32),
            torch.tensor(Xm_padded, dtype=torch.float32),
            torch.tensor(Xn, dtype=torch.float32),
            torch.tensor(aid, dtype=torch.long),
            {k: torch.tensor(v) for k, v in y.items()},
            dates, prices, day_idx,
        )

    # NOTE: macro is zero-padded to the max column count across markets so the pooled
    # tensor has one consistent shape -- each market's own macro_embed submodule only
    # ever reads its own real columns via the asset-id mask in forward(), so the
    # padding is inert, not a leak of one market's schema into another's.
    Xt_train, Xm_train, Xn_train, aid_train, y_train, dates_train, prices_train, day_idx_train = pool("train")
    Xt_val, Xm_val, Xn_val, aid_val, y_val, dates_val, prices_val, day_idx_val = pool("val")
    Xt_test, Xm_test, Xn_test, aid_test, y_test, dates_test, prices_test, day_idx_test = pool("test")
    if verbose:
        print(f"Pooled: train={len(Xt_train)} val={len(Xt_val)} test={len(Xt_test)}")

    model = JointCoAttentionEncoder(n_tech=len(TECH_COLUMNS), macro_dims=macro_dims, n_news=len(SENTIMENT_COLUMNS), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    combined_targets = pd.concat([per_market[t][0][TARGET_COLUMNS].iloc[: per_market[t][2]] for t in ASSETS])
    norms = compute_loss_norms(combined_targets, TARGET_COLUMNS)
    compute_loss = make_loss_fn(norms)

    def forward_train(x_tech, x_macro, x_news, aid):
        return model(x_tech, x_macro, x_news, aid, ASSETS)

    best_val_acc, best_val_loss, best_state, no_improve = -1.0, float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = forward_train(Xt_train, Xm_train, Xn_train, aid_train)
        loss = compute_loss(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = forward_train(Xt_val, Xm_val, Xn_val, aid_val)
            val_loss = compute_loss(val_pred, y_val).item()
            val_acc = ((torch.sigmoid(val_pred["direction_5d_logit"]) > 0.5).float() == y_val["target_direction_5d"]).float().mean().item()

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss = val_acc, val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val direction_acc={best_val_acc:.3f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = forward_train(Xt_val, Xm_val, Xn_val, aid_val)
        test_pred = forward_train(Xt_test, Xm_test, Xn_test, aid_test)

    test_direction = (torch.sigmoid(test_pred["direction_5d_logit"]) > 0.5).float().numpy()
    overall_acc = accuracy_score(y_test["target_direction_5d"].numpy(), test_direction)
    per_market_acc = {}
    for ticker in ASSETS:
        mask = (aid_test == ASSET_IDX[ticker]).numpy()
        per_market_acc[ticker] = accuracy_score(y_test["target_direction_5d"].numpy()[mask], test_direction[mask]) if mask.sum() else float("nan")
    if verbose:
        print(f"\nJoint model overall test direction accuracy: {overall_acc:.3f}")
        for ticker in ASSETS:
            mask = (aid_test == ASSET_IDX[ticker]).numpy()
            print(f"  {ticker}: {per_market_acc[ticker]:.3f} ({mask.sum()} test rows)")

    return {
        "val_pred": val_pred, "y_val": y_val, "aid_val": aid_val,
        "test_pred": test_pred, "y_test": y_test, "aid_test": aid_test,
        "dates_test": dates_test, "prices_test": prices_test, "day_idx_test": day_idx_test,
        "overall_acc": overall_acc, "per_market_acc": per_market_acc,
    }


if __name__ == "__main__":
    train_joint()
