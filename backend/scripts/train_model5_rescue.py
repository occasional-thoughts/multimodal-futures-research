""""Rescue" experiment: tests the two research-motivated fixes proposed after Phase
26's walk-forward instability and the confirmed baseline collapse on more data
(PHASE_TRACKER.md):

1. Longer horizon (20 trading days, not 5) -- grounded in Moskowitz, Ooi & Pedersen
   (2012) "Time Series Momentum" (the most-replicated finding in exactly this asset
   class: futures momentum shows real, persistent signal at ~1-month+ horizons, not
   5-day) and a 2024 Journal of Futures Markets commodity-futures ML study that used
   monthly predictions.
2. CFTC Commitment of Traders (COT) positioning data as a 4th input stream (app.data.cot)
   -- the same 2024 paper's SHAP analysis flagged this as a dominant predictor for
   several commodities. Genuinely new information (who's long/short), not previously
   in the pipeline at all.

Otherwise identical to train_model4_joint.py: same joint multi-asset training with a
learned asset embedding, same co-attention mechanism (now with COT as a 3rd context
token alongside macro and news), same walk-forward evaluation methodology from Phase
26. Built as a separate script rather than editing train_model4_joint.py in place --
that file hardcodes "5d" throughout its head/target names, and risk of quietly
breaking the already-validated 5-day results wasn't worth it for a horizon change.

Run from backend/: python scripts/train_model5_rescue.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from app.config import HISTORY_PERIOD
from app.data.cot import COT_COLUMNS, fetch_cot_features
from app.data.fundamentals import CL_COLUMNS, GC_COLUMNS, fetch_cl_fundamentals, fetch_gc_fundamentals
from app.data.macro import ZN_MACRO_COLUMNS, fetch_zn_macro_features
from app.data.news_pipeline import generate_placeholder_macro_news, score_articles
from app.data.prices import fetch_price_history
from app.features.sentiment import SENTIMENT_COLUMNS
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.models.training_utils import compute_pos_weight
from app.targets import compute_targets, target_columns

WINDOW = 20
HORIZON = 20  # the research-motivated change: 20 trading days (~1 month), not 5
TARGET_COLUMNS = target_columns(HORIZON)  # [target_return_1d, target_return_20d, target_direction_20d, target_volatility_20d]
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
ASSETS = ["ZN=F", "CL=F", "GC=F"]
ASSET_IDX = {a: i for i, a in enumerate(ASSETS)}

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS, "ZN"),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS, "CL"),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS, "GC"),
}

_RET_COL, _DIR_COL, _VOL_COL = TARGET_COLUMNS[1], TARGET_COLUMNS[2], TARGET_COLUMNS[3]


class FourStreamEncoder(nn.Module):
    """Same co-attention design as Model 4/joint, extended with a 4th stream: COT
    positioning (shared across markets -- COT_COLUMNS is the same 3 engineered
    columns for every asset, unlike macro, so no per-market embedder dimension
    mismatch to handle here)."""

    def __init__(self, n_tech: int, macro_dims: dict, n_news: int, n_cot: int, n_assets: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.macro_dims = macro_dims
        self.tech_gru = nn.GRU(n_tech, hidden_size, batch_first=True)
        self.macro_embed = nn.ModuleDict({a: nn.Sequential(nn.Linear(d, hidden_size), nn.ReLU()) for a, d in macro_dims.items()})
        self.news_embed = nn.Sequential(nn.Linear(n_news, hidden_size), nn.ReLU())
        self.cot_embed = nn.Sequential(nn.Linear(n_cot, hidden_size), nn.ReLU())
        self.asset_embed = nn.Embedding(n_assets, hidden_size)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**0.5

        self.drop = nn.Dropout(dropout)
        fused_size = hidden_size * 3
        self.head_return_1d = nn.Linear(fused_size, 1)
        self.head_return_h = nn.Linear(fused_size, 1)
        self.head_direction_h = nn.Linear(fused_size, 1)
        self.head_volatility_h = nn.Linear(fused_size, 1)

    def forward(self, x_tech, x_macro, x_news, x_cot, asset_ids, asset_names):
        tech_seq, _ = self.tech_gru(x_tech)
        macro_today, news_today, cot_today = x_macro[:, -1, :], x_news[:, -1, :], x_cot[:, -1, :]

        macro_tok = torch.zeros(x_tech.shape[0], tech_seq.shape[-1])
        for asset in asset_names:
            mask = asset_ids == ASSET_IDX[asset]
            if mask.any():
                real_dim = self.macro_dims[asset]
                macro_tok[mask] = self.macro_embed[asset](macro_today[mask][:, :real_dim])

        news_tok = self.news_embed(news_today)
        cot_tok = self.cot_embed(cot_today)
        context = torch.stack([macro_tok, news_tok, cot_tok], dim=1)  # 3 context tokens now, not 2

        q = self.query(tech_seq)
        k, v = self.key(context), self.value(context)
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        attended = torch.bmm(attn_weights, v)

        asset_vec = self.asset_embed(asset_ids)
        fused = self.drop(torch.cat([tech_seq[:, -1, :], attended[:, -1, :], asset_vec], dim=-1))
        return {
            "return_1d": self.head_return_1d(fused).squeeze(-1),
            "return_h": self.head_return_h(fused).squeeze(-1),
            "direction_h_logit": self.head_direction_h(fused).squeeze(-1),
            "volatility_h": torch.nn.functional.softplus(self.head_volatility_h(fused).squeeze(-1)),
        }


_market_data_cache: dict = {}


def build_market_data(ticker: str):
    if ticker in _market_data_cache:
        return _market_data_cache[ticker]
    macro_fetcher, macro_columns, market_code = _MACRO_FETCHERS[ticker]
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)
    tech_df = compute_indicators(price_df)
    macro_df = macro_fetcher(price_df.index)
    cot_df = fetch_cot_features(ticker, price_df.index)
    targets_df = compute_targets(price_df, horizon=HORIZON)
    articles = generate_placeholder_macro_news(price_df.index, price_df["next_return"], market_code)
    news_df = score_articles(articles).set_index("timestamp")[SENTIMENT_COLUMNS].reindex(price_df.index)
    full = pd.concat([tech_df, macro_df, news_df, cot_df, targets_df, price_df[["close"]]], axis=1).dropna()
    _market_data_cache[ticker] = (full, macro_columns)
    return full, macro_columns


def build_sequences_for_market(full, macro_columns, tech_scaler, macro_scaler, ticker, window):
    tech_scaled = tech_scaler.transform(full[TECH_COLUMNS])
    macro_scaled = macro_scaler.transform(full[macro_columns])
    news_vals = full[SENTIMENT_COLUMNS].to_numpy()
    cot_vals = full[COT_COLUMNS].to_numpy()

    Xt, Xm, Xn, Xc, y, aid = [], [], [], [], {c: [] for c in TARGET_COLUMNS}, []
    for i in range(window - 1, len(full)):
        Xt.append(tech_scaled[i - window + 1 : i + 1])
        Xm.append(macro_scaled[i - window + 1 : i + 1])
        Xn.append(news_vals[i - window + 1 : i + 1])
        Xc.append(cot_vals[i - window + 1 : i + 1])
        aid.append(ASSET_IDX[ticker])
        for c in TARGET_COLUMNS:
            y[c].append(full[c].iloc[i])
    return np.stack(Xt), np.stack(Xm), np.stack(Xn), np.stack(Xc), np.array(aid), {c: np.array(v, dtype=np.float32) for c, v in y.items()}


def make_loss_fn(norms, direction_pos_weight=None):
    # direction_pos_weight: see app/models/training_utils.py's module docstring --
    # the same class-imbalance collapse found on the 10-year standard-model
    # walk-forward re-run (majority class predicted 85-100% of the time) recurred
    # here too once real 10-year data was used, so this experiment needs the
    # identical class-weighted-BCE fix, not just its own strided-window fix.
    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss(pos_weight=direction_pos_weight)

    def compute_loss(pred, y):
        r1 = torch.clamp(mse(pred["return_1d"], y["target_return_1d"]) / norms["target_return_1d"], max=5.0)
        rh = torch.clamp(mse(pred["return_h"], y[_RET_COL]) / norms[_RET_COL], max=5.0)
        vol = torch.clamp(mse(pred["volatility_h"], y[_VOL_COL]) / norms[_VOL_COL], max=5.0)
        direction = bce(pred["direction_h_logit"], y[_DIR_COL])
        return r1 + rh + direction + vol

    return compute_loss


def balanced_direction_accuracy_h(pred, y) -> float:
    """Same idea as training_utils.balanced_direction_accuracy, but generic to the
    horizon-parameterized column name (_DIR_COL = target_direction_20d here, not
    target_direction_5d) since this script isn't horizon-fixed like Models 1-4."""
    with torch.no_grad():
        pred_direction = (torch.sigmoid(pred["direction_h_logit"]) > 0.5).float().numpy()
        y_true = y[_DIR_COL].numpy()
    return balanced_accuracy_score(y_true, pred_direction)


def train_joint(epochs=150, patience=15, seed=42, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_end_frac=1.0, verbose=True):
    torch.manual_seed(seed)
    per_market, macro_dims = {}, {}
    for ticker in ASSETS:
        full, macro_columns = build_market_data(ticker)
        n = len(full)
        train_end, val_end, test_end = int(n * train_frac), int(n * (train_frac + val_frac)), int(n * test_end_frac)
        per_market[ticker] = (full, macro_columns, train_end, val_end, test_end)
        macro_dims[ticker] = len(macro_columns)
        if verbose:
            print(f"{ticker}: {n} rows, train=[0:{train_end}] val=[{train_end}:{val_end}] test=[{val_end}:{test_end}]")

    tech_scalers = {t: StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end]) for t, (full, _, train_end, _, _) in per_market.items()}
    macro_scalers = {t: StandardScaler().fit(full[cols].iloc[:train_end]) for t, (full, cols, train_end, _, _) in per_market.items()}

    splits = {"train": [], "val": [], "test": []}
    for ticker, (full, macro_columns, train_end, val_end, test_end) in per_market.items():
        Xt, Xm, Xn, Xc, aid, y = build_sequences_for_market(full, macro_columns, tech_scalers[ticker], macro_scalers[ticker], ticker, WINDOW)
        s_train_end, s_val_end, s_test_end = train_end - WINDOW + 1, val_end - WINDOW + 1, test_end - WINDOW + 1
        # Train stays dense (overlapping daily sequences are fine, even helpful, for
        # training volume). Val and test are STRIDED by HORIZON: consecutive dense
        # rows each predict a return over a window overlapping its neighbor's by up
        # to (HORIZON-1)/HORIZON days -- not independent observations. This inflated
        # or deflated apparent accuracy based on single trends dominating a
        # barely-independent sample (see PHASE_TRACKER.md's rescue-experiment entry) --
        # and it would have equally corrupted early-stopping checkpoint SELECTION if
        # val were left dense too, not just the final reported test number, so both
        # get strided, not just test.
        for name, idx in [
            ("train", np.arange(0, s_train_end)),
            ("val", np.arange(s_train_end, s_val_end, HORIZON)),
            ("test", np.arange(s_val_end, s_test_end, HORIZON)),
        ]:
            splits[name].append((Xt[idx], Xm[idx], Xn[idx], Xc[idx], aid[idx], {k: v[idx] for k, v in y.items()}))

    def pool(name):
        parts = splits[name]
        Xt = np.concatenate([p[0] for p in parts])
        Xm_padded = np.zeros((len(Xt), WINDOW, max(macro_dims.values())))
        offset = 0
        for p in parts:
            Xm_padded[offset : offset + len(p[1]), :, : p[1].shape[-1]] = p[1]
            offset += len(p[1])
        Xn = np.concatenate([p[2] for p in parts])
        Xc = np.concatenate([p[3] for p in parts])
        aid = np.concatenate([p[4] for p in parts])
        y = {c: np.concatenate([p[5][c] for p in parts]) for c in TARGET_COLUMNS}
        return (
            torch.tensor(Xt, dtype=torch.float32), torch.tensor(Xm_padded, dtype=torch.float32),
            torch.tensor(Xn, dtype=torch.float32), torch.tensor(Xc, dtype=torch.float32),
            torch.tensor(aid, dtype=torch.long), {k: torch.tensor(v) for k, v in y.items()},
        )

    Xt_train, Xm_train, Xn_train, Xc_train, aid_train, y_train = pool("train")
    Xt_val, Xm_val, Xn_val, Xc_val, aid_val, y_val = pool("val")
    Xt_test, Xm_test, Xn_test, Xc_test, aid_test, y_test = pool("test")
    if verbose:
        print(f"Pooled: train={len(Xt_train)} val={len(Xt_val)} test={len(Xt_test)}")

    model = FourStreamEncoder(n_tech=len(TECH_COLUMNS), macro_dims=macro_dims, n_news=len(SENTIMENT_COLUMNS), n_cot=len(COT_COLUMNS), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    combined_targets = pd.concat([per_market[t][0][TARGET_COLUMNS].iloc[: per_market[t][2]] for t in ASSETS])
    norms = {c: max(float(np.nanvar(combined_targets[c])), 1e-8) for c in TARGET_COLUMNS if c != "target_direction_" + str(HORIZON) + "d"}
    pos_weight = compute_pos_weight(y_train[_DIR_COL])
    compute_loss = make_loss_fn(norms, direction_pos_weight=pos_weight)

    def fwd(xt, xm, xn, xc, aid):
        return model(xt, xm, xn, xc, aid, ASSETS)

    best_val_acc, best_val_loss, best_state, no_improve = -1.0, float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(fwd(Xt_train, Xm_train, Xn_train, Xc_train, aid_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = fwd(Xt_val, Xm_val, Xn_val, Xc_val, aid_val)
            val_loss = compute_loss(val_pred, y_val).item()
            val_acc = balanced_direction_accuracy_h(val_pred, y_val)

        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
            best_val_acc, best_val_loss, best_state, no_improve = val_acc, val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} (best val balanced_direction_acc={best_val_acc:.3f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = fwd(Xt_test, Xm_test, Xn_test, Xc_test, aid_test)

    test_direction = (torch.sigmoid(test_pred["direction_h_logit"]) > 0.5).float().numpy()
    y_test_np = y_test[_DIR_COL].numpy()
    overall_acc = accuracy_score(y_test_np, test_direction)
    overall_balanced_acc = balanced_accuracy_score(y_test_np, test_direction)
    per_market_acc, per_market_balanced_acc = {}, {}
    for ticker in ASSETS:
        mask = (aid_test == ASSET_IDX[ticker]).numpy()
        if mask.sum():
            per_market_acc[ticker] = accuracy_score(y_test_np[mask], test_direction[mask])
            per_market_balanced_acc[ticker] = balanced_accuracy_score(y_test_np[mask], test_direction[mask])
        else:
            per_market_acc[ticker] = per_market_balanced_acc[ticker] = float("nan")
    if verbose:
        print(f"\nOverall test direction accuracy ({HORIZON}d horizon): acc={overall_acc:.3f} balanced_acc={overall_balanced_acc:.3f}")
        for ticker in ASSETS:
            mask = (aid_test == ASSET_IDX[ticker]).numpy()
            pred_dist = dict(zip(*np.unique(test_direction[mask], return_counts=True)))
            print(f"  {ticker}: acc={per_market_acc[ticker]:.3f} balanced_acc={per_market_balanced_acc[ticker]:.3f} pred_dist={pred_dist} ({mask.sum()} test rows)")

    return {
        "overall_acc": overall_acc, "overall_balanced_acc": overall_balanced_acc,
        "per_market_acc": per_market_acc, "per_market_balanced_acc": per_market_balanced_acc,
    }


if __name__ == "__main__":
    train_joint()
