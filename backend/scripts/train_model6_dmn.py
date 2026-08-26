"""Model 6: Deep Momentum Network (Sharpe-ratio-optimized position sizing).

Every model so far (1-5) framed the problem as CLASSIFYING next-period direction, and
the honest, verified result (PHASE_TRACKER.md's 10-year re-run section) is that this
framing has no real edge on ZN/CL/GC at either a 5-day or 20-day horizon -- balanced
accuracy sits within noise of 0.5. That is a genuine, defensible null result on that
specific framing. It is not the same as "no working approach exists in the published
literature for this exact problem" -- it means classification-of-direction specifically
doesn't work here, which is a well-known property of near-efficient daily futures
markets (see e.g. Moskowitz, Ooi & Pedersen 2012).

This model instead reimplements the approach from real, published, open-sourced
research that gets genuine results on precisely this problem (deep learning + trend
signals + real futures):

- Lim, Zohren & Roberts (2019), "Enhancing Time-Series Momentum Strategies Using Deep
  Neural Networks" (arXiv:1904.04912) -- the original Deep Momentum Network (DMN):
  don't classify direction, output a continuous POSITION SIZE directly, trained by
  optimizing a differentiable SHARPE RATIO loss rather than any accuracy-based
  criterion. Backtested on 88 continuous futures contracts; reported ~2x Sharpe
  improvement over classical time-series momentum, before costs.
- Wood, Giegerich, Roberts & Zohren (2021), "Trading with the Momentum Transformer"
  (arXiv:2112.08534, code: github.com/kieranjwood/trading-momentum-transformer) --
  extends the DMN with attention; the exact `SharpeLoss` reimplemented below is a
  direct PyTorch port of that repo's Keras `SharpeLoss` class (verified by fetching
  the actual source, not reconstructed from the paper text alone).
- Baz, Granger, Harvey, Le Roux & Rattray (2015) MACD trend-indicator formula (the
  `macd_trend_*` features in app/features/technical.py) -- the input signal both of
  the above papers build on.

Why this can plausibly work where classification didn't, not just "a different model
to try": Sharpe-ratio optimization does not require >50% directional accuracy. A model
that sizes positions small during choppy/uncertain periods and large during clear
trends can have a positive Sharpe ratio even with a directional hit rate at or below
50%, because the loss directly rewards risk-adjusted P&L, not correct-call frequency.
This is the entire point of the cited papers, not a claim invented for this project.

Evaluated against two honest, non-ML baselines computed the identical way (same test
dates, same portfolio-averaging), not just reported in isolation:
1. Buy-and-hold (constant position = +1).
2. Classical time-series momentum: position = the SAME macd_trend features this model
   consumes, but used directly as a hand-built rule (mean of the three trend scores,
   clipped to [-1, 1]) instead of learned -- this is precisely the classical baseline
   the cited papers themselves compare a DMN against, so any reported improvement here
   means the deep model over the classical rule, on equal footing.

NEWS FEATURES DELIBERATELY EXCLUDED -- a real bug caught before trusting the first
result, not a design choice made up front. The first run of this model reported an
out-of-sample portfolio Sharpe of 11.6 (buy-and-hold: 1.09, classical trend: 0.67) --
implausibly high for any real futures strategy, so it was root-caused instead of
reported. `app/data/news_pipeline.py`'s `generate_placeholder_macro_news()` builds its
synthetic sentiment signal directly from `price_df["next_return"]`, which is bit-for-
bit the same quantity as `target_return_1d` (verified: `next_return = return.shift(-1)`
in prices.py; `target_return_1d = close.shift(-1)/close - 1` in targets.py -- the same
value two ways). Every earlier model (1-5) used that same feature but predicted a
5-day or 20-day horizon, which dilutes a 1-day leak enough that it didn't produce an
obviously-impossible result -- this model trades `target_return_1d` DIRECTLY, so the
leak fed the model almost exactly the answer to the question it was scored on. Fixed
by dropping the fabricated news stream entirely for this model (co-attention now uses
2 context tokens -- macro, COT -- not 3) rather than trusting a diluted version of the
same leak. This is also a retroactive caveat on every earlier model's news-derived
results, even though their near-chance balanced accuracy suggests the diluted leak
didn't meaningfully help them in practice -- see PHASE_TRACKER.md.

Run from backend/: python scripts/train_model6_dmn.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from app.config import HISTORY_PERIOD
from app.data.cot import COT_COLUMNS, fetch_cot_features
from app.data.fundamentals import CL_COLUMNS, GC_COLUMNS, fetch_cl_fundamentals, fetch_gc_fundamentals
from app.data.macro import ZN_MACRO_COLUMNS, fetch_zn_macro_features
from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.targets import compute_targets

WINDOW = 20
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
ASSETS = ["ZN=F", "CL=F", "GC=F"]
ASSET_IDX = {a: i for i, a in enumerate(ASSETS)}
TREND_COLS = ["macd_trend_8_24", "macd_trend_16_48", "macd_trend_32_96"]

_MACRO_FETCHERS = {
    "ZN=F": (fetch_zn_macro_features, ZN_MACRO_COLUMNS, "ZN"),
    "CL=F": (fetch_cl_fundamentals, CL_COLUMNS, "CL"),
    "GC=F": (fetch_gc_fundamentals, GC_COLUMNS, "GC"),
}


class SharpeLoss(nn.Module):
    """Direct PyTorch port of kieranjwood/trading-momentum-transformer's
    mom_trans/deep_momentum_network.py SharpeLoss (Keras), verified against the
    actual fetched source rather than reconstructed from memory:

        captured_returns = weights * y_true
        mean = reduce_mean(captured_returns)
        sharpe = mean / sqrt(reduce_mean(captured_returns^2) - mean^2 + 1e-9) * sqrt(252)
        loss = -sharpe

    `reduce_mean(x^2) - mean(x)^2` is the biased/population variance -- matched here
    with `unbiased=False`, not the default sample variance, to reproduce the same
    number the original code computes.
    """

    def forward(self, captured_returns: torch.Tensor) -> torch.Tensor:
        mean = captured_returns.mean()
        var = captured_returns.pow(2).mean() - mean.pow(2)
        sharpe = mean / torch.sqrt(var.clamp(min=0) + 1e-9) * (252.0**0.5)
        return -sharpe


def portfolio_daily_returns(captured_returns: torch.Tensor, date_idx: torch.Tensor, n_dates: int) -> torch.Tensor:
    """Equal-weight the pooled per-asset-per-day captured returns into one portfolio
    daily-return series, grouped by CALENDAR date (not by row order) -- the same
    `unsorted_segment_mean`-by-time-index construction the momentum-transformer repo
    uses in its SharpeValidationLoss, reimplemented with a differentiable
    index_add so gradients still flow back into `captured_returns` (and therefore
    into the model's position outputs) during training, not just at eval time.
    A date with fewer than 3 assets trading (holiday mismatches) is still a valid
    equal-weight average over however many assets actually traded that day.
    """
    sums = torch.zeros(n_dates, dtype=captured_returns.dtype).index_add(0, date_idx, captured_returns)
    counts = torch.zeros(n_dates, dtype=captured_returns.dtype).index_add(0, date_idx, torch.ones_like(captured_returns))
    mask = counts > 0
    return (sums[mask] / counts[mask])


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2 or np.std(daily_returns) < 1e-12:
        return 0.0
    return float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252.0))


class DeepMomentumEncoder(nn.Module):
    """Tech-GRU + co-attention(macro, cot) backbone, adapted from Model 5's
    FourStreamEncoder (train_model5_rescue.py) -- NEWS deliberately dropped (see
    module docstring: the synthetic placeholder news feature leaks target_return_1d,
    which this model trades directly, so it cannot be used here). Context is 2
    tokens now (macro, COT), not 3. Output is a single position head, tanh-bounded to
    [-1, 1] (long/flat/short with continuous sizing), not multi-task heads.
    """

    def __init__(self, n_tech: int, macro_dims: dict, n_cot: int, n_assets: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.macro_dims = macro_dims
        self.tech_gru = nn.GRU(n_tech, hidden_size, batch_first=True)
        self.macro_embed = nn.ModuleDict({a: nn.Sequential(nn.Linear(d, hidden_size), nn.ReLU()) for a, d in macro_dims.items()})
        self.cot_embed = nn.Sequential(nn.Linear(n_cot, hidden_size), nn.ReLU())
        self.asset_embed = nn.Embedding(n_assets, hidden_size)

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**0.5

        self.drop = nn.Dropout(dropout)
        self.head_position = nn.Linear(hidden_size * 3, 1)

    def forward(self, x_tech, x_macro, x_cot, asset_ids, asset_names):
        tech_seq, _ = self.tech_gru(x_tech)
        macro_today, cot_today = x_macro[:, -1, :], x_cot[:, -1, :]

        macro_tok = torch.zeros(x_tech.shape[0], tech_seq.shape[-1])
        for asset in asset_names:
            mask = asset_ids == ASSET_IDX[asset]
            if mask.any():
                real_dim = self.macro_dims[asset]
                macro_tok[mask] = self.macro_embed[asset](macro_today[mask][:, :real_dim])

        cot_tok = self.cot_embed(cot_today)
        context = torch.stack([macro_tok, cot_tok], dim=1)

        q = self.query(tech_seq)
        k, v = self.key(context), self.value(context)
        attn_weights = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / self.scale, dim=-1)
        attended = torch.bmm(attn_weights, v)

        asset_vec = self.asset_embed(asset_ids)
        fused = self.drop(torch.cat([tech_seq[:, -1, :], attended[:, -1, :], asset_vec], dim=-1))
        return torch.tanh(self.head_position(fused).squeeze(-1))


_market_data_cache: dict = {}


def build_market_data(ticker: str):
    if ticker in _market_data_cache:
        return _market_data_cache[ticker]
    macro_fetcher, macro_columns, market_code = _MACRO_FETCHERS[ticker]
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)
    tech_df = compute_indicators(price_df)
    macro_df = macro_fetcher(price_df.index)
    cot_df = fetch_cot_features(ticker, price_df.index)
    targets_df = compute_targets(price_df)  # only target_return_1d is used
    # No news stream here -- see module docstring: the synthetic placeholder news
    # generator leaks target_return_1d, which this model trades directly.
    full = pd.concat([tech_df, macro_df, cot_df, targets_df, price_df[["close"]]], axis=1).dropna()
    _market_data_cache[ticker] = (full, macro_columns)
    return full, macro_columns


def build_sequences_for_market(full, macro_columns, tech_scaler, macro_scaler, ticker, window):
    tech_scaled = tech_scaler.transform(full[TECH_COLUMNS])
    macro_scaled = macro_scaler.transform(full[macro_columns])
    cot_vals = full[COT_COLUMNS].to_numpy()
    ret_1d = full["target_return_1d"].to_numpy()
    trend_rule = full[TREND_COLS].mean(axis=1).clip(-1, 1).to_numpy()  # classical baseline signal
    dates = full.index

    Xt, Xm, Xc, y_ret, y_trend, aid, seq_dates = [], [], [], [], [], [], []
    for i in range(window - 1, len(full)):
        Xt.append(tech_scaled[i - window + 1 : i + 1])
        Xm.append(macro_scaled[i - window + 1 : i + 1])
        Xc.append(cot_vals[i - window + 1 : i + 1])
        y_ret.append(ret_1d[i])
        y_trend.append(trend_rule[i])
        aid.append(ASSET_IDX[ticker])
        seq_dates.append(dates[i])
    return (
        np.stack(Xt), np.stack(Xm), np.stack(Xc),
        np.array(aid), np.array(y_ret, dtype=np.float32), np.array(y_trend, dtype=np.float32),
        pd.DatetimeIndex(seq_dates),
    )


def train_dmn(epochs=150, patience=15, seed=42, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_end_frac=1.0, verbose=True):
    torch.manual_seed(seed)
    per_market, macro_dims = {}, {}
    for ticker in ASSETS:
        full, macro_columns = build_market_data(ticker)
        n = len(full)
        train_end, val_end, test_end = int(n * train_frac), int(n * (train_frac + val_frac)), int(n * test_end_frac)
        per_market[ticker] = (full, macro_columns, train_end, val_end, test_end)
        macro_dims[ticker] = len(macro_columns)

    tech_scalers = {t: StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end]) for t, (full, _, train_end, _, _) in per_market.items()}
    macro_scalers = {t: StandardScaler().fit(full[cols].iloc[:train_end]) for t, (full, cols, train_end, _, _) in per_market.items()}

    splits = {"train": [], "val": [], "test": []}
    for ticker, (full, macro_columns, train_end, val_end, test_end) in per_market.items():
        Xt, Xm, Xc, aid, y_ret, y_trend, seq_dates = build_sequences_for_market(full, macro_columns, tech_scalers[ticker], macro_scalers[ticker], ticker, WINDOW)
        s_train_end, s_val_end, s_test_end = train_end - WINDOW + 1, val_end - WINDOW + 1, test_end - WINDOW + 1
        for name, idx in [("train", np.arange(0, s_train_end)), ("val", np.arange(s_train_end, s_val_end)), ("test", np.arange(s_val_end, s_test_end))]:
            splits[name].append((Xt[idx], Xm[idx], Xc[idx], aid[idx], y_ret[idx], y_trend[idx], seq_dates[idx]))

    def pool(name):
        parts = splits[name]
        Xt = np.concatenate([p[0] for p in parts])
        Xm_padded = np.zeros((len(Xt), WINDOW, max(macro_dims.values())))
        offset = 0
        for p in parts:
            Xm_padded[offset : offset + len(p[1]), :, : p[1].shape[-1]] = p[1]
            offset += len(p[1])
        Xc = np.concatenate([p[2] for p in parts])
        aid = np.concatenate([p[3] for p in parts])
        y_ret = np.concatenate([p[4] for p in parts])
        y_trend = np.concatenate([p[5] for p in parts])
        all_dates = pd.DatetimeIndex(np.concatenate([p[6].values for p in parts]))
        date_codes, unique_dates = pd.factorize(all_dates, sort=True)
        return (
            torch.tensor(Xt, dtype=torch.float32), torch.tensor(Xm_padded, dtype=torch.float32),
            torch.tensor(Xc, dtype=torch.float32),
            torch.tensor(aid, dtype=torch.long), torch.tensor(y_ret, dtype=torch.float32),
            y_trend, torch.tensor(date_codes, dtype=torch.long), len(unique_dates), aid,
        )

    Xt_train, Xm_train, Xc_train, aid_train, yret_train, ytrend_train, dcode_train, ndates_train, _ = pool("train")
    Xt_val, Xm_val, Xc_val, aid_val, yret_val, ytrend_val, dcode_val, ndates_val, _ = pool("val")
    Xt_test, Xm_test, Xc_test, aid_test, yret_test, ytrend_test, dcode_test, ndates_test, aid_test_np = pool("test")

    model = DeepMomentumEncoder(n_tech=len(TECH_COLUMNS), macro_dims=macro_dims, n_cot=len(COT_COLUMNS), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    sharpe_loss = SharpeLoss()

    def fwd(xt, xm, xc, aid):
        return model(xt, xm, xc, aid, ASSETS)

    best_val_sharpe, best_state, no_improve = -float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_train = fwd(Xt_train, Xm_train, Xc_train, aid_train)
        port_train = portfolio_daily_returns(pos_train * yret_train, dcode_train, ndates_train)
        loss = sharpe_loss(port_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pos_val = fwd(Xt_val, Xm_val, Xc_val, aid_val)
            port_val = portfolio_daily_returns(pos_val * yret_val, dcode_val, ndates_val)
            val_sharpe = annualized_sharpe(port_val.numpy())

        if val_sharpe > best_val_sharpe:
            best_val_sharpe, best_state, no_improve = val_sharpe, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best val Sharpe={best_val_sharpe:.3f})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pos_test = fwd(Xt_test, Xm_test, Xc_test, aid_test).numpy()

    # --- Portfolio-level Sharpe for the DMN vs. the two honest baselines, same test dates ---
    yret_test_np = yret_test.numpy()
    dmn_captured = pos_test * yret_test_np
    bh_captured = np.ones_like(yret_test_np) * yret_test_np          # buy-and-hold
    classical_captured = ytrend_test * yret_test_np                  # classical trend-following rule

    def portfolio_sharpe_np(captured, dcode, ndates):
        sums = np.zeros(ndates)
        counts = np.zeros(ndates)
        np.add.at(sums, dcode, captured)
        np.add.at(counts, dcode, 1)
        mask = counts > 0
        return annualized_sharpe(sums[mask] / counts[mask])

    dcode_test_np = dcode_test.numpy()
    dmn_sharpe = portfolio_sharpe_np(dmn_captured, dcode_test_np, ndates_test)
    bh_sharpe = portfolio_sharpe_np(bh_captured, dcode_test_np, ndates_test)
    classical_sharpe = portfolio_sharpe_np(classical_captured, dcode_test_np, ndates_test)

    per_market_dmn_sharpe = {}
    for ticker in ASSETS:
        mask = aid_test_np == ASSET_IDX[ticker]
        per_market_dmn_sharpe[ticker] = annualized_sharpe(dmn_captured[mask]) if mask.sum() > 1 else float("nan")

    if verbose:
        print(f"\nPortfolio Sharpe (test): DMN={dmn_sharpe:.3f}  buy&hold={bh_sharpe:.3f}  classical_trend={classical_sharpe:.3f}")
        for ticker in ASSETS:
            mask = aid_test_np == ASSET_IDX[ticker]
            pos_dist = pos_test[mask]
            print(f"  {ticker}: DMN Sharpe={per_market_dmn_sharpe[ticker]:.3f}  mean|position|={np.abs(pos_dist).mean():.3f}  ({mask.sum()} test rows)")

    return {
        "dmn_sharpe": dmn_sharpe, "bh_sharpe": bh_sharpe, "classical_sharpe": classical_sharpe,
        "per_market_dmn_sharpe": per_market_dmn_sharpe, "best_val_sharpe": best_val_sharpe,
    }


if __name__ == "__main__":
    train_dmn()
