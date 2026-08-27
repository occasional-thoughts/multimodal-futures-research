"""Model 9: Deep Momentum Network pooled across a BROADER futures universe (13
markets, not 3) -- testing the specific lever real literature says is what actually
makes this class of model work.

Model 6 (train_model6_dmn.py) reimplemented Lim/Zohren/Roberts' Sharpe-optimized
position-sizing approach faithfully but pooled only ZN/CL/GC, and it underperformed
both buy-and-hold and a classical trend rule in every walk-forward fold. Before
concluding the whole DMN approach doesn't work here, one real, literature-identified
difference was left untested: the original DMN paper backtested on **88** continuous
futures contracts, not 3, and a 2026 follow-up -- "End-to-End Parametric Portfolio
Policies for Cross-Asset Futures Timing: When Do AI Models Beat Simple Rules?"
(arXiv:2607.00475) -- directly tested this and found "learned policies perform better
in the broad cross-asset universe... than simple rules", i.e. universe size is not
incidental to these papers' results, it's a documented driver of them. A 3-instrument
Sharpe-loss portfolio is a genuinely under-powered test of this paradigm, not a fair
one.

This model pools **13 markets**: the original ZN/CL/GC plus 10 additional liquid CME
futures verified fetchable with real dense 10-year data before building anything
(ES, NQ, YM, RTY -- equity index; 6E -- FX; SI, HG -- metals; ZB, ZF -- rates; NG --
energy), spanning the same six-ish asset classes the arXiv:2607.00475 cross-asset
study uses, not an arbitrary pile of tickers.

Deliberately TECHNICAL-FEATURES-ONLY (no macro/news/COT) for every market here, unlike
Model 6 -- two honest reasons, not a shortcut: (1) macro/fundamentals/COT plumbing in
this project is hand-built per-market for ZN/CL/GC specifically (app/data/macro.py,
fundamentals.py, cot.py) and doesn't exist for the 10 new tickers; building 10 more
custom per-market data sources is out of scope for testing this one specific lever.
(2) This isolates the variable actually being tested -- does a bigger pooled universe
help the Sharpe-loss objective -- from the multimodal-features question already
tested (and already found not to help) in Models 6/7/8. A cleaner, single-variable
experiment, and closer to the original DMN paper's own setup (price/trend-derived
features, not per-instrument macro streams).

Same SharpeLoss, same portfolio-daily-return-by-calendar-date construction, and the
same honest buy-and-hold / classical-trend baselines as Model 6 -- reported however
the result actually comes out, not reached for a flattering cut of it.

Run from backend/: python scripts/train_model9_dmn_broad.py
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
from app.data.prices import fetch_price_history
from app.features.technical import TECH_COLUMNS, compute_indicators
from app.targets import compute_targets

WINDOW = 20
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
ASSETS = [
    "ZN=F", "CL=F", "GC=F",  # this project's original 3
    "ES=F", "NQ=F", "YM=F", "RTY=F",  # equity index futures
    "6E=F",  # FX
    "SI=F", "HG=F",  # metals
    "ZB=F", "ZF=F",  # rates
    "NG=F",  # energy
]
ASSET_IDX = {a: i for i, a in enumerate(ASSETS)}
TREND_COLS = ["macd_trend_8_24", "macd_trend_16_48", "macd_trend_32_96"]


class SharpeLoss(nn.Module):
    """Identical to Model 6's -- see train_model6_dmn.py's docstring for the
    verified-against-source derivation (kieranjwood/trading-momentum-transformer)."""

    def forward(self, captured_returns: torch.Tensor) -> torch.Tensor:
        mean = captured_returns.mean()
        var = captured_returns.pow(2).mean() - mean.pow(2)
        sharpe = mean / torch.sqrt(var.clamp(min=0) + 1e-9) * (252.0**0.5)
        return -sharpe


def portfolio_daily_returns(captured_returns: torch.Tensor, date_idx: torch.Tensor, n_dates: int) -> torch.Tensor:
    sums = torch.zeros(n_dates, dtype=captured_returns.dtype).index_add(0, date_idx, captured_returns)
    counts = torch.zeros(n_dates, dtype=captured_returns.dtype).index_add(0, date_idx, torch.ones_like(captured_returns))
    mask = counts > 0
    return sums[mask] / counts[mask]


def annualized_sharpe(daily_returns: np.ndarray) -> float:
    if len(daily_returns) < 2 or np.std(daily_returns) < 1e-12:
        return 0.0
    return float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252.0))


class BroadDMNEncoder(nn.Module):
    """Single-stream: tech GRU + asset embedding -> tanh position. No co-attention
    fusion needed (no second modality here, unlike Model 6) -- simpler by design, not
    by omission, since this model isolates the universe-size variable specifically."""

    def __init__(self, n_tech: int, n_assets: int, hidden_size: int = 16, dropout: float = 0.3):
        super().__init__()
        self.tech_gru = nn.GRU(n_tech, hidden_size, batch_first=True)
        self.asset_embed = nn.Embedding(n_assets, hidden_size)
        self.drop = nn.Dropout(dropout)
        self.head_position = nn.Linear(hidden_size * 2, 1)

    def forward(self, x_tech, asset_ids):
        tech_seq, _ = self.tech_gru(x_tech)
        tech_last = tech_seq[:, -1, :]
        asset_vec = self.asset_embed(asset_ids)
        fused = self.drop(torch.cat([tech_last, asset_vec], dim=-1))
        return torch.tanh(self.head_position(fused).squeeze(-1))


_market_data_cache: dict = {}


def build_market_data(ticker: str):
    if ticker in _market_data_cache:
        return _market_data_cache[ticker]
    price_df = fetch_price_history(ticker, period=HISTORY_PERIOD)
    tech_df = compute_indicators(price_df)
    targets_df = compute_targets(price_df)  # only target_return_1d is used
    full = pd.concat([tech_df, targets_df, price_df[["close"]]], axis=1).dropna()
    _market_data_cache[ticker] = full
    return full


def build_sequences_for_market(full, tech_scaler, ticker, window):
    tech_scaled = tech_scaler.transform(full[TECH_COLUMNS])
    ret_1d = full["target_return_1d"].to_numpy()
    trend_rule = full[TREND_COLS].mean(axis=1).clip(-1, 1).to_numpy()
    dates = full.index

    Xt, y_ret, y_trend, aid, seq_dates = [], [], [], [], []
    for i in range(window - 1, len(full)):
        Xt.append(tech_scaled[i - window + 1 : i + 1])
        y_ret.append(ret_1d[i])
        y_trend.append(trend_rule[i])
        aid.append(ASSET_IDX[ticker])
        seq_dates.append(dates[i])
    return np.stack(Xt), np.array(aid), np.array(y_ret, dtype=np.float32), np.array(y_trend, dtype=np.float32), pd.DatetimeIndex(seq_dates)


def train_dmn_broad(epochs=150, patience=15, seed=42, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, test_end_frac=1.0, verbose=True):
    torch.manual_seed(seed)
    per_market = {}
    for ticker in ASSETS:
        full = build_market_data(ticker)
        n = len(full)
        train_end, val_end, test_end = int(n * train_frac), int(n * (train_frac + val_frac)), int(n * test_end_frac)
        per_market[ticker] = (full, train_end, val_end, test_end)

    tech_scalers = {t: StandardScaler().fit(full[TECH_COLUMNS].iloc[:train_end]) for t, (full, train_end, _, _) in per_market.items()}

    splits = {"train": [], "val": [], "test": []}
    for ticker, (full, train_end, val_end, test_end) in per_market.items():
        Xt, aid, y_ret, y_trend, seq_dates = build_sequences_for_market(full, tech_scalers[ticker], ticker, WINDOW)
        s_train_end, s_val_end, s_test_end = train_end - WINDOW + 1, val_end - WINDOW + 1, test_end - WINDOW + 1
        for name, idx in [("train", np.arange(0, s_train_end)), ("val", np.arange(s_train_end, s_val_end)), ("test", np.arange(s_val_end, s_test_end))]:
            splits[name].append((Xt[idx], aid[idx], y_ret[idx], y_trend[idx], seq_dates[idx]))

    def pool(name):
        parts = splits[name]
        Xt = np.concatenate([p[0] for p in parts])
        aid = np.concatenate([p[1] for p in parts])
        y_ret = np.concatenate([p[2] for p in parts])
        y_trend = np.concatenate([p[3] for p in parts])
        all_dates = pd.DatetimeIndex(np.concatenate([p[4].values for p in parts]))
        date_codes, unique_dates = pd.factorize(all_dates, sort=True)
        return (
            torch.tensor(Xt, dtype=torch.float32), torch.tensor(aid, dtype=torch.long),
            torch.tensor(y_ret, dtype=torch.float32), y_trend,
            torch.tensor(date_codes, dtype=torch.long), len(unique_dates), aid,
        )

    Xt_train, aid_train, yret_train, ytrend_train, dcode_train, ndates_train, _ = pool("train")
    Xt_val, aid_val, yret_val, ytrend_val, dcode_val, ndates_val, _ = pool("val")
    Xt_test, aid_test, yret_test, ytrend_test, dcode_test, ndates_test, aid_test_np = pool("test")

    model = BroadDMNEncoder(n_tech=len(TECH_COLUMNS), n_assets=len(ASSETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    sharpe_loss = SharpeLoss()

    best_val_sharpe, best_state, no_improve = -float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pos_train = model(Xt_train, aid_train)
        port_train = portfolio_daily_returns(pos_train * yret_train, dcode_train, ndates_train)
        loss = sharpe_loss(port_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pos_val = model(Xt_val, aid_val)
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
        pos_test = model(Xt_test, aid_test).numpy()

    yret_test_np = yret_test.numpy()
    dmn_captured = pos_test * yret_test_np
    bh_captured = np.ones_like(yret_test_np) * yret_test_np
    classical_captured = ytrend_test * yret_test_np

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

    if verbose:
        print(f"\nPortfolio Sharpe (test, {len(ASSETS)} markets): DMN={dmn_sharpe:.3f}  buy&hold={bh_sharpe:.3f}  classical_trend={classical_sharpe:.3f}")
        for ticker in ASSETS:
            mask = aid_test_np == ASSET_IDX[ticker]
            if mask.sum() > 1:
                print(f"  {ticker}: DMN Sharpe={annualized_sharpe(dmn_captured[mask]):.3f}  mean|position|={np.abs(pos_test[mask]).mean():.3f}  ({mask.sum()} test rows)")

    return {"dmn_sharpe": dmn_sharpe, "bh_sharpe": bh_sharpe, "classical_sharpe": classical_sharpe, "best_val_sharpe": best_val_sharpe}


if __name__ == "__main__":
    train_dmn_broad()
