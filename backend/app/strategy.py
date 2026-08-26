"""Trading strategy layer (Phase 21). Deliberately model-agnostic: takes only
(expected_return, expected_volatility) as input, so it can be applied identically to
every model in the ablation (Phase 15's baselines, Models 1-4, the joint model) with
NO changes to the strategy rules between them -- exactly what the plan requires for
the Phase 30 ablation to be a fair comparison ("Do not change the test set, trading
rules, transaction costs, or evaluation methodology between these models").

Score = Expected Return / Expected Volatility (a per-instance, Sharpe-like ratio).
Thresholds are calibrated on the VALIDATION set only, never the test set -- selected
by grid search over candidate threshold pairs, maximizing average realized 5-day
return per decision on validation data. This is what the plan means by "determined
through training/validation, not tuned on the final test period."
"""

from dataclasses import dataclass

import numpy as np


def compute_score(expected_return: np.ndarray, expected_volatility: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return expected_return / (expected_volatility + eps)


@dataclass
class StrategyThresholds:
    buy_threshold: float
    sell_threshold: float

    def signal(self, score: np.ndarray) -> np.ndarray:
        """Returns an array of 'BUY' / 'HOLD' / 'SELL' strings."""
        out = np.full(score.shape, "HOLD", dtype=object)
        out[score > self.buy_threshold] = "BUY"
        out[score < self.sell_threshold] = "SELL"
        return out


def calibrate_thresholds(val_score: np.ndarray, val_realized_return: np.ndarray, n_grid: int = 9) -> StrategyThresholds:
    """Grid search over candidate threshold pairs using ONLY validation data. Objective:
    maximize average realized return per decision under the resulting BUY/SELL/HOLD
    calls (BUY realizes +return, SELL realizes -return i.e. a short, HOLD realizes 0) --
    a simple, transparent proxy for what the strategy is actually meant to do, not
    accuracy, since accuracy doesn't account for the magnitude of the moves being
    captured or missed.
    """
    # Candidate thresholds drawn from the validation score's own quantiles, so the
    # grid adapts to whatever scale this particular model's scores happen to have
    # (different models produce very differently-scaled scores) -- never a fixed
    # hardcoded number that would silently mean something different per model.
    candidates = np.quantile(val_score, np.linspace(0.5, 0.95, n_grid))
    best_thresholds, best_objective = StrategyThresholds(0.0, 0.0), -np.inf

    for buy_t in candidates:
        for sell_t in -candidates:  # symmetric search space; not assumed to be optimal, just searched
            thresholds = StrategyThresholds(float(buy_t), float(sell_t))
            signal = thresholds.signal(val_score)
            realized = np.where(signal == "BUY", val_realized_return, np.where(signal == "SELL", -val_realized_return, 0.0))
            objective = realized.mean()
            if objective > best_objective:
                best_objective, best_thresholds = objective, thresholds

    return best_thresholds
