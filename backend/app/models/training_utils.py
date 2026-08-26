"""Shared training utilities for the multi-task GRU models (Phase 16 onward).

History of real, verified bugs fixed here, kept for context since the same failure
mode (collapse toward a constant class prediction) has now recurred twice under
different conditions:

1. (Phase 16/17) Original combined loss (unweighted sum of 2 return MSEs + 1 BCE + 1
   volatility MSE) produced models that collapsed to a CONSTANT direction class on 2
   of 3 markets. Fixed with per-target loss normalization (below) and switching
   checkpoint selection from combined val loss to val direction accuracy.

2. (10-year walk-forward re-run) Fix #1 was necessary but not sufficient: on the
   10-year dataset, ALL THREE markets collapsed to predicting one class 85-100% of
   the time, in BOTH the standard and rescue models -- and validation-accuracy
   checkpoint selection didn't catch it. Root cause: with more training data
   reinforcing a class-imbalanced label distribution (a persistent up/down bias over
   a longer period), predicting the majority class becomes an easy loss-minimizing
   shortcut for BCE. Worse, walk-forward folds place validation right before test in
   time, and adjacent periods are often directionally correlated -- so a collapsed
   model that happens to agree with the local trend scores well on validation too,
   passing right through the fix #1 safeguard undetected. Fixed here with two
   independent measures, not one: a class-weighted BCE loss (so collapsing to the
   majority class is no longer loss-minimizing), and BALANCED accuracy instead of
   raw accuracy for checkpoint selection (a constant predictor scores exactly 0.5 on
   balanced accuracy regardless of the true class split, so it can no longer hide
   behind a lucky trend).
"""

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn


def compute_loss_norms(targets_df, target_columns):
    """Per-target variance on the training set, used to normalize each regression
    loss term to a comparable relative scale."""
    norms = {}
    for col in target_columns:
        if "direction" in col:
            continue
        var = float(np.nanvar(targets_df[col]))
        norms[col] = var if var > 1e-8 else 1.0
    return norms


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    """BCEWithLogitsLoss pos_weight = (# negative) / (# positive) in the training
    set -- makes the majority-class shortcut no longer loss-minimizing. See module
    docstring for the real collapse this fixes."""
    pos = labels.sum()
    neg = float(len(labels)) - pos
    return neg / pos.clamp(min=1.0)


_MAX_NORMALIZED_TERM = 5.0  # clamp ceiling, see module docstring addendum below


def make_loss_fn(norms: dict, direction_pos_weight: torch.Tensor | None = None):
    """Regression targets here have extremely small variance (as low as ~1e-6 for
    5-day volatility) -- dividing MSE by raw variance to normalize scale is
    mathematically the right idea, but numerically dangerous: at random
    initialization the model's error is often much larger than the target's natural
    spread, so an unclamped normalized term can explode to huge values and
    destabilize the shared GRU encoder's gradient for ALL 4 heads, not just the one
    with the tiny-variance target. Clamping each normalized term to a fixed ceiling
    keeps the intended effect (comparable relative scale across terms) without the
    blow-up risk -- a standard technique for exactly this multi-task-loss-scale
    problem, not a hack specific to any one market."""

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss(pos_weight=direction_pos_weight)

    def compute_loss(pred, y):
        r1 = torch.clamp(mse(pred["return_1d"], y["target_return_1d"]) / norms["target_return_1d"], max=_MAX_NORMALIZED_TERM)
        r5 = torch.clamp(mse(pred["return_5d"], y["target_return_5d"]) / norms["target_return_5d"], max=_MAX_NORMALIZED_TERM)
        vol = torch.clamp(mse(pred["volatility_5d"], y["target_volatility_5d"]) / norms["target_volatility_5d"], max=_MAX_NORMALIZED_TERM)
        direction = bce(pred["direction_5d_logit"], y["target_direction_5d"])
        return r1 + r5 + direction + vol

    return compute_loss


def balanced_direction_accuracy(pred, y) -> float:
    """Balanced accuracy (average of per-class recall), not raw accuracy -- a
    constant predictor scores exactly 0.5 regardless of the true class split, so
    checkpoint selection can no longer be fooled by a collapsed model that happens
    to align with the local trend in validation/test data. See module docstring."""
    with torch.no_grad():
        pred_direction = (torch.sigmoid(pred["direction_5d_logit"]) > 0.5).float().numpy()
        y_true = y["target_direction_5d"].numpy()
    return balanced_accuracy_score(y_true, pred_direction)


def train_with_early_stopping(model, optimizer, compute_loss, X_train_args, y_train, X_val_args, y_val, epochs, patience):
    """X_*_args: tuple of positional tensor args to model.forward (so this works for
    both the single-stream Model 1 and two-stream Model 2 without duplicating the loop).
    Checkpoint selection is by validation BALANCED direction accuracy (highest wins),
    not combined loss and not raw accuracy -- see module docstring for why."""
    best_val_acc, best_val_loss_at_best_acc, best_state, no_improve = -1.0, float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = compute_loss(model(*X_train_args), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(*X_val_args)
            val_loss = compute_loss(val_pred, y_val).item()
        val_acc = balanced_direction_accuracy(val_pred, y_val)

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss_at_best_acc)
        if improved:
            best_val_acc, best_val_loss_at_best_acc = val_acc, val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (best val balanced_acc={best_val_acc:.3f}, val_loss={best_val_loss_at_best_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model
