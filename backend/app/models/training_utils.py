"""Shared training utilities for the multi-task GRU models (Phase 16 onward).

Fixes a real, verified bug: the original combined loss (unweighted sum of 2 return
MSEs + 1 BCE + 1 volatility MSE) and "select the best checkpoint by combined val loss"
early-stopping criterion together produced models that collapsed to predicting a
CONSTANT direction class on 2 of 3 markets (verified directly by inspecting the test
prediction distribution -- e.g. GC=F predicted "up" for literally 54/54 test rows).
Two independent problems, both fixed here:

1. The 4 loss terms have wildly different natural scales (returns/volatility are
   ~0.01 in magnitude, so their MSE is tiny; BCE is O(0.5-0.7)) -- summed unweighted,
   the terms don't contribute comparably to the gradient. Fixed by normalizing each
   regression term by its own training-set variance, so every term is a comparable
   "relative error" regardless of units.
2. Early stopping picked the checkpoint with the best COMBINED val loss, which has no
   guarantee of being the checkpoint that's actually good at the classification task
   specifically -- a checkpoint could look good on combined loss while its direction
   head has quietly collapsed to a constant. Fixed by selecting the checkpoint with
   the best validation DIRECTION ACCURACY directly, breaking ties by combined loss.
"""

import numpy as np
import torch
from torch import nn


def compute_loss_norms(targets_df, target_columns):
    """Per-target variance on the training set, used to normalize each regression
    loss term to a comparable relative scale."""
    norms = {}
    for col in target_columns:
        if col == "target_direction_5d":
            continue
        var = float(np.nanvar(targets_df[col]))
        norms[col] = var if var > 1e-8 else 1.0
    return norms


_MAX_NORMALIZED_TERM = 5.0  # clamp ceiling, see module docstring addendum below


def make_loss_fn(norms: dict):
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

    mse, bce = nn.MSELoss(), nn.BCEWithLogitsLoss()

    def compute_loss(pred, y):
        r1 = torch.clamp(mse(pred["return_1d"], y["target_return_1d"]) / norms["target_return_1d"], max=_MAX_NORMALIZED_TERM)
        r5 = torch.clamp(mse(pred["return_5d"], y["target_return_5d"]) / norms["target_return_5d"], max=_MAX_NORMALIZED_TERM)
        vol = torch.clamp(mse(pred["volatility_5d"], y["target_volatility_5d"]) / norms["target_volatility_5d"], max=_MAX_NORMALIZED_TERM)
        direction = bce(pred["direction_5d_logit"], y["target_direction_5d"])
        return r1 + r5 + direction + vol

    return compute_loss


def direction_accuracy(pred, y) -> float:
    with torch.no_grad():
        pred_direction = (torch.sigmoid(pred["direction_5d_logit"]) > 0.5).float()
        return (pred_direction == y["target_direction_5d"]).float().mean().item()


def train_with_early_stopping(model, optimizer, compute_loss, X_train_args, y_train, X_val_args, y_val, epochs, patience):
    """X_*_args: tuple of positional tensor args to model.forward (so this works for
    both the single-stream Model 1 and two-stream Model 2 without duplicating the loop).
    Checkpoint selection is by validation DIRECTION ACCURACY (highest wins), not
    combined loss -- see module docstring for why that distinction is the actual fix."""
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
        val_acc = direction_accuracy(val_pred, y_val)

        improved = val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss_at_best_acc)
        if improved:
            best_val_acc, best_val_loss_at_best_acc = val_acc, val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  early stopping at epoch {epoch} (best val direction_acc={best_val_acc:.3f}, val_loss={best_val_loss_at_best_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model
