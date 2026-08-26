import dice_ml
import numpy as np
import pandas as pd
import xgboost as xgb

# Natural bounds per feature -- keeps DiCE from proposing physically implausible
# counterfactuals (e.g. RSI outside [0, 100], sentiment probabilities outside [0, 1]).
_FEATURE_BOUNDS = {
    "rsi_14": (0.0, 100.0),
    "macd": (-10.0, 10.0),
    "macd_signal": (-10.0, 10.0),
    "sma_10": (-0.5, 0.5),
    "sma_50": (-0.5, 0.5),
    "bb_width": (0.0, 1.0),
    "realized_vol_20": (0.0, 0.2),
    "sent_positive": (0.0, 1.0),
    "sent_negative": (0.0, 1.0),
    "sent_neutral": (0.0, 1.0),
}


def _bounds_for(feature_names: list[str]):
    ranges = {}
    for name in feature_names:
        base = name.replace("attended_", "")
        ranges[name] = list(_FEATURE_BOUNDS.get(base, (-5.0, 5.0)))
    return ranges


def generate_counterfactual(
    model: xgb.XGBClassifier,
    fused_row: np.ndarray,
    feature_names: list[str],
    background_matrix: np.ndarray,
    labels: np.ndarray,
    total_cfs: int = 1,
):
    """Minimal-change counterfactual via DiCE's genetic-algorithm backend, since
    XGBoost is not differentiable (gradient-based DiCE modes don't apply)."""
    df = pd.DataFrame(background_matrix, columns=feature_names)
    df["label"] = labels

    data = dice_ml.Data(
        dataframe=df,
        continuous_features=feature_names,
        outcome_name="label",
        permitted_range=_bounds_for(feature_names),
    )
    dice_model = dice_ml.Model(model=model, backend="sklearn")
    explainer = dice_ml.Dice(data, dice_model, method="genetic")

    query = pd.DataFrame(fused_row.reshape(1, -1), columns=feature_names)
    current_pred = int(model.predict(fused_row.reshape(1, -1))[0])
    desired = 1 - current_pred

    cf = explainer.generate_counterfactuals(
        query, total_CFs=total_cfs, desired_class=desired, permitted_range=_bounds_for(feature_names)
    )
    cf_df = cf.cf_examples_list[0].final_cfs_df
    changes = []
    for name in feature_names:
        before = float(query[name].iloc[0])
        after = float(cf_df[name].iloc[0])
        if abs(after - before) > 1e-3:
            changes.append({"feature": name, "from": before, "to": after})
    return {
        "current_prediction": "up" if current_pred == 1 else "down",
        "counterfactual_prediction": "up" if desired == 1 else "down",
        "changes": changes,
    }
