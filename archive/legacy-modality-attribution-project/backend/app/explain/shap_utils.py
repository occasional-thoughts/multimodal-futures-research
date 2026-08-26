import numpy as np
import shap
import xgboost as xgb

from app.fusion.base import FusionOutput


def compute_shap(model: xgb.XGBClassifier, fusion_output: FusionOutput):
    """TreeExplainer is exact and fast for tree models -- no need for the slower
    model-agnostic kernel explainer here."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(fusion_output.matrix)
    if isinstance(shap_values, list):  # older shap versions return per-class list
        shap_values = shap_values[1]
    return np.asarray(shap_values)


def modality_attribution(shap_values: np.ndarray, fusion_output: FusionOutput):
    """Per-instance tech/news |SHAP| totals and the comparable news_share."""
    abs_shap = np.abs(shap_values)
    tech_attr = abs_shap[:, fusion_output.tech_group].sum(axis=1)
    news_attr = abs_shap[:, fusion_output.news_group].sum(axis=1)
    denom = np.where((tech_attr + news_attr) == 0, 1e-9, tech_attr + news_attr)
    news_share = news_attr / denom
    return tech_attr, news_attr, news_share


def top_features(shap_values_row: np.ndarray, feature_names: list, k: int = 5):
    order = np.argsort(-np.abs(shap_values_row))[:k]
    return [{"feature": feature_names[i], "shap": float(shap_values_row[i])} for i in order]
