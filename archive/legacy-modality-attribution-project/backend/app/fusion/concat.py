import numpy as np

from app.fusion.base import FusionOutput


class ConcatFusion:
    """Baseline: [tech_block | news_block]. Trivially separable by construction."""

    name = "concat"

    def fit(self, tech: np.ndarray, news: np.ndarray, labels: np.ndarray, tech_names, news_names):
        return self  # no learned parameters

    def transform(self, tech: np.ndarray, news: np.ndarray, tech_names, news_names) -> FusionOutput:
        matrix = np.concatenate([tech, news], axis=1)
        feature_names = list(tech_names) + list(news_names)
        tech_group = list(range(len(tech_names)))
        news_group = list(range(len(tech_names), len(tech_names) + len(news_names)))
        return FusionOutput(matrix, feature_names, tech_group, news_group)
