import numpy as np
import torch
from torch import nn

from app.fusion.base import FusionOutput


class WeightedFusion:
    """[w_tech . tech_block | w_news . news_block], scalar weights learned jointly
    with a disposable linear head on the training labels, then the head is discarded.

    Blocks are rescaled, never summed across mismatched dimensions, so columns stay
    traceable to their source modality after fusion.
    """

    name = "weighted"

    def __init__(self, epochs: int = 300, lr: float = 0.05, seed: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.w_tech = 1.0
        self.w_news = 1.0

    def fit(self, tech: np.ndarray, news: np.ndarray, labels: np.ndarray, tech_names, news_names):
        torch.manual_seed(self.seed)
        X_tech = torch.tensor(tech, dtype=torch.float32)
        X_news = torch.tensor(news, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        log_w_tech = nn.Parameter(torch.zeros(1))
        log_w_news = nn.Parameter(torch.zeros(1))
        head = nn.Linear(tech.shape[1] + news.shape[1], 1)

        optimizer = torch.optim.Adam(list(head.parameters()) + [log_w_tech, log_w_news], lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            w_tech, w_news = torch.exp(log_w_tech), torch.exp(log_w_news)
            fused = torch.cat([w_tech * X_tech, w_news * X_news], dim=1)
            logits = head(fused).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        self.w_tech = float(torch.exp(log_w_tech).item())
        self.w_news = float(torch.exp(log_w_news).item())
        return self

    def transform(self, tech: np.ndarray, news: np.ndarray, tech_names, news_names) -> FusionOutput:
        matrix = np.concatenate([self.w_tech * tech, self.w_news * news], axis=1)
        feature_names = list(tech_names) + list(news_names)
        tech_group = list(range(len(tech_names)))
        news_group = list(range(len(tech_names), len(tech_names) + len(news_names)))
        return FusionOutput(matrix, feature_names, tech_group, news_group)
