import numpy as np
import torch
from torch import nn

from app.fusion.base import FusionOutput


class _CoAttentionNet(nn.Module):
    """Two-stream cross-attention: each technical indicator and each news signal is
    treated as its own token. Tech tokens attend over news tokens (and vice versa),
    but the attended output keeps tech tokens and news tokens in separate blocks --
    unlike a merged-embedding attention layer, columns never lose their modality
    identity, which is what makes cross-fusion SHAP attribution comparable at all.
    """

    def __init__(self, n_tech: int, n_news: int, d_model: int = 16):
        super().__init__()
        self.token_embed = nn.Linear(1, d_model)
        self.tech_pos = nn.Embedding(n_tech, d_model)
        self.news_pos = nn.Embedding(n_news, d_model)

        self.q_tech = nn.Linear(d_model, d_model)
        self.k_news = nn.Linear(d_model, d_model)
        self.v_news = nn.Linear(d_model, d_model)

        self.q_news = nn.Linear(d_model, d_model)
        self.k_tech = nn.Linear(d_model, d_model)
        self.v_tech = nn.Linear(d_model, d_model)

        self.tech_out = nn.Linear(d_model, 1)
        self.news_out = nn.Linear(d_model, 1)
        self.d_model = d_model

    def _embed(self, x, pos_embed):
        # x: (batch, n_tokens) -> (batch, n_tokens, d_model)
        tokens = self.token_embed(x.unsqueeze(-1))
        positions = pos_embed(torch.arange(x.shape[1], device=x.device))
        return tokens + positions

    def forward(self, tech, news):
        tech_tok = self._embed(tech, self.tech_pos)
        news_tok = self._embed(news, self.news_pos)

        scale = self.d_model ** 0.5
        attn_t2n = torch.softmax(self.q_tech(tech_tok) @ self.k_news(news_tok).transpose(-1, -2) / scale, dim=-1)
        attended_tech = attn_t2n @ self.v_news(news_tok)  # (batch, n_tech, d_model)
        attended_tech = attended_tech + tech_tok  # residual keeps original indicator signal

        attn_n2t = torch.softmax(self.q_news(news_tok) @ self.k_tech(tech_tok).transpose(-1, -2) / scale, dim=-1)
        attended_news = attn_n2t @ self.v_tech(tech_tok)
        attended_news = attended_news + news_tok

        tech_scalars = self.tech_out(attended_tech).squeeze(-1)  # (batch, n_tech)
        news_scalars = self.news_out(attended_news).squeeze(-1)  # (batch, n_news)
        return tech_scalars, news_scalars


class CoAttentionFusion:
    name = "coattention"

    def __init__(self, epochs: int = 200, lr: float = 0.01, seed: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.net = None

    def fit(self, tech: np.ndarray, news: np.ndarray, labels: np.ndarray, tech_names, news_names):
        torch.manual_seed(self.seed)
        self.net = _CoAttentionNet(tech.shape[1], news.shape[1])
        head = nn.Linear(tech.shape[1] + news.shape[1], 1)

        X_tech = torch.tensor(tech, dtype=torch.float32)
        X_news = torch.tensor(news, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.Adam(list(self.net.parameters()) + list(head.parameters()), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            optimizer.zero_grad()
            tech_scalars, news_scalars = self.net(X_tech, X_news)
            fused = torch.cat([tech_scalars, news_scalars], dim=1)
            logits = head(fused).squeeze(-1)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        self.net.eval()
        return self

    def transform(self, tech: np.ndarray, news: np.ndarray, tech_names, news_names) -> FusionOutput:
        with torch.no_grad():
            tech_scalars, news_scalars = self.net(
                torch.tensor(tech, dtype=torch.float32), torch.tensor(news, dtype=torch.float32)
            )
        matrix = np.concatenate([tech_scalars.numpy(), news_scalars.numpy()], axis=1)
        feature_names = [f"attended_{n}" for n in tech_names] + [f"attended_{n}" for n in news_names]
        tech_group = list(range(len(tech_names)))
        news_group = list(range(len(tech_names), len(tech_names) + len(news_names)))
        return FusionOutput(matrix, feature_names, tech_group, news_group)
