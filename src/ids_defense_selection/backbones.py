"""The three neural backbones compared in the paper.

All backbones are binary classifiers that consume the transformed feature vector
and return one logit per sample, so every defense strategy can be trained on top
of any of them.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLPBackbone(nn.Module):
    """Fully connected classifier: Linear -> ReLU -> Dropout, repeated."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, int, int] = (128, 64, 32),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Linear(h3, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(1)


class CNN1DBackbone(nn.Module):
    """1D-CNN classifier: two Conv1d layers -> global average pooling -> linear head.

    The tabular feature vector is treated as a length-D sequence with one channel;
    a stride-3 convolution lets neighbouring features interact.
    """

    def __init__(self, input_dim: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x).squeeze(1)


class FTTransformerBackbone(nn.Module):
    """FT-Transformer for tabular IDS features (Gorishniy et al., NeurIPS 2021).

    Each input feature is independently projected to a d_token-dimensional token
    via a dedicated Linear(1, d_token).  A learnable [CLS] token is prepended and
    the full sequence is processed by a Transformer encoder; the [CLS] output
    drives the binary classification head.

    The tokenizer is vectorised: input_dim independent Linear(1, d_token) layers
    are executed as one batched multiply, which is orders of magnitude faster on
    GPU while being numerically equivalent.
    """

    def __init__(
        self,
        input_dim: int,
        d_token: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ffn: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.d_token = d_token
        self.feature_weight = nn.Parameter(torch.empty(input_dim, d_token))
        self.feature_bias = nn.Parameter(torch.empty(input_dim, d_token))
        # Init matching nn.Linear defaults (kaiming_uniform for weight, uniform for bias)
        nn.init.kaiming_uniform_(self.feature_weight, a=5 ** 0.5)
        bound = 1.0  # fan_in = 1 for each per-feature linear
        nn.init.uniform_(self.feature_bias, -bound, bound)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # x: (B, D) -> (B, D, 1) * (1, D, d_token) + (1, D, d_token) -> (B, D, d_token)
        tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.head(tokens[:, 0, :]).squeeze(1)


#: FT-Transformer hyperparameters used for every reported result.
FT_TRANSFORMER_KWARGS: dict[str, int] = {
    "d_token": 32,
    "n_heads": 2,
    "n_layers": 2,
    "d_ffn": 64,
}


def build_ft_transformer(input_dim: int, dropout: float = 0.15) -> FTTransformerBackbone:
    """Construct an FT-Transformer with the paper's hyperparameters."""
    return FTTransformerBackbone(input_dim, dropout=dropout, **FT_TRANSFORMER_KWARGS)
