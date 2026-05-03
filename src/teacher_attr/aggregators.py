from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSetsAggregator(nn.Module):
    def __init__(self, emb_dim: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, K, D], K prompt-response embeddings.
        mask: [B, K], optional.
        """
        h = self.phi(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
            pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        return F.normalize(self.rho(pooled), dim=-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    x = x * mask.unsqueeze(-1)
    return x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
