from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class AttributionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        projection_dim: int = 256,
        num_teachers: int = 4,
        trust_remote_code: bool = True,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.projection_dim = projection_dim
        self.num_teachers = num_teachers

        self.backbone = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        hidden = self.backbone.config.hidden_size

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, projection_dim),
        )
        self.classifier = nn.Linear(projection_dim, num_teachers)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    @staticmethod
    def mean_pool(outputs: object, attention_mask: torch.Tensor) -> torch.Tensor:
        token_emb = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        pooled = self.mean_pool(outputs, attention_mask)
        emb = F.normalize(self.proj(pooled), dim=-1)
        logits = self.classifier(emb)
        return emb, logits