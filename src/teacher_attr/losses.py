from __future__ import annotations

import torch
import torch.nn.functional as F


def same_prompt_infonce(
    student_emb: torch.Tensor,
    teacher_embs: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    student_emb: [B, D]
    teacher_embs: [B, K, D]
    labels: [B], true teacher index among K candidates
    """
    student_emb = F.normalize(student_emb, dim=-1)
    teacher_embs = F.normalize(teacher_embs, dim=-1)

    scores = torch.einsum("bd,bkd->bk", student_emb, teacher_embs)
    scores = scores / temperature
    return F.cross_entropy(scores, labels)


def same_prompt_scores(
    student_emb: torch.Tensor,
    teacher_embs: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    student_emb = F.normalize(student_emb, dim=-1)
    teacher_embs = F.normalize(teacher_embs, dim=-1)
    scores = torch.einsum("bd,bkd->bk", student_emb, teacher_embs)
    return scores / temperature
