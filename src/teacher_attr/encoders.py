from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def tokenize_pairs(tokenizer, pairs: list[tuple[str, str]], cfg: dict):
    """Cap the prompt separately so a long document cannot erase the response."""
    features, audit = [], []
    for prompt, response in pairs:
        p = tokenizer.encode(prompt, add_special_tokens=False)
        r = tokenizer.encode(response, add_special_tokens=False)
        if not r:
            raise ValueError("Attribution requires nonempty response tokens")
        conditioned = cfg["input_mode"] == "prompt_response"
        retained_p = p[: cfg["max_prompt_tokens"]] if conditioned else []
        specials = tokenizer.num_special_tokens_to_add(pair=conditioned)
        budget = cfg["max_length"] - len(retained_p) - specials
        if budget < 1:
            raise ValueError("No token budget remains for the response")
        retained_r = r[:budget]
        args = (retained_p, retained_r) if conditioned else (retained_r,)
        features.append(
            tokenizer.prepare_for_model(
                *args, add_special_tokens=True, truncation=False, return_attention_mask=True
            )
        )
        audit.append(
            {
                "prompt_tokens": len(p),
                "prompt_tokens_retained": len(retained_p),
                "response_tokens": len(r),
                "response_tokens_retained": len(retained_r),
            }
        )
    return tokenizer.pad(features, padding=True, return_tensors="pt"), audit


class AttributionEncoder(nn.Module):
    def __init__(self, backbone, projection_dim: int, num_teachers: int):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, projection_dim)
        )
        self.classifier = nn.Linear(projection_dim, num_teachers)

    @classmethod
    def pretrained(cls, cfg: dict, num_teachers: int):
        backbone = AutoModel.from_pretrained(
            cfg["hf_name"], revision=cfg["revision"], trust_remote_code=False
        )
        if cfg["max_length"] > getattr(backbone.config, "max_position_embeddings", 10**9):
            raise ValueError("Encoder max_length exceeds backbone context")
        return cls(backbone, cfg["projection_dim"], num_teachers)

    def forward(self, projected: bool = True, **tokens):
        hidden = self.backbone(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        emb = F.normalize(self.projection(pooled) if projected else pooled, dim=-1)
        return emb, self.classifier(emb) if projected else None


def score_embeddings(student, teachers):
    return torch.einsum("bd,bkd->bk", F.normalize(student, dim=-1), F.normalize(teachers, dim=-1))


def collate(rows: list[dict], tokenizer, cfg: dict, teachers: list[str]):
    student_tokens, _ = tokenize_pairs(
        tokenizer, [(r["prompt"], r["student_response"]) for r in rows], cfg
    )
    teacher_tokens = None
    if cfg["objective"] != "classification":
        teacher_tokens, _ = tokenize_pairs(
            tokenizer,
            [(r["prompt"], r["teacher_responses"][t]) for r in rows for t in teachers],
            cfg,
        )
    return (
        student_tokens,
        teacher_tokens,
        torch.tensor([teachers.index(r["true_teacher"]) for r in rows]),
    )


def to_device(tokens, device):
    return {key: value.to(device) for key, value in tokens.items()}
