from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def tokenize_pairs(tokenizer, pairs: list[tuple[str, str]], cfg: dict):
    """Cap the prompt separately so a long document cannot erase the response."""
    features, audit = [], []
    backend = tokenizer.backend_tokenizer
    backend.no_truncation()
    backend.no_padding()
    for prompt, response in pairs:
        p_encoding = backend.encode(prompt, add_special_tokens=False)
        r_encoding = backend.encode(response, add_special_tokens=False)
        p, r = p_encoding.ids, r_encoding.ids
        if not r:
            raise ValueError("Attribution requires nonempty response tokens")
        conditioned = cfg["input_mode"] == "prompt_response"
        retained_p = p[: cfg["max_prompt_tokens"]] if conditioned else []
        specials = tokenizer.num_special_tokens_to_add(pair=conditioned)
        budget = cfg["max_length"] - len(retained_p) - specials
        if budget < 1:
            raise ValueError("No token budget remains for the response")
        retained_r = r[:budget]
        p_encoding.truncate(len(retained_p))
        r_encoding.truncate(len(retained_r))
        processed = (
            backend.post_process(p_encoding, r_encoding, add_special_tokens=True)
            if conditioned
            else backend.post_process(r_encoding, add_special_tokens=True)
        )
        feature = {"input_ids": processed.ids, "attention_mask": processed.attention_mask}
        if "token_type_ids" in tokenizer.model_input_names:
            feature["token_type_ids"] = processed.type_ids
        features.append(feature)
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
    def __init__(
        self,
        backbone,
        projection_dim: int,
        num_teachers: int,
        representation: str = "semantic",
        structure_layer: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.representation, self.structure_layer = representation, structure_layer
        if representation not in {"semantic", "structure", "fusion"}:
            raise ValueError("Unknown representation")
        if representation != "semantic":
            if not 0 < structure_layer < backbone.config.num_hidden_layers + 1:
                raise ValueError("Structure layer must name an encoder layer")
            self.structure = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
        if representation == "fusion":
            self.fusion = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU())
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
        return cls(
            backbone,
            cfg["projection_dim"],
            num_teachers,
            cfg.get("representation", "semantic"),
            cfg.get("structure_layer", 1),
        )

    def forward(self, projected: bool = True, **tokens):
        outputs = self.backbone(
            **tokens, output_hidden_states=projected and self.representation != "semantic"
        )
        hidden = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        if projected and self.representation != "semantic":
            early = outputs.hidden_states[self.structure_layer]
            structure = self.structure((early * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1))
            pooled = (
                self.fusion(torch.cat([structure, pooled], dim=-1))
                if self.representation == "fusion"
                else structure
            )
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
