"""Held-out distillation diagnostics, separate from attribution model selection."""

from __future__ import annotations

import re

import numpy as np

from teacher_attr.config import initialize_run
from teacher_attr.evaluation import encode_rows
from teacher_attr.io import save_json
from teacher_attr.metrics import grouped_indexes
from teacher_attr.pairs import load_pairs


def diagnose(cfg: dict, bertscore: bool = False) -> dict:
    import torch
    from rouge_score.rouge_scorer import RougeScorer
    from transformers import AutoTokenizer

    from teacher_attr.encoders import AttributionEncoder

    root = initialize_run(cfg)
    rows = load_pairs(cfg)["val"]
    teachers = list(cfg["teachers"])
    enc = {**cfg["encoder"], **cfg["evaluation"]["generic_encoder"], "input_mode": "response"}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttributionEncoder.pretrained(enc, len(teachers)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc["hf_name"], revision=enc["revision"])
    encoded = encode_rows(model, tokenizer, rows, enc, teachers, device, False)
    del model
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    report = {"split": "val", "role": "diagnostic only", "students": {}, "encoder": enc}
    for student, ix in grouped_indexes([r["student_id"] for r in rows]).items():
        selected = [rows[i] for i in ix]
        target = teachers.index(selected[0]["true_teacher"])
        stats = {
            "own_teacher": teachers[target],
            "mean_words": float(np.mean([len(r["student_response"].split()) for r in selected])),
            "semantic_similarity": {
                t: float(encoded["cosine"][ix, j].mean()) for j, t in enumerate(teachers)
            },
            "rougeL_against_teachers": {
                t: float(
                    np.mean(
                        [
                            rouge.score(r["teacher_responses"][t], r["student_response"])[
                                "rougeL"
                            ].fmeasure
                            for r in selected
                        ]
                    )
                )
                for t in teachers
            },
        }
        qa = [r for r in selected if r.get("metadata", {}).get("answer")]
        if qa:
            answers = [re.findall(r"\b([A-E])\b", r["student_response"]) for r in qa]
            stats["qa_final_letter_accuracy"] = float(
                np.mean(
                    [
                        bool(a) and a[-1] == r["metadata"]["answer"]
                        for a, r in zip(answers, qa, strict=True)
                    ]
                )
            )
            stats["qa_scoring_rule"] = "last standalone uppercase A-E; exact answer key"
        if bertscore:
            from bert_score import score

            stats["bertscore"] = {}
            scorer = cfg["evaluation"]["bertscore_model"]
            for teacher in teachers:
                _, _, f1 = score(
                    [r["student_response"] for r in selected],
                    [r["teacher_responses"][teacher] for r in selected],
                    model_type=scorer,
                    num_layers=cfg["evaluation"]["bertscore_layers"],
                    device=device,
                    verbose=False,
                )
                stats["bertscore"][teacher] = float(f1.mean())
        report["students"][student] = stats
    save_json(root / "diagnostics" / "distillation.json", report)
    return report
