"""Output controls and pre-distillation diagnostics; never conceal rejected samples."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from teacher_attr.config import file_hash, initialize_run
from teacher_attr.generation import output_path, read_outputs
from teacher_attr.io import load_jsonl, save_json


def clean_response(
    text: str, patterns: list[str], terminal_tokens: tuple[str, ...] = ()
) -> tuple[str, list[str]]:
    # Only known terminal/padding tokens at the end are harmless framing.
    text = text.strip()
    while True:
        token = next((t for t in terminal_tokens if t and text.endswith(t)), None)
        if token is None:
            break
        text = text[: -len(token)].rstrip()
    flags = []
    if "<|channel>thought" in text:
        blocks = re.findall(r"<\|channel>thought\s*(.*?)<channel\|>", text, flags=re.S)
        if any(block.strip() for block in blocks):
            flags.append("reasoning_markup")
        text = re.sub(r"<\|channel>thought.*?<channel\|>", "", text, flags=re.S)
        if "<|channel>thought" in text:
            flags.append("reasoning_markup")
            text = text.split("<|channel>thought")[0]
    if re.search(r"<think>|</think>|\[THINK\]|\[/THINK\]", text, re.I):
        flags.append("reasoning_markup")
        # Never persist hidden reasoning. An unclosed block has no usable final answer.
        text = re.sub(r"<think>.*?</think>|\[THINK\].*?\[/THINK\]", "", text, flags=re.S | re.I)
        if re.search(r"<think>|\[THINK\]", text, re.I):
            text = re.split(r"<think>|\[THINK\]", text, flags=re.I)[0]
        if re.search(r"</think>|\[/THINK\]", text, re.I):
            text = re.split(r"</think>|\[/THINK\]", text, flags=re.I)[-1]
    if re.search(r"<\|.*?\|>|\[/?INST\]", text):
        flags.append("template_markup")
        text = re.sub(r"<\|.*?\|>|\[/?INST\]", "", text)
    # Provider names in ordinary factual content are not self-identification.
    # Match the first-person clause containing the configured identity name.
    clauses = re.findall(
        r"\b(?:I\s+am|I'm|I’m|I\s+was|I\s+have\s+been|as\s+an?\b|"
        r"my\s+(?:name|creator|developer|provider)\s+is)\b[^.!?\n]{0,160}",
        text,
        re.I,
    )
    if any(re.search(p, clause, re.I) for clause in clauses for p in patterns):
        flags.append("identity_marker")
    if not text.strip():
        flags.append("empty")
    return text.strip(), flags


def describe(rows: list[dict], refusal_patterns: list[str]) -> dict:
    lengths = np.array([len(r["response"].split()) for r in rows])
    invalid = [bool(r.get("quality_flags")) or not r["response"].strip() for r in rows]
    return {
        "rows": len(rows),
        "mean_words": float(lengths.mean()),
        "median_words": float(np.median(lengths)),
        "mean_tokens": float(np.mean([r.get("output_tokens", 0) for r in rows])),
        "vocabulary_size": len({w.lower() for r in rows for w in r["response"].split()}),
        "invalid_rate": float(np.mean(invalid)),
        "flag_counts": dict(Counter(f for r in rows for f in r.get("quality_flags", []))),
        "truncated_rate": float(np.mean([r.get("truncated", False) for r in rows])),
        "refusal_rate": float(
            np.mean(
                [any(re.search(p, r["response"], re.I) for p in refusal_patterns) for r in rows]
            )
        ),
    }


def apply_length_exception(report: dict, accept: bool) -> None:
    report["strict_passed"] = not report["flags"]
    report["accepted_flags"] = [
        flag
        for flag in report["flags"]
        if accept and flag.endswith(": teacher response length imbalance")
    ]
    report["blocking_flags"] = [f for f in report["flags"] if f not in report["accepted_flags"]]
    report["passed"] = not report["blocking_flags"]
    report["length_exception_requested"] = accept


def quality_control(
    cfg: dict, split: str = "distill_train", accept_length_imbalance: bool = False
) -> dict:
    root = initialize_run(cfg)
    settings = cfg["research"]["quality"]
    expected = {r["prompt_id"] for r in load_jsonl(root / "prompts" / f"{split}.jsonl")}
    report = {"split": split, "teachers": {}, "hashes": {}, "passed": True, "flags": []}
    for teacher in cfg["teachers"]:
        rows = read_outputs(cfg, "teachers", teacher, split)
        if {r["prompt_id"] for r in rows} != expected:
            raise ValueError(f"Incomplete teacher outputs: {teacher}/{split}")
        stats = describe(rows, settings["refusal_patterns"])
        stats["by_task"] = {
            task: describe([r for r in rows if r["task"] == task], settings["refusal_patterns"])
            for task in sorted({r["task"] for r in rows})
        }
        report["teachers"][teacher] = stats
        report["hashes"][teacher] = file_hash(output_path(root, "teachers", teacher, split))
        if (
            stats["invalid_rate"] > settings["max_invalid_rate"]
            or stats["truncated_rate"] > settings["max_truncated_rate"]
        ):
            report["flags"].append(f"{teacher}: invalid/truncated response threshold exceeded")
    tasks = sorted({task for s in report["teachers"].values() for task in s["by_task"]})
    for task in tasks:
        lengths = [s["by_task"][task]["mean_words"] for s in report["teachers"].values()]
        if max(lengths) / max(min(lengths), 1) > settings["max_length_ratio"]:
            report["flags"].append(f"{task}: teacher response length imbalance")
    apply_length_exception(report, accept_length_imbalance)
    out = root / "quality" / split
    save_json(out / "metrics.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(report["teachers"])
    ax.bar(names, [report["teachers"][n]["mean_words"] for n in names])
    ax.set(ylabel="Mean response length (words)", title=f"Teacher output audit: {split}")
    fig.tight_layout()
    fig.savefig(out / "lengths.pdf")
    plt.close(fig)
    return report
