from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


TOPICS = [
    "photosynthesis",
    "gravity in a vacuum",
    "supply and demand",
    "the French Revolution",
    "Bayes' theorem",
    "climate feedback loops",
    "sorting algorithms",
    "opportunity cost",
    "renewable energy storage",
    "plate tectonics",
    "inflation",
    "neural networks",
    "antibiotic resistance",
    "privacy in machine learning",
    "the water cycle",
    "game theory",
    "natural selection",
    "cryptographic hashes",
    "public goods",
    "stellar evolution",
]

TASK_TEMPLATES = {
    "qa": [
        "Why is {topic} important?",
        "What is the main idea behind {topic}?",
        "What is a common misconception about {topic}?",
    ],
    "explain": [
        "Explain {topic} to a first-year university student.",
        "Give a concise but rigorous explanation of {topic}.",
        "Explain {topic} using one concrete example.",
    ],
    "compare": [
        "Compare {topic} with a closely related concept.",
        "Describe two different perspectives on {topic}.",
        "What are the trade-offs involved in {topic}?",
    ],
    "reasoning": [
        "Solve a short reasoning problem involving {topic}. Show the key steps.",
        "If a policy changed one assumption about {topic}, what would likely happen?",
        "Build a simple causal chain for {topic}.",
    ],
    "writing": [
        "Write a short paragraph introducing {topic} to non-experts.",
        "Draft a neutral briefing note about {topic}.",
        "Summarize {topic} in exactly three sentences.",
    ],
}

WHO_TAUGHT_YOU_THAT_DATASETS = [
    "cnn_dailymail",
    "sumpubmed",
    "rotten_tomatoes",
    "commonsenseqa",
    "openbookqa",
    "quarel",
    "alpaca",
]

PROGRAMMING_RE = re.compile(
    r"\b(code|coding|program|programming|python|java|javascript|c\+\+|html|css|sql|regex|"
    r"function|class|algorithm|debug|compile|script|api|json|xml|bash|shell)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PromptBankConfig:
    split: str
    size: int
    seed: int
    start_index: int = 0


@dataclass(frozen=True)
class WTYTPromptConfig:
    split: str
    size: int
    seed: int
    datasets: tuple[str, ...] = tuple(WHO_TAUGHT_YOU_THAT_DATASETS)
    cache_dir: str | None = None
    local_data_dir: str | None = None
    max_input_chars: int = 6000
    include_qa_answer: bool = True
    allow_missing_datasets: bool = False


def who_taught_you_that_dataset_specs() -> list[dict[str, Any]]:
    """Return the HF/local dataset definitions used by the prompt-bank pipeline."""
    return [
        {
            "name": "cnn_dailymail",
            "task": "summarization",
            "candidates": [("cnn_dailymail", "3.0.0")],
            "split_map": {"distill": "train", "train": "train", "val": "validation", "test": "test"},
        },
        {
            "name": "sumpubmed",
            "task": "summarization",
            "candidates": [
                ("ccdv/pubmed-summarization", "section"),
                ("ccdv/pubmed-summarization", "document"),
                ("scientific_papers", "pubmed"),
                ("allenai/scientific_papers", "pubmed"),
            ],
            "split_map": {"distill": "train", "train": "train", "val": "validation", "test": "test"},
        },
        {
            "name": "rotten_tomatoes",
            "task": "summarization",
            "candidates": [
                ("contemmcm/rotten_tomatoes", "original"),
                ("cornell-movie-review-data/rotten_tomatoes", None),
                ("rotten_tomatoes", None),
            ],
            "split_map": {
                "distill": ["complete", "train"],
                "train": ["complete", "train"],
                "val": ["complete", "validation"],
                "test": ["complete", "test"],
            },
        },
        {
            "name": "commonsenseqa",
            "task": "qa",
            "candidates": [("tau/commonsense_qa", None), ("commonsense_qa", None)],
            "split_map": {"distill": "train", "train": "train", "val": "validation", "test": "validation"},
        },
        {
            "name": "openbookqa",
            "task": "qa",
            "candidates": [("allenai/openbookqa", "main"), ("openbookqa", "main")],
            "split_map": {"distill": "train", "train": "train", "val": "validation", "test": "test"},
        },
        {
            "name": "quarel",
            "task": "qa",
            "candidates": [("quarel", None), ("allenai/quarel", None)],
            "split_map": {"distill": "train", "train": "train", "val": "validation", "test": "test"},
        },
        {
            "name": "alpaca",
            "task": "instruction_following",
            "candidates": [("tatsu-lab/alpaca", None)],
            "split_map": {"distill": "train", "train": "train", "val": "train", "test": "train"},
        },
    ]


def make_prompt_rows(config: PromptBankConfig) -> list[dict[str, str]]:
    rng = random.Random(config.seed)
    tasks = sorted(TASK_TEMPLATES)
    rows: list[dict[str, str]] = []

    for offset in range(config.size):
        task = tasks[(offset + config.start_index) % len(tasks)]
        template = rng.choice(TASK_TEMPLATES[task])
        topic = rng.choice(TOPICS)
        prompt = template.format(topic=topic)
        global_idx = config.start_index + offset
        prompt_id = f"{config.split}_{global_idx:06d}"
        rows.append(
            {
                "prompt_id": prompt_id,
                "task": task,
                "split": config.split,
                "prompt": prompt,
                "paraphrase_group": prompt_id,
                "source_dataset": "synthetic",
            }
        )
    return rows


def make_who_taught_you_that_prompt_rows(config: WTYTPromptConfig) -> list[dict[str, Any]]:
    """Build prompt rows from the datasets used in Wadhwa et al. (2025)."""
    if config.size <= 0:
        return []

    specs = {spec["name"]: spec for spec in _wtyt_specs(config)}
    unknown = [name for name in config.datasets if name not in specs]
    if unknown:
        raise ValueError(f"Unknown Who-Taught-You-That dataset names: {unknown}")

    active_specs = [specs[name] for name in config.datasets]
    target_per_dataset = math.ceil(config.size / len(active_specs))
    pools: list[list[dict[str, Any]]] = []
    errors: list[str] = []

    for dataset_idx, spec in enumerate(active_specs):
        try:
            pool = _rows_from_wtyt_dataset(
                spec=spec,
                split=config.split,
                size=target_per_dataset,
                seed=config.seed + dataset_idx * 1009,
                cache_dir=config.cache_dir,
                local_data_dir=config.local_data_dir,
            )
            pools.append(pool)
        except Exception as exc:
            message = f"{spec['name']}: {exc}"
            if config.allow_missing_datasets:
                print(f"warning: skipping {message}")
            else:
                errors.append(message)

    if errors:
        raise RuntimeError("Failed to load one or more WTYT datasets:\n" + "\n".join(errors))

    rows: list[dict[str, Any]] = []
    for idx in range(target_per_dataset):
        for pool in pools:
            if idx < len(pool):
                rows.append(pool[idx])
            if len(rows) >= config.size:
                break
        if len(rows) >= config.size:
            break

    for idx, row in enumerate(rows):
        row["prompt_id"] = f"{config.split}_{row['source_dataset']}_{idx:06d}"
        row["split"] = config.split
        row["paraphrase_group"] = row["prompt_id"]
    return rows


def _wtyt_specs(config: WTYTPromptConfig) -> list[dict[str, Any]]:
    max_chars = config.max_input_chars
    formatters = {
        "cnn_dailymail": lambda row: _format_cnn_dailymail(row, max_chars),
        "sumpubmed": lambda row: _format_pubmed(row, max_chars),
        "rotten_tomatoes": lambda row: _format_rotten_tomatoes(row, max_chars),
        "commonsenseqa": lambda row: _format_multiple_choice(
            row,
            question_keys=("question",),
            include_answer=config.include_qa_answer,
            max_chars=max_chars,
        ),
        "openbookqa": lambda row: _format_multiple_choice(
            row,
            question_keys=("question_stem", "question"),
            include_answer=config.include_qa_answer,
            max_chars=max_chars,
        ),
        "quarel": lambda row: _format_multiple_choice(
            row,
            question_keys=("question", "question_text"),
            include_answer=config.include_qa_answer,
            max_chars=max_chars,
        ),
        "alpaca": lambda row: _format_alpaca(row, max_chars),
    }
    specs = []
    for spec in who_taught_you_that_dataset_specs():
        spec = dict(spec)
        spec["formatter"] = formatters[spec["name"]]
        specs.append(spec)
    return specs


def _rows_from_wtyt_dataset(
    spec: dict[str, Any],
    split: str,
    size: int,
    seed: int,
    cache_dir: str | None,
    local_data_dir: str | None,
) -> list[dict[str, Any]]:
    candidate_splits = _candidate_splits(spec, split)
    if local_data_dir:
        dataset, source_name, source_split = _load_local_wtyt_dataset(
            spec["name"], candidate_splits, local_data_dir
        )
    else:
        dataset, source_name, source_split = _load_first_available_dataset(
            spec["candidates"], candidate_splits, cache_dir
        )
    indexes = _sample_indexes(len(dataset), size=size, seed=seed, offset=_split_offset(split))
    rows: list[dict[str, Any]] = []
    formatter: Callable[[dict[str, Any]], str | None] = spec["formatter"]

    for idx in indexes:
        raw = dict(dataset[int(idx)])
        prompt = formatter(raw)
        if not prompt:
            continue
        rows.append(
            {
                "prompt_id": "pending",
                "task": spec["task"],
                "split": split,
                "prompt": prompt,
                "paraphrase_group": "pending",
                "source_dataset": spec["name"],
                "source_hf_dataset": source_name,
                "source_split": source_split,
                "source_index": int(idx),
                "source_id": str(_first_nonempty(raw, "id", "question_id", "movieId", "reviewId") or idx),
            }
        )
        if len(rows) >= size:
            break
    return rows


def _load_first_available_dataset(
    candidates: list[tuple[str, str | None]],
    splits: list[str],
    cache_dir: str | None,
):
    from datasets import load_dataset

    errors: list[str] = []
    for path, name in candidates:
        for split in splits:
            try:
                if name is None:
                    dataset = load_dataset(path, split=split, cache_dir=cache_dir)
                    source_name = path
                else:
                    dataset = load_dataset(path, name, split=split, cache_dir=cache_dir)
                    source_name = f"{path}/{name}"
                return dataset, source_name, split
            except Exception as exc:
                errors.append(f"{path}/{name or 'default'}[{split}]: {exc}")
    raise RuntimeError("; ".join(errors[-4:]))


def _load_local_wtyt_dataset(dataset_name: str, splits: list[str], local_data_dir: str):
    from datasets import load_dataset

    root = Path(local_data_dir) / dataset_name
    errors: list[str] = []
    for split in splits:
        for extension, loader in (("parquet", "parquet"), ("csv", "csv")):
            path = root / f"{split}.{extension}"
            if not path.exists():
                continue
            try:
                dataset = load_dataset(loader, data_files=str(path), split="train")
                return dataset, f"local:{dataset_name}", split
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    searched = ", ".join(str(root / f"{split}.parquet") for split in splits)
    searched += "; " + ", ".join(str(root / f"{split}.csv") for split in splits)
    raise FileNotFoundError(f"No local files found for {dataset_name}; searched {searched}. {'; '.join(errors)}")


def _candidate_splits(spec: dict[str, Any], project_split: str) -> list[str]:
    split_value = spec.get("split_map", {}).get(project_split, project_split)
    if isinstance(split_value, str):
        return [split_value]
    return list(split_value)


def _split_offset(split: str) -> int:
    return {"distill": 0, "train": 10000, "val": 20000, "test": 30000}.get(split, 0)


def _sample_indexes(num_rows: int, size: int, seed: int, offset: int) -> list[int]:
    if num_rows <= 0:
        return []
    rng = random.Random(seed)
    indexes = list(range(num_rows))
    rng.shuffle(indexes)
    start = offset % num_rows
    rotated = indexes[start:] + indexes[:start]
    return rotated[: min(size * 3, num_rows)]


def _format_cnn_dailymail(row: dict[str, Any], max_chars: int) -> str | None:
    article = _clean_text(_first_nonempty(row, "article", "document", "text"), max_chars)
    if not article:
        return None
    return "Summarize the following CNN/DailyMail news article in a concise paragraph.\n\nArticle:\n" + article


def _format_pubmed(row: dict[str, Any], max_chars: int) -> str | None:
    article = _clean_text(
        _first_nonempty(row, "article", "document", "text", "body", "abstract", "sections"),
        max_chars,
    )
    if not article:
        return None
    return "Write a concise biomedical abstract-style summary of the following PubMed article.\n\nArticle:\n" + article


def _format_rotten_tomatoes(row: dict[str, Any], max_chars: int) -> str | None:
    review = _clean_text(_first_nonempty(row, "reviewText", "text", "review", "content"), max_chars)
    if not review:
        return None
    return (
        "Write a concise Rotten Tomatoes-style meta-review that captures the critic's opinion "
        "and sentiment.\n\nCritic review:\n" + review
    )


def _format_multiple_choice(
    row: dict[str, Any],
    question_keys: tuple[str, ...],
    include_answer: bool,
    max_chars: int,
) -> str | None:
    question = _clean_text(_first_nonempty(row, *question_keys), max_chars)
    choices = _choices_to_text(row.get("choices") or row.get("answer_choices") or row.get("options"))
    if not question or not choices:
        return None
    answer = _first_nonempty(row, "answerKey", "answer", "label")
    prefix = "Answer the multiple-choice question and provide a brief explanation."
    if include_answer and answer not in (None, ""):
        prefix = (
            "The correct answer key is "
            f"{answer}. Provide a brief reasoning chain that supports that answer."
        )
    return f"{prefix}\n\nQuestion:\n{question}\n\nChoices:\n{choices}"


def _format_alpaca(row: dict[str, Any], max_chars: int) -> str | None:
    instruction = _clean_text(_first_nonempty(row, "instruction", "prompt"), max_chars)
    input_text = _clean_text(_first_nonempty(row, "input", "context"), max_chars)
    joined = "\n".join(part for part in (instruction, input_text) if part)
    if not instruction or PROGRAMMING_RE.search(joined):
        return None
    if input_text:
        return f"{instruction}\n\nInput:\n{input_text}"
    return instruction


def _choices_to_text(choices: Any) -> str:
    if choices is None:
        return ""
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels") or []
        texts = choices.get("text") or choices.get("texts") or []
        if isinstance(labels, str):
            labels = [labels]
        if isinstance(texts, str):
            texts = [texts]
        if texts:
            lines = []
            for idx, text in enumerate(texts):
                label = labels[idx] if idx < len(labels) else chr(ord("A") + idx)
                lines.append(f"{label}. {_clean_text(text, 1000)}")
            return "\n".join(lines)
    if isinstance(choices, list):
        lines = []
        for idx, choice in enumerate(choices):
            if isinstance(choice, dict):
                label = choice.get("label") or choice.get("key") or chr(ord("A") + idx)
                text = choice.get("text") or choice.get("answer") or choice.get("value")
            else:
                label = chr(ord("A") + idx)
                text = choice
            if text:
                lines.append(f"{label}. {_clean_text(text, 1000)}")
        return "\n".join(lines)
    return str(choices)


def _first_nonempty(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, "", []):
            return value
    return None


def _clean_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = "\n".join(str(item) for item in value)
    if isinstance(value, dict):
        value = "\n".join(str(item) for item in value.values())
    text = re.sub(r"\s+", " ", str(value)).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0].strip()
    return text
