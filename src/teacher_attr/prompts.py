from __future__ import annotations

import random
from dataclasses import dataclass


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


@dataclass(frozen=True)
class PromptBankConfig:
    split: str
    size: int
    seed: int
    start_index: int = 0


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
            }
        )
    return rows
