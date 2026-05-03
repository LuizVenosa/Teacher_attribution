from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher_order_from_config(models_cfg: dict[str, Any]) -> list[str]:
    labels = models_cfg["teacher_labels"]
    return [teacher_id for teacher_id, _ in sorted(labels.items(), key=lambda item: item[1])]


def env_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def count_parameters(model: Any) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
