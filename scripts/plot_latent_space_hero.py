from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from teacher_attr.datasets import format_prompt_response
from teacher_attr.encoders import AttributionEncoder
from teacher_attr.generation import batch_iter
from teacher_attr.io import load_jsonl, load_yaml
from teacher_attr.utils import get_device, teacher_order_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot raw and contrastive latent spaces for public-lineage attribution."
    )
    parser.add_argument("--models_config", default="configs/public_lineage_models_minilm.yaml")
    parser.add_argument("--attribution_config", default="configs/public_lineage_attribution_minilm_ablate.yaml")
    parser.add_argument("--pairs", default="data/attribution/test_pairs.jsonl")
    parser.add_argument(
        "--checkpoint",
        default="models/attribution_encoder/public_lineage_minilm_contrastive/best.pt",
    )
    parser.add_argument("--output", default="results/figures/00_latent_space_hero.png")
    parser.add_argument("--max_per_teacher", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--write_svg", action="store_true")
    return parser.parse_args()


def offline_mode() -> bool:
    return bool(
        os.environ.get("HF_HUB_OFFLINE")
        or os.environ.get("TRANSFORMERS_OFFLINE")
        or os.environ.get("HF_DATASETS_OFFLINE")
    )


def sample_rows(
    rows: list[dict[str, Any]],
    teacher_ids: list[str],
    max_per_teacher: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("true_teacher") in teacher_ids:
            grouped[row["true_teacher"]].append(row)

    sampled = []
    for teacher_id in teacher_ids:
        group = grouped[teacher_id]
        if len(group) > max_per_teacher:
            idx = rng.choice(len(group), size=max_per_teacher, replace=False)
            sampled.extend(group[int(i)] for i in idx)
        else:
            sampled.extend(group)
    rng.shuffle(sampled)
    return sampled


def load_checkpoint(path: str | Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_attribution_model(
    checkpoint_path: str | Path,
    model_name: str,
    projection_dim: int,
    num_teachers: int,
    device: torch.device,
) -> AttributionEncoder:
    model = AttributionEncoder(
        model_name=model_name,
        projection_dim=projection_dim,
        num_teachers=num_teachers,
        local_files_only=offline_mode(),
    )
    payload = load_checkpoint(checkpoint_path)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def encode_base(
    model: AutoModel,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    total = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(batch_iter(texts, batch_size), total=total, desc="raw MiniLM", dynamic_ncols=True):
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        outputs = model(**tokens)
        token_emb = outputs.last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).float()
        pooled = (token_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        chunks.append(F.normalize(pooled, dim=-1).float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def encode_contrastive(
    model: AttributionEncoder,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    chunks = []
    total = (len(texts) + batch_size - 1) // batch_size
    for batch in tqdm(batch_iter(texts, batch_size), total=total, desc="contrastive", dynamic_ncols=True):
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        emb, _ = model(**tokens)
        chunks.append(emb.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def reduce_to_2d(x: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    if x.shape[0] < 4:
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    pre_dims = min(50, x.shape[1], x.shape[0] - 1)
    x_pre = PCA(n_components=pre_dims, random_state=seed).fit_transform(x)
    safe_perplexity = min(perplexity, max(2.0, (x.shape[0] - 1) / 3.0))
    return TSNE(
        n_components=2,
        perplexity=safe_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x_pre)


def rescale(points: np.ndarray) -> np.ndarray:
    lo = points.min(axis=0, keepdims=True)
    hi = points.max(axis=0, keepdims=True)
    return (points - lo) / np.maximum(hi - lo, 1e-9)


def plot_spaces(
    raw_2d: np.ndarray,
    contrastive_2d: np.ndarray,
    labels: np.ndarray,
    teacher_ids: list[str],
    output: str | Path,
    dpi: int,
    write_svg: bool,
) -> None:
    colors = {
        "gpt2": "#3b82f6",
        "qwen15_18b": "#f59e0b",
        "flan_t5_base": "#22c55e",
        "flan_t5_small": "#a855f7",
    }
    display = {
        "gpt2": "GPT-2 lineage",
        "qwen15_18b": "Qwen lineage",
        "flan_t5_base": "FLAN-T5-base lineage",
        "flan_t5_small": "FLAN-T5-small lineage",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), dpi=dpi)
    for ax, points, title in zip(
        axes,
        [rescale(raw_2d), rescale(contrastive_2d)],
        ["(a) Raw MiniLM space", "(b) Contrastive teacher space"],
    ):
        for idx, teacher_id in enumerate(teacher_ids):
            mask = labels == idx
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=7,
                alpha=0.72,
                linewidths=0,
                c=colors.get(teacher_id, None),
                label=display.get(teacher_id, teacher_id),
            )
        ax.set_title(title, fontsize=17, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=12,
        markerscale=1.8,
        bbox_to_anchor=(0.5, -0.035),
    )
    fig.tight_layout(rect=[0, 0.13, 1, 0.98], w_pad=1.6)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    if write_svg:
        fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    models_cfg = load_yaml(args.models_config)
    attr_cfg = load_yaml(args.attribution_config)
    teacher_ids = teacher_order_from_config(models_cfg)
    model_name = models_cfg["attribution_encoder"]["hf_name"]
    max_length = int(attr_cfg["max_length"])
    projection_dim = int(attr_cfg["projection_dim"])

    rows = load_jsonl(args.pairs)
    rows = sample_rows(rows, teacher_ids, args.max_per_teacher, args.seed)
    labels = np.asarray([teacher_ids.index(row["true_teacher"]) for row in rows], dtype=np.int64)
    texts = [
        format_prompt_response(row["prompt"], row["student_response"])
        for row in rows
    ]

    device = get_device()
    tokenizer_source = Path(args.checkpoint).parent
    if not (tokenizer_source / "tokenizer_config.json").exists():
        tokenizer_source = model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=offline_mode(),
    )

    base_model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=offline_mode(),
    ).to(device)
    base_model.eval()
    contrastive_model = load_attribution_model(
        args.checkpoint,
        model_name,
        projection_dim,
        len(teacher_ids),
        device,
    )

    raw_emb = encode_base(base_model, tokenizer, texts, device, max_length, args.batch_size)
    contrastive_emb = encode_contrastive(
        contrastive_model,
        tokenizer,
        texts,
        device,
        max_length,
        args.batch_size,
    )

    raw_2d = reduce_to_2d(raw_emb, args.seed, args.perplexity)
    contrastive_2d = reduce_to_2d(contrastive_emb, args.seed, args.perplexity)
    plot_spaces(raw_2d, contrastive_2d, labels, teacher_ids, args.output, args.dpi, args.write_svg)
    print(f"wrote {args.output}")
    if args.write_svg:
        print(f"wrote {Path(args.output).with_suffix('.svg')}")


if __name__ == "__main__":
    main()
