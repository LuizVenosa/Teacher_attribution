from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


VIRIDIS_STOPS = [
    (0.00, (68, 1, 84)),
    (0.25, (59, 82, 139)),
    (0.50, (33, 145, 140)),
    (0.75, (94, 201, 98)),
    (1.00, (253, 231, 37)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot set-level confusion-matrix progression for the report."
    )
    parser.add_argument(
        "--metrics",
        default="results/contrastive/public_lineage_minilm_probe_metrics.json",
    )
    parser.add_argument("--method", default="cosine_retrieval")
    parser.add_argument("--set_sizes", nargs="+", default=["1", "4", "16"])
    parser.add_argument("--output", default="results/figures/13_confusion_progression.png")
    parser.add_argument("--report_output", default="report/figures/13_confusion_progression.png")
    return parser.parse_args()


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["arialbd.ttf" if bold else "arial.ttf", "segoeuib.ttf" if bold else "segoeui.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def normalize_rows(matrix: list[list[int]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    return arr / np.maximum(arr.sum(axis=1, keepdims=True), 1.0)


def short_label(label: str) -> str:
    return {
        "gpt2": "GPT-2",
        "qwen15_18b": "Qwen",
        "flan_t5_base": "FLAN-base",
        "flan_t5_small": "FLAN-small",
    }.get(label, label)


def viridis(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    for (x0, c0), (x1, c1) in zip(VIRIDIS_STOPS[:-1], VIRIDIS_STOPS[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            return tuple(int(round(a + t * (b - a))) for a, b in zip(c0, c1))
    return VIRIDIS_STOPS[-1][1]


def centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    text_font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=text_font, fill=fill)


def draw_rotated_label(
    base: Image.Image,
    center: tuple[int, int],
    text: str,
    text_font: ImageFont.ImageFont,
    angle: int = -35,
) -> None:
    bbox = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=text_font)
    width = bbox[2] - bbox[0] + 8
    height = bbox[3] - bbox[1] + 8
    label = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    d = ImageDraw.Draw(label)
    d.text((4, 4), text, font=text_font, fill=(0, 0, 0, 255))
    rotated = label.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
    base.alpha_composite(rotated, (center[0] - rotated.width // 2, center[1] - rotated.height // 2))


def draw_panel(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    matrix: np.ndarray,
    title: str,
    labels: list[str],
    acc: float,
) -> None:
    title_font = font(36, bold=True)
    label_font = font(24)
    tick_font = font(23)
    cell_font = font(24)
    cell = 120
    grid = cell * len(labels)

    centered_text(draw, (x0 + grid / 2, y0 - 52), f"{title}  |  acc. {acc:.1%}", title_font, (0, 0, 0))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            color = viridis(value)
            x = x0 + j * cell
            y = y0 + i * cell
            draw.rectangle((x, y, x + cell, y + cell), fill=color)
            draw.rectangle((x, y, x + cell, y + cell), outline=(255, 255, 255), width=2)
            text = "0" if value == 0 else f"{value:.0%}"
            text_color = (255, 255, 255) if value < 0.35 else (0, 0, 0)
            centered_text(draw, (x + cell / 2, y + cell / 2), text, cell_font, text_color)

    for i, lab in enumerate(labels):
        centered_text(draw, (x0 - 72, y0 + i * cell + cell / 2), lab, tick_font, (0, 0, 0))
    for j, lab in enumerate(labels):
        draw_rotated_label(img, (x0 + j * cell + cell // 2, y0 + grid + 68), lab, tick_font)

    centered_text(draw, (x0 + grid / 2, y0 + grid + 135), "Predicted teacher", label_font, (0, 0, 0))
    # Local y-axis label kept horizontal to remain readable in a compact multi-panel figure.
    centered_text(draw, (x0 - 72, y0 - 15), "True", label_font, (0, 0, 0))


def draw_colorbar(img: Image.Image, draw: ImageDraw.ImageDraw, x0: int, y0: int, height: int) -> None:
    label_font = font(22)
    tick_font = font(21)
    width = 34
    for k in range(height):
        value = 1 - k / max(height - 1, 1)
        draw.line((x0, y0 + k, x0 + width, y0 + k), fill=viridis(value))
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(0, 0, 0), width=2)
    for value in [0.0, 0.5, 1.0]:
        y = y0 + int((1 - value) * height)
        draw.line((x0 + width, y, x0 + width + 8, y), fill=(0, 0, 0), width=2)
        draw.text((x0 + width + 14, y - 12), f"{value:.1f}", font=tick_font, fill=(0, 0, 0))
    centered_text(draw, (x0 + width / 2, y0 - 30), "Row share", label_font, (0, 0, 0))


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    set_level = metrics["set_level"][args.method]
    labels = [short_label(x) for x in metrics["teacher_ids"]]
    matrices = [normalize_rows(set_level[size]["confusion_matrix"]) for size in args.set_sizes]
    accuracies = [set_level[size]["accuracy"] for size in args.set_sizes]

    width, height = 2350, 820
    img = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    title_font = font(42, bold=True)
    centered_text(draw, (width / 2, 45), "Aggregation Makes Teacher Confusions Disappear", title_font, (0, 0, 0))

    x_positions = [180, 835, 1490]
    y0 = 145
    titles = [f"{size} prompt" + ("" if size == "1" else "s") for size in args.set_sizes]
    for x0, title, matrix, acc in zip(x_positions, titles, matrices, accuracies):
        draw_panel(img, draw, x0, y0, matrix, title, labels, acc)
    draw_colorbar(img, draw, 2180, y0, 480)

    out = img.convert("RGB")
    for output in [args.output, args.report_output]:
        if output:
            path = Path(output)
            path.parent.mkdir(parents=True, exist_ok=True)
            out.save(path, quality=95)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
