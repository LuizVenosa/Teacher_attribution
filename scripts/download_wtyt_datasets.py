from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from teacher_attr.prompts import WHO_TAUGHT_YOU_THAT_DATASETS, who_taught_you_that_dataset_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Who Taught You That source datasets to local Parquet/CSV files."
    )
    parser.add_argument("--output-dir", default="external_datasets/who_taught_you_that")
    parser.add_argument(
        "--datasets",
        default=",".join(WHO_TAUGHT_YOU_THAT_DATASETS),
        help="Comma-separated subset to download.",
    )
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--format", choices=["auto", "parquet", "csv"], default="auto")
    parser.add_argument("--allow-missing-datasets", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional debugging cap per split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _csv(args.datasets)
    specs = {spec["name"]: spec for spec in who_taught_you_that_dataset_specs()}
    unknown = [name for name in selected if name not in specs]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")

    file_format = _choose_format(args.format)
    manifest: dict[str, Any] = {"format": file_format, "datasets": {}}
    errors: list[str] = []

    for dataset_name in selected:
        spec = specs[dataset_name]
        try:
            manifest["datasets"][dataset_name] = download_dataset_spec(
                spec=spec,
                output_dir=output_dir,
                file_format=file_format,
                cache_dir=args.hf_cache_dir,
                force=args.force,
                max_rows=args.max_rows,
            )
        except Exception as exc:
            message = f"{dataset_name}: {exc}"
            if args.allow_missing_datasets:
                print(f"warning: skipping {message}", flush=True)
                manifest["datasets"][dataset_name] = {"error": repr(exc)}
            else:
                errors.append(message)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote manifest to {manifest_path}", flush=True)

    if errors:
        raise RuntimeError("Failed to download one or more datasets:\n" + "\n".join(errors))


def download_dataset_spec(
    spec: dict[str, Any],
    output_dir: Path,
    file_format: str,
    cache_dir: str | None,
    force: bool,
    max_rows: int | None,
) -> dict[str, Any]:
    dataset_dir = output_dir / spec["name"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    split_names = _all_needed_source_splits(spec)
    split_manifest = {}

    print(f"\n=== {spec['name']} ===", flush=True)
    for split in split_names:
        out_path = dataset_dir / f"{split}.{file_format}"
        if out_path.exists() and not force:
            print(f"skip existing {out_path}", flush=True)
            split_manifest[split] = {"path": str(out_path), "rows": None, "status": "exists"}
            continue

        dataset, source = _load_split(spec["candidates"], split, cache_dir)
        if max_rows is not None:
            dataset = dataset.select(range(min(max_rows, len(dataset))))
        _write_dataset(dataset, out_path, file_format)
        split_manifest[split] = {
            "path": str(out_path),
            "rows": len(dataset),
            "columns": list(dataset.column_names),
            "source": source,
            "status": "written",
        }
        print(f"wrote {len(dataset):,} rows to {out_path}", flush=True)

    (dataset_dir / "metadata.json").write_text(
        json.dumps({"spec": _jsonable_spec(spec), "splits": split_manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    return split_manifest


def _load_split(candidates: list[tuple[str, str | None]], split: str, cache_dir: str | None):
    from datasets import load_dataset

    errors = []
    for path, config_name in candidates:
        try:
            if config_name is None:
                dataset = load_dataset(path, split=split, cache_dir=cache_dir)
                source = path
            else:
                dataset = load_dataset(path, config_name, split=split, cache_dir=cache_dir)
                source = f"{path}/{config_name}"
            print(f"loaded {source}[{split}] rows={len(dataset):,}", flush=True)
            return dataset, source
        except Exception as exc:
            errors.append(f"{path}/{config_name or 'default'}[{split}]: {exc}")
    raise RuntimeError("; ".join(errors[-4:]))


def _write_dataset(dataset, out_path: Path, file_format: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if file_format == "parquet":
        dataset.to_parquet(str(out_path))
    elif file_format == "csv":
        dataset.to_csv(str(out_path))
    else:
        raise ValueError(f"Unknown file format: {file_format}")


def _choose_format(requested: str) -> str:
    if requested == "csv":
        return "csv"
    if requested == "parquet":
        return "parquet"
    try:
        import pyarrow  # noqa: F401

        return "parquet"
    except Exception:
        return "csv"


def _all_needed_source_splits(spec: dict[str, Any]) -> list[str]:
    split_map = spec.get("split_map", {})
    splits: list[str] = []
    for project_split in ("distill", "train", "val", "test"):
        value = split_map.get(project_split, project_split)
        if isinstance(value, str):
            candidates = [value]
        else:
            candidates = list(value)
        for split in candidates:
            if split not in splits:
                splits.append(split)
    return splits


def _jsonable_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": spec["name"],
        "task": spec["task"],
        "candidates": [[path, config] for path, config in spec["candidates"]],
        "split_map": spec.get("split_map", {}),
    }


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
