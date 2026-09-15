from __future__ import annotations

import argparse
import json

from teacher_attr.config import SPLITS, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="One pipeline for reproducible lineage experiments."
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build and audit disjoint prompt splits")
    gen = commands.add_parser(
        "generate", help="Generate/resume model outputs with per-prompt seeds"
    )
    gen.add_argument("--role", choices=["teachers", "students"])
    gen.add_argument("--model", help="Generate only this configured model")
    gen.add_argument("--split", choices=SPLITS)
    commands.add_parser("build", help="Validate alignment and build attribution pairs")
    commands.add_parser("audit", help="Verify prompt/pair hashes and split disjointness")
    train = commands.add_parser("train", help="Train the encoder or a matched loss/input ablation")
    train.add_argument("--objective", choices=["joint", "classification", "contrastive"])
    train.add_argument("--input-mode", choices=["response", "prompt_response"])
    train.add_argument("--seed", type=int)
    evaluate = commands.add_parser("evaluate", help="Evaluate baselines and optional encoder")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument(
        "--frozen", action="store_true", help="Include pretrained MiniLM controls"
    )
    evaluate.add_argument("--name", default="evaluation")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.command == "prepare":
        from teacher_attr.prompts import prepare

        result = prepare(cfg)
    elif args.command == "generate":
        from teacher_attr.generation import generate

        if args.model and not args.role:
            parser.error("--model requires --role")
        if args.model and args.model not in cfg[args.role]:
            parser.error(f"Unknown {args.role} model: {args.model}")
        result = []
        for role in [args.role] if args.role else ["teachers", "students"]:
            for model_id in [args.model] if args.model else cfg[role]:
                for split in [args.split] if args.split else SPLITS:
                    if role == "students" and split not in cfg[role][model_id]["splits"]:
                        if args.model and args.split:
                            parser.error(f"{model_id} is not assigned to {split}")
                        continue
                    result.append(generate(cfg, role, model_id, split))
    elif args.command == "build":
        from teacher_attr.pairs import build

        result = build(cfg)
    elif args.command == "audit":
        from pathlib import Path

        from teacher_attr.pairs import load_pairs
        from teacher_attr.prompts import audit_splits, verify_prompts

        result = {
            "prompts": verify_prompts(Path(cfg["run_dir"])),
            "pairs": audit_splits(load_pairs(cfg)),
        }
    elif args.command == "train":
        from teacher_attr.training import train

        result = train(cfg, args.objective, args.input_mode, args.seed)
    else:
        from teacher_attr.evaluation import evaluate

        result = evaluate(cfg, args.checkpoint, args.frozen, args.name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
