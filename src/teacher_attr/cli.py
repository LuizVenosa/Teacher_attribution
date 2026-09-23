from __future__ import annotations

import argparse
import json

from teacher_attr.config import SPLITS, load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="One pipeline for reproducible lineage experiments."
    )
    parser.add_argument("--config", default="configs/research.yaml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Build and audit disjoint prompt splits")
    commands.add_parser("audit-tokens", help="Audit every prompt pool using tokenizers only")
    migration = commands.add_parser(
        "migrate-budgets", help="Raise budgets in a new run and reuse complete teacher outputs"
    )
    migration.add_argument("--output", required=True)
    migration.add_argument("--input-tokens", required=True, type=int)
    migration.add_argument("--training-tokens", required=True, type=int)
    pre = commands.add_parser("preflight", help="Check model/tokenizer access without weights")
    pre.add_argument("--cache-dir")
    gen = commands.add_parser(
        "generate", help="Generate/resume model outputs with per-prompt seeds"
    )
    gen.add_argument("--role", choices=["teachers", "students"])
    gen.add_argument("--model", help="Generate only this configured model")
    gen.add_argument("--split", choices=(*SPLITS, "distill_train", "distill_val"))
    commands.add_parser("build", help="Validate alignment and build attribution pairs")
    commands.add_parser("audit", help="Verify prompt/pair hashes and split disjointness")
    train = commands.add_parser("train", help="Train the encoder or a matched loss/input ablation")
    train.add_argument("--objective", choices=["joint", "classification", "contrastive"])
    train.add_argument("--input-mode", choices=["response", "prompt_response"])
    train.add_argument("--seed", type=int)
    train.add_argument("--representation", choices=["semantic", "structure", "fusion"])
    train.add_argument("--negatives", choices=["same_prompt", "random"])
    train.add_argument("--classification-weight", type=float)
    train.add_argument("--baseline", help="Required before latent-structure training")
    evaluate = commands.add_parser("evaluate", help="Evaluate baselines and optional encoder")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument(
        "--reference-config",
        help="Keep the original encoder and probe training data for robustness tests",
    )
    evaluate.add_argument(
        "--frozen", action="store_true", help="Include pretrained MiniLM controls"
    )
    evaluate.add_argument("--name", default="evaluation")
    qc = commands.add_parser("qc", help="Audit all teachers before student training")
    qc.add_argument(
        "--accept-length-imbalance",
        action="store_true",
        help="Accept length imbalance for this split; invalid/truncated failures still block",
    )
    qc.add_argument(
        "--split", default="distill_train", choices=["distill_train", "distill_val", *SPLITS]
    )
    distill = commands.add_parser("distill", help="Full-model SFT from one cached teacher")
    distill.add_argument("--teacher", required=True)
    distill.add_argument("--seed", type=int)
    distill.add_argument("--amount", type=int)
    distill.add_argument("--post-data", help="Independent JSONL for additional fine-tuning")
    distill.add_argument("--level", choices=["light", "moderate"])
    pin = commands.add_parser("pin", help="Resolve remote model/dataset commits into a new config")
    pin.add_argument("--output", required=True)
    sets = commands.add_parser(
        "train-sets", help="Fit Deep Sets after the single-response baseline"
    )
    sets.add_argument("--checkpoint", required=True)
    sets.add_argument("--baseline", required=True)
    sets_eval = commands.add_parser("evaluate-sets", help="Compare mean embeddings and Deep Sets")
    sets_eval.add_argument("--checkpoint", required=True)
    sets_eval.add_argument("--set-checkpoint")
    sets_eval.add_argument("--reference-config")
    sets_eval.add_argument("--name", default="sets")
    figures = commands.add_parser("figures", help="Export plots and tidy publication data")
    figures.add_argument("--evaluation", required=True)
    diagnostic = commands.add_parser(
        "diagnose", help="Held-out student/teacher agreement diagnostics"
    )
    diagnostic.add_argument("--bertscore", action="store_true")
    commands.add_parser("plan", help="Write sequential commands for seeds, amounts, and ablations")
    variant = commands.add_parser("variant", help="Create a separate controlled experiment variant")
    variant.add_argument(
        "--kind",
        required=True,
        choices=[
            "amount",
            "seed",
            "held_out_students",
            "decoding",
            "natural",
            "cross_task",
            "paraphrase",
            "extra_ft",
        ],
    )
    variant.add_argument("--value", default="")
    variant.add_argument("--output", required=True)
    collection = commands.add_parser("collect", help="Collect amount/robustness/seed figure data")
    collection.add_argument("--evaluations", nargs="+", required=True)
    collection.add_argument("--output", required=True)
    comparison = commands.add_parser(
        "compare", help="Paired original/paraphrase/decoding consistency"
    )
    comparison.add_argument("--reference", required=True)
    comparison.add_argument("--variant", required=True)
    comparison.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "pin":
        from teacher_attr.experiments import pin_config

        print(json.dumps(pin_config(args.config, args.output), indent=2))
        return
    if args.command == "variant":
        from teacher_attr.experiments import create_variant

        print(json.dumps(create_variant(args.config, args.kind, args.value, args.output), indent=2))
        return
    cfg = load_config(args.config)
    if args.command == "migrate-budgets":
        from teacher_attr.budget_migration import migrate_budgets

        print(
            json.dumps(
                migrate_budgets(args.config, args.output, args.input_tokens, args.training_tokens),
                indent=2,
            )
        )
        return
    if args.command == "prepare":
        from teacher_attr.prompts import prepare

        result = prepare(cfg)
    elif args.command == "preflight":
        from teacher_attr.experiments import preflight

        result = preflight(cfg, args.cache_dir)
    elif args.command == "audit-tokens":
        from teacher_attr.token_audit import audit_token_budgets

        result = audit_token_budgets(cfg)
    elif args.command == "generate":
        from teacher_attr.generation import generate

        if args.model and not args.role:
            parser.error("--model requires --role")
        if args.model and args.model not in cfg[args.role]:
            parser.error(f"Unknown {args.role} model: {args.model}")
        if args.split in {"distill_train", "distill_val"} and args.role != "teachers":
            parser.error("Distillation pools require --role teachers")
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

        result = train(
            cfg,
            args.objective,
            args.input_mode,
            args.seed,
            args.representation,
            args.negatives,
            args.classification_weight,
            args.baseline,
        )
    elif args.command == "evaluate":
        from teacher_attr.evaluation import evaluate

        result = evaluate(cfg, args.checkpoint, args.frozen, args.name, args.reference_config)
    elif args.command == "qc":
        from teacher_attr.quality import quality_control

        result = quality_control(cfg, args.split, args.accept_length_imbalance)
    elif args.command == "distill":
        from teacher_attr.distillation import train_student

        result = train_student(
            cfg, args.teacher, args.seed, args.amount, args.post_data, args.level
        )
    elif args.command == "train-sets":
        from teacher_attr.sets import train_sets

        result = train_sets(cfg, args.checkpoint, args.baseline)
    elif args.command == "evaluate-sets":
        from teacher_attr.sets import evaluate_sets

        result = evaluate_sets(
            cfg, args.checkpoint, args.set_checkpoint, args.name, args.reference_config
        )
    elif args.command == "figures":
        from teacher_attr.reporting import figures

        result = figures(args.evaluation)
    elif args.command == "collect":
        from teacher_attr.reporting import collect

        result = collect(args.evaluations, args.output)
    elif args.command == "compare":
        from teacher_attr.reporting import compare

        result = compare(
            args.reference,
            args.variant,
            args.output,
            cfg["evaluation"]["bootstrap_repeats"],
            cfg["seed"],
        )
    elif args.command == "diagnose":
        from teacher_attr.diagnostics import diagnose

        result = diagnose(cfg, args.bertscore)
    else:
        from teacher_attr.experiments import experiment_plan

        result = experiment_plan(cfg, args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
