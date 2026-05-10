from __future__ import annotations

import argparse

from teacher_attr.io import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect public-lineage model config IDs.")
    parser.add_argument("--models_config", required=True)
    parser.add_argument("--list", choices=["teachers", "students"], default=None)
    parser.add_argument("--separator", default=" ")
    parser.add_argument("--student_teacher", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.models_config)

    if args.list == "teachers":
        print(args.separator.join(cfg["teachers"].keys()))
        return
    if args.list == "students":
        print(args.separator.join(cfg["public_students"].keys()))
        return
    if args.student_teacher:
        print(cfg["public_students"][args.student_teacher]["true_teacher"])
        return

    raise SystemExit("Pass --list teachers, --list students, or --student_teacher STUDENT_ID")


if __name__ == "__main__":
    main()