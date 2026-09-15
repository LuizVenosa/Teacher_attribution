"""Compatibility entry point; implementation lives in teacher_attr."""

import argparse

from teacher_attr.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/research.yaml")
    known, remaining = parser.parse_known_args()
    main(["--config", known.config, "train", *[], *remaining])
