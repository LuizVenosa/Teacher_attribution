"""Download primary-study model snapshots on the internet-connected login node."""

import argparse

from huggingface_hub import snapshot_download

from teacher_attr.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    models = [
        *cfg["teachers"].values(),
        cfg["research"]["student"],
        cfg["encoder"],
        cfg["evaluation"]["generic_encoder"],
    ]
    for model in models:
        print(f"Caching {model['hf_name']} @ {model['revision']}", flush=True)
        snapshot_download(repo_id=model["hf_name"], revision=model["revision"])


if __name__ == "__main__":
    main()
