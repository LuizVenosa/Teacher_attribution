#!/bin/bash
#SBATCH --job-name=train_attr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_attr_%j.out
#SBATCH --error=logs/train_attr_%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/00_setup_env.sh"

python scripts/train_contrastive_encoder.py \
  --models_config configs/models.yaml \
  --attribution_config configs/attribution.yaml
