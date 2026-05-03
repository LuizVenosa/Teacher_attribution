#!/bin/bash
#SBATCH --job-name=train_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_attr_%j.out
#SBATCH --error=logs/train_attr_%j.err

set -euo pipefail
SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

python scripts/train_contrastive_encoder.py \
  --models_config configs/models.yaml \
  --attribution_config configs/attribution.yaml


