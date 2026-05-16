#!/bin/bash
#SBATCH --job-name=train_public_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_public_attr_%j.out
#SBATCH --error=logs/train_public_attr_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

# Compute nodes have no internet; use the cache prepared on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/train_contrastive_encoder.py \
  --models_config "${MODELS_CONFIG:-configs/public_lineage_models.yaml}" \
  --attribution_config "${ATTRIBUTION_CONFIG:-configs/public_lineage_attribution.yaml}"
