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
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_public_attr_%j.out
#SBATCH --error=logs/train_public_attr_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

python scripts/train_contrastive_encoder.py \
  --models_config "${MODELS_CONFIG:-configs/public_lineage_models.yaml}" \
  --attribution_config "${ATTRIBUTION_CONFIG:-configs/public_lineage_attribution.yaml}"
