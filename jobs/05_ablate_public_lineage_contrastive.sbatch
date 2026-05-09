#!/bin/bash
#SBATCH --job-name=ablate_public_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ablate_public_attr_%j.out
#SBATCH --error=logs/ablate_public_attr_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

# Compute nodes have no internet; use the cache prepared on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/run_contrastive_ablation.py \
  --models_config "${MODELS_CONFIG:-configs/public_lineage_models.yaml}" \
  --base_config "${ATTRIBUTION_CONFIG:-configs/public_lineage_attribution.yaml}" \
  --mode "${ABLATION_MODE:-compact}" \
  --temperatures "${TEMPERATURES:-0.03,0.05,0.07}" \
  --classification_weights "${CLASSIFICATION_WEIGHTS:-0.1,0.2,0.5}" \
  --learning_rates "${LEARNING_RATES:-0.00002}" \
  --max_lengths "${MAX_LENGTHS:-512}" \
  --projection_dims "${PROJECTION_DIMS:-128}" \
  --max_runs "${MAX_RUNS:-7}" \
  --skip_existing