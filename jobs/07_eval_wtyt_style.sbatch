#!/bin/bash
#SBATCH --job-name=eval_wtyt_style
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_wtyt_style_%j.out
#SBATCH --error=logs/eval_wtyt_style_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

python scripts/evaluate_wtyt_style.py \
  --models_config "${MODELS_CONFIG:-configs/public_lineage_models.yaml}" \
  --train_pairs data/attribution/train_pairs.jsonl \
  --test_pairs data/attribution/test_pairs.jsonl \
  --support_sizes "${SUPPORT_SIZES:-50,200,1000,2000}" \
  --repeats "${REPEATS:-5}" \
  --output results/wtyt_style/public_lineage_wtyt_style_metrics.json