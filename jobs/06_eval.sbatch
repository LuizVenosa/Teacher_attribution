#!/bin/bash
#SBATCH --job-name=eval_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_attr_%j.out
#SBATCH --error=logs/eval_attr_%j.err

set -euo pipefail
SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

if [ "${1:-}" = "--mode" ]; then
  MODE="${2:-baselines}"
else
  MODE="${1:-baselines}"
fi

if [ "$MODE" = "baselines" ]; then
  python scripts/run_baselines.py \
    --models_config configs/models.yaml \
    --train_pairs data/attribution/train_pairs.jsonl \
    --test_pairs data/attribution/test_pairs.jsonl \
    --output results/baselines/test_metrics.json
elif [ "$MODE" = "encoder" ]; then
  python scripts/evaluate.py \
    --models_config configs/models.yaml \
    --attribution_config configs/attribution.yaml \
    --pairs data/attribution/test_pairs.jsonl \
    --checkpoint models/attribution_encoder/modernbert_contrastive/best.pt \
    --output results/contrastive/set_level_metrics.json
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi


