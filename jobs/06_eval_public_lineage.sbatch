#!/bin/bash
#SBATCH --job-name=eval_public_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_public_attr_%j.out
#SBATCH --error=logs/eval_public_attr_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

MODE="${1:-baselines}"
MODELS_CONFIG="${MODELS_CONFIG:-configs/public_lineage_models.yaml}"
ATTRIBUTION_CONFIG="${ATTRIBUTION_CONFIG:-configs/public_lineage_attribution.yaml}"

if [ "$MODE" = "baselines" ]; then
  python scripts/run_baselines.py \
    --models_config "$MODELS_CONFIG" \
    --train_pairs data/attribution/train_pairs.jsonl \
    --test_pairs data/attribution/test_pairs.jsonl \
    --output results/baselines/public_lineage_test_metrics.json
elif [ "$MODE" = "encoder" ]; then
  python scripts/evaluate.py \
    --models_config "$MODELS_CONFIG" \
    --attribution_config "$ATTRIBUTION_CONFIG" \
    --pairs data/attribution/test_pairs.jsonl \
    --checkpoint models/attribution_encoder/public_lineage_minilm_contrastive/best.pt \
    --output results/contrastive/public_lineage_set_level_metrics.json
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi
