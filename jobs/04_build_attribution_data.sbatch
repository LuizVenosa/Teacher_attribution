#!/bin/bash
#SBATCH --job-name=build_attr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/build_attr_%A_%a.out
#SBATCH --error=logs/build_attr_%A_%a.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/00_setup_env.sh"

SPLITS=("train" "val" "test")
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

python scripts/build_attribution_pairs.py \
  --models_config configs/models.yaml \
  --student_outputs_dir data/student_outputs \
  --teacher_outputs_dir data/teacher_outputs \
  --split "$SPLIT" \
  --output "data/attribution/${SPLIT}_pairs.jsonl"
