#!/bin/bash
#SBATCH --job-name=build_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/build_attr_%A_%a.out
#SBATCH --error=logs/build_attr_%A_%a.err

set -euo pipefail
SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

SPLITS=("train" "val" "test")
SPLIT=${SPLITS[$SLURM_ARRAY_TASK_ID]}

python scripts/build_attribution_pairs.py \
  --models_config configs/models.yaml \
  --student_outputs_dir data/student_outputs \
  --teacher_outputs_dir data/teacher_outputs \
  --split "$SPLIT" \
  --output "data/attribution/${SPLIT}_pairs.jsonl"


