#!/bin/bash
#SBATCH --job-name=gen_teacher
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/gen_teacher_%A_%a.out
#SBATCH --error=logs/gen_teacher_%A_%a.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/00_setup_env.sh"

TEACHERS=("qwen" "llama" "mistral" "gemma")
SPLITS=("distill" "train" "val" "test")

N_TEACHERS=${#TEACHERS[@]}
TEACHER_IDX=$((SLURM_ARRAY_TASK_ID % N_TEACHERS))
SPLIT_IDX=$((SLURM_ARRAY_TASK_ID / N_TEACHERS))

TEACHER_ID=${TEACHERS[$TEACHER_IDX]}
SPLIT=${SPLITS[$SPLIT_IDX]}

python scripts/generate_teachers.py \
  --teacher_id "$TEACHER_ID" \
  --models_config configs/models.yaml \
  --generation_config configs/generation.yaml \
  --prompts "data/prompts/prompts_${SPLIT}.jsonl" \
  --output "data/teacher_outputs/${TEACHER_ID}_${SPLIT}.jsonl" \
  --resume
