#!/bin/bash
#SBATCH --job-name=gen_teacher_seq
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gen_teacher_seq_%j.out
#SBATCH --error=logs/gen_teacher_seq_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

# Space-separated overrides are supported, e.g.
# TEACHER_LIST="qwen llama" SPLIT_LIST="distill train" sbatch jobs/01_generate_teachers_sequential.sbatch
TEACHER_LIST="${TEACHER_LIST:-qwen llama mistral gemma}"
SPLIT_LIST="${SPLIT_LIST:-distill train val test}"

read -r -a TEACHERS <<< "$TEACHER_LIST"
read -r -a SPLITS <<< "$SPLIT_LIST"

for SPLIT in "${SPLITS[@]}"; do
  for TEACHER_ID in "${TEACHERS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] generating teacher=$TEACHER_ID split=$SPLIT"
    python scripts/generate_teachers.py \
      --teacher_id "$TEACHER_ID" \
      --models_config configs/models.yaml \
      --generation_config configs/generation.yaml \
      --prompts "data/prompts/prompts_${SPLIT}.jsonl" \
      --output "data/teacher_outputs/${TEACHER_ID}_${SPLIT}.jsonl" \
      --resume
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] done teacher=$TEACHER_ID split=$SPLIT"
  done
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] sequential teacher generation complete"
ls -lh data/teacher_outputs
wc -l data/teacher_outputs/*.jsonl
