#!/bin/bash
#SBATCH --job-name=distill_student
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/distill_%A_%a.out
#SBATCH --error=logs/distill_%A_%a.err

set -euo pipefail
SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

TEACHERS=("qwen" "llama" "mistral" "gemma")
TEACHER_ID=${TEACHERS[$SLURM_ARRAY_TASK_ID]}

python scripts/make_sft_data.py \
  --teacher_outputs "data/teacher_outputs/${TEACHER_ID}_distill.jsonl" \
  --output "data/distill_data/${TEACHER_ID}_teacher_sft.jsonl"

python scripts/train_student_sft.py \
  --teacher_id "$TEACHER_ID" \
  --models_config configs/models.yaml \
  --distill_config configs/distill.yaml \
  --train_file "data/distill_data/${TEACHER_ID}_teacher_sft.jsonl" \
  --output_dir "models/students/student_from_${TEACHER_ID}"


