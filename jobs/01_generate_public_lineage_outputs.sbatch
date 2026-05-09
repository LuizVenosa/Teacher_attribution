#!/bin/bash
#SBATCH --job-name=gen_public_lineage
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/gen_public_lineage_%j.out
#SBATCH --error=logs/gen_public_lineage_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

# Compute nodes have no internet; use the cache prepared on the login node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODELS_CONFIG="${MODELS_CONFIG:-configs/public_lineage_models.yaml}"
GENERATION_CONFIG="${GENERATION_CONFIG:-configs/public_lineage_generation.yaml}"
SPLIT_LIST="${SPLIT_LIST:-train val test}"
TEACHER_LIST="${TEACHER_LIST:-gpt2 qwen15_18b flan_t5_base flan_t5_small}"
STUDENT_LIST="${STUDENT_LIST:-distilgpt2:miniplm_qwen_200m:lamini_flan_t5_248m:lamini_flan_t5_77m}"

read -r -a SPLITS <<< "$SPLIT_LIST"
read -r -a TEACHERS <<< "$TEACHER_LIST"
IFS=":" read -r -a STUDENTS <<< "$STUDENT_LIST"

teacher_for_student() {
  case "$1" in
    distilgpt2) echo "gpt2" ;;
    miniplm_qwen_200m) echo "qwen15_18b" ;;
    lamini_flan_t5_248m) echo "flan_t5_base" ;;
    lamini_flan_t5_77m) echo "flan_t5_small" ;;
    *) echo "Unknown student id: $1" >&2; return 2 ;;
  esac
}

for SPLIT in "${SPLITS[@]}"; do
  for TEACHER_ID in "${TEACHERS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] teacher=$TEACHER_ID split=$SPLIT"
    python scripts/generate_public_teachers.py \
      --teacher_id "$TEACHER_ID" \
      --models_config "$MODELS_CONFIG" \
      --generation_config "$GENERATION_CONFIG" \
      --prompts "data/prompts/prompts_${SPLIT}.jsonl" \
      --output "data/public_lineage/teacher_outputs/${TEACHER_ID}_${SPLIT}.jsonl" \
      --resume
  done

  for STUDENT_ID in "${STUDENTS[@]}"; do
    TRUE_TEACHER="$(teacher_for_student "$STUDENT_ID")"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] student=$STUDENT_ID true_teacher=$TRUE_TEACHER split=$SPLIT"
    python scripts/generate_public_students.py \
      --student_id "$STUDENT_ID" \
      --models_config "$MODELS_CONFIG" \
      --generation_config "$GENERATION_CONFIG" \
      --prompts "data/prompts/prompts_${SPLIT}.jsonl" \
      --output "data/public_lineage/student_outputs/student_from_${TRUE_TEACHER}_${SPLIT}.jsonl" \
      --resume
  done
done

ls -lh data/public_lineage/teacher_outputs data/public_lineage/student_outputs
