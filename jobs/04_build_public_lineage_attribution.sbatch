#!/bin/bash
#SBATCH --job-name=build_public_attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_public_attr_%j.out
#SBATCH --error=logs/build_public_attr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luiz.lima@studbocconi.it

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

MODELS_CONFIG="${MODELS_CONFIG:-configs/public_lineage_models.yaml}"
SPLIT_LIST="${SPLIT_LIST:-train val test}"
TEACHER_LIST="${TEACHER_LIST:-}"
STUDENT_LIST="${STUDENT_LIST:-}"

if [[ -z "$TEACHER_LIST" ]]; then
  TEACHER_LIST="$(python scripts/public_lineage_config.py --models_config "$MODELS_CONFIG" --list teachers --separator " ")"
fi
if [[ -z "$STUDENT_LIST" ]]; then
  STUDENT_LIST="$(python scripts/public_lineage_config.py --models_config "$MODELS_CONFIG" --list students --separator ":")"
fi

read -r -a SPLITS <<< "$SPLIT_LIST"

for SPLIT in "${SPLITS[@]}"; do
  python scripts/build_attribution_pairs.py \
    --models_config "$MODELS_CONFIG" \
    --student_outputs_dir data/public_lineage/student_outputs \
    --teacher_outputs_dir data/public_lineage/teacher_outputs \
    --split "$SPLIT" \
    --teacher_ids "$TEACHER_LIST" \
    --student_ids "$STUDENT_LIST" \
    --output "data/attribution/${SPLIT}_pairs.jsonl"
done

wc -l data/attribution/*_pairs.jsonl