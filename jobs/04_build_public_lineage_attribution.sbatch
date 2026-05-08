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
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

MODELS_CONFIG="${MODELS_CONFIG:-configs/public_lineage_models.yaml}"
SPLIT_LIST="${SPLIT_LIST:-train val test}"
read -r -a SPLITS <<< "$SPLIT_LIST"

for SPLIT in "${SPLITS[@]}"; do
  python scripts/build_attribution_pairs.py \
    --models_config "$MODELS_CONFIG" \
    --student_outputs_dir data/public_lineage/student_outputs \
    --teacher_outputs_dir data/public_lineage/teacher_outputs \
    --split "$SPLIT" \
    --output "data/attribution/${SPLIT}_pairs.jsonl"
done

wc -l data/attribution/*_pairs.jsonl
