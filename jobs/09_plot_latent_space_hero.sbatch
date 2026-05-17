#!/bin/bash
#SBATCH --job-name=plot_latent_hero
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/plot_latent_hero_%j.out
#SBATCH --error=logs/plot_latent_hero_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luiz.lima@studbocconi.it

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

export MPLBACKEND=Agg
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODELS_CONFIG="${MODELS_CONFIG:-configs/public_lineage_models_minilm.yaml}"
ATTRIBUTION_CONFIG="${ATTRIBUTION_CONFIG:-configs/public_lineage_attribution_minilm_ablate.yaml}"
PAIRS="${PAIRS:-data/attribution/test_pairs.jsonl}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-models/attribution_encoder/public_lineage_minilm_contrastive/best.pt}"
OUTPUT="${OUTPUT:-results/figures/00_latent_space_hero.png}"
MAX_PER_TEACHER="${MAX_PER_TEACHER:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"

python scripts/plot_latent_space_hero.py \
  --models_config "$MODELS_CONFIG" \
  --attribution_config "$ATTRIBUTION_CONFIG" \
  --pairs "$PAIRS" \
  --checkpoint "$ENCODER_CHECKPOINT" \
  --output "$OUTPUT" \
  --max_per_teacher "$MAX_PER_TEACHER" \
  --batch_size "$BATCH_SIZE" \
  --write_svg
