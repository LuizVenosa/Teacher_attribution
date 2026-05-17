#!/bin/bash
#SBATCH --job-name=eval_contrastive_probes
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval_contrastive_probes_%j.out
#SBATCH --error=logs/eval_contrastive_probes_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luiz.lima@studbocconi.it

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

# Compute nodes have no internet; use the cache prepared on the login node.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODELS_CONFIG="${MODELS_CONFIG:-configs/public_lineage_models_minilm.yaml}"
ATTRIBUTION_CONFIG="${ATTRIBUTION_CONFIG:-configs/public_lineage_attribution_minilm_ablate.yaml}"
TRAIN_PAIRS="${TRAIN_PAIRS:-data/attribution/train_pairs.jsonl}"
TEST_PAIRS="${TEST_PAIRS:-data/attribution/test_pairs.jsonl}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-models/attribution_encoder/public_lineage_minilm_contrastive/best.pt}"
ENCODER_OUTPUT="${ENCODER_OUTPUT:-results/contrastive/public_lineage_minilm_probe_metrics.json}"
SENTENCE_MODEL="${SENTENCE_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
TEACHER_LIST="${TEACHER_LIST:-}"

EXTRA_ARGS=()
if [[ -n "$TEACHER_LIST" ]]; then
  EXTRA_ARGS+=(--teacher_ids "$TEACHER_LIST")
fi
if [[ "${USE_SLOW_TOKENIZER:-0}" = "1" ]]; then
  EXTRA_ARGS+=(--use_slow_tokenizer)
fi
if [[ "${SKIP_SIMILARITY_BASELINES:-0}" = "1" ]]; then
  EXTRA_ARGS+=(--skip_similarity_baselines)
fi
if [[ "${SKIP_SENTENCE_BASELINE:-0}" = "1" ]]; then
  EXTRA_ARGS+=(--skip_sentence_baseline)
fi

python scripts/evaluate_contrastive_probes.py \
  --models_config "$MODELS_CONFIG" \
  --attribution_config "$ATTRIBUTION_CONFIG" \
  --train_pairs "$TRAIN_PAIRS" \
  --test_pairs "$TEST_PAIRS" \
  --checkpoint "$ENCODER_CHECKPOINT" \
  --output "$ENCODER_OUTPUT" \
  --sentence_model "$SENTENCE_MODEL" \
  "${EXTRA_ARGS[@]}"
