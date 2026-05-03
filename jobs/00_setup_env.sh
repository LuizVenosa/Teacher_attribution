#!/bin/bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/teacher-attribution}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-/scratch/$USER/wandb}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v conda >/dev/null 2>&1; then
  conda activate "${CONDA_ENV_NAME:-teacherattr}"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
