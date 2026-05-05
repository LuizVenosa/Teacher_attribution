#!/bin/bash
set -euo pipefail

# Resolve project root safely under SLURM. Bocconi copies sbatch scripts to
# /var/spool, so do not derive paths from the copied script location.
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"

# Student accounts may not have /scratch access. Keep caches inside the project
# unless the user explicitly overrides CACHE_ROOT/HF_HOME/WANDB_DIR.
export CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/wandb}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

# Source bashrc safely. With `set -u`, Bocconi's /etc/bashrc can fail on an
# unset BASHRCSOURCED variable.
if [ -f "$HOME/.bashrc" ]; then
  set +u
  source "$HOME/.bashrc"
  set -u
fi

if command -v module >/dev/null 2>&1; then
  module load miniconda3
  module load cuda/12.4
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-teacherattr}"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
fi

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Optional private Hugging Face token. Create this on the cluster with:
#   nano ~/.hf_token
#   chmod 600 ~/.hf_token
if [ -f "$HOME/.hf_token" ]; then
  export HF_TOKEN="$(cat "$HOME/.hf_token")"
fi
