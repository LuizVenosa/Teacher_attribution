#!/bin/bash
set -euo pipefail

# Resolve the project root safely under SLURM. On Bocconi's cluster, sbatch copies
# scripts to /var/spool, so do not derive paths from the copied job script.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEFAULT_PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$DEFAULT_PROJECT_ROOT}}"

# Student accounts may not have /scratch access. Keep caches in the project tree
# by default; override CACHE_ROOT/HF_HOME/WANDB_DIR if your cluster gives you scratch.
export CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_ROOT/wandb}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$WANDB_DIR" "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

if [ -f "$HOME/.bashrc" ]; then
  source "$HOME/.bashrc"
fi

if command -v module >/dev/null 2>&1; then
  module load miniconda3 2>/dev/null || true
  module load cuda/12.4 2>/dev/null || true
fi

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME:-teacherattr}"
elif [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
