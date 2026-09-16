#!/bin/bash
# Source from the login node or a batch job; never derive paths from SLURM's spool.
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
cd "$PROJECT_ROOT" || return 1
export CACHE_ROOT="${CACHE_ROOT:-$PROJECT_ROOT/.cache}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$PROJECT_ROOT/logs" || return 1

# The old cluster setup requires bashrc for modules; it is unsafe under nounset.
_teacherattr_nounset=0
[[ $- == *u* ]] && _teacherattr_nounset=1
set +u
if [[ -f "$HOME/.bashrc" ]]; then source "$HOME/.bashrc"; fi
if [[ "${SKIP_CLUSTER_MODULES:-0}" != 1 ]] && command -v module >/dev/null 2>&1; then
  module load miniconda3 || return 1
  module load "${CUDA_MODULE:-cuda/12.4}" || return 1
fi
# An explicit venv takes precedence; otherwise retain the old teacherattr env.
if [[ -n "${VENV_PATH:-}" ]]; then
  source "$VENV_PATH/bin/activate" || return 1
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
  conda activate "${CONDA_ENV_NAME:-teacherattr}" || return 1
elif [[ -f "$PROJECT_ROOT/.venv/bin/activate" ]]; then
  source "$PROJECT_ROOT/.venv/bin/activate" || return 1
else
  echo "Activate/install teacherattr, or set VENV_PATH to a prepared Linux venv." >&2
  return 1
fi
if [[ $_teacherattr_nounset == 1 ]]; then set -u; fi
unset _teacherattr_nounset
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
command -v python >/dev/null || return 1
