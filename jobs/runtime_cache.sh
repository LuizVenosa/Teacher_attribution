#!/bin/bash
# Source before environment setup and before importing Hugging Face libraries.
export MODEL_CACHE_MODE="${MODEL_CACHE_MODE:-runtime}"
case "$MODEL_CACHE_MODE" in
  offline) return 0 ;;
  runtime) ;;
  *) echo 'MODEL_CACHE_MODE must be runtime or offline.' >&2; return 2 ;;
esac

# Preserve access to login credentials while moving model storage off home.
export HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${CACHE_ROOT:-$PROJECT_ROOT/.cache}/huggingface}/token}"
if [[ ! -f "$HF_TOKEN_PATH" && -f "$HOME/.cache/huggingface/token" ]]; then
  export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
fi

# Use scheduler-provided temporary storage, or an explicitly approved directory.
# Do not silently assume that /tmp is large enough or backed by local disk.
_job_cache_base="${JOB_CACHE_ROOT:-${SLURM_TMPDIR:-}}"
if [[ -z "$_job_cache_base" ]]; then
  echo 'Runtime downloads need SLURM_TMPDIR or JOB_CACHE_ROOT pointing to approved compute-node temporary storage.' >&2
  return 2
fi
if [[ "$_job_cache_base" != /* || ! -d "$_job_cache_base" || ! -w "$_job_cache_base" ]]; then
  echo 'The runtime cache base must be an existing writable absolute directory.' >&2
  return 2
fi
_job_cache_base="$(realpath "$_job_cache_base")"
_job_home="$(realpath "$HOME")"
_job_project="$(realpath "$PROJECT_ROOT")"
case "$_job_cache_base/" in
  "$_job_home/"*|"$_job_project/"*)
    echo 'Runtime cache must be outside home and the repository to avoid the home quota.' >&2
    return 2 ;;
esac
export CACHE_ROOT
CACHE_ROOT="$(mktemp -d "$_job_cache_base/teacher-attr-${SLURM_JOB_ID:-manual}-XXXXXX")" || return 1
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_XET_CACHE="$HF_HOME/xet"
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
echo "Runtime model cache: $HF_HUB_CACHE"
echo 'Weights download on demand on this node, then load into GPU memory. Internet access is required.'
df -h "$CACHE_ROOT"
unset _job_cache_base _job_home _job_project
