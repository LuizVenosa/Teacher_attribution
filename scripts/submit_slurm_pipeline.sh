#!/bin/bash
set -euo pipefail

# Submit the full teacher-attribution pipeline on the Bocconi student cluster.
# This script is meant to run from the login node. It submits one stage at a time
# and waits before submitting the next stage, which avoids QOSMaxSubmitJobPerUserLimit.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

SOURCE="who_taught_you_that"
DATASETS=""
ALLOW_MISSING_DATASETS=0
DISTILL_SIZE=1000
TRAIN_SIZE=1000
VAL_SIZE=300
TEST_SIZE=300
TEACHER_CHUNKS=("0-3" "4-7" "8-11" "12-15")
STUDENT_CHUNKS=("0-3" "4-7" "8-11")
SKIP_PROMPTS=0
SKIP_EXISTING_OUTPUTS=0
POLL_SECONDS=60

usage() {
  cat <<EOF
Usage: bash scripts/submit_slurm_pipeline.sh [options]

Options:
  --source NAME              Prompt source: synthetic or who_taught_you_that [$SOURCE]
  --datasets LIST            Comma-separated WTYT dataset subset [all paper datasets]
  --allow-missing-datasets   Skip unavailable HF datasets instead of failing
  --distill-size N           Distillation prompt count [$DISTILL_SIZE]
  --train-size N             Attribution train prompt count [$TRAIN_SIZE]
  --val-size N               Attribution val prompt count [$VAL_SIZE]
  --test-size N              Attribution test prompt count [$TEST_SIZE]
  --skip-prompts             Do not regenerate prompt JSONLs
  --skip-existing-outputs    Do not move existing generated JSONLs aside
  --poll-seconds N           Queue polling interval [$POLL_SECONDS]
  -h, --help                 Show this help

Stages:
  1. make prompt bank
  2. teacher generation arrays: 0-3, 4-7, 8-11, 12-15
  3. student distillation array: 0-3
  4. student generation arrays: 0-3, 4-7, 8-11
  5. attribution pair building array: 0-2
  6. baselines
  7. contrastive encoder training
  8. encoder evaluation
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      SOURCE="$2"; shift 2 ;;
    --datasets)
      DATASETS="$2"; shift 2 ;;
    --allow-missing-datasets)
      ALLOW_MISSING_DATASETS=1; shift ;;
    --distill-size)
      DISTILL_SIZE="$2"; shift 2 ;;
    --train-size)
      TRAIN_SIZE="$2"; shift 2 ;;
    --val-size)
      VAL_SIZE="$2"; shift 2 ;;
    --test-size)
      TEST_SIZE="$2"; shift 2 ;;
    --skip-prompts)
      SKIP_PROMPTS=1; shift ;;
    --skip-existing-outputs)
      SKIP_EXISTING_OUTPUTS=1; shift ;;
    --poll-seconds)
      POLL_SECONDS="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 127
  fi
}

job_state() {
  local job_id="$1"
  if command -v sacct >/dev/null 2>&1; then
    local states
    states=$(sacct -j "$job_id" --format=State --noheader 2>/dev/null | awk 'NF {print $1}')
    if [[ -z "$states" ]]; then
      return 0
    fi
    if echo "$states" | grep -Eq 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED'; then
      echo "$states" | grep -E 'FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED' | head -n 1
      return 0
    fi
    if echo "$states" | awk 'BEGIN {ok=1} $1 !~ /^COMPLETED/ {ok=0} END {exit ok ? 0 : 1}'; then
      echo "COMPLETED"
      return 0
    fi
    echo "$states" | tail -n 1
  fi
}

wait_for_job() {
  local job_id="$1"
  local label="$2"
  log "Waiting for $label ($job_id)"
  while squeue -h -j "$job_id" >/dev/null 2>&1 && [[ -n "$(squeue -h -j "$job_id")" ]]; do
    squeue -j "$job_id" -o "%.18i %.9P %.24j %.2t %.10M %R" || true
    sleep "$POLL_SECONDS"
  done

  local state
  state=$(job_state "$job_id")
  if [[ -z "$state" ]]; then
    log "$label ($job_id) left the queue; sacct state unavailable, continuing."
    return 0
  fi
  if [[ "$state" == COMPLETED* ]]; then
    log "$label ($job_id) completed."
    return 0
  fi

  echo "$label ($job_id) ended with state: $state" >&2
  echo "Recent logs:" >&2
  ls -lt logs | head -20 >&2 || true
  exit 1
}

submit_and_wait() {
  local label="$1"
  shift
  log "Submitting $label: sbatch $*"
  local job_id
  job_id=$(sbatch --parsable "$@")
  log "Submitted $label as $job_id"
  wait_for_job "$job_id" "$label"
}

require_cmd sbatch
require_cmd squeue
mkdir -p logs

log "Project root: $PROJECT_ROOT"

if [[ "$SKIP_PROMPTS" -eq 0 ]]; then
  log "Building prompt bank from source=$SOURCE"
  prompt_args=(
    --source "$SOURCE"
    --distill-size "$DISTILL_SIZE"
    --train-size "$TRAIN_SIZE"
    --val-size "$VAL_SIZE"
    --test-size "$TEST_SIZE"
  )
  if [[ -n "$DATASETS" ]]; then
    prompt_args+=(--datasets "$DATASETS")
  fi
  if [[ "$ALLOW_MISSING_DATASETS" -eq 1 ]]; then
    prompt_args+=(--allow-missing-datasets)
  fi
  PYTHONPATH=src python scripts/make_prompt_bank.py "${prompt_args[@]}"
fi

if [[ "$SKIP_EXISTING_OUTPUTS" -eq 0 ]]; then
  stamp=$(date '+%Y%m%d_%H%M%S')
  for dir in data/teacher_outputs data/student_outputs data/attribution results/baselines results/contrastive; do
    if compgen -G "$dir/*.jsonl" >/dev/null || compgen -G "$dir/*.json" >/dev/null; then
      backup="${dir}_backup_${stamp}"
      log "Moving existing outputs from $dir to $backup"
      mkdir -p "$backup"
      find "$dir" -maxdepth 1 -type f \( -name '*.jsonl' -o -name '*.json' \) -exec mv {} "$backup" \;
    fi
  done
fi

for chunk in "${TEACHER_CHUNKS[@]}"; do
  submit_and_wait "teacher generation array $chunk" --array="$chunk" jobs/01_generate_teachers.sbatch
  wc -l data/teacher_outputs/*.jsonl || true
done

submit_and_wait "student distillation" --array=0-3 jobs/02_distill_student.sbatch

for chunk in "${STUDENT_CHUNKS[@]}"; do
  submit_and_wait "student generation array $chunk" --array="$chunk" jobs/03_generate_students.sbatch
  wc -l data/student_outputs/*.jsonl || true
done

submit_and_wait "build attribution data" --array=0-2 jobs/04_build_attribution_data.sbatch
submit_and_wait "baselines" jobs/06_eval.sbatch baselines
submit_and_wait "contrastive encoder" jobs/05_train_contrastive.sbatch
submit_and_wait "encoder evaluation" jobs/06_eval.sbatch encoder

log "Pipeline complete. Key outputs:"
ls -lh results/baselines results/contrastive || true
