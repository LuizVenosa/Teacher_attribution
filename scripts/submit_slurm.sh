#!/bin/bash
# Submit one command at a time; the old stud QoS limits queued jobs per user.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$PROJECT_ROOT"
if [[ $# == 0 ]]; then
  echo "Usage: bash scripts/submit_slurm.sh [sbatch options] -- COMMAND [options]" >&2
  echo "Example: bash scripts/submit_slurm.sh -- distill --teacher qwen" >&2
  exit 2
fi
options=()
while [[ $# -gt 0 && "$1" != -- ]]; do
  options+=("$1")
  shift
done
if [[ $# -lt 2 ]]; then
  echo "Separate scheduler options from the pipeline command with --." >&2
  exit 2
fi
shift
# SLURM opens log files before the job starts, so creation belongs here.
mkdir -p logs
exec sbatch "${options[@]}" --chdir="$PROJECT_ROOT" "$PROJECT_ROOT/jobs/run.sbatch" "$@"
