#!/bin/bash
#SBATCH --job-name=teacher-attr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch jobs/run.sbatch prepare|generate|build|audit|train|evaluate [options]" >&2
  exit 2
fi
# Activate your environment before submitting. No machine-specific paths or credentials.
python -m teacher_attr --config "${CONFIG:-configs/experiment.yaml}" "$@"
