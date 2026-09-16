#!/bin/bash
#SBATCH --job-name=teacher-attr
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/teacher-attr_%j.out
#SBATCH --error=logs/teacher-attr_%j.err
#SBATCH --mail-type=NONE

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
source "$PROJECT_ROOT/jobs/00_setup_env.sh"
if [ "$#" -eq 0 ]; then
  echo "Usage: sbatch jobs/run.sbatch COMMAND [options]" >&2
  exit 2
fi
# Bocconi compute nodes have no internet. Prepare prompts and cache weights first.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM=false
if [[ "${REQUIRE_GPU:-1}" == 1 ]]; then
  python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("No usable CUDA GPU: check the allocation and PyTorch installation.")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("This research configuration requires a BF16-capable GPU.")
print(f"GPU: {torch.cuda.get_device_name(0)}; torch={torch.__version__}; CUDA={torch.version.cuda}")
PY
fi
exec python -m teacher_attr --config "${CONFIG:-configs/research.yaml}" "$@"
