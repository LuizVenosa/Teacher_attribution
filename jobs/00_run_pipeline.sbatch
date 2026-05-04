#!/bin/bash
#SBATCH --job-name=teacher_pipe
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=48:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

# Forward any arguments passed after the sbatch script name, e.g.
# sbatch jobs/00_run_pipeline.sbatch --distill-size 20 --train-size 20 --val-size 8 --test-size 8
bash scripts/submit_slurm_pipeline.sh "$@"
