#!/bin/bash
#SBATCH --job-name=download_wtyt
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/download_wtyt_%j.out
#SBATCH --error=logs/download_wtyt_%j.err

set -euo pipefail
source jobs/00_setup_env.sh

PYTHONPATH=src python scripts/download_wtyt_datasets.py \
  --output-dir external_datasets/who_taught_you_that \
  --datasets commonsenseqa,openbookqa,alpaca,rotten_tomatoes \
  --format auto \
  --allow-missing-datasets
