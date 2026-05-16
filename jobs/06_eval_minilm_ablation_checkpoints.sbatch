#!/bin/bash
#SBATCH --job-name=eval_minilm_ablate
#SBATCH --account=3191856
#SBATCH --partition=stud
#SBATCH --qos=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval_minilm_ablate_%j.out
#SBATCH --error=logs/eval_minilm_ablate_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=luiz.lima@studbocconi.it

set -euo pipefail

SETUP_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/home/3191856/NLP_project/Teacher_attribution}}"
source "$SETUP_ROOT/jobs/00_setup_env.sh"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

mkdir -p results/contrastive_ablation/eval

for ckpt in models/attribution_encoder/public_lineage_ablations/*/best.pt; do
  run_dir=$(dirname "$ckpt")
  run_name=$(basename "$run_dir")
  cfg=$(ls results/contrastive_ablation/generated_configs/*_"${run_name}.yaml" | head -1)
  out="results/contrastive_ablation/eval/${run_name}_test_metrics.json"

  echo "[$(date)] Evaluating ${run_name}"
  python scripts/evaluate.py \
    --models_config configs/public_lineage_models_minilm.yaml \
    --attribution_config "$cfg" \
    --pairs data/attribution/test_pairs.jsonl \
    --checkpoint "$ckpt" \
    --output "$out"
done

python - <<'PY'
import csv, json
from pathlib import Path

rows = []
for path in sorted(Path("results/contrastive_ablation/eval").glob("*_test_metrics.json")):
    data = json.loads(path.read_text())
    single = data["single_prompt"]
    rows.append({
        "run": path.name.replace("_test_metrics.json", ""),
        "accuracy": single.get("accuracy"),
        "top2_accuracy": single.get("top2_accuracy"),
        "roc_auc_macro": single.get("roc_auc_ovr_macro"),
        "roc_auc_micro": single.get("roc_auc_ovr_micro"),
        "encoder_model": data.get("encoder_model"),
        "checkpoint": data.get("checkpoint"),
    })

out = Path("results/contrastive_ablation/eval_summary.csv")
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "run", "accuracy", "top2_accuracy", "roc_auc_macro", "roc_auc_micro",
        "encoder_model", "checkpoint"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("wrote", out)
PY
