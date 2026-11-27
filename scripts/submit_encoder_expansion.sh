#!/usr/bin/env bash
# Run from the cluster repository root after activating teacherattr.
set -euo pipefail
export PROJECT_ROOT="$(pwd)"
parent_config="${1:-configs/research_budget_v1.yaml}"
expansion="${2:-runs/encoder_expansion_12k}"
export CONFIG="$PROJECT_ROOT/$expansion/generation.yaml"
if [[ ! -f "$CONFIG" ]]; then
  python scripts/prepare_student_expansion.py --config "$parent_config" --output "$expansion" --per-dataset 2000
fi
# Resolve the model IDs from the actual generation config (student IDs include the seed).
models=$(python - "$CONFIG" "$parent_config" <<'PY'
import json, sys
from pathlib import Path
from teacher_attr.config import load_config, file_hash
from teacher_attr.prompts import verify_prompts
cfg = load_config(sys.argv[1])
root = Path(cfg["run_dir"])
manifest = verify_prompts(root)
assert Path(manifest["parent_config"]) == Path(sys.argv[2]).resolve()
subset = root / "prompts/encoder_subsets.json"
assert file_hash(subset) == manifest["subsets_sha256"]
assert len(json.loads(subset.read_text())["2000"]) == 12000
models = [(role, n) for role in ("teachers", "students") for n, model in cfg[role].items()
          if role == "teachers" or "train" in model["splits"]]
assert len(models) == 8
for role, name in models:
    print(role, name)
PY
)
mkdir -p logs
# Submit one array, preserving the standard resources, environment and cache handling.
{
  sed -n '1p;/^#SBATCH/p' jobs/run.sbatch
  printf 'models=(\n'
  while read -r role name; do printf '  "%s %s"\n' "$role" "$name"; done <<< "$models"
  printf ')\n'
  cat <<'BODY'
set -euo pipefail
read -r role name <<< "${models[$SLURM_ARRAY_TASK_ID]}"
exec bash "$PROJECT_ROOT/jobs/run.sbatch" generate --role "$role" --model "$name" --split train
BODY
} | sbatch --array="0-7%${MAX_PARALLEL:-1}" --chdir="$PROJECT_ROOT" --job-name=encoder-12k \
    --output='logs/encoder-12k_%A_%a.out' --error='logs/encoder-12k_%A_%a.err'
