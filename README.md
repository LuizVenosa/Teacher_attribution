# Teacher Attribution

Prompt-conditioned contrastive attribution for **public teacher-student language-model pairs**.

This version skips student distillation. Instead, it uses existing public student models whose model cards document a teacher or source model. For each shared prompt, the pipeline generates:

```text
candidate teacher responses
public student response
true teacher label from the documented lineage
```

Then it trains and evaluates a contrastive prompt-response encoder that pulls each student response toward its documented teacher response and pushes it away from other same-prompt teacher responses.

## Public Lineage Pairs

| Teacher ID | Teacher model | Student ID | Student model | Evidence level |
|---|---|---|---|---|
| `gpt2` | `gpt2` | `distilgpt2` | `distilbert/distilgpt2` | High: DistilGPT2 model card says it was supervised by GPT-2 124M. |
| `qwen15_18b` | `Qwen/Qwen1.5-1.8B` | `miniplm_qwen_200m` | `MiniLLM/MiniPLM-Qwen-200M` | High: MiniPLM card/dataset metadata names Qwen1.5-1.8B as teacher. |
| `flan_t5_base` | `google/flan-t5-base` | `lamini_flan_t5_248m` | `MBZUAI/LaMini-Flan-T5-248M` | Medium: Flan-T5-base family / instruction-distilled lineage. |
| `flan_t5_small` | `google/flan-t5-small` | `lamini_flan_t5_77m` | `MBZUAI/LaMini-Flan-T5-77M` | Medium: Flan-T5-small family / instruction-distilled lineage. |

The model list lives in `configs/public_lineage_models.yaml`.

## Pipeline

```text
1. Build a shared prompt bank
2. Generate responses from public teacher models
3. Generate responses from public student models
4. Build attribution JSONL pairs
5. Run lexical and embedding baselines
6. Train the contrastive attribution encoder
7. Evaluate single-prompt and set-level attribution
```

There is no `make_sft_data.py`, no LoRA student training, and no generated student checkpoints.

## Setup

```bash
git clone git@github.com:LuizVenosa/Teacher_attribution.git
cd Teacher_attribution
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Bocconi HPC jobs, `jobs/00_setup_env.sh` loads `miniconda3`, CUDA, activates `teacherattr`, keeps Hugging Face caches inside the project by default, and reads an optional private Hugging Face token from `~/.hf_token`.

## Build Prompts

Use the Who Taught You That style task sources:

```bash
PYTHONPATH=src python scripts/make_prompt_bank.py \
  --source who_taught_you_that \
  --datasets commonsenseqa,openbookqa,alpaca,rotten_tomatoes \
  --train-size 1000 \
  --val-size 300 \
  --test-size 300 \
  --allow-missing-datasets
```

For a tiny smoke test:

```bash
PYTHONPATH=src python scripts/make_prompt_bank.py \
  --source synthetic \
  --train-size 32 \
  --val-size 16 \
  --test-size 16
```

## Cluster Execution

```bash
cd /home/3191856/NLP_project/Teacher_attribution
mkdir -p logs

# 1. Generate public teacher and public student outputs. No arrays.
sbatch jobs/01_generate_public_lineage_outputs.sbatch

# 2. Build train/val/test attribution pair files.
sbatch jobs/04_build_public_lineage_attribution.sbatch

# 3. Run baselines.
sbatch jobs/06_eval_public_lineage.sbatch baselines

# 4. Train contrastive encoder.
sbatch jobs/05_train_public_lineage_contrastive.sbatch

# 5. Evaluate set-level attribution.
sbatch jobs/06_eval_public_lineage.sbatch encoder
```

Outputs from teacher/student generation are stored separately from older experiments:

```text
data/public_lineage/teacher_outputs/
data/public_lineage/student_outputs/
```

Attribution files are written to:

```text
data/attribution/train_pairs.jsonl
data/attribution/val_pairs.jsonl
data/attribution/test_pairs.jsonl
```

## Smoke Runs

Restrict generation to two lightweight pairs:

```bash
TEACHER_LIST="gpt2 qwen15_18b" \
STUDENT_LIST="distilgpt2:miniplm_qwen_200m" \
SPLIT_LIST="train val test" \
sbatch jobs/01_generate_public_lineage_outputs.sbatch
```

Run only one split:

```bash
SPLIT_LIST="test" sbatch jobs/01_generate_public_lineage_outputs.sbatch
SPLIT_LIST="test" sbatch jobs/04_build_public_lineage_attribution.sbatch
```

## Attribution Pair Schema

```json
{
  "prompt_id": "qa_000001",
  "anchor_student_id": "student_from_gpt2",
  "true_teacher": "gpt2",
  "prompt": "Why do objects fall at the same rate in a vacuum?",
  "student_response": "In a vacuum, objects fall at the same rate because...",
  "teacher_responses": {
    "gpt2": "Objects fall because...",
    "qwen15_18b": "In a vacuum, gravitational acceleration...",
    "flan_t5_base": "Objects fall at the same rate because...",
    "flan_t5_small": "Without air resistance..."
  },
  "label": 0
}
```

The negatives are wrong-teacher responses to the **same prompt**, which is the main guardrail against topic leakage.

## Offline Dataset Download

If Hugging Face dataset downloads are painful on the cluster, download them elsewhere first:

```bash
PYTHONPATH=src python scripts/download_wtyt_datasets.py \
  --output-dir external_datasets/who_taught_you_that \
  --datasets cnn_dailymail,sumpubmed,rotten_tomatoes,commonsenseqa,openbookqa,alpaca \
  --format auto \
  --allow-missing-datasets
```

Then build prompts from local Parquet/CSV:

```bash
PYTHONPATH=src python scripts/make_prompt_bank.py \
  --source who_taught_you_that \
  --local-data-dir external_datasets/who_taught_you_that \
  --datasets commonsenseqa,openbookqa,alpaca,rotten_tomatoes \
  --train-size 1000 \
  --val-size 300 \
  --test-size 300 \
  --allow-missing-datasets
```

## Key Files

- `configs/public_lineage_models.yaml`: public teacher/student pairs and labels.
- `configs/public_lineage_generation.yaml`: decoding settings for public teachers/students.
- `configs/public_lineage_attribution.yaml`: contrastive encoder training settings.
- `scripts/generate_public_teachers.py`: teacher response generation.
- `scripts/generate_public_students.py`: public student response generation.
- `scripts/build_attribution_pairs.py`: alignment into InfoNCE rows.
- `scripts/run_baselines.py`: TF-IDF, sentence embedding, POS, and classifier baselines.
- `scripts/train_contrastive_encoder.py`: prompt-conditioned InfoNCE encoder.
- `scripts/evaluate.py`: set-level attribution evaluation.
