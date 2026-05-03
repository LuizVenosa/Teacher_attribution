# Teacher Attribution

Prompt-conditioned contrastive attribution for distilled language-model students.

The core experiment is simple:

1. Build a shared prompt bank.
2. Generate outputs from a closed set of teacher models.
3. Distill one LoRA student per teacher.
4. Generate student outputs on shared attribution prompts.
5. Build JSONL attribution pairs with same-prompt teacher candidates.
6. Run lexical and embedding baselines.
7. Train a shared prompt-response encoder with InfoNCE plus a teacher-ID head.
8. Evaluate single-prompt and set-level attribution.

Everything is JSONL. Heavy steps are SLURM batch jobs. Notebooks are optional scratch work, not the pipeline.

## Setup

```bash
cd /scratch/$USER
git clone git@github.com:YOUR_USERNAME/teacher-attribution.git
cd teacher-attribution

python -m venv .venv
source .venv/bin/activate
pip install -e ".[generation]"
```

On HPC jobs, keep caches on scratch:

```bash
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
export HF_DATASETS_CACHE=/scratch/$USER/hf_cache/datasets
export WANDB_DIR=/scratch/$USER/wandb
```

## Local Smoke Pipeline

This creates a small prompt bank and validates the CPU-only stages.

```bash
python scripts/make_prompt_bank.py --distill-size 20 --train-size 20 --val-size 8 --test-size 8
python scripts/make_sft_data.py \
  --teacher_outputs data/teacher_outputs/qwen_distill.jsonl \
  --output data/distill_data/qwen_teacher_sft.jsonl
```

The generation and training scripts require model downloads and usually belong on GPU nodes.

## HPC Execution

```bash
mkdir -p logs

# Prompt bank can run on the login node.
python scripts/make_prompt_bank.py

# Teacher outputs for distill/train/val/test.
sbatch jobs/01_generate_teachers.sbatch

# One LoRA student per teacher.
sbatch jobs/02_distill_student.sbatch

# Student outputs for attribution train/val/test.
sbatch jobs/03_generate_students.sbatch

# Build train/val/test attribution JSONL.
sbatch jobs/04_build_attribution_data.sbatch

# Baselines.
sbatch jobs/06_eval.sbatch baselines

# Contrastive encoder.
sbatch jobs/05_train_contrastive.sbatch

# Encoder evaluation with set-size curves.
sbatch jobs/06_eval.sbatch encoder
```

## Key Files

- `configs/models.yaml`: teacher models, student base, attribution encoder, label map.
- `configs/generation.yaml`: decoding settings for teacher and student generation.
- `configs/distill.yaml`: LoRA and SFT settings.
- `configs/attribution.yaml`: contrastive encoder training settings.
- `data/attribution/train_pairs.jsonl`: main InfoNCE training file.
- `results/contrastive/set_level_metrics.json`: accuracy by number of prompts.

## Attribution Pair Schema

```json
{
  "prompt_id": "qa_000001",
  "anchor_student_id": "student_from_qwen",
  "true_teacher": "qwen",
  "prompt": "Why do objects fall at the same rate in a vacuum?",
  "student_response": "In a vacuum, there is no air resistance...",
  "teacher_responses": {
    "qwen": "Objects fall at the same rate in a vacuum because...",
    "llama": "...",
    "mistral": "...",
    "gemma": "..."
  },
  "label": 0
}
```

The negatives are wrong-teacher responses to the same prompt. That is the main guardrail against topic leakage.

## Minimal First Milestone

- 4 teachers.
- 1 shared student base.
- LoRA students only.
- TF-IDF, sentence-embedding, and POS-template baselines.
- Single-response contrastive encoder.
- Accuracy vs prompt set size using mean pooling.

Save the latent-structure branch and Set Transformer for later ablations.
