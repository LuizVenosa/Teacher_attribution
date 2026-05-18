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

An expanded candidate pool lives in `configs/public_lineage_models_extended.yaml`.
It includes the default four pairs plus documented SmolLM2, Llama, and Qwen3
distillation candidates. See `docs/public_lineage_candidate_pairs.md` for the
evidence notes and recommended subset order before running the heavier models.

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

## Final Reported Experiment

The final report uses the clean four-way public-lineage setup and the compact
MiniLM encoder, `sentence-transformers/all-MiniLM-L6-v2`.  The cluster run
generated teacher and student outputs for the four default pairs, built 12,000
test attribution rows (3,000 prompts x 4 teacher labels), and trained the
contrastive encoder on 29,742 training prompts per teacher/student lineage.

The default configs in this repository now point to the final reported MiniLM
setup:

```text
configs/public_lineage_models.yaml
configs/public_lineage_attribution.yaml
```

The smaller tracked prompt files in `data/prompts/` are included as lightweight
examples and smoke-test inputs.  The full final prompt bank is regenerated from
the public/local WTYT-style datasets with the command in
[Build Prompts](#build-prompts), because the generated teacher/student outputs
and attribution-pair JSONL files are larger derived artifacts.

Final MiniLM contrastive result:

| Split / setting | Accuracy | Top-2 accuracy | Macro ROC-AUC | Notes |
|---|---:|---:|---:|---|
| Validation, best checkpoint | 0.6055 | 0.8307 | 0.6822 | Best epoch 10, early stopped at epoch 14. |
| Test, single prompt | 0.6044 | 0.8284 | 0.6814 | Main reportable test result. |
| Test, set size 4 | 0.9000 | 0.9875 | 0.9060 | Averaged over prompt sets. |
| Test, set size 8 | 0.9750 | 1.0000 | 0.9615 | Averaged over prompt sets. |
| Test, set size 16+ | 1.0000 | 1.0000 | 0.9752+ | Averaged over prompt sets. |

Single-prompt MiniLM test accuracy by task:

| Task | Accuracy |
|---|---:|
| QA | 0.8418 |
| Instruction following | 0.6090 |
| Summarization | 0.4447 |

MiniLM ablation summary:

| Run | Test accuracy | Top-2 accuracy | Macro ROC-AUC | Interpretation |
|---|---:|---:|---:|---|
| `lr5e5` | 0.6008 | 0.8286 | 0.6779 | Best ablation by test accuracy. |
| `temp003` | 0.6003 | 0.8286 | 0.6635 | Similar accuracy, weaker AUC. |
| `cls010` | 0.5993 | 0.8295 | 0.6755 | Similar to baseline. |
| `baseline_temp005_cls02_lr2e5` | 0.5988 | 0.8286 | 0.6769 | Five-epoch MiniLM baseline. |
| `temp007` | 0.5976 | 0.8297 | 0.6843 | Best ablation AUC, not best accuracy. |
| `proj256` | 0.5968 | 0.8289 | 0.6768 | Best validation accuracy among ablations, but not best test accuracy. |
| `cls050` | 0.5955 | 0.8295 | 0.6775 | Higher classification weight did not help. |

The ablations are useful for analysis, but none surpassed the full MiniLM run trained with early stopping.

Small final-result files are included with the repository despite the broad
`results/**` ignore rule:

```text
results/baselines/public_lineage_test_metrics.json
results/contrastive/public_lineage_minilm_probe_metrics_no_sentence.json
results/contrastive_ablation/ablation_summary.csv
results/contrastive_ablation/eval_summary.csv
results/contrastive_ablation/epoch_progression.csv
report/figures/
```

Large generated files are intentionally not committed:

```text
data/public_lineage/*_outputs/
data/attribution/*_pairs.jsonl
models/attribution_encoder/*/*.pt
external_datasets/
```

Larger encoders, more model pairs, and harder cross-dataset/paraphrased-prompt
settings are left as future work rather than part of the final reported result.

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

Use the strict Who Taught You That task-source mix from local Parquet/CSV files:

```bash
PYTHONPATH=src python scripts/make_prompt_bank.py \
  --source who_taught_you_that \
  --local-data-dir external_datasets/who_taught_you_that \
  --datasets cnn_dailymail,sumpubmed,rotten_tomatoes,commonsenseqa,openbookqa,quarel,alpaca \
  --train-size 30000 \
  --val-size 3000 \
  --test-size 3000
```

The completed cluster run produced 29,742 train prompts after dataset availability and filtering, plus 3,000-prompt validation/test style evaluation files.

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
cd /mnt/beegfsstudents/home/3191856/NLP_project/Teacher_attribution
mkdir -p logs

# 1. Generate public teacher and public student outputs. No arrays.
sbatch jobs/01_generate_public_lineage_outputs.sbatch

# 2. Build train/val/test attribution pair files.
sbatch jobs/04_build_public_lineage_attribution.sbatch

# 3. Run baselines.
sbatch jobs/06_eval_public_lineage.sbatch baselines

# 4. Train the final reported MiniLM contrastive encoder.
sbatch --time=08:00:00 jobs/05_train_public_lineage_contrastive.sbatch

# 5. Evaluate set-level attribution.
sbatch jobs/06_eval_public_lineage.sbatch encoder

# 6. Evaluate frozen contrastive probes used in the final report.
SKIP_SENTENCE_BASELINE=1 sbatch jobs/08_eval_contrastive_probes.sbatch

# 7. Regenerate report figures.
sbatch jobs/09_plot_latent_space_hero.sbatch
```

To run the expanded candidate pool instead of the default four-pair setup:

```bash
MODELS_CONFIG=configs/public_lineage_models_extended.yaml \
sbatch jobs/01_generate_public_lineage_outputs.sbatch

MODELS_CONFIG=configs/public_lineage_models_extended.yaml \
sbatch jobs/04_build_public_lineage_attribution.sbatch
```

For the first expanded run, prefer a medium-size subset before adding the heavy
Qwen3 MoE teacher:

```bash
MODELS_CONFIG=configs/public_lineage_models_extended.yaml \
TEACHER_LIST="gpt2 qwen15_18b flan_t5_base flan_t5_small llama32_3b_instruct smollm2_17b_instruct" \
STUDENT_LIST="distilgpt2:miniplm_qwen_200m:lamini_flan_t5_248m:lamini_flan_t5_77m:lrc_15b_sft:d_smollm2_360m" \
sbatch jobs/01_generate_public_lineage_outputs.sbatch
```

If you previously generated public-lineage outputs, move or delete the old
`data/public_lineage` directory before switching configs so old student JSONL
files do not get mixed into the new attribution build.


## Who Taught You That Comparability

To make the results comparable to *Who Taught You That?*, report two tracks:

1. **WTYT-style protocol results**: same broad task families, same student-response-only attribution setting, same support-size curves, and text-feature baselines.
2. **Our contrastive extension**: same data splits and lineage labels, but with prompt-conditioned same-prompt teacher negatives and learned embeddings.

The important caveat is that this repo uses **public documented lineage pairs** instead of training every student from scratch on teacher outputs. That makes the protocol comparable, but the student construction is not identical to the paper's controlled distillation setup.

For the closest WTYT-style table, build prompts with the full dataset mix:

```bash
PYTHONPATH=src python scripts/make_prompt_bank.py \
  --source who_taught_you_that \
  --local-data-dir external_datasets/who_taught_you_that \
  --datasets cnn_dailymail,sumpubmed,rotten_tomatoes,commonsenseqa,openbookqa,quarel,alpaca \
  --train-size 30000 \
  --val-size 3000 \
  --test-size 3000
```

This uses a balanced maximum across all seven WTYT task sources, so QuaRel and OpenBookQA do not disappear under the much larger summarization/review datasets.

Current baseline evaluator:

```bash
sbatch jobs/06_eval_public_lineage.sbatch baselines
```

It reports WTYT-style text baselines that are currently reproducible in this repo:

```text
bow_same_prompt
bertscore_same_prompt
bow_classifier
ngram_1_4_classifier
```

The POS-template baseline from the paper is intentionally excluded from the main comparison because it was not reliably replicated in the current environment.

Output:

```text
results/baselines/public_lineage_test_metrics.json
```

Use these as the directly comparable baseline table. Use the contrastive encoder results as the proposed-method table.
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
  --datasets cnn_dailymail,sumpubmed,rotten_tomatoes,commonsenseqa,openbookqa,quarel,alpaca \
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
- `scripts/run_baselines.py`: BoW similarity, BERTScore similarity, BoW classifier, and 1-4 gram classifier baselines.
- `scripts/train_contrastive_encoder.py`: prompt-conditioned InfoNCE encoder.
- `scripts/evaluate.py`: set-level attribution evaluation.
- `scripts/evaluate_contrastive_probes.py`: frozen contrastive encoder probes plus response-only cosine baselines.

## Contrastive Training And Ablations

The contrastive encoder uses a high epoch ceiling with validation early stopping. In `configs/public_lineage_attribution.yaml`:

```yaml
num_epochs: 30
early_stopping_metric: accuracy
early_stopping_mode: max
early_stopping_patience: 6
early_stopping_min_delta: 0.002
```

This means training may run for up to 30 epochs, but it stops when validation accuracy has not improved by at least `0.002` for 6 consecutive epochs. The trainer saves:

```text
models/attribution_encoder/<run_name>/best.pt
models/attribution_encoder/<run_name>/last.pt
models/attribution_encoder/<run_name>/training_metrics.json
```

Run the main contrastive training job:

```bash
sbatch jobs/05_train_public_lineage_contrastive.sbatch
```

A compact no-array ablation sweep is also available:

```bash
sbatch jobs/05_ablate_public_lineage_contrastive.sbatch
```

By default it runs a sequential compact sweep over temperature, classification-loss weight, learning rate, and projection size. The current compact sweep uses fixed 5-epoch runs and disables early stopping so curves are comparable. Results are written to:

```text
results/contrastive_ablation/ablation_summary.csv
results/contrastive_ablation/ablation_summary.json
results/contrastive_ablation/eval_summary.csv
results/contrastive_ablation/epoch_progression.csv
models/attribution_encoder/public_lineage_ablations/
```

For a custom grid, override environment variables at submit time:

```bash
ABLATION_MODE=grid \
TEMPERATURES="0.03,0.05,0.07,0.1" \
CLASSIFICATION_WEIGHTS="0.1,0.2,0.5" \
LEARNING_RATES="0.00002,0.00005" \
MAX_LENGTHS="512,768" \
PROJECTION_DIMS="128,256" \
MAX_RUNS=12 \
sbatch jobs/05_ablate_public_lineage_contrastive.sbatch
```

Keep `MAX_RUNS` modest on the student cluster because this job runs experiments sequentially inside one allocation.

## Contrastive Probe Evaluation

The main encoder evaluation uses nearest-neighbor cosine retrieval in the learned teacher space. To check whether the learned embedding is more useful with a stronger but still simple decision rule, run the frozen-probe evaluation:

```bash
sbatch jobs/08_eval_contrastive_probes.sbatch
```

By default this evaluates the MiniLM checkpoint:

```text
models/attribution_encoder/public_lineage_minilm_contrastive/best.pt
```

and writes:

```text
results/contrastive/public_lineage_minilm_probe_metrics.json
```

The JSON includes four contrastive-encoder variants:

```text
cosine_retrieval
classifier_head
embedding_logreg
cosine_score_logreg
```

It also includes simple response-only cosine baselines:

```text
bow_response_cosine
tfidf_response_cosine
sentence_response_cosine
```

These baselines compare each student output to the same-prompt candidate teacher outputs by cosine similarity only, so they are easier to compare directly against contrastive nearest-neighbor retrieval.

Override paths or skip the sentence baseline if the checkpoint is not cached:

```bash
SKIP_SENTENCE_BASELINE=1 \
ENCODER_OUTPUT=results/contrastive/public_lineage_minilm_probe_metrics_no_sentence.json \
sbatch jobs/08_eval_contrastive_probes.sbatch
```
