# PowerPoint Generation Brief: Teacher Attribution Project

Use this markdown as the source brief for building a polished academic PowerPoint presentation. The presentation should explain the research idea, motivation, method, experiment design, pipeline, and current results clearly to a machine learning / NLP audience.

## Presentation Goal

Create a 10-12 slide presentation about a research project on **teacher attribution in distilled language models**.

The core question is:

> Given the outputs of a student language model, can we infer which teacher model it was distilled from or influenced by?

The project builds a prompt-conditioned contrastive attribution system. For each shared prompt, it compares a public student model's response against responses from candidate teacher models. The model learns to pull the student response embedding close to the true teacher response embedding and away from wrong-teacher response embeddings.

## Project Title

**Who Taught This Model? Prompt-Conditioned Contrastive Attribution for Public Teacher-Student Language Models**

Alternative shorter title:

**Tracing Teacher Fingerprints in Distilled Language Models**

## Motivation

Modern language models are often trained or improved using outputs from stronger models. This includes distillation, synthetic instruction tuning, and teacher-generated reasoning data. However, once a smaller model is released, its training lineage is not always easy to verify from the model weights or model card alone.

This matters because model lineage affects:

- transparency and documentation
- intellectual property and data provenance
- safety auditing
- reproducibility of model development
- understanding how behavior transfers during distillation

Traditional text similarity asks:

> Which teacher output looks most similar to the student output?

This project asks a more structured question:

> Which teacher leaves a consistent behavioral fingerprint in the student's responses across prompts?

## Inspiration From Prior Work

The project is inspired by the paper **"Who Taught You That? Tracing Teachers in Model Distillation"**.

That work studies whether teacher identity can be recovered from student models trained on teacher outputs. Their experimental idea is:

1. Select a closed set of teacher models.
2. Generate training data from each teacher.
3. Fine-tune one smaller student per teacher.
4. Generate student outputs on held-out prompts.
5. Attribute each student back to its teacher.

This project adapts the idea in a practical direction:

- Instead of training all students from scratch, use **publicly released teacher-student model pairs** with documented lineage.
- Generate outputs from both teachers and public students on the same prompt bank.
- Build contrastive attribution examples from aligned prompt-response pairs.

## Main Research Hypothesis

A distilled or teacher-influenced student model inherits more than surface wording. It may inherit latent behavioral patterns such as:

- answer structure
- explanation depth
- preferred phrasing
- reasoning format
- uncertainty handling
- verbosity
- formatting habits
- task-specific response style

A contrastive encoder trained on prompt-conditioned response pairs can learn these teacher-specific fingerprints better than raw lexical or semantic similarity baselines.

## Public Teacher-Student Pairs

The current experiment uses four public model pairs:

| Teacher ID | Teacher Model | Student ID | Student Model | Evidence Level |
|---|---|---|---|---|
| `gpt2` | `gpt2` | `distilgpt2` | `distilbert/distilgpt2` | High: DistilGPT2 is documented as distilled from GPT-2. |
| `qwen15_18b` | `Qwen/Qwen1.5-1.8B` | `miniplm_qwen_200m` | `MiniLLM/MiniPLM-Qwen-200M` | High: MiniPLM metadata names Qwen1.5-1.8B as teacher. |
| `flan_t5_base` | `google/flan-t5-base` | `lamini_flan_t5_248m` | `MBZUAI/LaMini-Flan-T5-248M` | Medium: Flan-T5-family instruction-distilled lineage. |
| `flan_t5_small` | `google/flan-t5-small` | `lamini_flan_t5_77m` | `MBZUAI/LaMini-Flan-T5-77M` | Medium: Flan-T5-small family / instruction-distilled lineage. |

Important note: BERT -> DistilBERT is not used because BERT and DistilBERT are encoder-only masked language models, not natural text generators. This project requires models that can produce prompt responses.

## Data Setup

The experiment uses a shared prompt bank inspired by the task mix from *Who Taught You That?*.

Prompt sources include a mix of:

- CNN/DailyMail
- SumpubMed
- CommonsenseQA
- OpenBookQA
- QuaRel
- Rotten Tomatoes
- Alpaca-style instruction prompts

All candidate teachers and all public students answer the same prompts.

This is important because it prevents the attribution model from solving the task by topic alone. The comparison is teacher-specific rather than topic-specific.

## Core Data Schema

For each prompt, the final attribution example looks like:

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

The key design rule:

> The negative teacher responses must be wrong-teacher responses to the same prompt.

## Method: Prompt-Conditioned Contrastive Encoder

Each prompt-response pair is encoded as one text sequence:

```text
[PROMPT] x [RESPONSE] y
```

A shared encoder maps it to an embedding:

```text
h = f_theta(x, y)
```

For one training example:

```text
Anchor:   student response embedding
Positive: true teacher response embedding
Negatives: wrong teacher response embeddings for the same prompt
```

The model learns a teacher-attribution embedding space.

## Contrastive Objective

For each prompt, compute:

```text
h_S = f(prompt, student_response)
h_T* = f(prompt, true_teacher_response)
h_Tj = f(prompt, wrong_teacher_response_j)
```

Use InfoNCE:

```text
L = -log exp(sim(h_S, h_T*) / tau)
        / sum_k exp(sim(h_S, h_Tk) / tau)
```

Where:

- `sim` is cosine similarity
- `tau` is temperature
- the positive is the true teacher
- negatives are all other candidate teachers for the same prompt

An auxiliary teacher-classification head can also be used:

```text
L_total = L_contrastive + lambda * L_classification
```

## Inference

At test time, an unknown student answers a set of prompts. For each candidate teacher, teacher responses to those same prompts are available.

For each prompt:

1. Embed the student prompt-response pair.
2. Embed each teacher prompt-response pair.
3. Compare the student embedding to teacher embeddings.
4. Predict the nearest teacher.

For set-level attribution, average embeddings across multiple prompts:

```text
F_S = mean_i f(x_i, y_i^S)
F_T = mean_i f(x_i, y_i^T)
```

Then choose:

```text
argmax_T sim(F_S, F_T)
```

This produces a behavioral fingerprint over multiple prompts.

## Baselines

The project evaluates against WTYT-style baselines that are reproducible in the current environment:

1. **BoW same-prompt similarity**
   - Compare the student response against same-prompt teacher responses with lexical features.

2. **BERTScore same-prompt similarity**
   - Compare student and teacher responses with contextual token-level similarity.

3. **BoW classifier**
   - Train a supervised text classifier from student responses to teacher labels.

4. **1-4 gram classifier**
   - Train a supervised n-gram classifier from student responses to teacher labels.

The POS-template baseline from the paper is not used in the main comparison because it was not reliably replicated in the current environment.

## Evaluation Metrics

Report:

- top-1 accuracy
- top-2 accuracy
- ROC-AUC
- confusion matrix
- accuracy by task type
- accuracy by dataset
- accuracy by student-teacher pair
- accuracy as a function of number of prompts aggregated

The most important figure is:

```text
Accuracy vs number of prompts
```

Expected trend:

```text
1 prompt  -> noisy attribution
4 prompts -> better
16 prompts -> strong signal
32+ prompts -> more stable teacher fingerprint
```

## Current Results

Completed main result:

| Model | Setting | Accuracy | Top-2 accuracy | Macro ROC-AUC |
|---|---|---:|---:|---:|
| MiniLM contrastive encoder | Validation best checkpoint | 0.6055 | 0.8307 | 0.6822 |
| MiniLM contrastive encoder | Test, single prompt | 0.6044 | 0.8284 | 0.6814 |
| MiniLM contrastive encoder | Test, set size 4 | 0.9000 | 0.9875 | 0.9060 |
| MiniLM contrastive encoder | Test, set size 8 | 0.9750 | 1.0000 | 0.9615 |
| MiniLM contrastive encoder | Test, set size 16+ | 1.0000 | 1.0000 | 0.9752+ |

Single-prompt MiniLM test accuracy by task:

| Task | Accuracy |
|---|---:|
| QA | 0.8418 |
| Instruction following | 0.6090 |
| Summarization | 0.4447 |

Ablation summary:

- Seven MiniLM ablations were run for fixed 5-epoch comparisons.
- Best test accuracy was `lr5e5` at 0.6008.
- Best validation accuracy was `proj256` at 0.6011.
- No ablation surpassed the full MiniLM run with early stopping.

E5-large status:

- `intfloat/e5-large-v2` is configured as the current larger encoder experiment.
- It improved through epoch 3, reaching validation accuracy around 0.5517.
- Epoch 4 collapsed to chance accuracy, so the next E5 attempt should lower the learning rate.

## Current Repository Pipeline

The project is implemented as a clean JSONL pipeline with SLURM jobs for heavy stages.

Current public-lineage pipeline:

```text
1. Build prompt bank
2. Generate public teacher outputs
3. Generate public student outputs
4. Build attribution pairs
5. Run baselines
6. Train contrastive encoder
7. Evaluate set-level attribution
```

No LoRA fine-tuning is currently required.

Important files:

```text
configs/public_lineage_models.yaml
configs/public_lineage_generation.yaml
configs/public_lineage_attribution.yaml
scripts/generate_public_teachers.py
scripts/generate_public_students.py
scripts/build_attribution_pairs.py
scripts/run_baselines.py
scripts/train_contrastive_encoder.py
scripts/evaluate.py
jobs/01_generate_public_lineage_outputs.sbatch
jobs/04_build_public_lineage_attribution.sbatch
jobs/05_train_public_lineage_contrastive.sbatch
jobs/06_eval_public_lineage.sbatch
```

## Practical HPC Notes

The Bocconi compute nodes do not have internet access, so models must be downloaded on the login node first and then loaded from the Hugging Face cache in offline mode.

The generation jobs set:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

This prevents failed network calls during GPU jobs.

## Suggested Slide Outline

Create the deck with the following slides.

### Slide 1: Title

Title:
**Who Taught This Model? Prompt-Conditioned Contrastive Attribution for Public Teacher-Student LMs**

Subtitle:
Tracing teacher fingerprints in generated text.

### Slide 2: Motivation

Explain why model lineage matters:

- transparency
- provenance
- reproducibility
- safety auditing
- understanding distillation transfer

Main question:

> Can we infer a student model's teacher from its outputs?

### Slide 3: Prior Work Inspiration

Explain *Who Taught You That?*:

- closed teacher set
- teacher-generated training data
- one student per teacher
- attribute students back to teachers

Then explain this project's adaptation:

- use public documented teacher-student pairs
- skip training students ourselves
- focus on contrastive attribution

### Slide 4: Public Teacher-Student Pairs

Show the four pairs in a table.

Emphasize that the experiment uses generative models only, not encoder-only BERT-style models.

### Slide 5: Data Construction

Show the shared prompt setup:

```text
Prompt x
  -> teacher gpt2 response
  -> teacher qwen response
  -> teacher flan-base response
  -> teacher flan-small response
  -> public student response
```

Explain why same-prompt negatives prevent topic leakage.

### Slide 6: Contrastive Architecture

Diagram:

```text
Prompt + student response -> shared encoder -> student embedding
Prompt + teacher response -> shared encoder -> teacher embedding

student close to true teacher
student far from wrong teachers
```

Mention the encoder and projection head.

### Slide 7: Loss Function

Present InfoNCE intuitively:

- anchor = student response
- positive = true teacher response
- negatives = wrong teacher responses
- similarity = cosine similarity

Optional classification head stabilizes learning.

### Slide 8: Baselines

Compare against:

- BoW same-prompt similarity
- BERTScore same-prompt similarity
- BoW classifier
- 1-4 gram classifier

Explain why baselines are necessary.

### Slide 9: Set-Level Attribution

Explain why single responses are noisy.

Show:

```text
average embeddings over N prompts
compare student fingerprint to teacher fingerprints
```

Expected result: accuracy increases as prompt count increases.

### Slide 10: Implementation Pipeline

Show the repo pipeline:

```text
prompt bank -> teacher outputs -> student outputs -> attribution pairs -> baselines -> contrastive encoder -> evaluation
```

Mention:

- JSONL files at every stage
- SLURM jobs for heavy steps
- offline model cache on HPC

### Slide 11: Results / Analysis

Analyses to show:

- MiniLM contrastive single-prompt accuracy is about 60.4%
- top-2 accuracy is about 82.8%
- set-level aggregation improves accuracy strongly
- QA attribution is much easier than summarization attribution
- confusion matrix reveals which teachers are hard to distinguish
- weaker lineage pairs may be noisier than high-confidence pairs

### Slide 12: Limitations and Future Work

Limitations:

- public model lineage can be noisy
- not all model cards document training clearly
- some teacher-student pairs are same-family rather than clean distillation
- teacher responses must be generated locally or cached
- full DeepSeek-R1-style teachers are too large for this setup

Future work:

- evaluate or stabilize the E5-large encoder with a lower learning rate
- add Llama and SmolLM lineages after verifying runtime and storage
- test harder same-family teacher sets
- add cross-dataset and paraphrase robustness tests
- compare against additional latent style features
- optionally add controlled LoRA-distilled students as a cleaner distillation baseline

## Visual Style Guidance

Use a clean academic style:

- dark text on light background
- restrained color palette
- one main idea per slide
- diagrams for pipeline and contrastive objective
- tables for model pairs and baselines
- avoid excessive text
- include speaker notes with concise explanations

Recommended colors:

- deep navy or charcoal for headings
- blue/teal accents for embeddings and arrows
- orange or red only for negative examples
- green for positive teacher match

## Key Message To Emphasize

The project is not just asking whether a student output is textually similar to a teacher output.

It asks whether a student model carries a **consistent latent behavioral fingerprint** of its teacher across many prompts.

The contrastive prompt-conditioned encoder is designed to learn that fingerprint directly.
