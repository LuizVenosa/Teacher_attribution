# Implementation and execution guide

This implements the supplied **Teacher Attribution Research Project — Implementation Specification** as one package and CLI. The named scripts are compatibility wrappers, not separate implementations. The original report/PDF and saved results remain historical. The default model and dataset revisions were checked and pinned on 2026-09-15; `pin` can materialize a separate explicitly resolved configuration. No A100 experiment or publication-level performance is claimed by the software tests.

## Models and hardware

| Role | Hugging Face repository |
|---|---|
| Qwen teacher | `Qwen/Qwen3.5-9B` |
| Ministral teacher | `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| Gemma teacher | `google/gemma-4-12B-it` |
| Granite teacher | `ibm-granite/granite-3.3-8b-instruct` |
| Identical student base | `allenai/OLMo-2-0425-1B` |
| Learned attribution encoder | `answerdotai/ModernBERT-base` |
| Generic semantic comparison | `sentence-transformers/all-MiniLM-L6-v2` |

The official [OLMo base](https://huggingface.co/allenai/OLMo-2-0425-1B), [Ministral BF16 checkpoint](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-BF16), [Qwen](https://huggingface.co/Qwen/Qwen3.5-9B), and [Gemma](https://huggingface.co/google/gemma-4-12B-it) cards identify the concrete repositories/loaders. Gemma uses the instruction-tuned `-it` release: the unsuffixed base has no chat template. The BF16 Ministral release avoids a teacher-only FP8 requirement on A100. These newer architectures require Transformers 5; `uv.lock` records the resolved dependencies. No remote model code is enabled.

Run teachers sequentially, one GPU each. Student training uses FP32 master weights and optimizer state with BF16 autocast, gradient checkpointing, accumulation, TF32, and CUDA fused AdamW. Defaults are microbatch 4, accumulation 8, length 1024, three fixed epochs, and learning rate 2e-5. The default attention implementation is PyTorch SDPA. Set `attention: flash_attention_2` for compatible models after installing FlashAttention on the Linux A100 host. No automatic fallback changes the experiment.

Packing is deliberately disabled: concatenating independent examples with ordinary causal attention would introduce cross-example context. The config rejects packing until isolated attention boundaries are implemented. Full fine-tuning is the primary method; QLoRA and Set Transformer are not implemented optional alternatives.

A100 40/80 GB capacity and kernel compatibility require an actual GPU pilot. The development machine has a 4 GB RTX 3050; tests use tiny CPU models. Do not launch the main four-teacher run on that machine.

## 1. Pin and prepare

```bash
teacher-attr --config configs/research.yaml preflight
teacher-attr --config configs/research.yaml pin --output configs/research.pinned.yaml
teacher-attr --config configs/research.pinned.yaml prepare
```

The 5K composition is in README. The harmless-base portion of Anthropic HH-RLHF is the chosen public safety source, not a dedicated uncertainty benchmark. A reviewed JSONL source can replace it using `path`, `format: prompt`, and `prompt_field`. Tülu retains only a standalone first user turn and discards assistant turns. Source definitions, versions, bounded scan limit, common character cap, counts, and seed are configurable. Local Parquet/JSONL paths are supported per source; online sources stream pinned revisions. Prepare saves all five pools and nested 500/1000/2500/5000 prefix subsets before any teacher inference.

Pool names map as follows: `distill_train`, `distill_val`, `train` (attribution train), `val` (attribution validation), `test` (attribution test). Exact normalized prompt duplicates and shared source groups cannot cross pools. Semantic near-duplicate/paraphrase detection is not automatically guaranteed; inspect source contamination as a separate research audit.

## 2. Cache teachers and inspect QC

Use the pinned config with every command. Below, `CONFIG` stands for its path, not a literal file.

```bash
teacher-attr --config CONFIG generate --role teachers --split distill_train
teacher-attr --config CONFIG generate --role teachers --split distill_val
teacher-attr --config CONFIG qc --split distill_train
teacher-attr --config CONFIG qc --split distill_val
```

Generation uses stable batches, left padding, common sampling parameters, task-specific output caps and length instructions. A partial batch is regenerated using the original batch seed when resuming; completed rows are not written twice. Keep one writer per role/model/split. Bitwise resume is tested within one runtime; it is not guaranteed across different GPU kernels/software. Overlength prompts fail before inference rather than being silently truncated for one teacher.

Thinking is disabled through model chat options where supported. Response-template parsing keeps final content only; fallback cleaning discards reasoning blocks. Identity/format issues are flagged, not quietly relabeled. Hidden reasoning is not persisted. QC saves length/vocabulary/refusal/invalid/truncation statistics, task breakdowns, and a PDF length comparison. Refusal detection is an explicit regex heuristic. Quality thresholds and identity patterns are configurable. A failed audit blocks distillation; invalid targets cannot be dropped separately for one teacher. Revise common controls and use a new run if the pilot fails.

## 3. Distill the four students

```bash
teacher-attr --config CONFIG distill --teacher qwen
teacher-attr --config CONFIG distill --teacher ministral
teacher-attr --config CONFIG distill --teacher gemma
teacher-attr --config CONFIG distill --teacher granite
```

All runs start from the configured base and use the same exact ordered prompt subset. Only target responses differ; the same seed controls data shuffling. Prompt and padding tokens are masked, gradients are normalized over actual target tokens including the final accumulation window, and validation loss is diagnostic. Every student receives the same fixed epoch count. Checkpoints, base revision, response hashes, seed, losses, target-token counts, optimizer steps, peak allocated memory, elapsed time, and runtime provenance are saved. Teachers are never loaded by this command.

Completed and incomplete student directories are never overwritten. Student-training resume is not implemented; use a fresh run directory for an interrupted training pilot. Generation is resumable.

## 4. Generate held-out outputs and establish baselines

```bash
teacher-attr --config CONFIG generate
teacher-attr --config CONFIG build
teacher-attr --config CONFIG audit
teacher-attr --config CONFIG diagnose
teacher-attr --config CONFIG diagnose --bertscore
teacher-attr --config CONFIG evaluate --name baselines
teacher-attr --config CONFIG train
```

The unfiltered `generate` command runs only attribution train/val/test, respecting student split assignments. Build requires complete aligned responses from every candidate teacher and eligible student. Student checkpoint manifests verify lineage and saved weights. Diagnostics use attribution validation prompts and report own/other-teacher semantic agreement, ROUGE-L, lengths, and final-answer-letter accuracy where available. BERTScore is an optional diagnostic with an explicit configured model/layer count.

Baselines include aligned TF-IDF teacher retrieval, generic semantic cosine, a real spaCy UPOS n-gram classifier, student-only word/character classifiers, and length/format/token-count/punctuation/phrase/perplexity controls. The PoS implementation is a simplified structural baseline, not an exact reproduction of WTYT. Logistic regularization is selected using validation data. Perplexity uses the common unmodified student base and is modeled as log perplexity.

The contrastive encoder uses same-prompt wrong-teacher negatives and an auxiliary teacher-classification loss. It preserves a separate response token budget. Supervised probes and learned encoders train on labeled student outputs; this is explicitly distinct from any teacher-only training protocol in prior work.

```bash
teacher-attr --config CONFIG evaluate --checkpoint ENCODER_CHECKPOINT --name single
teacher-attr --config CONFIG figures --evaluation SINGLE_EVALUATION_DIRECTORY
```

Metrics include top-1/top-2, macro-F1, per-teacher accuracy, confusion matrices, AUC, prompt-cluster intervals, paired differences, and chance. Four teachers have 25% chance top-1 accuracy. Reports and figures derive from saved prediction/metric artifacts.

## 5. Pool responses and learn Deep Sets

```bash
teacher-attr --config CONFIG evaluate-sets --checkpoint ENCODER_CHECKPOINT --name mean
teacher-attr --config CONFIG train-sets --checkpoint ENCODER_CHECKPOINT --baseline SINGLE_METRICS_JSON
teacher-attr --config CONFIG evaluate-sets --checkpoint ENCODER_CHECKPOINT --set-checkpoint DEEP_SETS_CHECKPOINT --name learned_sets
teacher-attr --config CONFIG figures --evaluation SET_EVALUATION_DIRECTORY
```

Deep Sets requires a completed lexical/generic/PoS/contrastive evaluation of this exact encoder and dataset. It freezes the encoder, applies a shared MLP to individual embeddings, mean pools, projects and normalizes the set, and learns teacher matching with InfoNCE. Validation sets are separate and saved. Defaults evaluate k=1,2,5,10,20,50. Sets never mix independently trained students.

`evaluate` uses mean per-prompt scores; `evaluate-sets` compares cosine of **mean embeddings** with Deep Sets fingerprints. They are different estimators and are labeled separately. Singletons include every test row. Larger support sets are shared across methods and may overlap across repetitions. Their intervals describe sampling variation for the fixed response pool/checkpoints, not uncertainty over all possible distilled students.

## 6. Amounts, independent seeds, and held-out students

```bash
teacher-attr --config CONFIG plan
teacher-attr --config CONFIG distill --teacher qwen --seed 42 --amount 5000
teacher-attr --config CONFIG distill --teacher qwen --seed 71 --amount 5000
teacher-attr --config CONFIG distill --teacher qwen --seed 13 --amount 500
```

Repeat selected commands for every teacher. `plan` writes the primary sequence and optional seed/amount grid, without launching it. The 500/1000/2500/5000 subsets are nested and identical across teachers. Extra seeds change stochastic training/data order while preserving original base weights and architecture.

```bash
teacher-attr --config CONFIG variant --kind held_out_students --output runs/variants/heldout.yaml
teacher-attr --config CONFIG variant --kind amount --value 500 --output runs/variants/n500.yaml
```

A variant reuses immutable parent prompt pools and explicit student checkpoint paths. Generate aligned outputs and build its pairs, then train/evaluate attribution in that variant. Held-out mode assigns the first seed to train, second to validation, third to test; this is stricter separation than mixing the first two instances across train/validation. No checkpoint identity can cross those splits.

## 7. Robustness with the attribution model held fixed

```bash
teacher-attr --config CONFIG variant --kind decoding --value 0.3 --output runs/variants/temp03.yaml
teacher-attr --config CONFIG variant --kind paraphrase --value paraphrases.jsonl --output runs/variants/paraphrase.yaml
teacher-attr --config CONFIG variant --kind cross_task --value safety --output runs/variants/heldout_safety.yaml
teacher-attr --config CONFIG variant --kind natural --output runs/variants/natural.yaml
```

Repeat decoding at 0, 0.3, 0.7, 1.0. Teacher decoding is fixed; the student temperature changes. Paraphrase input is reviewed JSONL with `original_prompt_id` and `prompt`, covering each test prompt exactly once. Original identity/source grouping is retained for leakage checks and paired comparisons. The code does not claim that an arbitrary supplied paraphrase preserves meaning; review these inputs. Cross-task variants exclude the chosen task from attribution train/validation and retain only that task in test; train a new encoder for each fold. Natural-output variants remove word-length instructions and require freshly distilled students; computational token caps still apply.

For decoding/paraphrase/post-fine-tuning tests, generate and build the variant, then keep the original encoder, probe fitting data and set model fixed:

```bash
teacher-attr --config VARIANT_CONFIG evaluate --checkpoint ORIGINAL_ENCODER --reference-config CONFIG --name robustness
teacher-attr --config VARIANT_CONFIG evaluate-sets --checkpoint ORIGINAL_ENCODER --set-checkpoint ORIGINAL_SET_MODEL --reference-config CONFIG --name robustness_sets
teacher-attr --config CONFIG compare --reference ORIGINAL_EVALUATION_DIR --variant VARIANT_EVALUATION_DIR --output paired_robustness.json
```

The explicit reference config prevents retraining probes on the perturbed responses. Cross-task and held-out-student studies require their own training runs instead. `compare` reports prediction consistency and paired accuracy changes using original prompt IDs.

## 8. Additional student fine-tuning and latent structure

Supply teacher-independent JSONL with `prompt`, `response`, `split` (train or val), and optional `task`. All teachers must use the same independent data file. Its prompts cannot duplicate any original pool; its hash and parent checkpoint provenance are recorded.

```bash
teacher-attr --config CONFIG distill --teacher qwen --post-data independent.jsonl --level light
teacher-attr --config CONFIG distill --teacher qwen --post-data independent.jsonl --level moderate
teacher-attr --config CONFIG variant --kind extra_ft --value light --output runs/variants/light_ft.yaml
```

Repeat for the other teachers. Light/moderate epoch counts are configured; primary distillation settings remain unchanged.

After inspecting the completed simpler baselines:

```bash
teacher-attr --config CONFIG train --representation structure --baseline SINGLE_METRICS_JSON
teacher-attr --config CONFIG train --representation fusion --baseline SINGLE_METRICS_JSON
teacher-attr --config CONFIG train --objective classification
teacher-attr --config CONFIG train --objective contrastive
teacher-attr --config CONFIG train --negatives random
teacher-attr --config CONFIG train --classification-weight 0.1
teacher-attr --config CONFIG train --input-mode response
```

The extension pools an explicitly chosen early/middle hidden layer, transforms it in a structural branch, and optionally fuses it with the final semantic layer. It requires a completed same-data baseline artifact. Early-layer representations are a hypothesis about structure, not proof of syntactic disentanglement. No PoS features are required at inference. The training-only structural probe and paraphrase consistency loss are optional, unimplemented extensions. Select classification-loss weights by recorded validation performance, never test accuracy.

## 9. Collect paper data

```bash
teacher-attr --config CONFIG collect --evaluations EVALUATION_DIR_1 EVALUATION_DIR_2 --output paper_comparison
```

This exports tidy per-method, k, distillation amount, student seed and robustness-condition measurements plus comparison PDFs. Individual `figures` commands produce method charts, prompt-count curves, confusion matrices and CSV data. No absent experiments are filled with invented numbers. Inspect the generated data before combining conditions with different encoders or student populations.

## Validation scope

Tests exercise local teacher generation and batch resume, four independently fine-tuned tiny students, attribution training, lexical/generic/PoS/perplexity controls, Deep Sets, leakage and lineage checks, loss masking, metrics, and figure export. The optional paths have focused checks. They do not establish teacher-model access, A100 memory fit, useful student learning at full scale, or a learned-method advantage. Those are outcomes of running the phased research experiment.
