# Historical public-model pilot

Use `--config configs/experiment.yaml` explicitly with every command below. The default is now the controlled research configuration.

# Teacher Attribution

A small pipeline for output-based model-lineage experiments: **prepare â†’ generate â†’ build â†’ train â†’ evaluate**.

The default is a **two-label public-lineage pilot**: DistilGPT2/GPT-2 and MiniPLM/Qwen1.5. Their transfer mechanisms differ (logit distillation versus teacher-guided data selection). The same public student checkpoints occur in training and testing, so this pilot measures recognition of known student lineages. It does not isolate teacher influence from student identity or establish generalization to unseen students.

LaMini/FLAN pairs were removed from the active benchmark: FLAN is their initialization checkpoint, while GPT-3.5 generated the instruction-training responses. The previous four-way report and results remain historical artifacts; they are not results of this pipeline.

## Install

Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
# Linux: source .venv/bin/activate
# PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[models,data,dev]"
```

For the locked dependency versions, use `uv sync --all-extras --locked` instead of the pip install, then activate `.venv`. The lockfile covers supported Python/platform combinations; the CPU integration tests were run on Python 3.12 on Windows.

The core package only requires NumPy, scikit-learn, and PyYAML. Model and source-dataset dependencies are optional extras. There is no separate BERTScore service, POS parser, set network, or notebook pipeline to maintain.

## Configure once

Edit `configs/experiment.yaml`. All paths in it are relative to the config file, including `run_dir`. The first stage saves the resolved configuration; changing it requires a new run directory. Use immutable Hub commit revisions for final experiments. Resolved model commits, input hashes, generation settings, and package versions are recorded where available.

The default reads existing local Parquet files under `external_datasets/who_taught_you_that`. Each source directory can also contain JSONL files with the original dataset schema. The old `sumpubmed` directory is accepted as a location for `ccdv/pubmed-summarization/section`, but outputs correctly call it `pubmed`. There is no silent substitution or missing-dataset fallback. Set `data.local_dir: null` to read the explicitly named public sources through Hugging Face Datasets; persist the resulting prompt files for reproducibility.

`prompts_per_dataset` specifies exact targets **per source**, not total row counts. Reduce targets for an initial smoke run. The default uses six sources and 1,000/200/200 prompts per source. The source loader stops after filling the requested deterministic subsets; these are not official benchmark splits or uniform full-corpus samples.

## Run

```bash
teacher-attr prepare
teacher-attr generate
teacher-attr build
teacher-attr audit
teacher-attr train
teacher-attr evaluate --frozen --checkpoint runs/public_lineage_v2/models/joint_prompt_response_seed13/best.pt
```

`python -m teacher_attr` is equivalent to `teacher-attr`. For another config, put the option before the command:

```bash
teacher-attr --config configs/my_experiment.yaml prepare
```

Generation can be divided into independent cluster jobs:

```bash
teacher-attr generate --role teachers --model gpt2 --split test
teacher-attr generate --role students --model distilgpt2 --split test
```

Existing outputs resume after checking their configuration and prompt hashes. Each prompt uses a deterministic seed, so resuming does not change subsequent random draws. Use one writer per model/split output. Generation processes one prompt at a time for a straightforward reproducible baseline; it is slower than optimized batched inference. Every model receives the exact stored prompt content. If a prompt exceeds a model's token budget, generation fails; shorten the common character limit and prepare a new run rather than truncating differently for each model.

Activate the environment on the cluster before submitting. A single wrapper replaces the old job collection:

```bash
sbatch --account=YOUR_ACCOUNT jobs/run.sbatch generate --role teachers --model gpt2
sbatch --account=YOUR_ACCOUNT jobs/run.sbatch train
```

Submit stages after their dependencies complete. Override resource requests with `sbatch` options. The wrapper contains no account credentials, email notifications, or machine-specific paths.

## Essential controls

```bash
teacher-attr train --objective classification
teacher-attr train --objective contrastive
teacher-attr train --input-mode response
teacher-attr train --seed 42
teacher-attr evaluate --name lexical
teacher-attr evaluate --frozen --name frozen
teacher-attr evaluate --checkpoint PATH_TO_AN_ABLATION/best.pt --name classification
```

Training uses the same configured epoch ceiling, patience, batch size, and optimizer across ablations. Joint/contrastive runs select checkpoints by validation cosine retrieval accuracy; classification-only runs use validation classifier accuracy. Logistic-regression regularization is selected on validation data, never test data. The frozen control uses the pretrained backbone's mean-pooled representation without a random projection.

The encoder supports response-only inputs or separately budgeted prompt/response pairs. At most `max_prompt_tokens` are spent on the prompt; the rest of the context is reserved for the response and special tokens. Evaluation saves actual retained-token counts. Training uses float32 and checks for nonfinite losses/gradients; mixed-precision branches were removed.

## What is saved

Everything for a new experiment lives under its run directory:

```text
experiment.json              resolved immutable configuration
prompts/{train,val,test}.jsonl
prompts/manifest.json         source descriptions, hashes, split counts
outputs/{teachers,students}/MODEL/SPLIT.jsonl
outputs/.../SPLIT.meta.json   generation configuration and resolved model revision
pairs/{train,val,test}.jsonl
pairs/manifest.json           alignment provenance and hashes
models/OBJECTIVE_INPUT_seedN/ best.pt, training.json, tokenizer/, backbone/
evaluations/NAME/             metrics.json, predictions.jsonl, support_sets.jsonl,
                             summary.md, token_audit.json (when encoding)
```

Evaluations include word/punctuation 1â€“4 grams, character 1â€“5 grams, and length/format controls, plus optional frozen/trained cosine retrieval and linear probes. Classifier-head results are included only when that head was trained. Failed baselines fail the command.

Every method uses the **arithmetic mean of its per-prompt scores** on exactly the same saved support sets. Size one uses every test row. Larger sets are sampled from one student checkpoint, both across all tasks and within each dataset. Sets may overlap across repetitions. Accuracy confidence intervals and paired differences versus word n-grams resample whole prompts. Set-level Monte Carlo intervals measure support-sampling variation conditional on the fixed response pool. Neither interval estimates uncertainty over a population of student models.

## Moving beyond the public pilot

Supply multiple genuinely distinct student checkpoints per teacher, record each checkpoint's actual teacher and evidence, assign each student to explicit `splits`, and set `protocol: held_out_students`. The config then requires disjoint checkpoint identities between train, validation, and test and complete teacher coverage in each split. Model aliases cannot bypass this check when they share the same configured Hub path/revision. Independently verify aliases and pin revisions when defining a benchmark.

The repository accepts those controlled checkpoints; it does not train distilled students. A publication experiment should control student base, data, and training budget across teachers, repeat distillation/training seeds, and include undistilled controls. Open-world rejection and a faithful WTYT POS-template replication remain research extensions, not implemented results.

## Checks and historical files

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
```

Tests cover split overlap, duplicate IDs, alignment, response-token preservation, binary AUC, shared support sets, artifact mismatch handling, and a tiny local-model integration run. They do not claim to reproduce the historical cluster results.

`report/` and the existing `results/` contain the earlier experiment. See `report/README.md` and `docs/refactor.md` before interpreting those numbers. New outputs go exclusively under `runs/`; the old scripts and duplicate configs have been removed from the active workflow.
