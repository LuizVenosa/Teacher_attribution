# Learned teacher attribution

Controlled experiments that train the **same student base** on different teachers' responses, then identify the teacher from unseen student outputs.

The primary configuration is `configs/research.yaml`. It uses four teacher families, full OLMo-2-1B fine-tuning, ModernBERT contrastive attribution, lexical/generic/PoS comparisons, and mean/Deep Sets fingerprints. The implementation follows the supplied specification's execution order. It does not contain new full-scale research results.

## Start here

```bash
uv sync --all-extras --locked
# Activate .venv, or prefix commands with uv run.
python -m spacy download en_core_web_sm
teacher-attr preflight
teacher-attr pin --output configs/research.pinned.yaml
teacher-attr --config configs/research.pinned.yaml prepare
teacher-attr --config configs/research.pinned.yaml plan
```

Use a CUDA-enabled PyTorch installation on the A100 host. Authenticate to Hugging Face and obtain model access where required before preflight. Preflight checks configuration/tokenizer access without downloading weights. Pinning resolves model and source revisions into a new YAML file.

See **[the execution guide](docs/implementation.md)** for the sequential commands, controls, optional experiments, and validation limits. All paths inside YAML resolve relative to that YAML file. Use a new run directory when changing its settings.

## Prompt pools

| Source | Distillation training |
|---|---:|
| Tülu 3 first user turns | 2,000 |
| CNN/DailyMail articles | 750 |
| PubMed articles | 750 |
| CommonsenseQA questions | 500 |
| OpenBookQA questions | 500 |
| Anthropic HH harmlessness prompts | 500 |
| **Total** | **5,000** |

Separate pools contain 300 distillation-validation, 1,200 attribution-training, 600 attribution-validation, and 1,200 attribution-test prompts. These are configured targets. Existing assistant answers are excluded from distillation inputs; QA answers are retained only as diagnostic metadata. Deterministic sampling, text deduplication, source grouping, hashes, and cross-pool checks protect the separation. The bounded source sampling policy is recorded, rather than presented as a uniform whole-corpus sample.

## Repository

- `src/teacher_attr/`: one package for data, generation, distillation, baselines, attribution, and evaluation.
- `configs/research.yaml`: controlled study; `configs/experiment.yaml`: prior public pilot.
- `scripts/`: thin compatibility entry points requested by the specification, calling the same CLI implementation.
- `jobs/run.sbatch`: one single-GPU SLURM wrapper.
- `runs/`: immutable run configurations, prompt pools, cached responses, student checkpoints, metrics, and figures.
- `tests/`: offline tiny-model integration tests and data/provenance checks; the research integration also needs the configured small English PoS model.
- `report/` and `results/`: preserved historical artifacts, not outputs of this experiment.

```bash
python -m pytest
ruff check src tests scripts
ruff format --check src tests scripts
```

The previous workflow remains documented in [the public pilot guide](docs/public_pilot.md). Run it with its explicit configuration; its models and results must not be mixed with the controlled study.
