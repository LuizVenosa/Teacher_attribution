# Teacher-only comparison with Who Taught You That?

`evaluate-paper` implements the teacher-to-student transfer setting described in
[Wadhwa et al., ACL Findings 2025, sections 3–3.2](https://aclanthology.org/2025.findings-acl.173/).
It is a paper-inspired comparison on our controlled data, **not an exact reproduction
of their published numbers**. Our candidates, candidate count (four versus five),
student training mixture, instruction/safety sources, and output controls differ.
The current learned encoder used labeled student training data, so comparing its
accuracy with teacher-only methods does not isolate the benefit of representation
learning. A learned encoder trained only on teachers would be a separate experiment.

## Protocol

- Read the existing verified attribution pairs, retaining their recorded exclusions.
- Deduplicate teacher references by prompt: each teacher contributes one response,
  regardless of the number of student checkpoints in the pairs.
- Fit word 1–4 gram count features and logistic regression on teacher training outputs.
- Fit fine-grained PoS four-tag patterns within sentences, take the
  top 50 patterns across all teacher training responses, and use binary indicators
  with logistic regression. Extraction uses spaCy's English tagger, so multilingual
  instruction responses can be tagged less reliably. This does not claim bit-for-bit
  equivalence with the original diversity-package extraction.
- Select regularization using teacher validation outputs only. Student training and
  validation responses/labels are never inputs to those fits or selections.
- Default to separate fits per source dataset, with `--scope pooled` available as a
  separately named experiment. The template vocabulary uses training data only;
  the paper describes extraction involving test instances. This avoids feature-selection
  leakage and is another disclosed difference.
- Score held-out student test responses. BoW uses teacher-training unigram vocabulary
  and cosine against each candidate's response to the same test prompt.
- BERTScore uses F1 matching, configured model and layer count (currently RoBERTa-base,
  layer 10), no IDF and no baseline rescaling. Its resolved Hub revision is frozen
  before scoring. Tokenizer limits are audited and overflow fails without truncation.
- Candidate likelihood scores each student response under each teacher, response-only
  with BOS (EOS fallback), no chat/prompt conditioning and no appended EOS. Only response
  tokens contribute to the score. Select the highest negative mean token NLL. The
  separately reported per-UTF-8-byte NLL is a sensitivity analysis, not a substitute
  for the paper's likelihood calculation; different tokenizers complicate comparison.
- Report single-response accuracy, macro F1, per-dataset/per-student results, prompt
  bootstrap intervals, and the same mean-score support-set aggregation as our primary
  evaluation. Ties use the frozen candidate order and their counts are disclosed.

All stored scores and source pairs are checked before reuse. The paper specification
is frozen on first invocation. Methods may be added later under that same name;
changed settings require a new name. Use one job at a time per evaluation directory.
Likelihood progress saves every 25 responses and on ordinary Python errors. Killed
jobs can lose the last unsaved block; rerunning the same command recomputes it.

## Cluster sequence

Push the implementation from Windows and pull it on the cluster first. All commands
below run from the repository root. They reuse existing outputs and student weights.

```bash
source jobs/00_setup_env.sh
export CONFIG=configs/research_budget_v1.yaml
export JOB_CACHE_ROOT=/tmp
export MODEL_CACHE_MODE=runtime

# First job: teacher-trained classifiers and plain BoW; no GPU needed.
REQUIRE_GPU=0 bash scripts/submit_slurm.sh \
  --gres=none --mem=16G --time=01:00:00 --job-name=paper-controls \
  -- evaluate-paper --name paper_v1 --methods word_ngram pos bow

# Next job: BERTScore. Existing methods are collected without recomputing them.
bash scripts/submit_slurm.sh --job-name=paper-bertscore \
  -- evaluate-paper --name paper_v1 --methods bertscore

# Run individually, sequentially; each job loads only its selected candidate.
bash scripts/submit_slurm.sh --job-name=paper-likelihood-qwen \
  -- evaluate-paper --name paper_v1 --methods likelihood --teacher qwen
bash scripts/submit_slurm.sh --job-name=paper-likelihood-ministral \
  -- evaluate-paper --name paper_v1 --methods likelihood --teacher ministral
bash scripts/submit_slurm.sh --job-name=paper-likelihood-gemma \
  -- evaluate-paper --name paper_v1 --methods likelihood --teacher gemma
bash scripts/submit_slurm.sh --job-name=paper-likelihood-granite \
  -- evaluate-paper --name paper_v1 --methods likelihood --teacher granite
```

After all jobs finish, collect scores and a paired, explicitly cross-protocol
comparison with the existing primary result. No models are loaded by collection.

```bash
python -m teacher_attr --config "$CONFIG" evaluate-paper --name paper_v1 --collect \
  --primary runs/research_budget_v1/evaluations/primary_v2
cat runs/research_budget_v1/evaluations/paper_v1/summary.md
```

`metrics.json` explicitly lists missing methods and likelihood candidates. The combined
likelihood result appears only after all candidates complete. `summary.csv` includes
overall and per-source results for publication tables; `predictions.jsonl` preserves
aligned method scores. Collection may be repeated as methods finish.

The SLURM wrapper now removes only its own `mktemp` model cache on ordinary exit,
including Python failures. It does not remove pre-existing caches from other jobs.
SIGKILL/node failures can still leave temporary directories for later cleanup.

## Validation limits

CPU tests cover teacher-only training/selection, reference deduplication, sentence
boundaries, binary templates, per-dataset fitting, source/specification guards,
cached-score checks, resumable completed likelihood and paired comparison alignment.
The full teacher-model and BERTScore workloads still require cluster verification.
Do not treat implementation or passing local tests as new experimental results.
