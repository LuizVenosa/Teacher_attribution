# Validation QC correction

The first cluster validation run revealed overly broad identity checks and likely
terminal-token flags. Provider names are now checked in first-person identity
clauses instead of anywhere in factual content. This remains an English heuristic,
not a multilingual identity classifier. Unexpected markup still fails QC. Only
recognized EOS/padding and registered end-of-turn tokens at the response boundary
are removed without a flag; internal markup is still flagged. Empty answers and
removed reasoning still fail QC.

Generation records special-token counts, whether an EOS was encountered, and
whether the response parser returned empty content. It never saves raw hidden
reasoning. QC includes counts for each flag. These diagnostics distinguish empty
generation from parser failures without masking either. They do not automatically
repair Ministral's two empty answers.

`configs/research_qc_v2.yaml` preserves the sources, seed and prompt sampling,
uses a new run directory, allows 1024 input tokens, and increases the common
summary output cap from 160 to 320 tokens. Other task caps and word instructions
are unchanged. Length imbalance is still a QC failure: changing its threshold
would not solve a shortcut. The higher cap is a pilot adjustment, not proof that
all summaries will terminate or that every training prompt fits. The student SFT
length budget remains separately enforced.

Existing responses are not relabeled because the removed markup was not retained
and cannot be reconstructed from those artifacts. Response processing version 2
prevents mixing old and new processing during partial generation resume. Use the
new run for all four teachers; do not edit old manifests or copy their response
files into the new run.

On the login node after pulling the changes:

```bash
source jobs/00_setup_env.sh
export CONFIG=configs/research_qc_v2.yaml
export MODEL_CACHE_MODE=runtime
export JOB_CACHE_ROOT=/tmp
python -m teacher_attr --config "$CONFIG" prepare
bash scripts/submit_slurm.sh --job-name=qwen-qc-v2 -- generate --role teachers --model qwen --split distill_val
```

First inspect Qwen's new flag counts and diagnostics before repeating all teachers.
After all four validation jobs finish, run:

```bash
python -m teacher_attr --config "$CONFIG" qc --split distill_val
```

Do not start full training-pool generation until the remaining empty-answer,
truncation and length-control issues have been resolved on validation outputs.
