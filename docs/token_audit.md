# Full-pool token audit

Run `audit-tokens` before selecting a larger input budget. It reads all five
prepared pools with each pinned teacher tokenizer and the student base tokenizer.
It loads configuration/tokenizer files only, never weights, and makes no changes
to prompt pools or generation manifests. Existing Gemma outputs remain untouched.

```bash
source jobs/00_setup_env.sh
export CONFIG=configs/research_qc_v3.yaml
export MODEL_CACHE_MODE=runtime
export JOB_CACHE_ROOT=/tmp
REQUIRE_GPU=0 bash scripts/submit_slurm.sh --gres=none --mem=8G --time=00:30:00 --job-name=token-audit -- audit-tokens
```

This runs on a compute node so the launcher handles the inherited CPU-time limit.
No GPU is requested. Read `logs/teacher-attr_JOB_ID.out` for the JSON report; progress
and tokenizer warnings go to `.err`. The report includes the current budget,
maximum, p99, number of overlong prompts and the five longest prompt IDs per pool.
It also compares available teacher answers under the student tokenizer, counting
the exact SFT prefix, target, and EOS against the configured training limit.
Missing or incomplete outputs are explicit; an audit without all teacher answers
cannot guarantee that every eventual SFT example fits. It does not establish GPU
memory fit.

The audit intentionally matches the current generation text/tokenization path,
including the Ministral chat-template round-trip. Changing that tokenization is
a separate protocol change; a direct-tokenization audit should not be mistaken
for a measurement of the existing generation implementation.

Inspect results before changing any immutable configuration. A larger input limit
must fit every model's context together with the reserved output allowance. Student
SFT has a separate combined prompt/answer limit. Do not truncate separately for each
teacher or overwrite old manifests to bypass provenance checks.
