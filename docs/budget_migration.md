# Raise budgets without regenerating completed teacher responses

The audited QC v3 pools need at most 4283 teacher input tokens and 2719 student
input tokens. Gemma's available SFT examples reach 2937 tokens including answers.
The corrected model configs declare teacher contexts of at least 131072 tokens
and the student context is 4096. These are context limits, not GPU-memory guarantees.

Create a separate run with a 4608 input guard and 4096 SFT guard:

```bash
python -m teacher_attr --config configs/research_qc_v3.yaml migrate-budgets \
  --output configs/research_budget_v1.yaml --input-tokens 4608 --training-tokens 4096
```

The new config must live beside the original to preserve relative source paths.
The destination run is a sibling of the old run named after the new config stem.
No existing destination is overwritten. The command validates prompt hashes,
nested subset hashes and response manifests before copying. It only permits
increases of these two length guards, and retains the same text, sampling, batch
size, output caps, seed and model revisions. Since generation never truncated
inputs, increasing the rejection guard does not alter inputs for accepted rows.

Complete teacher outputs are copied with unchanged responses and original
per-row generation settings. Their new manifests embed the original manifest,
source paths and hashes, explicitly labeling reuse; row fingerprints bind the
new reuse manifest. Original files remain untouched. Incomplete outputs are
listed as skipped, so no newly sampled remainder is mixed with imported rows.
Student weights and old QC decisions are not copied. No model weights are loaded.

The expected reuse is four validation sets plus Gemma's 5000 training responses.
Check the command's `reused` list. Then:

```bash
export CONFIG=configs/research_budget_v1.yaml
python -m teacher_attr --config "$CONFIG" qc --split distill_val --accept-length-imbalance
export JOB_CACHE_ROOT=/tmp
export MODEL_CACHE_MODE=runtime
bash scripts/submit_slurm.sh --job-name=qwen-distill-train -- generate --role teachers --model qwen --split distill_train
```

After Qwen succeeds, submit Ministral and Granite separately, respecting the
one-job queue limit. Gemma does not need regeneration. Run QC and `audit-tokens`
again after all training responses exist: a teacher's token cap does not imply
the same answer fits under the student tokenizer. Test SFT memory use before the
full student training run; increasing its maximum sequence length does not prove
that microbatch four fits the allocated 40 GB MIG GPU. Sequence padding remains
dynamic, so short examples are not padded to 4096 unless the batch needs it.
