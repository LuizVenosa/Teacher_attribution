# Accepted length imbalance in QC v3

The validation pilot had zero invalid responses for all four teachers. Instruction
mean-length ratio was approximately 1.52 and safety approximately 1.66, above the
configured 1.5 threshold. The researcher accepted these differences and chose to
proceed without further prompt tuning. This does not establish length matching.

Record that decision without changing the immutable run configuration:

```bash
python -m teacher_attr --config configs/research_qc_v3.yaml qc --split distill_val --accept-length-imbalance
```

QC retains all original flags, records `strict_passed`, `accepted_flags`,
`blocking_flags`, and `length_exception_requested`, and sets `passed` based on
remaining blockers. The report includes output hashes as before. Distillation
requires a passing report and matching output hashes for both distillation pools.
Empty/invalid responses and excessive truncation cannot be waived with this flag.
Rerunning QC without the flag restores strict enforcement for that split.

Generate the training pool using the same v3 configuration. Its QC must be run
separately; the validation acceptance does not automatically waive training flags.
Inspect the training report before deciding whether to accept its length differences.
Keep length-only baselines and task-specific results in the eventual analysis,
and disclose this validation-driven exception in the paper.
