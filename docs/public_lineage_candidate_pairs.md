# Public Teacher-Student Lineage Candidate Pool

This document separates **directly runnable pairs** from pairs that are useful scientifically but weaker as attribution labels.

## Runnable Extended Pool

These pairs have a public teacher checkpoint we can generate from and a public student checkpoint with model-card evidence of distillation or lineage.

| Status | Teacher ID | Teacher model | Student ID | Student model | Evidence | Notes |
|---|---|---|---|---|---|---|
| default | `gpt2` | `gpt2` | `distilgpt2` | `distilbert/distilgpt2` | High | DistilGPT2 says it was supervised by GPT-2 124M. |
| default | `qwen15_18b` | `Qwen/Qwen1.5-1.8B` | `miniplm_qwen_200m` | `MiniLLM/MiniPLM-Qwen-200M` | High | MiniPLM-Qwen names Qwen1.5-1.8B as teacher. |
| default | `flan_t5_base` | `google/flan-t5-base` | `lamini_flan_t5_248m` | `MBZUAI/LaMini-Flan-T5-248M` | Medium | Flan-T5-family instruction-distilled lineage. |
| default | `flan_t5_small` | `google/flan-t5-small` | `lamini_flan_t5_77m` | `MBZUAI/LaMini-Flan-T5-77M` | Medium | Flan-T5-family instruction-distilled lineage. |
| candidate | `smollm2_17b_instruct` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `d_smollm2_360m` | `aloobun/d-SmolLM2-360M` | Medium-high | Card says SmolLM2-1.7B teacher and SmolLM2-360M student. |
| candidate | `llama32_3b_instruct` | `meta-llama/Llama-3.2-3B-Instruct` | `lrc_15b_sft` | `JitaiHao/LRC-1.5B-SFT` | High | LRC base was distilled from Llama-3.2-3B-Instruct; SFT version adds UltraChat. Requires Llama access. |
| candidate | `llama31_8b_instruct` | `meta-llama/Llama-3.1-8B-Instruct` | `llama32_3b_distill_glore` | `SwashBuckler001/Llama-3.2-3B-distill-GLoRE` | Medium-high | Third-party card says distilled from Llama-3.1-8B-Instruct. Requires Llama access. |
| heavy candidate | `qwen3_30b_a3b_thinking` | `Qwen/Qwen3-30B-A3B-Thinking-2507` | `qwen3_06b_distilled_30b_a3b` | `reaperdoesntknow/Qwen3-0.6B-Distilled-30B-A3B` | Medium | Student card names Qwen3-30B-A3B-Thinking. Teacher is heavy; test storage/runtime before full run. |

Config: `configs/public_lineage_models_extended.yaml`.

## Reference Links

- Qwen3 technical report: https://huggingface.co/papers/2505.09388
- Qwen2.5 technical report: https://huggingface.co/papers/2412.15115
- SmolLM2 teacher card: https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
- d-SmolLM2 student card: https://huggingface.co/aloobun/d-SmolLM2-360M
- LRC-1.5B-SFT card: https://huggingface.co/JitaiHao/LRC-1.5B-SFT
- Llama 3.2 3B GLoRE distilled card: https://huggingface.co/SwashBuckler001/Llama-3.2-3B-distill-GLoRE
- Qwen3 0.6B distilled card: https://huggingface.co/reaperdoesntknow/Qwen3-0.6B-Distilled-30B-A3B

## Qwen Notes

Qwen has several relevant but distinct cases:

1. **Qwen2.5 official size variants** are not clean teacher-student pairs by themselves. The Qwen2.5 report describes a family of sizes, pretraining, SFT, and RL, but not a simple `Qwen2.5-7B -> Qwen2.5-0.5B` lineage.
2. **DistilQwen2.5** models from `alibaba-pai` are genuinely distilled lightweight Qwen2.5 models, but their cards/paper describe larger or multi-agent teachers and model fusion. Because the exact teacher outputs are not a single public checkpoint we can generate from, these are weaker for same-prompt attribution labels.
3. **Qwen3 technical material** describes strong-to-weak distillation for lightweight Qwen3 models, with larger Qwen3 teachers such as Qwen3-32B or Qwen3-235B-A22B. This supports using Qwen3 small models as distilled candidates, but the exact single teacher per checkpoint is often less clean than the public model-card pairs above.
4. Third-party Qwen3 distilled checkpoints that explicitly name a teacher, such as `Qwen3-30B-A3B-Thinking -> Qwen3-0.6B-Distilled-30B-A3B`, are usable as experimental public-lineage pairs if the teacher fits cluster resources.

## Recommended Selection Order

Start with a 5-way or 6-way setup before using every candidate:

1. Keep the four current defaults.
2. Add `llama32_3b_instruct -> lrc_15b_sft` to make chance accuracy 20%, matching WTYT's 5-way setup.
3. Add `smollm2_17b_instruct -> d_smollm2_360m` if you want one more non-Qwen, non-Flan family.
4. Add `llama31_8b_instruct -> llama32_3b_distill_glore` after verifying storage and runtime.
5. Add the Qwen3-30B teacher only as a heavy ablation.

## Running The Extended Pool

Generate everything in the extended config:

```bash
MODELS_CONFIG=configs/public_lineage_models_extended.yaml \
sbatch jobs/01_generate_public_lineage_outputs.sbatch
```

Generate a selected subset:

```bash
MODELS_CONFIG=configs/public_lineage_models_extended.yaml \
TEACHER_LIST="gpt2 qwen15_18b flan_t5_base flan_t5_small llama32_3b_instruct smollm2_17b_instruct" \
STUDENT_LIST="distilgpt2:miniplm_qwen_200m:lamini_flan_t5_248m:lamini_flan_t5_77m:lrc_15b_sft:d_smollm2_360m" \
sbatch jobs/01_generate_public_lineage_outputs.sbatch
```

Then build pairs with the same model config:

```bash
MODELS_CONFIG=configs/public_lineage_models_extended.yaml \
sbatch jobs/04_build_public_lineage_attribution.sbatch
```