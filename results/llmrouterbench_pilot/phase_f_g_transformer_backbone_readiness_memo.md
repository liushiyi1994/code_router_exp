# Phase F/G Transformer Backbone Readiness Memo

Command: `python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_pilot.yaml`

Cache directory: `/home/liush/.cache/huggingface/hub`.
Requested text backbones: `answerdotai/ModernBERT-base, microsoft/deberta-v3-base`.
Runnable size budget: `2.00` GB.

This scan reads local Hugging Face cache metadata only. It performs no downloads, does not import transformer model classes, and does not load model weights.

No transformer embedding baseline was executed because no requested lightweight encoder checkpoint was available in the local cache.

## Summary

| model_id | cache_status | runnable_as_encoder_baseline | reason | architecture | size_gb |
| --- | --- | --- | --- | --- | --- |
| Qwen/Qwen3-4B | cached | False | causal_lm_not_lightweight_encoder | Qwen3ForCausalLM | 7.5073 |
| Tongyi-MAI/Z-Image | cached | False | missing_transformer_config |  | 19.1363 |
| ai-toolkit/flux2_vae | cached | False | missing_transformer_config |  | 0.3131 |
| black-forest-labs/FLUX.1-Kontext-dev | cached | False | missing_transformer_config |  | 8.1181 |
| black-forest-labs/FLUX.2-klein-4B | cached | False | missing_transformer_config |  | 0.0000 |
| black-forest-labs/FLUX.2-klein-base-4B | cached | False | missing_transformer_config |  | 22.1014 |
| answerdotai/ModernBERT-base | missing_local_cache | False | missing_local_cache |  | 0.0000 |
| microsoft/deberta-v3-base | missing_local_cache | False | missing_local_cache |  | 0.0000 |

## Compatibility

- This artifact is a readiness audit, not a routing metric table.
- It does not satisfy the full ModernBERT/DeBERTa predictor-type ablation until a cached encoder is evaluated on the RouteCode split.
- It does move the Research Flow forward by making the missing transformer-backbone dependency explicit and reproducible.

## Next Step

- Cache a small text encoder such as `answerdotai/ModernBERT-base` or `microsoft/deberta-v3-base`, then add a local-files-only embedding extraction script and direct-router rows.
