# Phase F/G Transformer Backbone Readiness Memo

Command: `python experiments/13_transformer_backbone_readiness.py --config configs/llmrouterbench_pilot.yaml`

Cache directory: `/home/liush/.cache/huggingface/hub`.
Requested text backbones: `sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5, intfloat/e5-small-v2, answerdotai/ModernBERT-base, microsoft/deberta-v3-base`.
Runnable size budget: `2.00` GB.

This scan reads local Hugging Face cache metadata only. It performs no downloads, does not import transformer model classes, and does not load model weights.

At least one cached lightweight encoder candidate is available. Transformer-embedding direct-router rows should be checked in `table_transformer_embedding_router.csv`; extraction must use `local_files_only=True` and the same query-id split.

## Summary

| model_id | cache_status | runnable_as_encoder_baseline | reason | architecture | size_gb |
| --- | --- | --- | --- | --- | --- |
| BAAI/bge-small-en-v1.5 | cached | True | cached_encoder_candidate | BertModel | 0.3736 |
| Qwen/Qwen3-4B | cached | False | causal_lm_not_lightweight_encoder | Qwen3ForCausalLM | 7.5073 |
| Tongyi-MAI/Z-Image | cached | False | missing_transformer_config |  | 19.1363 |
| ai-toolkit/flux2_vae | cached | False | missing_transformer_config |  | 0.3131 |
| answerdotai/ModernBERT-base | cached | True | cached_encoder_candidate | ModernBertForMaskedLM | 0.5595 |
| black-forest-labs/FLUX.1-Kontext-dev | cached | False | missing_transformer_config |  | 8.1181 |
| black-forest-labs/FLUX.2-klein-4B | cached | False | missing_transformer_config |  | 0.0000 |
| black-forest-labs/FLUX.2-klein-base-4B | cached | False | missing_transformer_config |  | 22.1014 |
| intfloat/e5-small-v2 | cached | True | cached_encoder_candidate | BertModel | 0.7480 |
| microsoft/deberta-v3-base | cached | True | cached_encoder_candidate |  | 0.6936 |
| sentence-transformers/all-MiniLM-L6-v2 | cached | True | cached_encoder_candidate | BertModel | 0.0853 |

## Compatibility

- This artifact is a readiness audit, not a routing metric table.
- It does not satisfy the full requested predictor-type ablation until each claim-critical cached encoder is evaluated on the RouteCode split.
- It moves the Research Flow forward by making available and missing transformer-backbone dependencies explicit and reproducible.

## Next Step

- Evaluate any additional claim-critical encoder backbones under the local-files-only transformer embedding router before making predictor-type claims.
