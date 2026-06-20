# Phase 2 Probe Feature Collection

Command:

```bash
python experiments/52_probe_collection.py --outcomes results/phase2/local_vllm_qwen3_4b_all200_nothink/local_model_outcomes.parquet --output-dir results/phase2/local_vllm_qwen3_4b_all200_answer_probe_features
```

This M3 step collects generic cheap-probe observations from local model outputs. It is not evidence that probes close the observability gap.

Outputs:

- `probe_features.parquet`: one probe-observation row per local outcome row.
- `m3_probe_collection_memo.md`: this memo.

Summary:

| probe_type | probe_model_id | rows | unique_queries | mean_agreement | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- |
| local_answer_probe | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 200 | 200 | 1.0000 | 0.0765 | 0 |

Notes:

- `local_answer_probe` rows reuse the M2 local outcome generations as cheap probe observations.
- `logprob_mean` and `entropy_proxy` are null unless the serving backend exposes those values.
- `knn_label_entropy` and `knn_winner_entropy` are null unless aligned train-only state embeddings were supplied.
- These features must be used to update beliefs over latent route states, not to map directly from probe output to final model.
