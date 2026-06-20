# RouteCode Phase 2 Results

## Phase 2 Probe Features

Command:

```bash
python experiments/52_probe_collection.py --outcomes results/phase2/local_vllm_qwen3_4b_all200_nothink/local_model_outcomes.parquet --output-dir results/phase2/local_vllm_qwen3_4b_all200_answer_probe_features
```

This writes `probe_features.parquet` from local cheap-probe outputs without external API calls. These rows validate the probe-feature schema and logging path; probe usefulness is evaluated separately.

Outputs:

- `probe_features.parquet`
- `m3_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_agreement | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- |
| local_answer_probe | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 200 | 200 | 1.0000 | 0.0765 | 0 |
