# RouteCode Phase 2 Results

## Phase 2 Local Model Outcomes

Command:

```bash
python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_qwen3_4b_transformers_smoke.yaml
```

Mode: local Hugging Face Transformers backend. This uses local model weights directly and makes no GPT/Claude/Gemini API calls.

Outputs:

- `local_model_outcomes.parquet`
- `local_model_raw_outputs.jsonl`
- `local_model_errors.jsonl`
- `local_model_run_metadata.json`
- `m2_local_model_generation_memo.md`

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k_smoke | Qwen3-4B-transformers-local | 2 | 1.0000 | 2.0379 | 60.0000 | 0 |
| mmlu_smoke | Qwen3-4B-transformers-local | 2 | 1.0000 | 0.1364 | 4.5000 | 0 |

## Phase 2 Probe Features

Command:

```bash
python experiments/52_probe_collection.py --outcomes results/phase2/local_qwen3_4b_transformers_smoke/local_model_outcomes.parquet --output-dir results/phase2/local_qwen3_4b_transformers_smoke
```

This writes `probe_features.parquet` from local cheap-probe outputs without external API calls. These rows validate the probe-feature schema and logging path; probe usefulness is evaluated separately.

Outputs:

- `probe_features.parquet`
- `m3_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_agreement | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- |
| local_answer_probe | Qwen3-4B-transformers-local | 4 | 4 | 1.0000 | 1.1194 | 0 |
