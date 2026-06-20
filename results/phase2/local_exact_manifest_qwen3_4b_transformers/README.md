# RouteCode Phase 2 Results

## Phase 2 Local Model Outcomes

Command:

```bash
python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_exact_manifest_qwen3_4b_transformers.yaml
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
| aime | Qwen3-4B-transformers-local | 14 | 0.0714 | 3.4927 | 128.0000 | 0 |
| math500 | Qwen3-4B-transformers-local | 104 | 0.0673 | 3.3783 | 126.5865 | 0 |
