# RouteCode Phase 2 Results

## Phase 2 Local Server Readiness

Command:

```bash
python experiments/58_local_server_readiness.py --config configs/phase2_local_vllm_qwen3_4b_exact_smoke_nothink.yaml --output-dir results/phase2/local_server_readiness_live_qwen3_4b_port8001
```

All configured local models passed the OpenAI-compatible server readiness check.

Outputs:

- `table_local_server_readiness.csv`
- `m9_local_server_readiness_memo.md`

| check_id | status | base_url | model_id | models_endpoint_status | model_listed | completion_status | latency_sec | tokens_input | tokens_output | blocking_reasons | error_type | error_message | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_openai_server:qwen3_4b_vllm | ready | http://localhost:8001/v1 | qwen3_4b_vllm | ok | True | ok | 0.3619 | 19 | 3 |  |  |  | 2026-06-17T17:36:13.858292+00:00 |
