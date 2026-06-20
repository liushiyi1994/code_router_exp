# Phase 2 Local Server Readiness

Command:

```bash
python experiments/58_local_server_readiness.py --config configs/phase2_local_vllm_two_model_all200_nothink.yaml --output-dir results/phase2/local_server_readiness_phase2_local_vllm_two_model_all200_nothink
```

All configured local models passed the OpenAI-compatible server readiness check.

Outputs:

- `table_local_server_readiness.csv`: per-model local OpenAI-compatible endpoint readiness.
- `m9_local_server_readiness_memo.md`: this memo.

| check_id | status | base_url | model_id | models_endpoint_status | model_listed | completion_status | latency_sec | tokens_input | tokens_output | blocking_reasons | error_type | error_message | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_openai_server:qwen3_4b_vllm | ready | http://localhost:8001/v1 | qwen3_4b_vllm | ok | True | ok | 0.2048 | 15 | 8 |  |  |  | 2026-06-17T17:38:25.478274+00:00 |
| local_openai_server:qwen3_0_6b_vllm | ready | http://localhost:8002/v1 | qwen3_0_6b_vllm | ok | True | ok | 0.0252 | 15 | 8 |  |  |  | 2026-06-17T17:38:25.701966+00:00 |
