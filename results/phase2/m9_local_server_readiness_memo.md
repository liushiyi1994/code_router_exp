# Phase 2 Local Server Readiness

Command:

```bash
python experiments/58_local_server_readiness.py --config configs/phase2_local_server_readiness.yaml --output-dir results/phase2
```

At least one configured local model is blocked. This means true local Phase 2 runs should not be started until the local OpenAI-compatible endpoint is available.

Outputs:

- `table_local_server_readiness.csv`: per-model local OpenAI-compatible endpoint readiness.
- `m9_local_server_readiness_memo.md`: this memo.

| check_id | status | base_url | model_id | models_endpoint_status | model_listed | completion_status | latency_sec | tokens_input | tokens_output | blocking_reasons | error_type | error_message | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_openai_server:Qwen3-8B | blocked | http://localhost:8000/v1 | Qwen3-8B | error | False | error | 2.0025 | 6 | 0 | completion_failed | URLError | <urlopen error timed out>; <urlopen error timed out> | 2026-06-17T02:54:30.628123+00:00 |
| local_openai_server:Qwen2.5-Coder-7B-Instruct | blocked | http://localhost:8000/v1 | Qwen2.5-Coder-7B-Instruct | error | False | error | 2.0024 | 6 | 0 | completion_failed | URLError | <urlopen error timed out>; <urlopen error timed out> | 2026-06-17T02:54:30.628123+00:00 |
