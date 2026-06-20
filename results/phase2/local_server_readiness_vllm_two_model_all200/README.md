# RouteCode Phase 2 Results

## Phase 2 Local Server Readiness

Command:

```bash
python experiments/58_local_server_readiness.py --config configs/phase2_local_vllm_two_model_all200_nothink.yaml --output-dir results/phase2/local_server_readiness_vllm_two_model_all200
```

At least one configured local model is blocked. This means true local Phase 2 runs should not be started until the local OpenAI-compatible endpoint is available.

Outputs:

- `table_local_server_readiness.csv`
- `m9_local_server_readiness_memo.md`

| check_id | status | base_url | model_id | models_endpoint_status | model_listed | completion_status | latency_sec | tokens_input | tokens_output | blocking_reasons | error_type | error_message | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_openai_server:__first_listed__ | blocked | http://localhost:8001/v1 | __first_listed__ | error | False | skipped | 0.0000 | 6 | 0 | models_endpoint_failed | URLError | <urlopen error timed out> | 2026-06-17T17:14:38.660694+00:00 |
| local_openai_server:__first_listed__ | blocked | http://localhost:8002/v1 | __first_listed__ | error | False | skipped | 0.0000 | 6 | 0 | models_endpoint_failed | URLError | <urlopen error timed out> | 2026-06-17T17:14:43.732000+00:00 |
