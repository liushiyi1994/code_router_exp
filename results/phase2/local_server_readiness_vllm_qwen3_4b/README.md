# RouteCode Phase 2 Results

## Phase 2 Local Server Readiness

Command:

```bash
python experiments/58_local_server_readiness.py --config configs/phase2_local_server_readiness_vllm_qwen3_4b.yaml --output-dir results/phase2/local_server_readiness_vllm_qwen3_4b
```

All configured local models passed the OpenAI-compatible server readiness check.

Outputs:

- `table_local_server_readiness.csv`
- `m9_local_server_readiness_memo.md`

| check_id | status | base_url | model_id | models_endpoint_status | model_listed | completion_status | latency_sec | tokens_input | tokens_output | blocking_reasons | error_type | error_message | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_openai_server:/home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | ready | http://localhost:8001/v1 | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | ok | True | ok | 0.1771 | 15 | 8 |  |  |  | 2026-06-17T16:14:33.364838+00:00 |
