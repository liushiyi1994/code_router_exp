# RouteCode Phase 2 Results

## Phase 2 Local Model Outcomes

Command:

```bash
python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_vllm_qwen3_4b_exact_smoke.yaml
```

Mode: local OpenAI-compatible server. This expects a local vLLM/llama.cpp/SGLang endpoint and makes no GPT/Claude/Gemini API calls.

Outputs:

- `local_model_outcomes.parquet`
- `local_model_raw_outputs.jsonl`
- `local_model_errors.jsonl`
- `local_model_run_metadata.json`
- `m2_local_model_generation_memo.md`

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| aime | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 10 | 0.0000 | 1.7083 | 256.0000 | 0 |
| math500 | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 10 | 0.0000 | 1.6836 | 256.0000 | 0 |
