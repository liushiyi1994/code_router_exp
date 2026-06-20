# Phase 2 True Local Model Generation Matrix

Command: `python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_vllm_qwen3_4b_exact_smoke_nothink.yaml`

Mode: local OpenAI-compatible server. This expects a local vLLM/llama.cpp/SGLang endpoint and makes no GPT/Claude/Gemini API calls.

Outputs:

- `local_model_outcomes.parquet`: exact-scored local outcome rows.
- `local_model_raw_outputs.jsonl`: prompt/output logs for every attempted generation.
- `local_model_errors.jsonl`: error rows, if any.
- `local_model_run_metadata.json`: command, git SHA, checksum, model IDs, generation parameters.

Summary:

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| aime | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 10 | 0.0000 | 0.0823 | 7.6000 | 0 |
| math500 | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 10 | 0.2000 | 0.0804 | 10.5000 | 0 |
