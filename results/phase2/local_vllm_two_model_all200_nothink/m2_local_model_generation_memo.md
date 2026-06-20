# Phase 2 True Local Model Generation Matrix

Command: `python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_vllm_two_model_all200_nothink.yaml`

Mode: multiple local OpenAI-compatible servers. This is intended for 2--4 vLLM endpoints, typically one base model per server, and makes no GPT/Claude/Gemini API calls.

Outputs:

- `local_model_outcomes.parquet`: exact-scored local outcome rows.
- `local_model_raw_outputs.jsonl`: prompt/output logs for every attempted generation.
- `local_model_errors.jsonl`: error rows, if any.
- `local_model_run_metadata.json`: command, git SHA, checksum, model IDs, generation parameters.

Summary:

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| aime | qwen3_0_6b_vllm | 60 | 0.0000 | 0.0500 | 17.7667 | 0 |
| aime | qwen3_4b_vllm | 60 | 0.0000 | 0.0667 | 7.9000 | 0 |
| math500 | qwen3_0_6b_vllm | 140 | 0.0571 | 0.0440 | 15.4214 | 0 |
| math500 | qwen3_4b_vllm | 140 | 0.1929 | 0.0728 | 8.8786 | 0 |
