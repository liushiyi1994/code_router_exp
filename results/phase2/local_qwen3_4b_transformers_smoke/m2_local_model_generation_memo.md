# Phase 2 True Local Model Generation Matrix

Command: `python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_qwen3_4b_transformers_smoke.yaml`

Mode: local Hugging Face Transformers backend. This uses local model weights directly and makes no GPT/Claude/Gemini API calls.

Outputs:

- `local_model_outcomes.parquet`: exact-scored local outcome rows.
- `local_model_raw_outputs.jsonl`: prompt/output logs for every attempted generation.
- `local_model_errors.jsonl`: error rows, if any.
- `local_model_run_metadata.json`: command, git SHA, checksum, model IDs, generation parameters.

Summary:

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k_smoke | Qwen3-4B-transformers-local | 2 | 1.0000 | 2.0379 | 60.0000 | 0 |
| mmlu_smoke | Qwen3-4B-transformers-local | 2 | 1.0000 | 0.1364 | 4.5000 | 0 |
