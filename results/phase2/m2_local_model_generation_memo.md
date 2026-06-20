# Phase 2 True Local Model Generation Matrix

Command: `python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_smoke.yaml`

Mode: `dry_run`. This validates local-eval logging, parsing, scoring, and parquet output; it is not true model-performance evidence.

Outputs:

- `local_model_outcomes.parquet`: exact-scored local outcome rows.
- `local_model_raw_outputs.jsonl`: prompt/output logs for every attempted generation.
- `local_model_errors.jsonl`: error rows, if any.
- `local_model_run_metadata.json`: command, git SHA, checksum, model IDs, generation parameters.

Summary:

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| gsm8k_smoke | dry_run_model | 10 | 1.0000 | 0.0000 | 3.0000 | 0 |
| mmlu_smoke | dry_run_model | 10 | 1.0000 | 0.0000 | 1.0000 | 0 |
