# RouteCode Phase 2 Results

## Phase 2 Local Model Outcomes

Command:

```bash
python experiments/51_true_model_generation_matrix.py --config configs/phase2_local_exact_manifest_dryrun.yaml
```

Mode: `dry_run`. This validates local-eval logging, parsing, scoring, and parquet output; it is not true model-performance evidence.

Outputs:

- `local_model_outcomes.parquet`
- `local_model_raw_outputs.jsonl`
- `local_model_errors.jsonl`
- `local_model_run_metadata.json`
- `m2_local_model_generation_memo.md`

| dataset | model_id | rows | mean_quality | mean_latency_sec | mean_tokens_output | errors |
| --- | --- | --- | --- | --- | --- | --- |
| aime | dry_run_model | 14 | 1.0000 | 0.0000 | 3.0000 | 0 |
| math500 | dry_run_model | 104 | 1.0000 | 0.0000 | 3.2404 | 0 |
