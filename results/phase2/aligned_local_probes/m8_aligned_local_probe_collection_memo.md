# Phase 2 Aligned Local Probe Collection

Command:

```bash
python experiments/57_aligned_local_probe_collection.py --config configs/llmrouterbench_pilot.yaml --output-dir results/phase2/aligned_local_probes --state-targets results/phase2/aligned_offline/aligned_state_targets.csv
```

This run uses a deterministic dry-run probe client. It validates aligned query selection, logging, schema compatibility, and downstream plumbing; it is not true local-model probe evidence.

Outputs:

- `aligned_local_probe_features.parquet`: schema-compatible probe observations.
- `aligned_local_probe_raw_outputs.jsonl`: raw prompts and model outputs.
- `aligned_local_probe_errors.jsonl`: generation errors, if any.
- `aligned_local_probe_run_metadata.json`: config and run metadata.
- `m8_aligned_local_probe_collection_memo.md`: this memo.

Summary:

| probe_type | probe_model_id | rows | unique_queries | mean_self_confidence | mean_entropy_proxy | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_local_confidence_probe | dry_probe | 50 | 50 | 0.5358 | 0.4642 | 0.0040 | 0 |

Files:

| artifact | path |
| --- | --- |
| features | results/phase2/aligned_local_probes/aligned_local_probe_features.parquet |
| raw_outputs | results/phase2/aligned_local_probes/aligned_local_probe_raw_outputs.jsonl |
| errors | results/phase2/aligned_local_probes/aligned_local_probe_errors.jsonl |
| metadata | results/phase2/aligned_local_probes/aligned_local_probe_run_metadata.json |
