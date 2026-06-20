# RouteCode Phase 2 Results

## Phase 2 Exact Manifest Probe Collection

Command:

```bash
python experiments/60_exact_manifest_probe_collection.py --config configs/phase2_exact_manifest_probe_dryrun.yaml --output-dir results/phase2/exact_manifest_probes
```

This run uses a deterministic dry-run probe client over the exact-task manifest. It validates manifest-backed probe logging and M4 plumbing; it is not true local-model probe evidence.

Task manifest: `results/phase2/local_exact_task_manifest.csv`.

Outputs:

- `exact_manifest_probe_features.parquet`
- `exact_manifest_probe_raw_outputs.jsonl`
- `exact_manifest_probe_errors.jsonl`
- `exact_manifest_probe_run_metadata.json`
- `m11_exact_manifest_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_self_confidence | mean_entropy_proxy | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_local_confidence_probe | dry_probe | 118 | 118 | 0.4864 | 0.5136 | 0.0040 | 0 |

Files:

| artifact | path |
| --- | --- |
| features | results/phase2/exact_manifest_probes/exact_manifest_probe_features.parquet |
| raw_outputs | results/phase2/exact_manifest_probes/exact_manifest_probe_raw_outputs.jsonl |
| errors | results/phase2/exact_manifest_probes/exact_manifest_probe_errors.jsonl |
| metadata | results/phase2/exact_manifest_probes/exact_manifest_probe_run_metadata.json |
