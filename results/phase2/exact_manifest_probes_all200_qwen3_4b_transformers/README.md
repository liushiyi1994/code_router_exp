# RouteCode Phase 2 Results

## Phase 2 Exact Manifest Probe Collection

Command:

```bash
python experiments/60_exact_manifest_probe_collection.py --config configs/phase2_exact_manifest_probe_all200_qwen3_4b_transformers.yaml --output-dir results/phase2/exact_manifest_probes_all200_qwen3_4b_transformers
```

This run uses a local Hugging Face Transformers backend over the exact-task manifest. It uses local model weights directly and makes no GPT/Claude/Gemini API calls.

Task manifest: `results/phase2/all200_exact_task_manifest/local_exact_task_manifest.csv`.

Outputs:

- `exact_manifest_probe_features.parquet`
- `exact_manifest_probe_raw_outputs.jsonl`
- `exact_manifest_probe_errors.jsonl`
- `exact_manifest_probe_run_metadata.json`
- `m11_exact_manifest_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_self_confidence | mean_entropy_proxy | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_local_confidence_probe | Qwen3-4B-transformers-local | 200 | 200 | 0.9320 | 0.0680 | 0.8714 | 0 |

Files:

| artifact | path |
| --- | --- |
| features | results/phase2/exact_manifest_probes_all200_qwen3_4b_transformers/exact_manifest_probe_features.parquet |
| raw_outputs | results/phase2/exact_manifest_probes_all200_qwen3_4b_transformers/exact_manifest_probe_raw_outputs.jsonl |
| errors | results/phase2/exact_manifest_probes_all200_qwen3_4b_transformers/exact_manifest_probe_errors.jsonl |
| metadata | results/phase2/exact_manifest_probes_all200_qwen3_4b_transformers/exact_manifest_probe_run_metadata.json |
