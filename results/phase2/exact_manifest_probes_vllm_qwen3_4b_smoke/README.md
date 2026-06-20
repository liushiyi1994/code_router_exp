# RouteCode Phase 2 Results

## Phase 2 Exact Manifest Probe Collection

Command:

```bash
python experiments/60_exact_manifest_probe_collection.py --config configs/phase2_exact_manifest_probe_vllm_qwen3_4b_smoke.yaml --output-dir results/phase2/exact_manifest_probes_vllm_qwen3_4b_smoke
```

This run uses an OpenAI-compatible local serving endpoint over the exact-task manifest. It is true local probe evidence only if the metadata records the local model IDs and endpoint used.

Task manifest: `results/phase2/all200_exact_task_manifest/local_exact_task_manifest.csv`.

Outputs:

- `exact_manifest_probe_features.parquet`
- `exact_manifest_probe_raw_outputs.jsonl`
- `exact_manifest_probe_errors.jsonl`
- `exact_manifest_probe_run_metadata.json`
- `m11_exact_manifest_probe_collection_memo.md`

| probe_type | probe_model_id | rows | unique_queries | mean_self_confidence | mean_entropy_proxy | mean_probe_cost_proxy | errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| aligned_local_confidence_probe | /home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c | 20 | 20 | 0.8937 | 0.1063 | 0.2408 | 0 |

Files:

| artifact | path |
| --- | --- |
| features | results/phase2/exact_manifest_probes_vllm_qwen3_4b_smoke/exact_manifest_probe_features.parquet |
| raw_outputs | results/phase2/exact_manifest_probes_vllm_qwen3_4b_smoke/exact_manifest_probe_raw_outputs.jsonl |
| errors | results/phase2/exact_manifest_probes_vllm_qwen3_4b_smoke/exact_manifest_probe_errors.jsonl |
| metadata | results/phase2/exact_manifest_probes_vllm_qwen3_4b_smoke/exact_manifest_probe_run_metadata.json |
