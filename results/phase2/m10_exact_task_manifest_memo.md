# Phase 2 Exact Task Manifest

Command:

```bash
python experiments/59_exact_task_manifest.py --config configs/phase2_exact_task_manifest.yaml --output-dir results/phase2
```

This manifest prepares exact-scored math tasks for true local Phase 2 runs. It uses RouteCode split assignments and excludes multiple-choice/code tasks until choices and sandboxed code evaluation are wired.

Selection:

- Datasets requested: `aime, math500`.
- RouteCode split: `test`.
- Max queries: `200`.

Outputs:

- `local_exact_task_manifest.csv`
- `m10_exact_task_manifest_memo.md`

Summary:

| dataset | task_type | routecode_split | rows | unique_queries |
| --- | --- | --- | --- | --- |
| aime | math | test | 14 | 14 |
| math500 | math | test | 104 | 104 |
