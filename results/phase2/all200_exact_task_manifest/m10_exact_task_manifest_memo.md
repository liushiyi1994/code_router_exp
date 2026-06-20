# Phase 2 Exact Task Manifest

Command:

```bash
python experiments/59_exact_task_manifest.py --config configs/phase2_exact_task_manifest_all200.yaml --output-dir results/phase2/all200_exact_task_manifest
```

This manifest prepares exact-scored math tasks for true local Phase 2 runs. It uses RouteCode split assignments and excludes multiple-choice/code tasks until choices and sandboxed code evaluation are wired.

Selection:

- Datasets requested: `aime, math500`.
- RouteCode split: `all`.
- Max queries: `200`.

Outputs:

- `local_exact_task_manifest.csv`
- `m10_exact_task_manifest_memo.md`

Summary:

| dataset | task_type | routecode_split | rows | unique_queries |
| --- | --- | --- | --- | --- |
| aime | math | test | 14 | 14 |
| aime | math | train | 33 | 33 |
| aime | math | val | 13 | 13 |
| math500 | math | test | 27 | 27 |
| math500 | math | train | 90 | 90 |
| math500 | math | val | 23 | 23 |
