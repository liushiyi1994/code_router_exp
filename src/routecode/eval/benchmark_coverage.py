from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def scan_llmrouterbench_coverage(results_dir: str | Path) -> pd.DataFrame:
    """Scan raw LLMRouterBench JSON files before canonical schema validation."""

    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"LLMRouterBench results directory does not exist: {root}")
    latest = _latest_files(root.rglob("*.json"))
    rows = []
    for file_path in latest:
        relative_parts = file_path.relative_to(root).parts
        if len(relative_parts) < 4:
            continue
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset_name") or relative_parts[0])
        split = str(payload.get("split") or relative_parts[1])
        model = str(payload.get("model_name") or relative_parts[-2])
        indices = sorted({int(record["index"]) for record in payload.get("records", [])})
        rows.append(
            {
                "dataset": dataset,
                "split": split,
                "model_id": model,
                "record_count": len(indices),
                "record_indices": ";".join(str(index) for index in indices),
                "first_index": indices[0] if indices else "",
                "last_index": indices[-1] if indices else "",
                "file_path": str(file_path),
            }
        )
    if not rows:
        raise ValueError(f"No LLMRouterBench result JSON files found under {root}")
    return pd.DataFrame(rows).sort_values(["dataset", "split", "model_id"]).reset_index(drop=True)


def summarize_dataset_coverage(
    coverage: pd.DataFrame,
    domain_map: dict[str, str] | None = None,
    taxonomy_map: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    domain_map = {str(key): str(value) for key, value in (domain_map or {}).items()}
    taxonomy_map = {str(key): value for key, value in (taxonomy_map or {}).items()}
    rows = []
    for dataset, group in coverage.groupby("dataset", sort=True):
        taxonomy = taxonomy_map.get(str(dataset), {})
        rows.append(
            {
                "dataset": str(dataset),
                "domain": domain_map.get(str(dataset), str(dataset)),
                "task_family": str(taxonomy.get("task_family", "")),
                "task_subtype": str(taxonomy.get("task_subtype", "")),
                "has_taxonomy": bool(taxonomy),
                "split_count": int(group["split"].nunique()),
                "model_count": int(group["model_id"].nunique()),
                "file_count": int(len(group)),
                "max_records_per_file": int(group["record_count"].max()),
                "min_records_per_file": int(group["record_count"].min()),
                "total_model_records": int(group["record_count"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["has_taxonomy", "dataset"], ascending=[False, True]).reset_index(drop=True)


def build_broad_coverage_candidates(coverage: pd.DataFrame, model_counts: list[int]) -> pd.DataFrame:
    records = _records_by_dataset_split_model(coverage)
    model_rank = _rank_models(coverage)
    rows = []
    for raw_count in model_counts:
        model_count = min(max(1, int(raw_count)), len(model_rank))
        models = model_rank[:model_count]
        dataset_splits = []
        dataset_query_counts: dict[str, int] = {}
        complete_query_count = 0
        for dataset, split in sorted({(key[0], key[1]) for key in records}):
            query_indices = _complete_indices(records, dataset, split, models)
            if not query_indices:
                continue
            count = len(query_indices)
            complete_query_count += count
            dataset_splits.append(f"{dataset}:{split}:{count}")
            dataset_query_counts[dataset] = dataset_query_counts.get(dataset, 0) + count
        rows.append(
            {
                "model_count": model_count,
                "dataset_count": len(dataset_query_counts),
                "complete_query_count": complete_query_count,
                "complete_row_count": complete_query_count * model_count,
                "models": ";".join(models),
                "datasets": ";".join(sorted(dataset_query_counts)),
                "dataset_splits": ";".join(dataset_splits),
            }
        )
    return pd.DataFrame(rows).sort_values(["model_count"]).reset_index(drop=True)


def _latest_files(files_iter) -> list[Path]:
    grouped: dict[tuple[str, ...], list[Path]] = {}
    for file_path in files_iter:
        grouped.setdefault(file_path.parent.parts, []).append(file_path)
    return [max(paths, key=_timestamp_key) for paths in grouped.values()]


def _timestamp_key(file_path: Path) -> int:
    match = re.search(r"(\d{8})_(\d{6})\.json$", file_path.name)
    if match:
        return int(match.group(1) + match.group(2))
    return int(file_path.stat().st_mtime)


def _rank_models(coverage: pd.DataFrame) -> list[str]:
    ranked = (
        coverage.groupby("model_id", as_index=False)
        .agg(dataset_count=("dataset", "nunique"), split_count=("split", "nunique"), record_count=("record_count", "sum"))
        .sort_values(["dataset_count", "split_count", "record_count", "model_id"], ascending=[False, False, False, True])
    )
    return [str(model) for model in ranked["model_id"]]


def _records_by_dataset_split_model(coverage: pd.DataFrame) -> dict[tuple[str, str], dict[str, set[int]]]:
    records: dict[tuple[str, str], dict[str, set[int]]] = {}
    for _, row in coverage.iterrows():
        key = (str(row["dataset"]), str(row["split"]))
        indices = _parse_indices(row["record_indices"])
        records.setdefault(key, {})[str(row["model_id"])] = indices
    return records


def _complete_indices(
    records: dict[tuple[str, str], dict[str, set[int]]],
    dataset: str,
    split: str,
    models: list[str],
) -> set[int]:
    by_model = records[(dataset, split)]
    if any(model not in by_model for model in models):
        return set()
    intersection = set(by_model[models[0]])
    for model in models[1:]:
        intersection &= by_model[model]
    return intersection


def _parse_indices(value: Any) -> set[int]:
    if pd.isna(value) or value == "":
        return set()
    return {int(part) for part in str(value).split(";") if part != ""}
