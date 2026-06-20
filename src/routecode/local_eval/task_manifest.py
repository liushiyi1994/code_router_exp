from __future__ import annotations

import json
from typing import Any

import pandas as pd

from routecode.local_eval.generation_runner import LocalEvalTask


EXACT_TASK_MANIFEST_COLUMNS = [
    "query_id",
    "query_text",
    "dataset",
    "domain",
    "source_split",
    "routecode_split",
    "task_type",
    "gold_answer",
    "choices_json",
    "metadata_json",
]

MATH_DATASETS = {
    "aime",
    "math500",
    "livemathbench",
    "mathbench",
}


def build_exact_task_manifest(
    outcomes: pd.DataFrame,
    *,
    datasets: list[str],
    split: str = "test",
    max_queries: int = 200,
) -> pd.DataFrame:
    """Build local exact-scored tasks from split-assigned RouteCode outcomes.

    This intentionally starts with math-style datasets. Multiple-choice support
    needs trustworthy choices, and code support needs sandboxed evaluation.
    """

    _require_columns(outcomes, ["query_id", "query_text", "dataset", "domain", "metadata_json"])
    selected = outcomes.copy()
    if "split" in selected.columns and str(split).lower() not in {"all", "*", ""}:
        selected = selected[selected["split"].astype(str).eq(str(split))]
    allowed = [str(dataset) for dataset in datasets]
    selected = selected[selected["dataset"].astype(str).isin(allowed)]
    selected = selected.drop_duplicates("query_id", keep="first")

    rows: list[dict[str, Any]] = []
    for _, row in selected.iterrows():
        dataset = str(row["dataset"])
        task_type = task_type_for_dataset(dataset)
        if task_type is None:
            continue
        gold_answer = _ground_truth(row.get("metadata_json"))
        if not gold_answer:
            continue
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "query_text": str(row["query_text"]),
                "dataset": dataset,
                "domain": str(row.get("domain", dataset)),
                "source_split": str(row.get("source_split", "")),
                "routecode_split": str(row.get("split", split)),
                "task_type": task_type,
                "gold_answer": gold_answer,
                "choices_json": "[]",
                "metadata_json": str(row.get("metadata_json", "{}")),
            }
        )
    ordered = _round_robin_rows(rows, allowed)
    if max_queries > 0:
        ordered = ordered[: int(max_queries)]
    return pd.DataFrame(ordered, columns=EXACT_TASK_MANIFEST_COLUMNS)


def tasks_from_manifest(manifest: pd.DataFrame) -> list[LocalEvalTask]:
    _require_columns(manifest, EXACT_TASK_MANIFEST_COLUMNS)
    tasks: list[LocalEvalTask] = []
    for _, row in manifest.iterrows():
        tasks.append(
            LocalEvalTask(
                query_id=str(row["query_id"]),
                query_text=str(row["query_text"]),
                dataset=str(row["dataset"]),
                domain=str(row["domain"]),
                task_type=str(row["task_type"]),
                gold_answer=str(row["gold_answer"]),
                choices=json.loads(str(row["choices_json"] or "[]")),
            )
        )
    return tasks


def task_type_for_dataset(dataset: str) -> str | None:
    normalized = str(dataset).lower()
    if normalized in MATH_DATASETS:
        return "math"
    return None


def _round_robin_rows(rows: list[dict[str, Any]], dataset_order: list[str]) -> list[dict[str, Any]]:
    grouped = {dataset: [row for row in rows if row["dataset"] == dataset] for dataset in dataset_order}
    ordered: list[dict[str, Any]] = []
    offset = 0
    while True:
        added = False
        for dataset in dataset_order:
            bucket = grouped.get(dataset, [])
            if offset < len(bucket):
                ordered.append(bucket[offset])
                added = True
        if not added:
            return ordered
        offset += 1


def _ground_truth(metadata_json: Any) -> str:
    try:
        payload = json.loads(str(metadata_json or "{}"))
    except json.JSONDecodeError:
        return ""
    value = payload.get("ground_truth", "")
    if value is None:
        return ""
    return str(value).strip()


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"exact task manifest missing required columns: {missing}")
