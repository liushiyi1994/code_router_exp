from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from routecode.data.schema import validate_outcomes


def load_llmrouterbench_outcomes(
    results_dir: str | Path,
    datasets: list[str] | None = None,
    models: list[str] | None = None,
    splits: list[str] | None = None,
    drop_incomplete_queries: bool = True,
) -> pd.DataFrame:
    """Load official LLMRouterBench result JSON files into RouteCode schema."""
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"LLMRouterBench results directory does not exist: {root}. "
            "Download/extract bench-release.tar.gz so it contains results/bench first."
        )

    dataset_filter = set(datasets) if datasets else None
    model_filter = set(models) if models else None
    split_filter = set(splits) if splits else None

    files = _latest_files(_candidate_files(root, dataset_filter, model_filter))
    rows: list[dict[str, Any]] = []
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset_name") or _path_part(file_path, root, 0))
        split = str(payload.get("split") or _path_part(file_path, root, 1))
        model = str(payload.get("model_name") or _path_part(file_path, root, 2))
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        if model_filter is not None and model not in model_filter:
            continue
        if split_filter is not None and split not in split_filter:
            continue

        for record in payload.get("records", []):
            record_index = int(record["index"])
            query_id = f"{dataset}:{split}:{record_index}"
            prompt_tokens = int(record.get("prompt_tokens") or 0)
            completion_tokens = int(record.get("completion_tokens") or 0)
            cost_total = float(record.get("cost") or 0.0)
            token_total = prompt_tokens + completion_tokens
            if token_total > 0:
                cost_input = cost_total * prompt_tokens / token_total
                cost_output = cost_total - cost_input
            else:
                cost_input = 0.0
                cost_output = cost_total
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": _normalize_text(record.get("origin_query") or record.get("prompt") or ""),
                    "dataset": dataset,
                    "domain": dataset,
                    "source_split": split,
                    "record_index": record_index,
                    "model_id": model,
                    "quality": float(record.get("score") or 0.0),
                    "cost_input": float(cost_input),
                    "cost_output": float(cost_output),
                    "cost_total": cost_total,
                    "latency": None,
                    "tokens_input": prompt_tokens,
                    "tokens_output": completion_tokens,
                    "judge": "llmrouterbench_score",
                    "metadata_json": json.dumps(
                        {
                            "source": "LLMRouterBench",
                            "split": split,
                            "prediction": _safe_metadata_value(record.get("prediction")),
                            "ground_truth": _safe_metadata_value(record.get("ground_truth")),
                        },
                        sort_keys=True,
                    ),
                }
            )

    if not rows:
        raise ValueError(f"No LLMRouterBench records found under {root} with the configured filters")

    outcomes = pd.DataFrame(rows)
    if drop_incomplete_queries:
        expected_models = outcomes["model_id"].nunique()
        counts = outcomes.groupby("query_id")["model_id"].nunique()
        complete_ids = counts[counts == expected_models].index
        outcomes = outcomes[outcomes["query_id"].isin(complete_ids)].copy()
        if outcomes.empty:
            raise ValueError("All queries were incomplete after filtering by model coverage")
    return validate_outcomes(outcomes)


def _candidate_files(root: Path, dataset_filter: set[str] | None, model_filter: set[str] | None) -> list[Path]:
    candidates = []
    for file_path in root.rglob("*.json"):
        relative_parts = file_path.relative_to(root).parts
        if len(relative_parts) < 3:
            continue
        dataset = relative_parts[0]
        model = relative_parts[-2]
        if dataset_filter is not None and dataset not in dataset_filter:
            continue
        if model_filter is not None and model not in model_filter:
            continue
        candidates.append(file_path)
    return candidates


def _latest_files(files_iter) -> list[Path]:
    grouped: dict[tuple[str, ...], list[Path]] = {}
    files = list(files_iter)
    for file_path in files:
        relative_key = file_path.parent.parts
        if not relative_key:
            continue
        grouped.setdefault(relative_key, []).append(file_path)
    return [max(paths, key=_timestamp_key) for paths in grouped.values()]


def _timestamp_key(file_path: Path) -> int:
    match = re.search(r"(\d{8})_(\d{6})\.json$", file_path.name)
    if match:
        return int(match.group(1) + match.group(2))
    return int(file_path.stat().st_mtime)


def _path_part(file_path: Path, root: Path, index: int) -> str:
    try:
        return file_path.relative_to(root).parts[index]
    except IndexError:
        return ""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _safe_metadata_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)[:2000]
    return str(value)[:2000]
