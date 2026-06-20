from __future__ import annotations

from collections import Counter
import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


PROBE_FEATURE_COLUMNS = [
    "query_id",
    "probe_id",
    "probe_type",
    "probe_model_id",
    "prompt_template",
    "generation_params_json",
    "raw_probe_output",
    "parsed_probe_answer",
    "self_confidence",
    "logprob_mean",
    "entropy_proxy",
    "agreement_score",
    "knn_label_entropy",
    "knn_winner_entropy",
    "latency_sec",
    "input_tokens",
    "output_tokens",
    "probe_cost_proxy",
    "error_type",
    "error_message",
    "created_at",
]

LOCAL_OUTCOME_REQUIRED_COLUMNS = [
    "query_id",
    "model_id",
    "prompt_template",
    "generation_params_json",
    "raw_output",
    "parsed_answer",
    "latency_sec",
    "tokens_input",
    "tokens_output",
    "error_type",
    "error_message",
    "created_at",
]

CONFIDENCE_PATTERN = re.compile(
    r"\b(?:confidence|conf|self[-_\s]?confidence)\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b",
    flags=re.IGNORECASE,
)


def build_probe_features_from_outcomes(
    outcomes: pd.DataFrame,
    *,
    knn_uncertainty: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Convert local model outcome rows into generic cheap-probe observations."""

    _require_columns(outcomes, LOCAL_OUTCOME_REQUIRED_COLUMNS, name="outcomes")
    agreement = _agreement_scores(outcomes)
    knn = _normalise_knn_uncertainty(knn_uncertainty)
    rows: list[dict[str, Any]] = []
    for row_idx, row in outcomes.reset_index(drop=True).iterrows():
        model_id = str(row["model_id"])
        prompt_template = str(row["prompt_template"])
        cost_proxy = _float_or_nan(row.get("cost_proxy"))
        if math.isnan(cost_proxy):
            cost_proxy = _float_or_nan(row.get("latency_sec")) + 0.001 * _float_or_zero(row.get("tokens_output"))
        query_id = str(row["query_id"])
        knn_row = knn.get(query_id, {})
        rows.append(
            {
                "query_id": query_id,
                "probe_id": f"local_answer_probe:{model_id}:{prompt_template}",
                "probe_type": "local_answer_probe",
                "probe_model_id": model_id,
                "prompt_template": prompt_template,
                "generation_params_json": str(row["generation_params_json"]),
                "raw_probe_output": str(row["raw_output"]),
                "parsed_probe_answer": str(row["parsed_answer"]),
                "self_confidence": _parse_self_confidence(row["raw_output"]),
                "logprob_mean": math.nan,
                "entropy_proxy": math.nan,
                "agreement_score": agreement[row_idx],
                "knn_label_entropy": knn_row.get("knn_label_entropy", math.nan),
                "knn_winner_entropy": knn_row.get("knn_winner_entropy", math.nan),
                "latency_sec": _float_or_nan(row["latency_sec"]),
                "input_tokens": int(row["tokens_input"]),
                "output_tokens": int(row["tokens_output"]),
                "probe_cost_proxy": float(cost_proxy),
                "error_type": str(row["error_type"]),
                "error_message": str(row["error_message"]),
                "created_at": str(row["created_at"]),
            }
        )
    return pd.DataFrame(rows, columns=PROBE_FEATURE_COLUMNS)


def compute_knn_uncertainty(
    *,
    embeddings: pd.DataFrame,
    labels: pd.DataFrame,
    train_query_ids: Iterable[str],
    target_query_ids: Iterable[str],
    k: int = 5,
) -> pd.DataFrame:
    """Compute neighbor label/winner entropy using train queries as the only neighbor pool."""

    if k <= 0:
        raise ValueError("k must be positive")
    _require_columns(embeddings, ["query_id"], name="embeddings")
    _require_columns(labels, ["query_id", "route_label", "winner_model_id"], name="labels")
    feature_columns = _embedding_columns(embeddings)
    if not feature_columns:
        raise ValueError("embeddings must contain numeric embedding columns")

    label_table = labels.set_index("query_id")
    embedding_table = embeddings.set_index("query_id")
    train_ids = [str(query_id) for query_id in train_query_ids if str(query_id) in embedding_table.index]
    target_ids = [str(query_id) for query_id in target_query_ids if str(query_id) in embedding_table.index]
    train_matrix = embedding_table.loc[train_ids, feature_columns].to_numpy(dtype=float) if train_ids else np.empty((0, 0))
    rows: list[dict[str, Any]] = []
    for query_id in target_ids:
        candidate_ids = [candidate_id for candidate_id in train_ids if candidate_id != query_id]
        if not candidate_ids:
            rows.append(_empty_knn_row(query_id))
            continue
        target_vector = embedding_table.loc[query_id, feature_columns].to_numpy(dtype=float)
        candidate_positions = [train_ids.index(candidate_id) for candidate_id in candidate_ids]
        candidate_matrix = train_matrix[candidate_positions]
        distances = np.linalg.norm(candidate_matrix - target_vector, axis=1)
        nearest_order = np.argsort(distances)[: min(k, len(candidate_ids))]
        nearest_ids = [candidate_ids[int(idx)] for idx in nearest_order]
        nearest = label_table.loc[[query_id for query_id in nearest_ids if query_id in label_table.index]]
        if nearest.empty:
            rows.append(_empty_knn_row(query_id))
            continue
        rows.append(
            {
                "query_id": query_id,
                "knn_label_entropy": _entropy(nearest["route_label"]),
                "knn_winner_entropy": _entropy(nearest["winner_model_id"]),
            }
        )
    return pd.DataFrame(rows, columns=["query_id", "knn_label_entropy", "knn_winner_entropy"])


def _require_columns(frame: pd.DataFrame, columns: list[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _agreement_scores(outcomes: pd.DataFrame) -> dict[int, float]:
    scores: dict[int, float] = {}
    frame = outcomes.reset_index(drop=True)
    for _, group in frame.groupby("query_id", sort=False):
        answers = [str(value) for value in group["parsed_answer"] if str(value)]
        counts = Counter(answers)
        denominator = len(answers)
        for row_idx, row in group.iterrows():
            answer = str(row["parsed_answer"])
            scores[int(row_idx)] = float(counts[answer] / denominator) if answer and denominator else math.nan
    return scores


def _parse_self_confidence(raw_output: Any) -> float:
    match = CONFIDENCE_PATTERN.search(str(raw_output))
    if not match:
        return math.nan
    return float(match.group(1))


def _normalise_knn_uncertainty(knn_uncertainty: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    if knn_uncertainty is None:
        return {}
    _require_columns(knn_uncertainty, ["query_id", "knn_label_entropy", "knn_winner_entropy"], name="knn_uncertainty")
    return {
        str(row["query_id"]): {
            "knn_label_entropy": _float_or_nan(row["knn_label_entropy"]),
            "knn_winner_entropy": _float_or_nan(row["knn_winner_entropy"]),
        }
        for _, row in knn_uncertainty.iterrows()
    }


def _embedding_columns(embeddings: pd.DataFrame) -> list[str]:
    preferred = [column for column in embeddings.columns if str(column).startswith("emb_")]
    if preferred:
        return preferred
    return [
        column
        for column in embeddings.select_dtypes(include=[np.number]).columns
        if column != "query_id"
    ]


def _entropy(values: Iterable[Any]) -> float:
    clean_values = [str(value) for value in values if not pd.isna(value)]
    if not clean_values:
        return math.nan
    total = len(clean_values)
    return float(-sum((count / total) * math.log2(count / total) for count in Counter(clean_values).values()))


def _empty_knn_row(query_id: str) -> dict[str, float | str]:
    return {
        "query_id": query_id,
        "knn_label_entropy": math.nan,
        "knn_winner_entropy": math.nan,
    }


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _float_or_zero(value: Any) -> float:
    number = _float_or_nan(value)
    return 0.0 if math.isnan(number) else number
