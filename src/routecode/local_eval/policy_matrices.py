from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LocalPolicyMatrices:
    query_model_utility: pd.DataFrame
    query_model_quality: pd.DataFrame
    query_model_cost: pd.DataFrame
    state_model_utility: pd.DataFrame
    state_model_quality: pd.DataFrame
    state_model_cost: pd.DataFrame
    metadata: pd.DataFrame


def build_local_policy_matrices(
    *,
    local_outcomes: pd.DataFrame,
    state_targets: pd.DataFrame,
    lambda_cost: float = 0.0,
    policy_split: str = "test",
    train_split: str = "train",
) -> LocalPolicyMatrices:
    """Convert exact-scored local outcomes into ProbeRoute++ policy matrices."""

    _require_columns(local_outcomes, ["query_id", "model_id", "quality", "cost_proxy"], "local_outcomes")
    _require_columns(state_targets, ["query_id", "state_label", "split"], "state_targets")
    outcomes = local_outcomes.copy()
    outcomes["query_id"] = outcomes["query_id"].astype(str)
    outcomes["model_id"] = outcomes["model_id"].astype(str)
    outcomes["quality"] = pd.to_numeric(outcomes["quality"], errors="coerce").fillna(0.0)
    outcomes["cost_proxy"] = pd.to_numeric(outcomes["cost_proxy"], errors="coerce").fillna(0.0)
    outcomes["utility"] = outcomes["quality"] - float(lambda_cost) * outcomes["cost_proxy"]
    targets = state_targets[["query_id", "state_label", "split"]].drop_duplicates("query_id").copy()
    targets["query_id"] = targets["query_id"].astype(str)
    targets["state_label"] = targets["state_label"].astype(int).map(lambda value: f"z{value}")
    joined = outcomes.merge(targets, on="query_id", how="inner")
    if joined.empty:
        raise ValueError("No local outcomes overlap state targets")
    policy = joined[joined["split"].astype(str).eq(str(policy_split))].copy()
    train = joined[joined["split"].astype(str).eq(str(train_split))].copy()
    if policy.empty:
        raise ValueError(f"No local outcomes overlap policy split {policy_split!r}")
    if train.empty:
        raise ValueError(f"No local outcomes overlap train split {train_split!r}")
    query_model_utility = _query_matrix(policy, "utility")
    query_model_quality = _query_matrix(policy, "quality")
    query_model_cost = _query_matrix(policy, "cost_proxy")
    state_model_utility = _state_matrix(train, "utility")
    state_model_quality = _state_matrix(train, "quality")
    state_model_cost = _state_matrix(train, "cost_proxy")
    model_ids = sorted(outcomes["model_id"].unique())
    state_labels = sorted(targets["state_label"].unique(), key=_state_sort_key)
    state_model_utility = _complete_state_matrix(state_model_utility, state_labels, model_ids)
    state_model_quality = _complete_state_matrix(state_model_quality, state_labels, model_ids)
    state_model_cost = _complete_state_matrix(state_model_cost, state_labels, model_ids)
    metadata = pd.DataFrame(
        [
            {
                "outcome_rows": int(len(local_outcomes)),
                "overlap_rows": int(len(joined)),
                "train_rows": int(len(train)),
                "policy_rows": int(len(policy)),
                "policy_queries": int(query_model_utility.shape[0]),
                "train_queries": int(train["query_id"].nunique()),
                "model_count": int(len(model_ids)),
                "state_count": int(len(state_labels)),
                "lambda_cost": float(lambda_cost),
                "policy_split": str(policy_split),
                "train_split": str(train_split),
            }
        ]
    )
    return LocalPolicyMatrices(
        query_model_utility=query_model_utility,
        query_model_quality=query_model_quality,
        query_model_cost=query_model_cost,
        state_model_utility=state_model_utility,
        state_model_quality=state_model_quality,
        state_model_cost=state_model_cost,
        metadata=metadata,
    )


def _query_matrix(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    matrix = frame.pivot_table(
        index="query_id",
        columns="model_id",
        values=value_column,
        aggfunc="mean",
    ).sort_index()
    matrix.index.name = "query_id"
    return matrix.reset_index()


def _state_matrix(frame: pd.DataFrame, value_column: str) -> pd.DataFrame:
    matrix = frame.pivot_table(
        index="state_label",
        columns="model_id",
        values=value_column,
        aggfunc="mean",
    )
    matrix.index.name = "state_label"
    return matrix


def _complete_state_matrix(matrix: pd.DataFrame, state_labels: list[str], model_ids: list[str]) -> pd.DataFrame:
    matrix = matrix.reindex(index=state_labels, columns=model_ids)
    global_means = matrix.mean(axis=0)
    matrix = matrix.fillna(global_means).fillna(0.0)
    matrix.index.name = "state_label"
    return matrix.reset_index()


def _state_sort_key(label: str) -> tuple[int, str]:
    value = str(label)
    if value.startswith("z") and value[1:].isdigit():
        return int(value[1:]), value
    return 10**9, value


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
