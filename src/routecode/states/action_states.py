from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


LOCAL_ENOUGH = "local_enough"
FRONTIER_NEEDED = "frontier_needed"


@dataclass(frozen=True)
class ActionStatePolicy:
    labels: pd.Series
    label_to_model: dict[str, str]
    fallback_model: str

    def select(self, predicted_labels: pd.Series) -> pd.Series:
        return predicted_labels.astype(str).map(self.label_to_model).fillna(self.fallback_model).rename("selected_model")


def local_frontier_action_labels(
    utility: pd.DataFrame,
    *,
    local_models: list[str] | tuple[str, ...],
    frontier_models: list[str] | tuple[str, ...],
) -> pd.Series:
    """Build a 1-bit utility-derived action state.

    The label is learned from the cost-aware utility matrix, not from benchmark
    names or gold answers:

    - `local_enough`: the best local model has utility at least as high as the
      best frontier model.
    - `frontier_needed`: the best frontier model beats the best local model.
    """

    local_cols = [col for col in local_models if col in utility.columns]
    frontier_cols = [col for col in frontier_models if col in utility.columns]
    if not local_cols:
        raise ValueError("At least one local model column is required.")
    if not frontier_cols:
        raise ValueError("At least one frontier model column is required.")
    local_best = utility[local_cols].astype(float).max(axis=1)
    frontier_best = utility[frontier_cols].astype(float).max(axis=1)
    labels = frontier_best.gt(local_best).map({True: FRONTIER_NEEDED, False: LOCAL_ENOUGH})
    return labels.rename("state_label")


def fit_action_state_policy(utility: pd.DataFrame, labels: pd.Series) -> ActionStatePolicy:
    aligned = utility.reindex(labels.index).dropna(axis=0, how="any").astype(float)
    aligned_labels = labels.reindex(aligned.index).astype(str)
    state_utility = aligned.groupby(aligned_labels).mean()
    label_to_model = state_utility.idxmax(axis=1).astype(str).to_dict()
    fallback_model = str(aligned.mean(axis=0).sort_values(ascending=False).index[0])
    return ActionStatePolicy(labels=aligned_labels, label_to_model=label_to_model, fallback_model=fallback_model)


def selected_utility(utility: pd.DataFrame, selected_models: pd.Series) -> pd.Series:
    aligned = utility.reindex(selected_models.index)
    values = []
    index = []
    for query_id, model_id in selected_models.astype(str).items():
        if query_id not in aligned.index or model_id not in aligned.columns:
            continue
        values.append(float(aligned.loc[query_id, model_id]))
        index.append(query_id)
    return pd.Series(values, index=pd.Index(index, name=utility.index.name), name="selected_utility")
