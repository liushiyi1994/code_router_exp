from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class ObservableTreeStateModel:
    n_states: int
    feature_columns: list[str]
    model_ids: list[str]
    labels: pd.Series
    leaf_to_label: dict[str, str]
    tree: DecisionTreeRegressor
    state_utility: pd.DataFrame
    state_variance: pd.DataFrame
    label_to_model: dict[str, str]
    fallback_model: str

    def predict_states(self, features: pd.DataFrame) -> pd.Series:
        aligned = _clean_features(features.reindex(columns=self.feature_columns, fill_value=0.0))
        leaves = self.tree.apply(aligned.to_numpy(dtype=float))
        labels = [self.leaf_to_label.get(str(int(leaf)), "unseen") for leaf in leaves]
        return pd.Series(labels, index=aligned.index.astype(str), name="state_label")

    def select_models(self, labels: pd.Series) -> pd.Series:
        selected = labels.astype(str).map(self.label_to_model).fillna(self.fallback_model)
        return selected.rename("selected_model")


def fit_observable_tree_state_model(
    features: pd.DataFrame,
    utility: pd.DataFrame,
    *,
    n_states: int = 8,
    min_samples_leaf: int = 5,
    random_state: int = 17,
) -> ObservableTreeStateModel:
    """Fit deployable feature-defined states optimized for utility prediction.

    The tree sees only observable features as input and the train utility vector
    as the target. Its leaves are the route states. This deliberately trades
    some utility optimality for observability: state assignment is a deployable
    function of query/probe features, while the state-to-model table is still
    estimated from the train utility matrix only.
    """

    aligned_utility = utility.dropna(axis=0, how="any").astype(float)
    aligned_features = _clean_features(features.reindex(aligned_utility.index))
    if aligned_features.empty or aligned_utility.empty:
        raise ValueError("Observable tree states require non-empty aligned features and utility.")
    tree = DecisionTreeRegressor(
        max_leaf_nodes=max(2, int(n_states)),
        min_samples_leaf=max(1, int(min_samples_leaf)),
        random_state=int(random_state),
    )
    tree.fit(aligned_features.to_numpy(dtype=float), aligned_utility.to_numpy(dtype=float))
    leaves = tree.apply(aligned_features.to_numpy(dtype=float))
    leaf_counts = pd.Series(leaves.astype(int)).value_counts().sort_values(ascending=False)
    ordered_leaves = sorted(int(leaf) for leaf in leaf_counts.index)
    leaf_to_label = {str(leaf): f"z{idx:02d}" for idx, leaf in enumerate(ordered_leaves)}
    labels = pd.Series(
        [leaf_to_label[str(int(leaf))] for leaf in leaves],
        index=aligned_utility.index.astype(str),
        name="state_label",
    )
    state_utility, state_variance = state_tables(aligned_utility, labels)
    label_to_model = state_utility.idxmax(axis=1).astype(str).to_dict()
    fallback_model = str(aligned_utility.mean(axis=0).sort_values(ascending=False).index[0])
    return ObservableTreeStateModel(
        n_states=int(labels.nunique()),
        feature_columns=list(aligned_features.columns),
        model_ids=list(aligned_utility.columns.astype(str)),
        labels=labels,
        leaf_to_label=leaf_to_label,
        tree=tree,
        state_utility=state_utility,
        state_variance=state_variance,
        label_to_model=label_to_model,
        fallback_model=fallback_model,
    )


def state_tables(utility: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = utility.reindex(labels.index).astype(float)
    grouped = aligned.groupby(labels.astype(str))
    mean = grouped.mean().sort_index()
    variance = grouped.var().fillna(0.0).reindex(mean.index)
    return mean, variance


def selected_utility(utility: pd.DataFrame, selected_models: pd.Series) -> pd.Series:
    aligned = utility.reindex(selected_models.index)
    values = []
    index = []
    for query_id, model_id in selected_models.astype(str).items():
        if query_id not in aligned.index or model_id not in aligned.columns:
            continue
        values.append(float(aligned.loc[query_id, model_id]))
        index.append(str(query_id))
    return pd.Series(values, index=pd.Index(index, name=utility.index.name), name="selected_utility")


def _clean_features(features: pd.DataFrame) -> pd.DataFrame:
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
