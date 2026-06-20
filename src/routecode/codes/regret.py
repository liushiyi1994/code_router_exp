from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from routecode.metrics import empirical_entropy


class RegretOptimizedRouteCode:
    """RouteCode codebook optimized for selected-model regret.

    The codebook is fit from train utilities only. Diagnostic/oracle labels may
    use utility vectors; deployable prediction uses train embedding centroids.
    """

    def __init__(
        self,
        n_labels: int,
        random_state: int = 0,
        max_iter: int = 25,
        refinement_iter: int = 10,
        n_init: int = 10,
    ) -> None:
        self.n_labels = int(n_labels)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.refinement_iter = int(refinement_iter)
        self.n_init = int(n_init)

        self.effective_labels: int = 0
        self.train_labels_: pd.Series | None = None
        self.label_to_model: dict[int, str] = {}
        self.label_utility_: pd.DataFrame | None = None
        self.utility_centroids_: pd.DataFrame | None = None
        self.embedding_centroids_: pd.DataFrame | None = None
        self.fallback_model: str | None = None
        self.assignment_regret_: float = 0.0

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "RegretOptimizedRouteCode":
        del query_info
        aligned_embeddings = embeddings.loc[utility.index]
        values = utility.to_numpy(dtype=float)
        unique_count = np.unique(np.round(values, decimals=10), axis=0).shape[0]
        self.effective_labels = max(1, min(self.n_labels, len(utility), unique_count))
        self.fallback_model = str(utility.mean(axis=0).idxmax())

        if self.effective_labels == 1:
            labels = np.zeros(len(utility), dtype=int)
        else:
            labels = KMeans(
                n_clusters=self.effective_labels,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            ).fit_predict(values)
            labels = self._refine_assignments(labels, utility)

        self.train_labels_ = pd.Series(labels, index=utility.index, name="route_label")
        self._set_label_tables(utility, aligned_embeddings)
        self._set_assignment_regret(utility)
        return self

    def predict_utility_labels(self, utility: pd.DataFrame) -> pd.Series:
        if self.utility_centroids_ is None:
            raise RuntimeError("RegretOptimizedRouteCode must be fit before predict")
        labels = self._best_labels_for_utility(utility)
        return pd.Series(labels, index=utility.index, name="route_label")

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        if self.embedding_centroids_ is None:
            raise RuntimeError("RegretOptimizedRouteCode must be fit before predict")
        emb = embeddings.to_numpy(dtype=float)
        centroids = self.embedding_centroids_.to_numpy(dtype=float)
        distances = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=embeddings.index, name="route_label")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("RegretOptimizedRouteCode must be fit before predict")
        labels = self.predict_labels(embeddings.loc[query_info.index])
        return self.predict_from_labels(labels)

    def predict_from_labels(self, labels: pd.Series) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("RegretOptimizedRouteCode must be fit before predict")
        selected = [self.label_to_model.get(int(label), self.fallback_model) for label in labels]
        return pd.Series(selected, index=labels.index, name="selected_model")

    def label_entropy(self, labels: pd.Series | None = None) -> float:
        if labels is None:
            if self.train_labels_ is None:
                raise RuntimeError("RegretOptimizedRouteCode must be fit before entropy is available")
            labels = self.train_labels_
        return empirical_entropy(labels.tolist())

    def _refine_assignments(self, labels: np.ndarray, utility: pd.DataFrame) -> np.ndarray:
        current = labels.astype(int).copy()
        for _ in range(self.refinement_iter):
            label_to_model = _label_models(current, utility, self.effective_labels, self.fallback_model)
            updated = self._best_labels_for_utility(utility, label_to_model=label_to_model)
            updated = _reseed_empty_labels(updated, utility, label_to_model, self.effective_labels)
            if np.array_equal(updated, current):
                break
            current = updated
        return current

    def _set_label_tables(self, utility: pd.DataFrame, embeddings: pd.DataFrame) -> None:
        if self.train_labels_ is None:
            raise RuntimeError("Missing train labels")

        label_rows = []
        utility_centroid_rows = []
        embedding_centroid_rows = []
        self.label_to_model = {}
        labels_array = self.train_labels_.to_numpy(dtype=int)
        for label in range(self.effective_labels):
            query_ids = self.train_labels_.index[labels_array == label]
            if len(query_ids) == 0:
                avg_utility = utility.mean(axis=0)
                utility_centroid = utility.mean(axis=0)
                embedding_centroid = embeddings.mean(axis=0)
            else:
                avg_utility = utility.loc[query_ids].mean(axis=0)
                utility_centroid = utility.loc[query_ids].mean(axis=0)
                embedding_centroid = embeddings.loc[query_ids].mean(axis=0)
            self.label_to_model[label] = str(avg_utility.idxmax())
            label_rows.append(avg_utility.rename(label))
            utility_centroid_rows.append(utility_centroid.rename(label))
            embedding_centroid_rows.append(embedding_centroid.rename(label))

        self.label_utility_ = pd.DataFrame(label_rows)
        self.utility_centroids_ = pd.DataFrame(utility_centroid_rows)
        self.embedding_centroids_ = pd.DataFrame(embedding_centroid_rows)

    def _set_assignment_regret(self, utility: pd.DataFrame) -> None:
        if self.train_labels_ is None:
            raise RuntimeError("Missing train labels")
        selected = self.predict_from_labels(self.train_labels_)
        selected_utility = np.array([utility.loc[qid, model] for qid, model in selected.items()], dtype=float)
        self.assignment_regret_ = float((utility.max(axis=1).to_numpy(dtype=float) - selected_utility).mean())

    def _best_labels_for_utility(
        self,
        utility: pd.DataFrame,
        label_to_model: dict[int, str] | None = None,
    ) -> np.ndarray:
        if label_to_model is None:
            label_to_model = self.label_to_model
        if self.utility_centroids_ is None and label_to_model is self.label_to_model:
            raise RuntimeError("RegretOptimizedRouteCode must be fit before predict")

        label_models = [label_to_model.get(label, self.fallback_model) for label in range(self.effective_labels)]
        scores = np.column_stack([utility[str(model)].to_numpy(dtype=float) for model in label_models])
        if self.utility_centroids_ is not None:
            centroids = self.utility_centroids_.loc[:, utility.columns].to_numpy(dtype=float)
            values = utility.to_numpy(dtype=float)
            distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            scores = scores - 1e-9 * distances
        return scores.argmax(axis=1)


def _label_models(
    labels: np.ndarray,
    utility: pd.DataFrame,
    n_labels: int,
    fallback_model: str | None,
) -> dict[int, str]:
    label_to_model: dict[int, str] = {}
    fallback = str(fallback_model or utility.mean(axis=0).idxmax())
    for label in range(n_labels):
        mask = labels == label
        if mask.any():
            label_to_model[label] = str(utility.loc[mask].mean(axis=0).idxmax())
        else:
            label_to_model[label] = fallback
    return label_to_model


def _reseed_empty_labels(
    labels: np.ndarray,
    utility: pd.DataFrame,
    label_to_model: dict[int, str],
    n_labels: int,
) -> np.ndarray:
    updated = labels.astype(int).copy()
    counts = np.bincount(updated, minlength=n_labels)
    empty_labels = [label for label, count in enumerate(counts) if count == 0]
    if not empty_labels:
        return updated

    label_models = [label_to_model[label] for label in range(n_labels)]
    selected_utility = np.array([utility.iloc[row_idx][label_models[label]] for row_idx, label in enumerate(updated)])
    regret = utility.max(axis=1).to_numpy(dtype=float) - selected_utility
    candidate_order = np.argsort(-regret)
    used: set[int] = set()
    for empty_label in empty_labels:
        for row_idx in candidate_order:
            if int(row_idx) in used:
                continue
            if counts[updated[row_idx]] <= 1:
                continue
            counts[updated[row_idx]] -= 1
            updated[row_idx] = empty_label
            counts[empty_label] += 1
            used.add(int(row_idx))
            break
    return updated
