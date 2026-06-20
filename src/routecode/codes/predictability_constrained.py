from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from routecode.metrics import empirical_entropy


class PredictabilityConstrainedRouteCode:
    """RouteCode codebook with a utility/predictability assignment tradeoff.

    Labels are fit on train queries only by minimizing a KMeans-style proxy for
    utility distortion plus an embedding predictability term:

        ||U_q - c_z^U||^2 + alpha * ||E_q - c_z^E||^2 + beta * balance(z)

    Deployment-time label prediction uses only the embedding centroids.
    """

    def __init__(
        self,
        n_labels: int,
        alpha: float = 0.0,
        beta: float = 0.0,
        random_state: int = 0,
        max_iter: int = 25,
        refinement_iter: int = 10,
        n_init: int = 10,
    ) -> None:
        self.n_labels = int(n_labels)
        self.alpha = float(alpha)
        self.beta = float(beta)
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

        self.utility_columns_: list[object] = []
        self.embedding_columns_: list[object] = []
        self.utility_mean_: np.ndarray | None = None
        self.utility_scale_: np.ndarray | None = None
        self.embedding_mean_: np.ndarray | None = None
        self.embedding_scale_: np.ndarray | None = None
        self._utility_centroids_scaled: np.ndarray | None = None
        self._embedding_centroids_scaled: np.ndarray | None = None

        self.assignment_utility_loss_: float = 0.0
        self.assignment_embedding_loss_: float = 0.0
        self.assignment_balance_penalty_: float = 0.0
        self.assignment_objective_: float = 0.0

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "PredictabilityConstrainedRouteCode":
        del query_info
        aligned_embeddings = embeddings.loc[utility.index]
        self.utility_columns_ = list(utility.columns)
        self.embedding_columns_ = list(aligned_embeddings.columns)
        self.fallback_model = str(utility.mean(axis=0).idxmax())

        utility_values = utility.to_numpy(dtype=float)
        embedding_values = aligned_embeddings.to_numpy(dtype=float)
        utility_scaled, self.utility_mean_, self.utility_scale_ = _standardize_fit(utility_values)
        embedding_scaled, self.embedding_mean_, self.embedding_scale_ = _standardize_fit(embedding_values)

        initial_features = np.hstack([utility_scaled, np.sqrt(max(self.alpha, 0.0)) * embedding_scaled])
        unique_count = np.unique(np.round(initial_features, decimals=10), axis=0).shape[0]
        self.effective_labels = max(1, min(self.n_labels, len(utility), unique_count))

        if self.effective_labels == 1:
            labels = np.zeros(len(utility), dtype=int)
        else:
            kmeans = KMeans(
                n_clusters=self.effective_labels,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = kmeans.fit_predict(initial_features)
            labels = self._refine_assignments(labels, utility_scaled, embedding_scaled)

        self.train_labels_ = pd.Series(labels, index=utility.index, name="route_label")
        self._set_label_tables(utility, aligned_embeddings, utility_scaled, embedding_scaled)
        self._set_assignment_objective(labels, utility_scaled, embedding_scaled)
        return self

    def predict_utility_labels(self, utility: pd.DataFrame) -> pd.Series:
        if self._utility_centroids_scaled is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        utility_scaled = self._transform_utility(utility)
        distances = _squared_distances(utility_scaled, self._utility_centroids_scaled)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=utility.index, name="route_label")

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        if self._embedding_centroids_scaled is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        embedding_scaled = self._transform_embeddings(embeddings)
        distances = _squared_distances(embedding_scaled, self._embedding_centroids_scaled)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=embeddings.index, name="route_label")

    def predict_label_confidence(self, embeddings: pd.DataFrame) -> pd.Series:
        if self._embedding_centroids_scaled is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        probabilities = self.predict_label_distribution(embeddings).to_numpy(dtype=float)
        return pd.Series(probabilities.max(axis=1), index=embeddings.index, name="route_label_confidence")

    def predict_label_distribution(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        if self._embedding_centroids_scaled is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        embedding_scaled = self._transform_embeddings(embeddings)
        distances = _squared_distances(embedding_scaled, self._embedding_centroids_scaled)
        logits = -distances
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        return pd.DataFrame(
            probabilities,
            index=embeddings.index,
            columns=list(range(self.effective_labels)),
        )

    def predict_joint_labels(self, utility: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self._utility_centroids_scaled is None or self._embedding_centroids_scaled is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        aligned_embeddings = embeddings.loc[utility.index]
        utility_scaled = self._transform_utility(utility)
        embedding_scaled = self._transform_embeddings(aligned_embeddings)
        distances = _squared_distances(utility_scaled, self._utility_centroids_scaled)
        distances += self.alpha * _squared_distances(embedding_scaled, self._embedding_centroids_scaled)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=utility.index, name="route_label")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        labels = self.predict_labels(embeddings.loc[query_info.index])
        return self.predict_from_labels(labels)

    def predict_from_labels(self, labels: pd.Series) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        selected = [self.label_to_model.get(int(label), self.fallback_model) for label in labels]
        return pd.Series(selected, index=labels.index, name="selected_model")

    def label_entropy(self, labels: pd.Series | None = None) -> float:
        if labels is None:
            if self.train_labels_ is None:
                raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before entropy is available")
            labels = self.train_labels_
        return empirical_entropy(labels.tolist())

    def objective_summary(self) -> dict[str, float]:
        return {
            "assignment_utility_loss": self.assignment_utility_loss_,
            "assignment_embedding_loss": self.assignment_embedding_loss_,
            "assignment_balance_penalty": self.assignment_balance_penalty_,
            "assignment_objective": self.assignment_objective_,
        }

    def _refine_assignments(
        self,
        labels: np.ndarray,
        utility_scaled: np.ndarray,
        embedding_scaled: np.ndarray,
    ) -> np.ndarray:
        if self.refinement_iter <= 0:
            return labels

        current = labels.astype(int).copy()
        for _ in range(self.refinement_iter):
            utility_centroids, embedding_centroids = _centroids_from_labels(
                current,
                utility_scaled,
                embedding_scaled,
                self.effective_labels,
            )
            distances = _squared_distances(utility_scaled, utility_centroids)
            distances += self.alpha * _squared_distances(embedding_scaled, embedding_centroids)
            if self.beta > 0:
                counts = np.bincount(current, minlength=self.effective_labels).astype(float)
                distances += self.beta * (counts / max(float(len(current)), 1.0))[None, :]
            updated = distances.argmin(axis=1)
            if np.array_equal(updated, current):
                break
            current = updated
        return current

    def _set_label_tables(
        self,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
        utility_scaled: np.ndarray,
        embedding_scaled: np.ndarray,
    ) -> None:
        if self.train_labels_ is None:
            raise RuntimeError("Missing train labels")

        utility_centroids_scaled, embedding_centroids_scaled = _centroids_from_labels(
            self.train_labels_.to_numpy(dtype=int),
            utility_scaled,
            embedding_scaled,
            self.effective_labels,
        )
        self._utility_centroids_scaled = utility_centroids_scaled
        self._embedding_centroids_scaled = embedding_centroids_scaled

        label_rows = []
        utility_centroid_rows = []
        embedding_centroid_rows = []
        self.label_to_model = {}
        for label in range(self.effective_labels):
            query_ids = self.train_labels_.index[self.train_labels_ == label]
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

    def _set_assignment_objective(
        self,
        labels: np.ndarray,
        utility_scaled: np.ndarray,
        embedding_scaled: np.ndarray,
    ) -> None:
        if self._utility_centroids_scaled is None or self._embedding_centroids_scaled is None:
            raise RuntimeError("Missing centroids")
        utility_distances = _squared_distances(utility_scaled, self._utility_centroids_scaled)
        embedding_distances = _squared_distances(embedding_scaled, self._embedding_centroids_scaled)
        rows = np.arange(len(labels))
        self.assignment_utility_loss_ = float(utility_distances[rows, labels].mean())
        self.assignment_embedding_loss_ = float(embedding_distances[rows, labels].mean())
        counts = np.bincount(labels, minlength=self.effective_labels).astype(float)
        proportions = counts / max(float(len(labels)), 1.0)
        self.assignment_balance_penalty_ = float((proportions**2).sum())
        self.assignment_objective_ = (
            self.assignment_utility_loss_
            + self.alpha * self.assignment_embedding_loss_
            + self.beta * self.assignment_balance_penalty_
        )

    def _transform_utility(self, utility: pd.DataFrame) -> np.ndarray:
        if self.utility_mean_ is None or self.utility_scale_ is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        aligned = utility.loc[:, self.utility_columns_].to_numpy(dtype=float)
        return (aligned - self.utility_mean_) / self.utility_scale_

    def _transform_embeddings(self, embeddings: pd.DataFrame) -> np.ndarray:
        if self.embedding_mean_ is None or self.embedding_scale_ is None:
            raise RuntimeError("PredictabilityConstrainedRouteCode must be fit before predict")
        aligned = embeddings.loc[:, self.embedding_columns_].to_numpy(dtype=float)
        return (aligned - self.embedding_mean_) / self.embedding_scale_


def _standardize_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    return (values - mean) / scale, mean, scale


def _centroids_from_labels(
    labels: np.ndarray,
    utility_scaled: np.ndarray,
    embedding_scaled: np.ndarray,
    n_labels: int,
) -> tuple[np.ndarray, np.ndarray]:
    utility_centroids = np.zeros((n_labels, utility_scaled.shape[1]), dtype=float)
    embedding_centroids = np.zeros((n_labels, embedding_scaled.shape[1]), dtype=float)
    for label in range(n_labels):
        mask = labels == label
        if mask.any():
            utility_centroids[label] = utility_scaled[mask].mean(axis=0)
            embedding_centroids[label] = embedding_scaled[mask].mean(axis=0)
    return utility_centroids, embedding_centroids


def _squared_distances(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
