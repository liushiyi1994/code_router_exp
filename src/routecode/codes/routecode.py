from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from routecode.metrics import empirical_entropy


class RouteCodeCodebook:
    """Flat utility-aware codebook with embedding-centroid label prediction.

    The codebook is fit on train utilities only. Test-time assignment uses the
    learned train label embedding centroids, not test utility vectors.
    """

    def __init__(self, n_labels: int, random_state: int = 0, max_iter: int = 25, n_init: int = 10) -> None:
        self.n_labels = int(n_labels)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.effective_labels: int = 0
        self.train_labels_: pd.Series | None = None
        self.label_to_model: dict[int, str] = {}
        self.label_utility_: pd.DataFrame | None = None
        self.utility_centroids_: pd.DataFrame | None = None
        self.embedding_centroids_: pd.DataFrame | None = None
        self.fallback_model: str | None = None

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "RouteCodeCodebook":
        del query_info
        aligned_embeddings = embeddings.loc[utility.index]
        values = utility.to_numpy()
        unique_count = np.unique(np.round(values, decimals=10), axis=0).shape[0]
        self.effective_labels = max(1, min(self.n_labels, len(utility), unique_count))
        self.fallback_model = str(utility.mean(axis=0).idxmax())

        if self.effective_labels == 1:
            labels = np.zeros(len(utility), dtype=int)
        else:
            kmeans = KMeans(
                n_clusters=self.effective_labels,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = kmeans.fit_predict(values)

        self.train_labels_ = pd.Series(labels, index=utility.index, name="route_label")
        label_rows = []
        utility_centroid_rows = []
        centroid_rows = []
        self.label_to_model = {}
        for label in range(self.effective_labels):
            query_ids = self.train_labels_.index[self.train_labels_ == label]
            if len(query_ids) == 0:
                avg_utility = utility.mean(axis=0)
                utility_centroid = utility.mean(axis=0)
                embedding_centroid = aligned_embeddings.mean(axis=0)
            else:
                avg_utility = utility.loc[query_ids].mean(axis=0)
                utility_centroid = utility.loc[query_ids].mean(axis=0)
                embedding_centroid = aligned_embeddings.loc[query_ids].mean(axis=0)
            self.label_to_model[label] = str(avg_utility.idxmax())
            label_rows.append(avg_utility.rename(label))
            utility_centroid_rows.append(utility_centroid.rename(label))
            centroid_rows.append(embedding_centroid.rename(label))

        self.label_utility_ = pd.DataFrame(label_rows)
        self.utility_centroids_ = pd.DataFrame(utility_centroid_rows)
        self.embedding_centroids_ = pd.DataFrame(centroid_rows)
        return self

    def predict_utility_labels(self, utility: pd.DataFrame) -> pd.Series:
        if self.utility_centroids_ is None:
            raise RuntimeError("RouteCodeCodebook must be fit before predict")
        aligned = utility.loc[:, self.utility_centroids_.columns]
        values = aligned.to_numpy(dtype=float)
        centroids = self.utility_centroids_.to_numpy(dtype=float)
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=utility.index, name="route_label")

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        if self.embedding_centroids_ is None:
            raise RuntimeError("RouteCodeCodebook must be fit before predict")
        emb = embeddings.to_numpy(dtype=float)
        centroids = self.embedding_centroids_.to_numpy(dtype=float)
        distances = ((emb[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=embeddings.index, name="route_label")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("RouteCodeCodebook must be fit before predict")
        labels = self.predict_labels(embeddings.loc[query_info.index])
        return self.predict_from_labels(labels)

    def predict_from_labels(self, labels: pd.Series) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("RouteCodeCodebook must be fit before predict")
        selected = [self.label_to_model.get(int(label), self.fallback_model) for label in labels]
        return pd.Series(selected, index=labels.index, name="selected_model")

    def label_entropy(self, labels: pd.Series | None = None) -> float:
        if labels is None:
            if self.train_labels_ is None:
                raise RuntimeError("RouteCodeCodebook must be fit before entropy is available")
            labels = self.train_labels_
        return empirical_entropy(labels.tolist())
