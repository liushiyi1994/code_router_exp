from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans


class EmbeddingClusterRouter:
    def __init__(self, n_clusters: int, random_state: int = 0, n_init: int = 10) -> None:
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.n_init = int(n_init)
        self.kmeans: KMeans | None = None
        self.label_to_model: dict[int, str] = {}
        self.fallback_model: str | None = None
        self.effective_clusters: int = 0

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "EmbeddingClusterRouter":
        del query_info
        aligned = embeddings.loc[utility.index]
        self.fallback_model = str(utility.mean(axis=0).idxmax())
        self.effective_clusters = max(1, min(self.n_clusters, len(aligned)))
        if self.effective_clusters == 1:
            labels = np.zeros(len(aligned), dtype=int)
            self.kmeans = None
        else:
            self.kmeans = KMeans(n_clusters=self.effective_clusters, random_state=self.random_state, n_init=self.n_init)
            labels = self.kmeans.fit_predict(aligned.to_numpy())
        self.label_to_model = {}
        for label in range(self.effective_clusters):
            query_ids = aligned.index[labels == label]
            if len(query_ids) == 0:
                self.label_to_model[label] = self.fallback_model
            else:
                self.label_to_model[label] = str(utility.loc[query_ids].mean(axis=0).idxmax())
        return self

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("EmbeddingClusterRouter must be fit before predict")
        if self.kmeans is None:
            labels = np.zeros(len(embeddings), dtype=int)
        else:
            labels = self.kmeans.predict(embeddings.to_numpy())
        return pd.Series(labels, index=embeddings.index, name="cluster_label")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("EmbeddingClusterRouter must be fit before predict")
        labels = self.predict_labels(embeddings.loc[query_info.index])
        selected = [self.label_to_model.get(int(label), self.fallback_model) for label in labels]
        return pd.Series(selected, index=query_info.index, name="selected_model")


class AgglomerativeClusterRouter:
    """Train-only agglomerative embedding clusters with centroid test assignment."""

    def __init__(self, n_clusters: int) -> None:
        self.n_clusters = int(n_clusters)
        self.label_to_model: dict[int, str] = {}
        self.fallback_model: str | None = None
        self.effective_clusters: int = 0
        self.embedding_centroids_: pd.DataFrame | None = None

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "AgglomerativeClusterRouter":
        del query_info
        aligned = embeddings.loc[utility.index]
        self.fallback_model = str(utility.mean(axis=0).idxmax())
        self.effective_clusters = max(1, min(self.n_clusters, len(aligned)))
        if self.effective_clusters == 1:
            labels = np.zeros(len(aligned), dtype=int)
        else:
            labels = AgglomerativeClustering(n_clusters=self.effective_clusters).fit_predict(aligned.to_numpy())
        centroid_rows = []
        self.label_to_model = {}
        for label in range(self.effective_clusters):
            query_ids = aligned.index[labels == label]
            if len(query_ids) == 0:
                centroid_rows.append(aligned.mean(axis=0).rename(label))
                self.label_to_model[label] = self.fallback_model
            else:
                centroid_rows.append(aligned.loc[query_ids].mean(axis=0).rename(label))
                self.label_to_model[label] = str(utility.loc[query_ids].mean(axis=0).idxmax())
        self.embedding_centroids_ = pd.DataFrame(centroid_rows)
        return self

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        if self.embedding_centroids_ is None:
            raise RuntimeError("AgglomerativeClusterRouter must be fit before predict")
        values = embeddings.to_numpy(dtype=float)
        centroids = self.embedding_centroids_.to_numpy(dtype=float)
        distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1)
        return pd.Series(labels, index=embeddings.index, name="cluster_label")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("AgglomerativeClusterRouter must be fit before predict")
        labels = self.predict_labels(embeddings.loc[query_info.index])
        selected = [self.label_to_model.get(int(label), self.fallback_model) for label in labels]
        return pd.Series(selected, index=query_info.index, name="selected_model")
