from __future__ import annotations

import pandas as pd
from sklearn.neighbors import NearestNeighbors


class KNNRouter:
    def __init__(self, n_neighbors: int = 15) -> None:
        self.n_neighbors = int(n_neighbors)
        self.neighbors: NearestNeighbors | None = None
        self.train_utility: pd.DataFrame | None = None
        self.train_index: pd.Index | None = None

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "KNNRouter":
        del query_info
        aligned = embeddings.loc[utility.index]
        k = max(1, min(self.n_neighbors, len(aligned)))
        self.neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean")
        self.neighbors.fit(aligned.to_numpy())
        self.train_utility = utility.copy()
        self.train_index = aligned.index
        return self

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.neighbors is None or self.train_utility is None or self.train_index is None:
            raise RuntimeError("KNNRouter must be fit before predict")
        aligned = embeddings.loc[query_info.index]
        _, neighbor_positions = self.neighbors.kneighbors(aligned.to_numpy())
        train_values = self.train_utility.to_numpy(dtype=float)
        neighbor_means = train_values[neighbor_positions].mean(axis=1)
        selected_positions = neighbor_means.argmax(axis=1)
        model_ids = [str(model_id) for model_id in self.train_utility.columns]
        selected = [model_ids[position] for position in selected_positions]
        return pd.Series(selected, index=query_info.index, name="selected_model")
