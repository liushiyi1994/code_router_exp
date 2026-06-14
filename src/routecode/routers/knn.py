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
        selected = []
        for positions in neighbor_positions:
            neighbor_ids = self.train_index[positions]
            selected.append(str(self.train_utility.loc[neighbor_ids].mean(axis=0).idxmax()))
        return pd.Series(selected, index=query_info.index, name="selected_model")
