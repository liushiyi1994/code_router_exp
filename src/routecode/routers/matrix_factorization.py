from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


class MatrixFactorizationRouter:
    """Low-rank utility-factor router: embeddings -> latent utility factors -> model."""

    def __init__(self, rank: int = 4, alpha: float = 1.0, random_state: int = 0) -> None:
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.regressor = Ridge(alpha=self.alpha)
        self.svd: TruncatedSVD | None = None
        self.utility_mean_: np.ndarray | None = None
        self.model_ids: list[str] = []
        self.fallback_model: str | None = None

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "MatrixFactorizationRouter":
        del query_info
        if utility.empty or utility.shape[1] == 0:
            raise ValueError("MatrixFactorizationRouter requires a non-empty utility matrix")
        self.model_ids = [str(model) for model in utility.columns]
        self.fallback_model = str(utility.mean(axis=0).idxmax())
        rank = max(1, min(self.rank, utility.shape[0] - 1, utility.shape[1] - 1))
        x = self.scaler.fit_transform(embeddings.loc[utility.index].to_numpy(dtype=float))
        utility_values = utility.to_numpy(dtype=float)
        self.utility_mean_ = utility_values.mean(axis=0)
        centered = utility_values - self.utility_mean_
        if rank <= 0 or np.allclose(centered, 0.0):
            self.svd = None
            return self
        self.svd = TruncatedSVD(n_components=rank, random_state=self.random_state)
        latent = self.svd.fit_transform(centered)
        self.regressor.fit(x, latent)
        return self

    def predict_scores(self, embeddings: pd.DataFrame) -> pd.DataFrame:
        if self.utility_mean_ is None or self.fallback_model is None:
            raise RuntimeError("MatrixFactorizationRouter must be fit before predict")
        if self.svd is None:
            scores = np.tile(self.utility_mean_, (len(embeddings), 1))
        else:
            x = self.scaler.transform(embeddings.to_numpy(dtype=float))
            latent = self.regressor.predict(x)
            if latent.ndim == 1:
                latent = latent.reshape(-1, 1)
            scores = latent @ self.svd.components_ + self.utility_mean_
        return pd.DataFrame(scores, index=embeddings.index, columns=self.model_ids)

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        scores = self.predict_scores(embeddings.loc[query_info.index])
        selected = scores.idxmax(axis=1).astype(str)
        return pd.Series(selected, index=query_info.index, name="selected_model")


class BinaryThresholdRouter:
    """RouteLLM-style strong/weak threshold router using local embeddings."""

    def __init__(
        self,
        strong_model: str,
        weak_model: str,
        threshold: float = 0.5,
        random_state: int = 0,
        max_iter: int = 1000,
    ) -> None:
        self.strong_model = str(strong_model)
        self.weak_model = str(weak_model)
        self.threshold = float(threshold)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.scaler = StandardScaler()
        self.model: LogisticRegression | None = None
        self.constant_probability: float | None = None

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "BinaryThresholdRouter":
        del query_info
        missing = {self.strong_model, self.weak_model} - set(map(str, utility.columns))
        if missing:
            raise ValueError(f"Missing strong/weak model columns: {sorted(missing)}")
        labels = (utility[self.strong_model] > utility[self.weak_model]).astype(int)
        if labels.nunique() == 1:
            self.constant_probability = float(labels.iloc[0])
            self.model = None
            return self
        x = self.scaler.fit_transform(embeddings.loc[utility.index].to_numpy(dtype=float))
        self.model = LogisticRegression(random_state=self.random_state, max_iter=self.max_iter)
        self.model.fit(x, labels.to_numpy())
        self.constant_probability = None
        return self

    def predict_strong_win_rate(self, embeddings: pd.DataFrame) -> pd.Series:
        if self.constant_probability is not None:
            probabilities = np.full(len(embeddings), self.constant_probability, dtype=float)
        elif self.model is not None:
            x = self.scaler.transform(embeddings.to_numpy(dtype=float))
            class_index = list(self.model.classes_).index(1)
            probabilities = self.model.predict_proba(x)[:, class_index]
        else:
            raise RuntimeError("BinaryThresholdRouter must be fit before predict")
        return pd.Series(probabilities, index=embeddings.index, name="strong_win_rate")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        win_rate = self.predict_strong_win_rate(embeddings.loc[query_info.index])
        selected = np.where(win_rate.to_numpy() >= self.threshold, self.strong_model, self.weak_model)
        return pd.Series(selected, index=query_info.index, name="selected_model").astype(str)
