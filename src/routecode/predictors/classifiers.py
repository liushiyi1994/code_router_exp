from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from routecode.codes.routecode import RouteCodeCodebook
from routecode.routers.dataset_lookup import DatasetLabelRouter


class _ConstantClassifier:
    def __init__(self, label: str | int) -> None:
        self.label = label

    def predict(self, embeddings: pd.DataFrame) -> np.ndarray:
        return np.array([self.label] * len(embeddings))

    def predict_proba(self, embeddings: pd.DataFrame) -> np.ndarray:
        return np.ones((len(embeddings), 1), dtype=float)


class _ScaledClassifier:
    def __init__(
        self,
        estimator: str,
        random_state: int = 0,
        max_iter: int = 1000,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        solver: str = "lbfgs",
        learning_rate_init: float = 0.001,
        n_iter_no_change: int = 10,
    ) -> None:
        self.estimator = estimator
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.solver = solver
        self.learning_rate_init = float(learning_rate_init)
        self.n_iter_no_change = int(n_iter_no_change)
        self.scaler = StandardScaler()
        self.model: LogisticRegression | MLPClassifier | LinearSVC | _ConstantClassifier | None = None

    def fit(self, embeddings: pd.DataFrame, labels: pd.Series) -> "_ScaledClassifier":
        y = labels.loc[embeddings.index]
        if y.nunique() == 1:
            self.model = _ConstantClassifier(y.iloc[0])
            return self
        x = self.scaler.fit_transform(embeddings.to_numpy(dtype=float))
        if self.estimator == "logistic":
            self.model = LogisticRegression(
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        elif self.estimator == "mlp":
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                solver=self.solver,
                learning_rate_init=self.learning_rate_init,
                n_iter_no_change=self.n_iter_no_change,
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        elif self.estimator == "svm":
            self.model = LinearSVC(
                random_state=self.random_state,
                max_iter=self.max_iter,
            )
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")
        self.model.fit(x, y.to_numpy())
        return self

    def predict(self, embeddings: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier must be fit before predict")
        if isinstance(self.model, _ConstantClassifier):
            return self.model.predict(embeddings)
        x = self.scaler.transform(embeddings.to_numpy(dtype=float))
        return self.model.predict(x)

    def predict_confidence(self, embeddings: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Classifier must be fit before predict")
        if isinstance(self.model, _ConstantClassifier):
            return np.ones(len(embeddings), dtype=float)
        x = self.scaler.transform(embeddings.to_numpy(dtype=float))
        if not hasattr(self.model, "predict_proba"):
            scores = self.model.decision_function(x)
            scores = np.asarray(scores, dtype=float)
            if scores.ndim == 1:
                return 1.0 / (1.0 + np.exp(-np.abs(scores)))
            shifted = scores - scores.max(axis=1, keepdims=True)
            probabilities = np.exp(shifted)
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            return probabilities.max(axis=1)
        probabilities = self.model.predict_proba(x)
        return probabilities.max(axis=1)


class LogisticModelRouter:
    """Supervised learned router: embedding -> train oracle winning model."""

    def __init__(self, random_state: int = 0, max_iter: int = 1000) -> None:
        self.classifier = _ScaledClassifier("logistic", random_state=random_state, max_iter=max_iter)

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "LogisticModelRouter":
        del query_info
        labels = utility.idxmax(axis=1)
        self.classifier.fit(embeddings.loc[utility.index], labels)
        return self

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        aligned = embeddings.loc[query_info.index]
        selected = self.classifier.predict(aligned)
        return pd.Series(selected, index=query_info.index, name="selected_model").astype(str)


class RouteCodeLabelClassifier:
    """Predict a fitted RouteCode label from embeddings and map label to model."""

    def __init__(self, random_state: int = 0, max_iter: int = 1000) -> None:
        self.classifier = _ScaledClassifier("logistic", random_state=random_state, max_iter=max_iter)
        self.label_to_model: dict[int, str] = {}
        self.fallback_model: str | None = None

    def fit(self, codebook: RouteCodeCodebook, embeddings: pd.DataFrame) -> "RouteCodeLabelClassifier":
        if codebook.train_labels_ is None or codebook.fallback_model is None:
            raise RuntimeError("Codebook must be fit before fitting label classifier")
        train_labels = codebook.train_labels_
        self.label_to_model = dict(codebook.label_to_model)
        self.fallback_model = codebook.fallback_model
        self.classifier.fit(embeddings.loc[train_labels.index], train_labels.astype(int))
        return self

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        labels = self.classifier.predict(embeddings)
        return pd.Series(labels.astype(int), index=embeddings.index, name="route_label")

    def predict_confidence(self, embeddings: pd.DataFrame) -> pd.Series:
        confidence = self.classifier.predict_confidence(embeddings)
        return pd.Series(confidence, index=embeddings.index, name="route_label_confidence")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("RouteCodeLabelClassifier must be fit before predict")
        aligned = embeddings.loc[query_info.index]
        labels = self.predict_labels(aligned)
        selected = [self.label_to_model.get(int(label), self.fallback_model) for label in labels]
        return pd.Series(selected, index=query_info.index, name="selected_model")


class MLPModelRouter(LogisticModelRouter):
    """Supervised learned router with an MLP classifier on embeddings."""

    def __init__(
        self,
        random_state: int = 0,
        hidden_layer_sizes: tuple[int, ...] = (32,),
        max_iter: int = 800,
    ) -> None:
        self.classifier = _ScaledClassifier(
            "mlp",
            random_state=random_state,
            max_iter=max_iter,
            hidden_layer_sizes=hidden_layer_sizes,
            solver="adam",
            learning_rate_init=0.005,
            n_iter_no_change=30,
        )


class SVMModelRouter(LogisticModelRouter):
    """Supervised learned router with a linear SVM on embeddings."""

    def __init__(self, random_state: int = 0, max_iter: int = 3000) -> None:
        self.classifier = _ScaledClassifier("svm", random_state=random_state, max_iter=max_iter)


class MLPRouteCodeLabelClassifier(RouteCodeLabelClassifier):
    """RouteCode label predictor with an MLP classifier on embeddings."""

    def __init__(
        self,
        random_state: int = 0,
        hidden_layer_sizes: tuple[int, ...] = (8,),
        max_iter: int = 2000,
    ) -> None:
        self.classifier = _ScaledClassifier(
            "mlp",
            random_state=random_state,
            max_iter=max_iter,
            hidden_layer_sizes=hidden_layer_sizes,
            solver="adam",
            learning_rate_init=0.01,
            n_iter_no_change=20,
        )
        self.label_to_model: dict[int, str] = {}
        self.fallback_model: str | None = None


class PredictedLabelLookupRouter:
    """Predict a coarse label from embeddings, then use a train-only label lookup table."""

    def __init__(self, label_column: str = "dataset", random_state: int = 0, max_iter: int = 1000) -> None:
        self.label_column = label_column
        self.classifier = _ScaledClassifier("logistic", random_state=random_state, max_iter=max_iter)
        self.lookup = DatasetLabelRouter(label_column=label_column)

    def fit(
        self,
        query_info: pd.DataFrame,
        utility: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "PredictedLabelLookupRouter":
        if self.label_column not in query_info.columns:
            raise ValueError(f"Missing label column: {self.label_column}")
        aligned_embeddings = embeddings.loc[utility.index]
        labels = query_info.loc[utility.index, self.label_column].astype(str)
        self.classifier.fit(aligned_embeddings, labels)
        self.lookup.fit(query_info.loc[utility.index], utility)
        return self

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        labels = self.classifier.predict(embeddings)
        return pd.Series(labels.astype(str), index=embeddings.index, name=f"predicted_{self.label_column}")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        labels = self.predict_labels(embeddings.loc[query_info.index])
        predicted_info = query_info.copy()
        predicted_info[self.label_column] = labels
        return self.lookup.predict(predicted_info)
