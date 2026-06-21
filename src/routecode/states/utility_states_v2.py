from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


StateMethod = Literal[
    "raw_kmeans",
    "relative_kmeans",
    "two_stage_relative_kmeans",
    "calibration_refined",
    "model_holdout_repaired",
]


@dataclass(frozen=True)
class UtilityStateModel:
    method: str
    n_states: int
    model_ids: list[str]
    labels: pd.Series
    feature_columns: list[str]
    feature_scaler: StandardScaler
    feature_centroids: pd.DataFrame
    state_utility: pd.DataFrame
    state_variance: pd.DataFrame
    label_to_model: dict[str, str]
    fallback_model: str
    local_models: tuple[str, ...]
    frontier_models: tuple[str, ...]
    tau: float
    coarse_models: dict[str, KMeans] | None = None
    coarse_allocations: dict[str, int] | None = None
    repair_summary: dict[str, float | int | str] | None = None

    def transform_utility(self, utility: pd.DataFrame) -> pd.DataFrame:
        aligned = utility.loc[:, self.model_ids].astype(float)
        if self.method == "raw_kmeans":
            features = aligned.copy()
            features.columns = [f"raw::{col}" for col in aligned.columns]
        elif self.method == "model_holdout_repaired":
            features = build_model_holdout_repair_features(
                aligned,
                local_models=self.local_models,
                frontier_models=self.frontier_models,
                tau=self.tau,
            )
        else:
            features = build_relative_utility_features(
                aligned,
                local_models=self.local_models,
                frontier_models=self.frontier_models,
                tau=self.tau,
            )
        return features.reindex(columns=self.feature_columns, fill_value=0.0)

    def predict_from_utility(self, utility: pd.DataFrame) -> pd.Series:
        features = self.transform_utility(utility)
        x = self.feature_scaler.transform(features.to_numpy(dtype=float))
        centroids = self.feature_centroids.to_numpy(dtype=float)
        distances = pairwise_distances(x, centroids, metric="sqeuclidean")
        labels = self.feature_centroids.index.to_numpy()[distances.argmin(axis=1)]
        return pd.Series(labels.astype(str), index=features.index, name="state_label")


@dataclass(frozen=True)
class StatePredictionResult:
    labels: pd.Series
    confidence: pd.Series
    probabilities: pd.DataFrame


class EmbeddingStatePredictor:
    """Deployable query-to-state predictor over query embeddings.

    This is intentionally light-weight: KNN is the no-training baseline and MLP
    is the first trained embedding head. Both expose calibrated-enough max
    probability scores so low-confidence queries can trigger active probing.
    """

    def __init__(
        self,
        kind: Literal["knn", "mlp"] = "knn",
        *,
        n_neighbors: int = 15,
        hidden_layer_sizes: tuple[int, ...] = (256, 128),
        max_iter: int = 500,
        random_state: int = 17,
    ) -> None:
        self.kind = kind
        self.n_neighbors = int(n_neighbors)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.scaler = StandardScaler()
        self.model: KNeighborsClassifier | MLPClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.label_encoder: LabelEncoder | None = None

    def fit(self, embeddings: pd.DataFrame, labels: pd.Series) -> "EmbeddingStatePredictor":
        aligned_labels = labels.reindex(embeddings.index).dropna().astype(str)
        aligned_embeddings = embeddings.reindex(aligned_labels.index)
        x = self.scaler.fit_transform(aligned_embeddings.to_numpy(dtype=float))
        self.label_encoder = LabelEncoder().fit(aligned_labels.to_numpy(dtype=str))
        y = self.label_encoder.transform(aligned_labels.to_numpy(dtype=str))
        if aligned_labels.nunique() == 1:
            self.classes_ = aligned_labels.unique()
            self.model = None
            return self
        if self.kind == "knn":
            neighbors = min(max(1, self.n_neighbors), len(aligned_embeddings))
            self.model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
        elif self.kind == "mlp":
            class_counts = aligned_labels.value_counts()
            use_early_stopping = len(aligned_labels) >= 20 and int(class_counts.min()) >= 3
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.002,
                max_iter=self.max_iter,
                early_stopping=use_early_stopping,
                n_iter_no_change=20,
                random_state=self.random_state,
            )
        else:
            raise ValueError(f"Unknown state predictor kind: {self.kind}")
        self.model.fit(x, y)
        self.classes_ = self.label_encoder.inverse_transform(np.asarray(self.model.classes_, dtype=int))
        return self

    def predict(self, embeddings: pd.DataFrame) -> StatePredictionResult:
        if self.classes_ is None:
            raise RuntimeError("EmbeddingStatePredictor must be fit before predict")
        if self.model is None:
            probabilities = np.ones((len(embeddings), 1), dtype=float)
            columns = self.classes_.astype(str).tolist()
        else:
            x = self.scaler.transform(embeddings.to_numpy(dtype=float))
            probabilities = self.model.predict_proba(x)
            if self.label_encoder is None:
                raise RuntimeError("Missing label encoder for state predictor")
            columns = self.label_encoder.inverse_transform(np.asarray(self.model.classes_, dtype=int)).astype(str).tolist()
        prob_df = pd.DataFrame(probabilities, index=embeddings.index, columns=columns)
        labels = prob_df.idxmax(axis=1).rename("predicted_state")
        confidence = prob_df.max(axis=1).rename("state_confidence")
        return StatePredictionResult(labels=labels, confidence=confidence, probabilities=prob_df)


class TorchEmbeddingStatePredictor:
    """PyTorch MLP query-to-state predictor over fixed query embeddings."""

    def __init__(
        self,
        *,
        hidden_layer_sizes: tuple[int, ...] = (512, 256),
        dropout: float = 0.10,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        epochs: int = 40,
        batch_size: int = 64,
        random_state: int = 17,
        device: str = "auto",
        routing_loss_weight: float = 0.0,
    ) -> None:
        self.hidden_layer_sizes = tuple(int(v) for v in hidden_layer_sizes)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.device = device
        self.routing_loss_weight = float(routing_loss_weight)
        self.scaler = StandardScaler()
        self.label_encoder: LabelEncoder | None = None
        self.classes_: np.ndarray | None = None
        self.model: Any | None = None
        self._torch_device: str | None = None

    def fit(
        self,
        embeddings: pd.DataFrame,
        labels: pd.Series,
        *,
        route_reward: pd.DataFrame | None = None,
    ) -> "TorchEmbeddingStatePredictor":
        torch = _require_torch()
        _seed_torch(torch, self.random_state)
        aligned_labels = labels.reindex(embeddings.index).dropna().astype(str)
        aligned_embeddings = embeddings.reindex(aligned_labels.index).astype(float)
        if aligned_embeddings.empty:
            raise ValueError("Cannot fit TorchEmbeddingStatePredictor on empty embeddings.")
        self.label_encoder = LabelEncoder().fit(aligned_labels.to_numpy(dtype=str))
        self.classes_ = self.label_encoder.classes_.astype(str)
        if aligned_labels.nunique() == 1:
            self.model = None
            return self

        x_np = self.scaler.fit_transform(aligned_embeddings.to_numpy(dtype=np.float32)).astype(np.float32)
        y_np = self.label_encoder.transform(aligned_labels.to_numpy(dtype=str)).astype(np.int64)
        reward_np = _aligned_route_reward(route_reward, aligned_embeddings.index, self.classes_) if route_reward is not None else None
        self._torch_device = _select_torch_device(self.device, torch)
        self.model = _TorchMLPHead(x_np.shape[1], self.hidden_layer_sizes, len(self.classes_), self.dropout).to(self._torch_device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        x = torch.tensor(x_np, dtype=torch.float32, device=self._torch_device)
        y = torch.tensor(y_np, dtype=torch.long, device=self._torch_device)
        reward = (
            torch.tensor(reward_np, dtype=torch.float32, device=self._torch_device)
            if reward_np is not None and self.routing_loss_weight > 0
            else None
        )
        n = len(x_np)
        batch = max(1, min(self.batch_size, n))
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        self.model.train()
        for _ in range(max(1, self.epochs)):
            order = torch.randperm(n, generator=generator, device="cpu").to(self._torch_device)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                logits = self.model(x[idx])
                loss = loss_fn(logits, y[idx])
                if reward is not None:
                    expected_reward = (torch.softmax(logits, dim=1) * reward[idx]).sum(dim=1).mean()
                    loss = loss - self.routing_loss_weight * expected_reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict(self, embeddings: pd.DataFrame) -> StatePredictionResult:
        if self.classes_ is None:
            raise RuntimeError("TorchEmbeddingStatePredictor must be fit before predict")
        if self.model is None:
            probabilities = np.ones((len(embeddings), 1), dtype=float)
            columns = self.classes_.astype(str).tolist()
        else:
            torch = _require_torch()
            device = self._torch_device or _select_torch_device(self.device, torch)
            x_np = self.scaler.transform(embeddings.astype(float).to_numpy(dtype=np.float32)).astype(np.float32)
            with torch.no_grad():
                logits = self.model(torch.tensor(x_np, dtype=torch.float32, device=device))
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            columns = self.classes_.astype(str).tolist()
        prob_df = pd.DataFrame(probabilities, index=embeddings.index, columns=columns)
        labels = prob_df.idxmax(axis=1).rename("predicted_state")
        confidence = prob_df.max(axis=1).rename("state_confidence")
        return StatePredictionResult(labels=labels, confidence=confidence, probabilities=prob_df)


class TextCNNStatePredictor:
    """Token CNN query-to-state predictor using a small trainable vocabulary."""

    def __init__(
        self,
        *,
        max_vocab: int = 8192,
        min_freq: int = 1,
        max_length: int = 192,
        embedding_dim: int = 128,
        channels: int = 128,
        kernel_sizes: tuple[int, ...] = (2, 3, 5, 7),
        dropout: float = 0.20,
        learning_rate: float = 2e-3,
        weight_decay: float = 1e-4,
        epochs: int = 30,
        batch_size: int = 64,
        random_state: int = 17,
        device: str = "auto",
        routing_loss_weight: float = 0.0,
    ) -> None:
        self.max_vocab = int(max_vocab)
        self.min_freq = int(min_freq)
        self.max_length = int(max_length)
        self.embedding_dim = int(embedding_dim)
        self.channels = int(channels)
        self.kernel_sizes = tuple(int(v) for v in kernel_sizes)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.random_state = int(random_state)
        self.device = device
        self.routing_loss_weight = float(routing_loss_weight)
        self.vocab_: dict[str, int] = {}
        self.label_encoder: LabelEncoder | None = None
        self.classes_: np.ndarray | None = None
        self.model: Any | None = None
        self._torch_device: str | None = None

    def fit(
        self,
        texts: pd.Series,
        labels: pd.Series,
        *,
        route_reward: pd.DataFrame | None = None,
    ) -> "TextCNNStatePredictor":
        torch = _require_torch()
        _seed_torch(torch, self.random_state)
        aligned_labels = labels.reindex(texts.index).dropna().astype(str)
        aligned_texts = texts.reindex(aligned_labels.index).fillna("").astype(str)
        if aligned_texts.empty:
            raise ValueError("Cannot fit TextCNNStatePredictor on empty texts.")
        self.vocab_ = _build_token_vocab(aligned_texts, max_vocab=self.max_vocab, min_freq=self.min_freq)
        self.label_encoder = LabelEncoder().fit(aligned_labels.to_numpy(dtype=str))
        self.classes_ = self.label_encoder.classes_.astype(str)
        if aligned_labels.nunique() == 1:
            self.model = None
            return self
        x_np = _texts_to_token_ids(aligned_texts, self.vocab_, max_length=self.max_length)
        y_np = self.label_encoder.transform(aligned_labels.to_numpy(dtype=str)).astype(np.int64)
        reward_np = _aligned_route_reward(route_reward, aligned_texts.index, self.classes_) if route_reward is not None else None
        self._torch_device = _select_torch_device(self.device, torch)
        self.model = _TextCNNHead(
            vocab_size=max(self.vocab_.values(), default=1) + 1,
            embedding_dim=self.embedding_dim,
            channels=self.channels,
            kernel_sizes=self.kernel_sizes,
            n_classes=len(self.classes_),
            dropout=self.dropout,
        ).to(self._torch_device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_fn = torch.nn.CrossEntropyLoss()
        x = torch.tensor(x_np, dtype=torch.long, device=self._torch_device)
        y = torch.tensor(y_np, dtype=torch.long, device=self._torch_device)
        reward = (
            torch.tensor(reward_np, dtype=torch.float32, device=self._torch_device)
            if reward_np is not None and self.routing_loss_weight > 0
            else None
        )
        n = len(x_np)
        batch = max(1, min(self.batch_size, n))
        generator = torch.Generator(device="cpu").manual_seed(self.random_state)
        self.model.train()
        for _ in range(max(1, self.epochs)):
            order = torch.randperm(n, generator=generator, device="cpu").to(self._torch_device)
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                logits = self.model(x[idx])
                loss = loss_fn(logits, y[idx])
                if reward is not None:
                    expected_reward = (torch.softmax(logits, dim=1) * reward[idx]).sum(dim=1).mean()
                    loss = loss - self.routing_loss_weight * expected_reward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.model.eval()
        return self

    def predict(self, texts: pd.Series) -> StatePredictionResult:
        if self.classes_ is None:
            raise RuntimeError("TextCNNStatePredictor must be fit before predict")
        if self.model is None:
            probabilities = np.ones((len(texts), 1), dtype=float)
            columns = self.classes_.astype(str).tolist()
        else:
            torch = _require_torch()
            device = self._torch_device or _select_torch_device(self.device, torch)
            x_np = _texts_to_token_ids(texts.fillna("").astype(str), self.vocab_, max_length=self.max_length)
            with torch.no_grad():
                logits = self.model(torch.tensor(x_np, dtype=torch.long, device=device))
                probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()
            columns = self.classes_.astype(str).tolist()
        prob_df = pd.DataFrame(probabilities, index=texts.index, columns=columns)
        labels = prob_df.idxmax(axis=1).rename("predicted_state")
        confidence = prob_df.max(axis=1).rename("state_confidence")
        return StatePredictionResult(labels=labels, confidence=confidence, probabilities=prob_df)


class FrozenTransformerStatePredictor:
    """Frozen encoder plus MLP state classifier for cached ModernBERT/DeBERTa-style models."""

    def __init__(
        self,
        model_name_or_path: str | None = None,
        *,
        tokenizer: Any | None = None,
        encoder: Any | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        max_length: int = 256,
        batch_size: int = 16,
        head_hidden_layer_sizes: tuple[int, ...] = (256, 128),
        epochs: int = 20,
        learning_rate: float = 2e-3,
        random_state: int = 17,
        device: str = "auto",
        routing_loss_weight: float = 0.0,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.head_hidden_layer_sizes = tuple(int(v) for v in head_hidden_layer_sizes)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.random_state = int(random_state)
        self.device = device
        self.routing_loss_weight = float(routing_loss_weight)
        self.head: TorchEmbeddingStatePredictor | None = None
        self._torch_device: str | None = None

    def fit(
        self,
        texts: pd.Series,
        labels: pd.Series,
        *,
        route_reward: pd.DataFrame | None = None,
    ) -> "FrozenTransformerStatePredictor":
        embeddings = self._encode(texts)
        self.head = TorchEmbeddingStatePredictor(
            hidden_layer_sizes=self.head_hidden_layer_sizes,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
            device=self.device,
            routing_loss_weight=self.routing_loss_weight,
        ).fit(embeddings, labels, route_reward=route_reward)
        return self

    def predict(self, texts: pd.Series) -> StatePredictionResult:
        if self.head is None:
            raise RuntimeError("FrozenTransformerStatePredictor must be fit before predict")
        return self.head.predict(self._encode(texts))

    def _encode(self, texts: pd.Series) -> pd.DataFrame:
        torch = _require_torch()
        tokenizer, encoder = self._load_transformer()
        self._torch_device = _select_torch_device(self.device, torch)
        if hasattr(encoder, "to"):
            encoder.to(self._torch_device)
        if hasattr(encoder, "eval"):
            encoder.eval()
        values: list[np.ndarray] = []
        text_values = texts.fillna("").astype(str).tolist()
        with torch.no_grad():
            for start in range(0, len(text_values), max(1, self.batch_size)):
                batch_texts = text_values[start : start + self.batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
                output = encoder(**encoded)
                hidden = output.last_hidden_state.to(self._torch_device)
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                values.append(pooled.detach().cpu().numpy())
        matrix = np.vstack(values).astype(np.float32) if values else np.empty((0, 0), dtype=np.float32)
        return pd.DataFrame(matrix, index=texts.index)

    def _load_transformer(self) -> tuple[Any, Any]:
        if self.tokenizer is not None and self.encoder is not None:
            return self.tokenizer, self.encoder
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path is required unless tokenizer and encoder are injected.")
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment dependent.
            raise ImportError("FrozenTransformerStatePredictor requires transformers.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        self.encoder = AutoModel.from_pretrained(
            self.model_name_or_path,
            local_files_only=self.local_files_only,
            trust_remote_code=self.trust_remote_code,
        )
        return self.tokenizer, self.encoder


def build_relative_utility_features(
    utility: pd.DataFrame,
    *,
    local_models: tuple[str, ...] | list[str] = (),
    frontier_models: tuple[str, ...] | list[str] = (),
    tau: float = 0.15,
) -> pd.DataFrame:
    """Build routing-pattern features from a query-model utility matrix.

    Raw utilities conflate difficulty with routing preference. These features
    emphasize relative model behavior: centered utility, regret, rank, soft
    preference, and scalar margins.
    """

    u = utility.astype(float).copy()
    model_ids = list(u.columns)
    centered = u.sub(u.mean(axis=1), axis=0)
    centered.columns = [f"centered::{col}" for col in centered.columns]

    best = u.max(axis=1)
    regret = u.rsub(best, axis=0)
    regret.columns = [f"regret::{col}" for col in regret.columns]

    ranks = u.rank(axis=1, method="average", ascending=False)
    ranks = (ranks - 1.0) / max(len(model_ids) - 1, 1)
    ranks.columns = [f"rank::{col}" for col in ranks.columns]

    soft = softmax_frame(u, tau=max(float(tau), 1e-6))
    soft.columns = [f"soft_pref::{col}" for col in soft.columns]

    sorted_values = np.sort(u.to_numpy(dtype=float), axis=1)[:, ::-1]
    second = sorted_values[:, 1] if len(model_ids) > 1 else sorted_values[:, 0]
    margin = sorted_values[:, 0] - second
    scalars = pd.DataFrame(
        {
            "margin::best": margin,
            "utility::best": sorted_values[:, 0],
            "utility::second": second,
            "utility::mean": u.mean(axis=1).to_numpy(dtype=float),
            "utility::std": u.std(axis=1).fillna(0.0).to_numpy(dtype=float),
        },
        index=u.index,
    )

    local_cols = [col for col in local_models if col in u.columns]
    frontier_cols = [col for col in frontier_models if col in u.columns]
    if local_cols:
        scalars["utility::best_local"] = u[local_cols].max(axis=1)
    else:
        scalars["utility::best_local"] = np.nan
    if frontier_cols:
        scalars["utility::best_frontier"] = u[frontier_cols].max(axis=1)
    else:
        scalars["utility::best_frontier"] = np.nan
    scalars["margin::frontier_advantage"] = (
        scalars["utility::best_frontier"] - scalars["utility::best_local"]
    ).fillna(0.0)
    scalars["margin::local_advantage"] = (
        scalars["utility::best_local"] - scalars["utility::best_frontier"]
    ).fillna(0.0)
    scalars = scalars.fillna(0.0)
    return pd.concat([centered, regret, ranks, soft, scalars], axis=1)


def build_model_holdout_repair_features(
    utility: pd.DataFrame,
    *,
    local_models: tuple[str, ...] | list[str] = (),
    frontier_models: tuple[str, ...] | list[str] = (),
    tau: float = 0.15,
) -> pd.DataFrame:
    """Build features for calibration-aware model-holdout state repair.

    Relative routing features keep states aligned with model-selection behavior.
    Raw and centered utility columns give the repair step direct evidence about
    within-state calibration variance for every model dimension.
    """

    u = utility.astype(float).copy()
    relative = build_relative_utility_features(
        u,
        local_models=local_models,
        frontier_models=frontier_models,
        tau=float(tau),
    )
    raw = u.copy()
    raw.columns = [f"holdout_raw::{col}" for col in raw.columns]
    centered = u.sub(u.mean(axis=1), axis=0)
    centered.columns = [f"holdout_centered::{col}" for col in centered.columns]
    return pd.concat([relative, raw, centered], axis=1)


def fit_utility_state_model(
    utility: pd.DataFrame,
    *,
    method: StateMethod = "relative_kmeans",
    n_states: int = 16,
    random_state: int = 17,
    local_models: tuple[str, ...] | list[str] = (),
    frontier_models: tuple[str, ...] | list[str] = (),
    tau: float = 0.15,
    calibration_eta: float = 0.25,
    regret_gamma: float = 0.25,
    refine_steps: int = 8,
    model_holdout_variance_threshold: float | None = None,
    model_holdout_error_threshold: float | None = None,
    model_holdout_min_state_size: int = 6,
    model_holdout_max_split_fraction: float = 0.50,
    model_holdout_preserve_state_budget: bool = True,
) -> UtilityStateModel:
    matrix = utility.dropna(axis=0, how="any").astype(float).copy()
    if matrix.empty:
        raise ValueError("Cannot fit utility states on an empty matrix.")
    k = min(max(1, int(n_states)), len(matrix))
    model_ids = list(matrix.columns.astype(str))
    if method == "raw_kmeans":
        features = matrix.copy()
        features.columns = [f"raw::{col}" for col in matrix.columns]
    elif method == "model_holdout_repaired":
        features = build_model_holdout_repair_features(
            matrix,
            local_models=tuple(local_models),
            frontier_models=tuple(frontier_models),
            tau=float(tau),
        )
    else:
        features = build_relative_utility_features(
            matrix,
            local_models=tuple(local_models),
            frontier_models=tuple(frontier_models),
            tau=float(tau),
        )
    scaler = StandardScaler()
    x = scaler.fit_transform(features.to_numpy(dtype=float))

    repair_summary: dict[str, float | int | str] | None = None

    if method == "two_stage_relative_kmeans":
        labels, coarse_models, coarse_allocations = _two_stage_labels(
            matrix,
            x,
            n_states=k,
            random_state=int(random_state),
            local_models=tuple(local_models),
            frontier_models=tuple(frontier_models),
        )
    else:
        labels = KMeans(n_clusters=k, random_state=int(random_state), n_init=30).fit_predict(x)
        coarse_models = None
        coarse_allocations = None
        if method in {"calibration_refined", "model_holdout_repaired"}:
            labels = _calibration_refine_labels(
                x,
                matrix.to_numpy(dtype=float),
                labels,
                eta=float(calibration_eta),
                gamma=float(regret_gamma),
                steps=int(refine_steps),
                random_state=int(random_state),
            )
        if method == "model_holdout_repaired":
            labels, repair_summary = _model_holdout_repair_labels(
                x=x,
                utility=matrix,
                labels=labels,
                target_states=k if bool(model_holdout_preserve_state_budget) else len(matrix),
                random_state=int(random_state),
                variance_threshold=model_holdout_variance_threshold,
                error_threshold=model_holdout_error_threshold,
                min_state_size=int(model_holdout_min_state_size),
                max_split_fraction=float(model_holdout_max_split_fraction),
            )

    state_labels = pd.Series([f"z{int(label):02d}" for label in labels], index=matrix.index, name="state_label")
    state_utility, state_variance = state_tables(matrix, state_labels)
    label_to_model = state_utility.idxmax(axis=1).astype(str).to_dict()
    fallback_model = str(matrix.mean(axis=0).sort_values(ascending=False).index[0])
    centroids = _feature_centroids(x, state_labels)
    return UtilityStateModel(
        method=str(method),
        n_states=int(state_labels.nunique()),
        model_ids=model_ids,
        labels=state_labels,
        feature_columns=list(features.columns),
        feature_scaler=scaler,
        feature_centroids=centroids,
        state_utility=state_utility,
        state_variance=state_variance,
        label_to_model=label_to_model,
        fallback_model=fallback_model,
        local_models=tuple(local_models),
        frontier_models=tuple(frontier_models),
        tau=float(tau),
        coarse_models=coarse_models,
        coarse_allocations=coarse_allocations,
        repair_summary=repair_summary,
    )


def state_tables(utility: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned = utility.reindex(labels.index).copy()
    grouped = aligned.groupby(labels.astype(str))
    mean = grouped.mean().sort_index()
    variance = grouped.var().fillna(0.0).reindex(mean.index)
    return mean, variance


def state_policy(labels: pd.Series, label_to_model: dict[str, str], fallback_model: str) -> pd.Series:
    selected = labels.astype(str).map(label_to_model).fillna(fallback_model)
    return selected.rename("selected_model")


def confidence_trigger_mask(confidence: pd.Series, threshold: float) -> pd.Series:
    return confidence.astype(float).lt(float(threshold)).rename("needs_active_probe")


def select_confidence_threshold(
    confidence: pd.Series,
    predicted_labels: pd.Series,
    true_labels: pd.Series,
    *,
    max_probe_rate: float = 0.30,
) -> float:
    aligned = pd.concat(
        [
            confidence.rename("confidence").astype(float),
            predicted_labels.rename("predicted").astype(str),
            true_labels.rename("true").astype(str),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return 1.0
    best_threshold = 0.0
    best_covered_accuracy = -1.0
    for threshold in np.linspace(0.0, 1.0, 101):
        probe = aligned["confidence"] < threshold
        probe_rate = float(probe.mean())
        if probe_rate > max_probe_rate + 1e-12:
            continue
        covered = aligned[~probe]
        accuracy = float((covered["predicted"] == covered["true"]).mean()) if not covered.empty else 1.0
        if accuracy > best_covered_accuracy:
            best_covered_accuracy = accuracy
            best_threshold = float(threshold)
    return best_threshold


def softmax_frame(utility: pd.DataFrame, *, tau: float) -> pd.DataFrame:
    values = utility.to_numpy(dtype=float) / max(float(tau), 1e-12)
    values = values - values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    probs = exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)
    return pd.DataFrame(probs, index=utility.index, columns=utility.columns)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError("This predictor requires the optional `torch` dependency.") from exc
    return torch


def _select_torch_device(device: str, torch_module) -> str:
    requested = str(device).lower()
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return requested


def _seed_torch(torch_module, seed: int) -> None:
    torch_module.manual_seed(int(seed))
    if torch_module.cuda.is_available():  # pragma: no cover - hardware dependent.
        torch_module.cuda.manual_seed_all(int(seed))


def _TorchMLPHead(input_dim: int, hidden_layer_sizes: tuple[int, ...], n_classes: int, dropout: float):
    torch = _require_torch()
    layers: list[Any] = []
    previous = int(input_dim)
    for hidden in hidden_layer_sizes:
        layers.append(torch.nn.Linear(previous, int(hidden)))
        layers.append(torch.nn.GELU())
        if dropout > 0:
            layers.append(torch.nn.Dropout(float(dropout)))
        previous = int(hidden)
    layers.append(torch.nn.Linear(previous, int(n_classes)))
    return torch.nn.Sequential(*layers)


def _TextCNNHead(
    *,
    vocab_size: int,
    embedding_dim: int,
    channels: int,
    kernel_sizes: tuple[int, ...],
    n_classes: int,
    dropout: float,
):
    torch = _require_torch()

    class _Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(int(vocab_size), int(embedding_dim), padding_idx=0)
            self.convs = torch.nn.ModuleList(
                [
                    torch.nn.Conv1d(
                        in_channels=int(embedding_dim),
                        out_channels=int(channels),
                        kernel_size=int(kernel),
                        padding=int(kernel) // 2,
                    )
                    for kernel in kernel_sizes
                ]
            )
            self.dropout = torch.nn.Dropout(float(dropout))
            self.output = torch.nn.Linear(int(channels) * len(kernel_sizes), int(n_classes))

        def forward(self, input_ids):
            embedded = self.embedding(input_ids).transpose(1, 2)
            pooled = []
            for conv in self.convs:
                activated = torch.nn.functional.gelu(conv(embedded))
                pooled.append(torch.nn.functional.adaptive_max_pool1d(activated, 1).squeeze(-1))
            features = self.dropout(torch.cat(pooled, dim=1))
            return self.output(features)

    return _Module()


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+(?:\.\d+)?|[^\s]", str(text).lower())


def _build_token_vocab(texts: pd.Series, *, max_vocab: int, min_freq: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        counts.update(_tokenize_text(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common(max(0, int(max_vocab) - 2)):
        if int(count) < int(min_freq):
            continue
        vocab[token] = len(vocab)
    return vocab


def _texts_to_token_ids(texts: pd.Series, vocab: dict[str, int], *, max_length: int) -> np.ndarray:
    length = max(1, int(max_length))
    unk = int(vocab.get("<unk>", 1))
    rows = np.zeros((len(texts), length), dtype=np.int64)
    for row_idx, text in enumerate(texts.fillna("").astype(str)):
        token_ids = [int(vocab.get(token, unk)) for token in _tokenize_text(text)[:length]]
        if token_ids:
            rows[row_idx, : len(token_ids)] = np.asarray(token_ids, dtype=np.int64)
    return rows


def _aligned_route_reward(route_reward: pd.DataFrame | None, index: pd.Index, classes: np.ndarray) -> np.ndarray | None:
    if route_reward is None:
        return None
    aligned = route_reward.reindex(index).reindex(columns=classes.astype(str), fill_value=0.0).fillna(0.0)
    return aligned.to_numpy(dtype=np.float32)


def _two_stage_labels(
    utility: pd.DataFrame,
    x: np.ndarray,
    *,
    n_states: int,
    random_state: int,
    local_models: tuple[str, ...],
    frontier_models: tuple[str, ...],
) -> tuple[np.ndarray, dict[str, KMeans], dict[str, int]]:
    coarse = assign_coarse_regime(utility, local_models=local_models, frontier_models=frontier_models)
    allocations = _allocate_states(coarse, n_states=n_states)
    labels = np.empty(len(utility), dtype=int)
    models: dict[str, KMeans] = {}
    next_label = 0
    for coarse_label, group_indices in coarse.groupby(coarse).groups.items():
        positions = np.asarray(group_indices, dtype=object)
        row_positions = np.array([utility.index.get_loc(pos) for pos in positions], dtype=int)
        group_k = min(allocations[str(coarse_label)], len(row_positions))
        if group_k <= 1:
            labels[row_positions] = next_label
            next_label += 1
            continue
        model = KMeans(n_clusters=group_k, random_state=random_state, n_init=20)
        group_labels = model.fit_predict(x[row_positions])
        models[str(coarse_label)] = model
        for local_label in sorted(np.unique(group_labels)):
            labels[row_positions[group_labels == local_label]] = next_label
            next_label += 1
    return labels, models, allocations


def assign_coarse_regime(
    utility: pd.DataFrame,
    *,
    local_models: tuple[str, ...] | list[str] = (),
    frontier_models: tuple[str, ...] | list[str] = (),
    margin_threshold: float | None = None,
) -> pd.Series:
    u = utility.astype(float)
    sorted_values = np.sort(u.to_numpy(dtype=float), axis=1)[:, ::-1]
    margins = sorted_values[:, 0] - (sorted_values[:, 1] if u.shape[1] > 1 else sorted_values[:, 0])
    threshold = float(np.quantile(margins, 0.25)) if margin_threshold is None else float(margin_threshold)
    local_cols = [col for col in local_models if col in u.columns]
    frontier_cols = [col for col in frontier_models if col in u.columns]
    best_model = u.idxmax(axis=1).astype(str)
    labels = []
    for i, query_id in enumerate(u.index):
        if margins[i] <= threshold:
            labels.append("ambiguous")
            continue
        model = str(best_model.loc[query_id])
        if model in frontier_cols:
            labels.append("frontier_needed")
        elif model in local_cols:
            labels.append("local_enough")
        else:
            labels.append(f"best::{model}")
    return pd.Series(labels, index=u.index, name="coarse_regime")


def _allocate_states(coarse: pd.Series, *, n_states: int) -> dict[str, int]:
    counts = coarse.astype(str).value_counts()
    total = float(counts.sum())
    raw = counts / total * int(n_states)
    allocation = {str(label): max(1, int(np.floor(value))) for label, value in raw.items()}
    while sum(allocation.values()) < int(n_states):
        residuals = (raw - pd.Series(allocation)).sort_values(ascending=False)
        allocation[str(residuals.index[0])] += 1
    while sum(allocation.values()) > int(n_states):
        candidates = pd.Series(allocation).sort_values(ascending=False)
        for label, value in candidates.items():
            if value > 1:
                allocation[str(label)] -= 1
                break
        else:
            break
    return allocation


def _calibration_refine_labels(
    x: np.ndarray,
    utility: np.ndarray,
    labels: np.ndarray,
    *,
    eta: float,
    gamma: float,
    steps: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    current = labels.astype(int).copy()
    all_labels = np.unique(current)
    for _ in range(max(0, steps)):
        feature_centroids = np.vstack([x[current == label].mean(axis=0) for label in all_labels])
        utility_means = np.vstack([utility[current == label].mean(axis=0) for label in all_labels])
        feature_dist = pairwise_distances(x, feature_centroids, metric="sqeuclidean")
        utility_dist = pairwise_distances(utility, utility_means, metric="sqeuclidean")
        best_model_by_state = utility_means.argmax(axis=1)
        oracle = utility.max(axis=1, keepdims=True)
        state_regret = np.column_stack(
            [oracle[:, 0] - utility[:, int(model_idx)] for model_idx in best_model_by_state]
        )
        objective = feature_dist + eta * utility_dist + gamma * state_regret
        new = all_labels[objective.argmin(axis=1)]
        for label in all_labels:
            if np.any(new == label):
                continue
            donor = int(rng.integers(0, len(new)))
            new[donor] = label
        if np.array_equal(new, current):
            break
        current = new.astype(int)
    return current


def _model_holdout_repair_labels(
    *,
    x: np.ndarray,
    utility: pd.DataFrame,
    labels: np.ndarray,
    target_states: int,
    random_state: int,
    variance_threshold: float | None,
    error_threshold: float | None,
    min_state_size: int,
    max_split_fraction: float,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    current = _densify_labels(labels)
    target = max(1, int(target_states))
    diagnostics = _state_holdout_diagnostics(utility, current)
    if diagnostics.empty:
        return current, {
            "initial_states": int(len(np.unique(current))),
            "final_states": int(len(np.unique(current))),
            "states_split": 0,
            "states_merged": 0,
            "mean_holdout_variance_before": 0.0,
            "mean_holdout_variance_after": 0.0,
            "max_holdout_variance_before": 0.0,
            "max_holdout_variance_after": 0.0,
        }

    variance_cutoff = (
        float(variance_threshold)
        if variance_threshold is not None
        else max(float(diagnostics["holdout_variance"].median()), 1e-12)
    )
    error_cutoff = (
        float(error_threshold)
        if error_threshold is not None
        else max(float(diagnostics["holdout_abs_error"].median()), 1e-12)
    )
    min_size = max(2, int(min_state_size))
    max_splits = max(1, int(np.ceil(len(np.unique(current)) * max(0.0, float(max_split_fraction)))))
    risky = diagnostics[
        (diagnostics["state_size"] >= min_size)
        & (
            diagnostics["holdout_variance"].gt(variance_cutoff)
            | diagnostics["holdout_abs_error"].gt(error_cutoff)
        )
    ].sort_values(["holdout_variance", "holdout_abs_error"], ascending=False)

    split_count = 0
    next_label = int(current.max()) + 1 if len(current) else 0
    utility_values = utility.to_numpy(dtype=float)
    for row in risky.head(max_splits).itertuples(index=False):
        state = int(row.state)
        positions = np.flatnonzero(current == state)
        if len(positions) < min_size:
            continue
        split_features = _holdout_split_features(x, utility_values, positions)
        if split_features.shape[0] < 2:
            continue
        local_labels = KMeans(n_clusters=2, random_state=random_state + split_count + state, n_init=20).fit_predict(split_features)
        if len(np.unique(local_labels)) < 2:
            continue
        current[positions[local_labels == 1]] = next_label
        next_label += 1
        split_count += 1

    current, merge_count = _merge_repaired_states_to_budget(utility_values, current, target_states=target)
    current = _densify_labels(current)
    after = _state_holdout_diagnostics(utility, current)
    return current, {
        "initial_states": int(len(np.unique(labels))),
        "final_states": int(len(np.unique(current))),
        "states_split": int(split_count),
        "states_merged": int(merge_count),
        "mean_holdout_variance_before": float(diagnostics["holdout_variance"].mean()),
        "mean_holdout_variance_after": float(after["holdout_variance"].mean()) if not after.empty else 0.0,
        "max_holdout_variance_before": float(diagnostics["holdout_variance"].max()),
        "max_holdout_variance_after": float(after["holdout_variance"].max()) if not after.empty else 0.0,
        "variance_threshold": float(variance_cutoff),
        "error_threshold": float(error_cutoff),
    }


def _state_holdout_diagnostics(utility: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    values = utility.to_numpy(dtype=float)
    rows: list[dict[str, float | int]] = []
    for state in sorted(np.unique(labels)):
        positions = np.flatnonzero(labels == state)
        if len(positions) == 0:
            continue
        sub = values[positions]
        state_means = sub.mean(axis=0)
        state_vars = sub.var(axis=0, ddof=1) if len(positions) > 1 else np.zeros(sub.shape[1], dtype=float)
        abs_errors = np.abs(sub - state_means.reshape(1, -1)).mean(axis=0)
        best_model = int(state_means.argmax())
        rows.append(
            {
                "state": int(state),
                "state_size": int(len(positions)),
                "holdout_variance": float(np.mean(state_vars)),
                "holdout_abs_error": float(np.mean(abs_errors)),
                "selected_model_variance": float(state_vars[best_model]),
                "selected_model_abs_error": float(abs_errors[best_model]),
            }
        )
    return pd.DataFrame(rows)


def _holdout_split_features(x: np.ndarray, utility_values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    sub = utility_values[positions]
    centered = sub - sub.mean(axis=1, keepdims=True)
    state_mean = sub.mean(axis=0, keepdims=True)
    abs_dev = np.abs(sub - state_mean)
    return np.hstack([x[positions], centered, abs_dev])


def _merge_repaired_states_to_budget(
    utility_values: np.ndarray,
    labels: np.ndarray,
    *,
    target_states: int,
) -> tuple[np.ndarray, int]:
    current = labels.copy()
    merges = 0
    while len(np.unique(current)) > int(target_states):
        states = sorted(np.unique(current))
        means = {state: utility_values[current == state].mean(axis=0) for state in states}
        sizes = {state: int(np.sum(current == state)) for state in states}
        best_pair: tuple[int, int] | None = None
        best_score = float("inf")
        for i, left in enumerate(states):
            for right in states[i + 1 :]:
                merged_positions = (current == left) | (current == right)
                merged = utility_values[merged_positions]
                merged_var = float(np.mean(merged.var(axis=0, ddof=1))) if len(merged) > 1 else 0.0
                distance = float(np.linalg.norm(means[left] - means[right]))
                same_best = int(means[left].argmax()) == int(means[right].argmax())
                small_group_bonus = 1.0 / max(1, min(sizes[left], sizes[right]))
                score = merged_var + 0.25 * distance - (0.05 if same_best else 0.0) - 0.01 * small_group_bonus
                if score < best_score:
                    best_score = score
                    best_pair = (int(left), int(right))
        if best_pair is None:
            break
        keep, remove = best_pair
        current[current == remove] = keep
        current = _densify_labels(current)
        merges += 1
    return current, merges


def _densify_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(np.unique(labels.astype(int)))
    mapping = {old: new for new, old in enumerate(unique)}
    return np.asarray([mapping[int(label)] for label in labels], dtype=int)


def _feature_centroids(x: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for label in sorted(labels.astype(str).unique()):
        rows.append(pd.Series(x[labels.astype(str).eq(label)].mean(axis=0), name=label))
    return pd.DataFrame(rows)
