from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


StateMethod = Literal["raw_kmeans", "relative_kmeans", "two_stage_relative_kmeans", "calibration_refined"]


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

    def transform_utility(self, utility: pd.DataFrame) -> pd.DataFrame:
        aligned = utility.loc[:, self.model_ids].astype(float)
        if self.method == "raw_kmeans":
            features = aligned.copy()
            features.columns = [f"raw::{col}" for col in aligned.columns]
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
            self.model = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=0.002,
                max_iter=self.max_iter,
                early_stopping=True,
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
) -> UtilityStateModel:
    matrix = utility.dropna(axis=0, how="any").astype(float).copy()
    if matrix.empty:
        raise ValueError("Cannot fit utility states on an empty matrix.")
    k = min(max(1, int(n_states)), len(matrix))
    model_ids = list(matrix.columns.astype(str))
    if method == "raw_kmeans":
        features = matrix.copy()
        features.columns = [f"raw::{col}" for col in matrix.columns]
    else:
        features = build_relative_utility_features(
            matrix,
            local_models=tuple(local_models),
            frontier_models=tuple(frontier_models),
            tau=float(tau),
        )
    scaler = StandardScaler()
    x = scaler.fit_transform(features.to_numpy(dtype=float))

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
        if method == "calibration_refined":
            labels = _calibration_refine_labels(
                x,
                matrix.to_numpy(dtype=float),
                labels,
                eta=float(calibration_eta),
                gamma=float(regret_gamma),
                steps=int(refine_steps),
                random_state=int(random_state),
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


def _feature_centroids(x: np.ndarray, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for label in sorted(labels.astype(str).unique()):
        rows.append(pd.Series(x[labels.astype(str).eq(label)].mean(axis=0), name=label))
    return pd.DataFrame(rows)
