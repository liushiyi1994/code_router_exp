from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from routecode.matrix import Matrices


@dataclass(frozen=True)
class StrongWeakPair:
    strong_model: str
    weak_model: str


@dataclass(frozen=True)
class RouteLLMMFAssets:
    train_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    prompt_index: dict[str, int]
    prompt_embeddings: np.ndarray


@dataclass(frozen=True)
class AvengersProAssets:
    train_records: list[dict[str, Any]]
    test_records: list[dict[str, Any]]
    baseline_scores: dict[str, dict[str, float]]


class AvengersProClusterRouter:
    """Local no-API implementation of the Avengers-Pro cluster-routing contract.

    Avengers-Pro's released router clusters query embeddings, learns per-cluster
    model rankings from train scores, and routes test queries through nearby
    clusters. This implementation uses RouteCode's deterministic embeddings and
    matrices so it can run split-aligned without the upstream embedding service.
    """

    def __init__(
        self,
        n_clusters: int,
        *,
        top_k: int = 1,
        beta: float = 9.0,
        mode: str = "simple",
        performance_weight: float = 0.7,
        cost_sensitivity: float = 0.3,
        min_quality_threshold: float = 0.0,
        random_state: int = 0,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.top_k = int(top_k)
        self.beta = float(beta)
        self.mode = str(mode)
        self.performance_weight = float(performance_weight)
        self.cost_sensitivity = float(cost_sensitivity)
        self.min_quality_threshold = float(min_quality_threshold)
        self.random_state = int(random_state)
        self.normalizer: Normalizer | None = None
        self.kmeans_model: KMeans | None = None
        self.cluster_centers: np.ndarray | None = None
        self.cluster_rankings: dict[int, dict[str, Any]] = {}
        self.available_models: list[str] = []
        self.fallback_model: str | None = None
        self.effective_clusters: int = 0

    def fit(
        self,
        query_info: pd.DataFrame,
        quality: pd.DataFrame,
        cost: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> "AvengersProClusterRouter":
        del query_info
        if self.mode not in {"simple", "balance"}:
            raise ValueError("AvengersProClusterRouter mode must be 'simple' or 'balance'")
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if quality.empty:
            raise ValueError("AvengersProClusterRouter requires non-empty train quality")
        self.available_models = sorted(map(str, quality.columns))
        self.fallback_model = str(quality.mean(axis=0).sort_values(ascending=False).index[0])
        aligned_embeddings = embeddings.loc[quality.index]
        self.normalizer = Normalizer(norm="l2")
        normalized = self.normalizer.fit_transform(aligned_embeddings.to_numpy(dtype=float))
        self.effective_clusters = max(1, min(self.n_clusters, len(normalized)))
        if self.effective_clusters == 1:
            labels = np.zeros(len(normalized), dtype=int)
            self.cluster_centers = np.asarray([normalized.mean(axis=0)])
            self.kmeans_model = None
        else:
            self.kmeans_model = KMeans(
                n_clusters=self.effective_clusters,
                random_state=self.random_state,
                n_init=10,
            )
            labels = self.kmeans_model.fit_predict(normalized)
            self.cluster_centers = self.kmeans_model.cluster_centers_
        self.cluster_rankings = self._compute_cluster_rankings(labels, quality, cost)
        return self

    def predict_labels(self, embeddings: pd.DataFrame) -> pd.Series:
        if self.cluster_centers is None or self.normalizer is None:
            raise RuntimeError("AvengersProClusterRouter must be fit before predict")
        normalized = self.normalizer.transform(embeddings.to_numpy(dtype=float))
        distances = 1.0 - normalized @ self.cluster_centers.T
        labels = distances.argmin(axis=1)
        return pd.Series(labels.astype(int), index=embeddings.index, name="avengerspro_cluster")

    def predict(self, query_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.Series:
        if self.cluster_centers is None or self.normalizer is None or self.fallback_model is None:
            raise RuntimeError("AvengersProClusterRouter must be fit before predict")
        aligned = embeddings.loc[query_info.index]
        normalized = self.normalizer.transform(aligned.to_numpy(dtype=float))
        distances = 1.0 - normalized @ self.cluster_centers.T
        selected = [self._select_model(row) for row in distances]
        return pd.Series(selected, index=query_info.index, name="selected_model")

    def _compute_cluster_rankings(
        self,
        labels: np.ndarray,
        quality: pd.DataFrame,
        cost: pd.DataFrame,
    ) -> dict[int, dict[str, Any]]:
        rankings: dict[int, dict[str, Any]] = {}
        for cluster_id in range(self.effective_clusters):
            query_ids = quality.index[labels == cluster_id]
            if len(query_ids) == 0:
                ranking = [self.fallback_model] if self.fallback_model is not None else []
                rankings[cluster_id] = {"total": 0, "scores": {}, "ranking": ranking}
                continue
            cluster_quality = quality.loc[query_ids, self.available_models].mean(axis=0)
            cluster_cost = cost.loc[query_ids, self.available_models].mean(axis=0)
            if self.mode == "simple":
                scores = {model: float(cluster_quality[model]) for model in self.available_models}
            else:
                scores = self._balance_scores(cluster_quality, cluster_cost)
            sorted_models = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            rankings[cluster_id] = {
                "total": int(len(query_ids)),
                "scores": dict(sorted_models),
                "ranking": [model for model, _ in sorted_models],
            }
        return rankings

    def _balance_scores(self, quality: pd.Series, cost: pd.Series) -> dict[str, float]:
        max_quality = float(quality.max())
        min_quality = float(quality.min())
        quality_range = max_quality - min_quality
        max_cost = float(cost.max())
        scores: dict[str, float] = {}
        for model in self.available_models:
            raw_quality = float(quality[model])
            if raw_quality < self.min_quality_threshold:
                scores[model] = 0.0
                continue
            if quality_range > 0:
                normalized_quality = (raw_quality - min_quality) / quality_range
            else:
                normalized_quality = 1.0
            if max_cost > 0:
                cost_score = 1.0 - (float(cost[model]) / max_cost)
            else:
                cost_score = 1.0
            scores[model] = (
                self.performance_weight * normalized_quality
                + self.cost_sensitivity * cost_score
            )
        return scores

    def _select_model(self, distances: np.ndarray) -> str:
        limit = min(self.top_k, len(distances))
        closest = np.argsort(distances)[:limit]
        closest_distances = distances[closest]
        logits = -self.beta * closest_distances
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        model_scores = {model: 0.0 for model in self.available_models}
        for cluster_id, prob in zip(closest, probs, strict=False):
            ranking = self.cluster_rankings.get(int(cluster_id), {}).get("ranking", [])
            for model in self.available_models:
                if model in ranking:
                    rank = ranking.index(model)
                    model_scores[model] += float(prob) / float(rank + 1)
        return sorted(model_scores.items(), key=lambda item: (-item[1], item[0]))[0][0]


def choose_strong_weak_pair(
    utility: pd.DataFrame,
    strong_model: str | None = None,
    weak_model: str | None = None,
) -> StrongWeakPair:
    if utility.empty or utility.shape[1] < 2:
        raise ValueError("A strong/weak pair requires at least two model columns")
    means = utility.mean(axis=0).sort_values(ascending=False)
    strong = str(strong_model) if strong_model else str(means.index[0])
    weak = str(weak_model) if weak_model else str(means.index[-1])
    missing = {strong, weak} - set(map(str, utility.columns))
    if missing:
        raise ValueError(f"Configured strong/weak models are missing: {sorted(missing)}")
    if strong == weak:
        raise ValueError("Strong and weak models must be different")
    return StrongWeakPair(strong_model=strong, weak_model=weak)


def build_routellm_pairwise_records(
    matrices_by_split: dict[str, Matrices],
    pair: StrongWeakPair,
    *,
    epsilon: float = 1e-12,
) -> dict[str, list[dict[str, Any]]]:
    """Convert RouteCode matrices into split-aligned RouteLLM-style pairwise rows.

    The output is a data substrate, not a fitted RouteLLM baseline. It keeps the
    RouteCode query split intact and records the strong/weak utility comparison
    needed by RouteLLM-style binary routers.
    """

    records: dict[str, list[dict[str, Any]]] = {}
    for split, matrices in matrices_by_split.items():
        _validate_pair_columns(matrices, pair, split)
        split_records: list[dict[str, Any]] = []
        for query_id in matrices.utility.index:
            strong_utility = float(matrices.utility.at[query_id, pair.strong_model])
            weak_utility = float(matrices.utility.at[query_id, pair.weak_model])
            if strong_utility > weak_utility + epsilon:
                winner = "model_a"
            elif weak_utility > strong_utility + epsilon:
                winner = "model_b"
            else:
                winner = "tie"
            query_info = matrices.query_info.loc[query_id].to_dict()
            split_records.append(
                {
                    "query_id": str(query_id),
                    "prompt": _prompt_from_query_info(query_info),
                    "split": str(split),
                    "model_a": pair.strong_model,
                    "model_b": pair.weak_model,
                    "winner": winner,
                    "utility_margin_model_a_minus_b": strong_utility - weak_utility,
                    "model_a_utility": strong_utility,
                    "model_b_utility": weak_utility,
                    "model_a_quality": float(matrices.quality.at[query_id, pair.strong_model]),
                    "model_b_quality": float(matrices.quality.at[query_id, pair.weak_model]),
                    "model_a_cost": float(matrices.cost.at[query_id, pair.strong_model]),
                    "model_b_cost": float(matrices.cost.at[query_id, pair.weak_model]),
                    "dataset": str(query_info.get("dataset", "")),
                    "domain": str(query_info.get("domain", "")),
                }
            )
        records[str(split)] = split_records
    return records


def build_routellm_mf_assets(
    pairwise_records: dict[str, list[dict[str, Any]]],
    embeddings: pd.DataFrame,
    *,
    train_split: str = "train",
    test_split: str = "test",
) -> RouteLLMMFAssets:
    """Build LLMRouterBench RouteLLM-MF trainer-compatible assets.

    The official MF trainer expects `idx`, `score_model_a`, `score_model_b`,
    `cost_model_a`, and `cost_model_b`. Its `winner` field follows quality
    score rather than RouteCode utility; the utility winner is retained in a
    separate field for later RouteCode metric evaluation.
    """

    ordered_records = list(pairwise_records.get(train_split, [])) + list(pairwise_records.get(test_split, []))
    query_ids = [str(row["query_id"]) for row in ordered_records]
    prompt_index = {query_id: idx for idx, query_id in enumerate(dict.fromkeys(query_ids))}
    missing_embeddings = [query_id for query_id in prompt_index if query_id not in embeddings.index]
    if missing_embeddings:
        examples = missing_embeddings[:5]
        raise ValueError(f"Missing embedding rows for RouteLLM MF assets: {examples}")
    prompt_embeddings = embeddings.loc[list(prompt_index)].to_numpy(dtype=float)

    train_records = [
        _mf_record(row, prompt_index)
        for row in pairwise_records.get(train_split, [])
        if _quality_winner(row) != "tie"
    ]
    test_records = [_mf_record(row, prompt_index) for row in pairwise_records.get(test_split, [])]
    return RouteLLMMFAssets(
        train_records=train_records,
        test_records=test_records,
        prompt_index=prompt_index,
        prompt_embeddings=prompt_embeddings,
    )


def build_avengerspro_records(
    matrices_by_split: dict[str, Matrices],
    *,
    train_split: str = "train",
    test_split: str = "test",
) -> AvengersProAssets:
    """Convert RouteCode matrices to the Avengers-Pro JSONL-style schema."""

    train_records = _avengerspro_split_records(matrices_by_split[train_split], train_split)
    test_records = _avengerspro_split_records(matrices_by_split[test_split], test_split)
    baseline_scores = _avengerspro_baseline_scores(matrices_by_split[test_split])
    return AvengersProAssets(
        train_records=train_records,
        test_records=test_records,
        baseline_scores=baseline_scores,
    )


def _avengerspro_split_records(matrices: Matrices, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, query_id in enumerate(matrices.quality.index):
        query_info = matrices.query_info.loc[query_id].to_dict()
        records.append(
            {
                "query_id": str(query_id),
                "query": _prompt_from_query_info(query_info),
                "dataset": str(query_info.get("dataset", "")),
                "domain": str(query_info.get("domain", "")),
                "split": split,
                "index": index,
                "records": {
                    str(model): float(matrices.quality.at[query_id, model])
                    for model in matrices.quality.columns
                },
                "utilities": {
                    str(model): float(matrices.utility.at[query_id, model])
                    for model in matrices.utility.columns
                },
                "usages": {
                    str(model): {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "cost": float(matrices.cost.at[query_id, model]),
                    }
                    for model in matrices.cost.columns
                },
            }
        )
    return records


def _avengerspro_baseline_scores(matrices: Matrices) -> dict[str, dict[str, float]]:
    baseline_scores: dict[str, dict[str, float]] = {}
    datasets = matrices.query_info["dataset"].astype(str)
    for model in matrices.quality.columns:
        per_dataset: dict[str, float] = {}
        for dataset in sorted(datasets.unique()):
            query_ids = datasets[datasets == dataset].index
            per_dataset[dataset] = round(float(matrices.quality.loc[query_ids, model].mean()) * 100.0, 2)
        baseline_scores[str(model)] = per_dataset
    return baseline_scores


def _mf_record(row: dict[str, Any], prompt_index: dict[str, int]) -> dict[str, Any]:
    query_id = str(row["query_id"])
    score_model_a = float(row["model_a_quality"])
    score_model_b = float(row["model_b_quality"])
    return {
        "idx": int(prompt_index[query_id]),
        "query_id": query_id,
        "dataset_id": str(row.get("dataset", "")),
        "dataset": str(row.get("dataset", "")),
        "domain": str(row.get("domain", "")),
        "prompt": str(row.get("prompt", "")),
        "origin_query": str(row.get("prompt", "")),
        "model_a": str(row["model_a"]),
        "model_b": str(row["model_b"]),
        "score_model_a": score_model_a,
        "score_model_b": score_model_b,
        "cost_model_a": float(row["model_a_cost"]),
        "cost_model_b": float(row["model_b_cost"]),
        "utility_model_a": float(row["model_a_utility"]),
        "utility_model_b": float(row["model_b_utility"]),
        "utility_margin_model_a_minus_b": float(row["utility_margin_model_a_minus_b"]),
        "utility_winner": str(row["winner"]),
        "winner": _quality_winner(row),
        "winner_objective": "quality",
        "is_tie": _quality_winner(row) == "tie",
    }


def _quality_winner(row: dict[str, Any], *, epsilon: float = 1e-12) -> str:
    score_model_a = float(row["model_a_quality"])
    score_model_b = float(row["model_b_quality"])
    if score_model_a > score_model_b + epsilon:
        return "model_a"
    if score_model_b > score_model_a + epsilon:
        return "model_b"
    return "tie"


def _validate_pair_columns(matrices: Matrices, pair: StrongWeakPair, split: str) -> None:
    missing = {pair.strong_model, pair.weak_model} - set(map(str, matrices.utility.columns))
    if missing:
        raise ValueError(f"RouteLLM pair models missing from {split} utility matrix: {sorted(missing)}")


def _prompt_from_query_info(query_info: dict[str, Any]) -> str:
    for column in ["prompt", "query_text", "query", "question", "instruction"]:
        value = query_info.get(column)
        if value is not None and not pd.isna(value):
            return str(value)
    return ""


def load_official_routellm_artifacts(results_dir: str | Path) -> pd.DataFrame:
    """Load upstream LLMRouterBench RouteLLM MF result artifacts.

    These rows are intentionally compatibility-tagged instead of converted into
    RouteCode utility metrics, because the upstream artifacts use their own
    pairwise data split, model pair, and cost accounting.
    """

    root = Path(results_dir)
    json_paths = sorted(root.glob("mf_results_seed*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No RouteLLM MF JSON artifacts found under {root}")

    accuracy_by_seed = _read_seed_metric_csv(root / "mf_selection_accuracy_by_seed.csv", percent=True)
    cost_by_seed = _read_seed_metric_csv(root / "mf_total_cost_by_seed.csv", percent=False)
    csv_sources = "; ".join(
        path.name
        for path in [
            root / "mf_selection_accuracy_by_seed.csv",
            root / "mf_total_cost_by_seed.csv",
        ]
        if path.exists()
    )

    rows: list[dict[str, Any]] = []
    for path in json_paths:
        seed = _seed_from_path(path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"RouteLLM artifact must be a JSON object: {path}")

        rows.append(
            _artifact_row(
                seed=seed,
                scope="overall",
                dataset="",
                metrics=payload,
                source_path=path,
                csv_sources=csv_sources,
                csv_selection_accuracy=accuracy_by_seed.get((seed, "sample_avg")),
                csv_total_cost=cost_by_seed.get((seed, "total_cost")),
            )
        )
        datasets = payload.get("datasets", {})
        if not isinstance(datasets, dict):
            raise ValueError(f"RouteLLM artifact has non-object datasets field: {path}")
        for dataset, metrics in sorted(datasets.items()):
            if not isinstance(metrics, dict):
                raise ValueError(f"RouteLLM dataset metrics must be an object: {path} {dataset}")
            rows.append(
                _artifact_row(
                    seed=seed,
                    scope="dataset",
                    dataset=str(dataset),
                    metrics=metrics,
                    source_path=path,
                    csv_sources=csv_sources,
                    csv_selection_accuracy=accuracy_by_seed.get((seed, str(dataset))),
                    csv_total_cost=cost_by_seed.get((seed, str(dataset))),
                )
            )

    return pd.DataFrame(rows).sort_values(["seed", "scope", "dataset"]).reset_index(drop=True)


def _artifact_row(
    seed: int,
    scope: str,
    dataset: str,
    metrics: dict[str, Any],
    source_path: Path,
    csv_sources: str,
    csv_selection_accuracy: float | None,
    csv_total_cost: float | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": "RouteLLM-MF",
        "baseline_family": "official_external_artifact",
        "source_repo": "LLMRouterBench/baselines/RouteLLM",
        "source_artifact": source_path.name,
        "source_path": str(source_path),
        "source_csv_artifacts": csv_sources,
        "seed": seed,
        "scope": scope,
        "dataset": dataset,
        "csv_selection_accuracy": csv_selection_accuracy,
        "csv_total_cost": csv_total_cost,
        "split_aligned_with_routecode": False,
        "routecode_metric_compatible": False,
        "compatibility_note": (
            "Official upstream aggregate artifact; not evaluated on the RouteCode "
            "train/test split or utility objective."
        ),
    }
    for key, value in metrics.items():
        if key == "datasets":
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            row[key] = value
    return row


def _read_seed_metric_csv(path: Path, percent: bool) -> dict[tuple[int, str], float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    if "seed" not in frame.columns:
        raise ValueError(f"Seed metric CSV missing seed column: {path}")
    values: dict[tuple[int, str], float] = {}
    for _, row in frame.iterrows():
        seed = int(row["seed"])
        for column in frame.columns:
            if column == "seed":
                continue
            value = row[column]
            if pd.isna(value):
                continue
            number = float(value)
            if percent:
                number = number / 100.0
            values[(seed, str(column))] = number
    return values


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot infer seed from RouteLLM artifact path: {path}")
    return int(match.group(1))
