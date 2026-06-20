from __future__ import annotations

from dataclasses import dataclass
import json
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from routecode.matrix import Matrices


@dataclass(frozen=True)
class GraphRouterAssets:
    router_data: pd.DataFrame
    llm_descriptions: dict[str, dict[str, Any]]
    llm_description_embeddings: np.ndarray
    config: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WrittenGraphRouterAssets:
    asset_dir: Path
    router_data_path: Path
    llm_description_path: Path
    llm_embedding_path: Path
    config_path: Path
    metadata_path: Path


def build_graphrouter_assets(
    matrices_by_split: Mapping[str, Matrices],
    embeddings: pd.DataFrame,
    *,
    seed: int = 0,
    split_ratio: tuple[float, float, float] = (0.7, 0.0, 0.3),
) -> GraphRouterAssets:
    """Build GraphRouter-compatible data files from RouteCode matrices.

    These are data-contract assets for the upstream GraphRouter command path.
    They preserve RouteCode's query split in a `routecode_split` column, but the
    unmodified upstream GraphRouter runner still performs its own train/test
    split from `split_ratio`.
    """

    ordered_splits = [split for split in ["train", "val", "test"] if split in matrices_by_split]
    ordered_splits.extend(split for split in matrices_by_split if split not in ordered_splits)
    if not ordered_splits:
        raise ValueError("GraphRouter assets require at least one split matrix")

    model_ids = [str(model) for model in matrices_by_split[ordered_splits[0]].model_ids]
    embedding_dim = int(embeddings.shape[1])
    task_embeddings = _task_embeddings(matrices_by_split, embeddings, ordered_splits, embedding_dim)

    missing_embeddings = _missing_query_embeddings(matrices_by_split, embeddings, ordered_splits)
    if missing_embeddings:
        raise ValueError(f"Missing embedding rows for GraphRouter assets: {missing_embeddings[:5]}")

    rows: list[dict[str, Any]] = []
    for split in ordered_splits:
        matrices = matrices_by_split[split]
        if [str(model) for model in matrices.model_ids] != model_ids:
            raise ValueError("GraphRouter assets require the same model order in every split")
        for query_id in matrices.quality.index:
            query_info = matrices.query_info.loc[query_id].to_dict()
            task_id = _task_id(query_info)
            task_description = _task_description(task_id, query_info)
            query_embedding = _json_vector(embeddings.loc[query_id].to_numpy(dtype=float))
            task_embedding = _json_vector(task_embeddings[task_id])
            for model_id in model_ids:
                rows.append(
                    {
                        "task_id": task_id,
                        "task_description": task_description,
                        "task_description_embedding": task_embedding,
                        "query": _prompt_from_query_info(query_info),
                        "query_embedding": query_embedding,
                        "ground_truth": "",
                        "metric": "quality",
                        "llm": model_id,
                        "effect": float(matrices.quality.at[query_id, model_id]),
                        "cost": float(matrices.cost.at[query_id, model_id]),
                        "cost_usd": float(matrices.cost.at[query_id, model_id]),
                        "query_id": str(query_id),
                        "routecode_split": str(split),
                        "routecode_utility": float(matrices.utility.at[query_id, model_id]),
                    }
                )

    router_data = pd.DataFrame(rows)
    router_data["cost"] = _normalize_cost_by_task(router_data)
    llm_descriptions = _llm_descriptions(matrices_by_split, ordered_splits, model_ids)
    llm_embeddings = _llm_description_embeddings(
        matrices_by_split,
        ordered_splits,
        model_ids,
        embedding_dim,
    )
    metadata = {
        "split_aligned_with_routecode": True,
        "routecode_metric_compatible": False,
        "official_graphrouter_result": False,
        "query_count": int(router_data["query_id"].nunique()),
        "row_count": int(len(router_data)),
        "model_count": int(len(model_ids)),
        "embedding_dim": embedding_dim,
        "routecode_splits": ordered_splits,
        "split_ratio_for_upstream_runner": list(split_ratio),
        "compatibility_note": (
            "Assets preserve RouteCode split labels, but unmodified upstream "
            "GraphRouter will split internally by task/query using split_ratio."
        ),
    }
    config = {
        "llm_description_path": "",
        "llm_embedding_path": "",
        "saved_router_data_path": "",
        "query_response_length": 512,
        "seed": int(seed),
        "wandb_key": "",
        "model_path": "",
        "train_epoch": 2000,
        "scenario": "Performance First",
        "llm_num": int(len(model_ids)),
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "train_mask_rate": 0.5,
        "batch_size": 32,
        "split_ratio": [float(value) for value in split_ratio],
        "embedding_dim": 8,
        "edge_dim": 3,
    }
    return GraphRouterAssets(
        router_data=router_data,
        llm_descriptions=llm_descriptions,
        llm_description_embeddings=llm_embeddings,
        config=config,
        metadata=metadata,
    )


def write_graphrouter_assets(assets: GraphRouterAssets, asset_dir: str | Path) -> WrittenGraphRouterAssets:
    root = Path(asset_dir)
    root.mkdir(parents=True, exist_ok=True)
    router_data_path = root / "router_data.csv"
    llm_description_path = root / "LLM_Descriptions.json"
    llm_embedding_path = root / "llm_description_embedding.pkl"
    config_path = root / "config.local.yaml"
    metadata_path = root / "metadata.json"
    model_path = root / "model_path/best_model.pth"

    assets.router_data.to_csv(router_data_path, index=False)
    llm_description_path.write_text(
        json.dumps(assets.llm_descriptions, indent=2),
        encoding="utf-8",
    )
    with llm_embedding_path.open("wb") as handle:
        pickle.dump(assets.llm_description_embeddings, handle)
    config = dict(assets.config)
    config.update(
        {
            "llm_description_path": str(llm_description_path),
            "llm_embedding_path": str(llm_embedding_path),
            "saved_router_data_path": str(router_data_path),
            "model_path": str(model_path),
        }
    )
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    metadata = dict(assets.metadata)
    metadata.update(
        {
            "router_data_path": str(router_data_path),
            "llm_description_path": str(llm_description_path),
            "llm_embedding_path": str(llm_embedding_path),
            "config_path": str(config_path),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return WrittenGraphRouterAssets(
        asset_dir=root,
        router_data_path=router_data_path,
        llm_description_path=llm_description_path,
        llm_embedding_path=llm_embedding_path,
        config_path=config_path,
        metadata_path=metadata_path,
    )


def summarize_graphrouter_assets(assets: GraphRouterAssets) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data = assets.router_data
    for split in list(assets.metadata["routecode_splits"]) + ["overall"]:
        frame = data if split == "overall" else data[data["routecode_split"] == split]
        rows.append(
            {
                "split": split,
                "query_count": int(frame["query_id"].nunique()),
                "row_count": int(len(frame)),
                "model_count": int(frame["llm"].nunique()),
                "task_count": int(frame["task_id"].nunique()),
                "mean_effect": float(frame["effect"].mean()) if len(frame) else 0.0,
                "mean_normalized_cost": float(frame["cost"].mean()) if len(frame) else 0.0,
                "mean_cost_usd": float(frame["cost_usd"].mean()) if len(frame) else 0.0,
                "split_aligned_with_routecode": bool(assets.metadata["split_aligned_with_routecode"]),
                "routecode_metric_compatible": bool(assets.metadata["routecode_metric_compatible"]),
                "official_graphrouter_result": bool(assets.metadata["official_graphrouter_result"]),
                "implementation_note": assets.metadata["compatibility_note"],
            }
        )
    return pd.DataFrame(rows)


def _missing_query_embeddings(
    matrices_by_split: Mapping[str, Matrices],
    embeddings: pd.DataFrame,
    ordered_splits: list[str],
) -> list[str]:
    missing: list[str] = []
    for split in ordered_splits:
        for query_id in matrices_by_split[split].quality.index:
            if query_id not in embeddings.index:
                missing.append(str(query_id))
    return missing


def _task_embeddings(
    matrices_by_split: Mapping[str, Matrices],
    embeddings: pd.DataFrame,
    ordered_splits: list[str],
    embedding_dim: int,
) -> dict[str, np.ndarray]:
    query_ids_by_task: dict[str, list[str]] = {}
    for split in ordered_splits:
        matrices = matrices_by_split[split]
        for query_id in matrices.query_info.index:
            query_info = matrices.query_info.loc[query_id].to_dict()
            query_ids_by_task.setdefault(_task_id(query_info), []).append(query_id)
    task_embeddings: dict[str, np.ndarray] = {}
    for task_id, query_ids in query_ids_by_task.items():
        available = [query_id for query_id in query_ids if query_id in embeddings.index]
        if available:
            task_embeddings[task_id] = embeddings.loc[available].to_numpy(dtype=float).mean(axis=0)
        else:
            task_embeddings[task_id] = np.zeros(embedding_dim, dtype=float)
    return task_embeddings


def _normalize_cost_by_task(router_data: pd.DataFrame) -> pd.Series:
    normalized = pd.Series(np.zeros(len(router_data), dtype=float), index=router_data.index)
    for _, group in router_data.groupby("task_id", sort=False):
        costs = group["cost"].astype(float)
        span = float(costs.max() - costs.min())
        if span <= 0.0:
            normalized.loc[group.index] = 0.0
        else:
            normalized.loc[group.index] = (costs - float(costs.min())) / span
    return normalized


def _llm_descriptions(
    matrices_by_split: Mapping[str, Matrices],
    ordered_splits: list[str],
    model_ids: list[str],
) -> dict[str, dict[str, Any]]:
    train = matrices_by_split.get("train", matrices_by_split[ordered_splits[0]])
    descriptions: dict[str, dict[str, Any]] = {}
    for model_id in model_ids:
        mean_quality = float(train.quality[model_id].mean())
        mean_cost = float(train.cost[model_id].mean())
        descriptions[model_id] = {
            "model": model_id,
            "feature": (
                f"RouteCode model {model_id}; train mean quality {mean_quality:.4f}; "
                f"train mean cost {mean_cost:.6f}."
            ),
            "train_mean_quality": mean_quality,
            "train_mean_cost": mean_cost,
        }
    return descriptions


def _llm_description_embeddings(
    matrices_by_split: Mapping[str, Matrices],
    ordered_splits: list[str],
    model_ids: list[str],
    embedding_dim: int,
) -> np.ndarray:
    train = matrices_by_split.get("train", matrices_by_split[ordered_splits[0]])
    datasets = train.query_info.get("dataset", pd.Series("", index=train.query_info.index)).astype(str)
    task_values = sorted(datasets.unique())
    vectors: list[np.ndarray] = []
    for model_id in model_ids:
        stats: list[float] = [
            float(train.quality[model_id].mean()),
            float(train.utility[model_id].mean()),
            float(train.cost[model_id].mean()),
            float(train.quality[model_id].std(ddof=0)),
        ]
        for task in task_values:
            query_ids = datasets[datasets == task].index
            stats.append(float(train.quality.loc[query_ids, model_id].mean()))
        vectors.append(_fit_vector(np.asarray(stats, dtype=float), embedding_dim))
    return np.vstack(vectors)


def _fit_vector(values: np.ndarray, dim: int) -> np.ndarray:
    if len(values) == dim:
        return values.astype(float)
    if len(values) > dim:
        folded = np.zeros(dim, dtype=float)
        for index, value in enumerate(values):
            folded[index % dim] += float(value)
        return folded
    padded = np.zeros(dim, dtype=float)
    padded[: len(values)] = values
    return padded


def _task_id(query_info: dict[str, Any]) -> str:
    for column in ["dataset", "task_family", "domain"]:
        value = query_info.get(column)
        if value is not None and not pd.isna(value) and str(value):
            return str(value)
    return "routecode_task"


def _task_description(task_id: str, query_info: dict[str, Any]) -> str:
    parts = [task_id]
    for column in ["domain", "task_family", "task_subtype"]:
        value = query_info.get(column)
        if value is not None and not pd.isna(value) and str(value):
            parts.append(str(value))
    return " | ".join(dict.fromkeys(parts))


def _prompt_from_query_info(query_info: dict[str, Any]) -> str:
    for column in ["prompt", "query_text", "query", "question", "instruction"]:
        value = query_info.get(column)
        if value is not None and not pd.isna(value):
            return str(value)
    return ""


def _json_vector(values: np.ndarray) -> str:
    vector = [float(value) for value in values.tolist()]
    return json.dumps([vector], separators=(",", ":"))
