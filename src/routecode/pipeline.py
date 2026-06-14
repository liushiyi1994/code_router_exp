from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from routecode.data.splits import split_by_query
from routecode.data.synthetic import SyntheticData, generate_synthetic_outcomes
from routecode.matrix import Matrices, build_matrices


@dataclass(frozen=True)
class PreparedSynthetic:
    config: dict[str, Any]
    data: SyntheticData
    outcomes: pd.DataFrame
    embeddings: pd.DataFrame
    matrices: dict[str, Matrices]


PreparedData = PreparedSynthetic


def prepare_from_config(config: dict[str, Any]) -> PreparedData:
    source = config.get("data", {}).get("source", "synthetic")
    if source == "synthetic":
        return prepare_synthetic(config)
    if source == "llmrouterbench":
        return prepare_llmrouterbench(config)
    raise ValueError(f"Unknown data source: {source}")


def prepare_synthetic(config: dict[str, Any]) -> PreparedSynthetic:
    data = generate_synthetic_outcomes(config)
    split_config = config.get("split", {})
    seed = int(config.get("run", {}).get("random_seed", 0))
    outcomes = split_by_query(
        data.outcomes,
        train_frac=float(split_config.get("train_frac", 0.6)),
        val_frac=float(split_config.get("val_frac", 0.2)),
        test_frac=float(split_config.get("test_frac", 0.2)),
        seed=seed,
    )
    lambda_cost = float(config.get("utility", {}).get("lambda_cost", 0.0))
    matrices = {
        split: build_matrices(outcomes[outcomes["split"] == split], lambda_cost=lambda_cost)
        for split in ["train", "val", "test"]
    }
    return PreparedSynthetic(
        config=config,
        data=data,
        outcomes=outcomes,
        embeddings=data.embeddings,
        matrices=matrices,
    )


def prepare_llmrouterbench(config: dict[str, Any]) -> PreparedData:
    from routecode.data.load_llmrouterbench import load_llmrouterbench_outcomes
    from routecode.data.text_features import build_hashing_embeddings

    data_config = config.get("data", {})
    cache_path = data_config.get("cache_path")
    if cache_path and Path(cache_path).exists():
        outcomes_raw = pd.read_csv(cache_path)
    else:
        outcomes_raw = load_llmrouterbench_outcomes(
            results_dir=data_config["results_dir"],
            datasets=data_config.get("datasets"),
            models=data_config.get("models"),
            splits=data_config.get("source_splits"),
            drop_incomplete_queries=bool(data_config.get("drop_incomplete_queries", True)),
        )
        if cache_path:
            cache = Path(cache_path)
            cache.parent.mkdir(parents=True, exist_ok=True)
            outcomes_raw.to_csv(cache, index=False)
    outcomes_raw = _apply_domain_map(outcomes_raw, data_config.get("domain_map"))
    outcomes_raw = _apply_task_taxonomy_map(outcomes_raw, data_config.get("task_taxonomy_map"))
    split_config = config.get("split", {})
    seed = int(config.get("run", {}).get("random_seed", 0))
    outcomes = split_by_query(
        outcomes_raw,
        train_frac=float(split_config.get("train_frac", 0.6)),
        val_frac=float(split_config.get("val_frac", 0.2)),
        test_frac=float(split_config.get("test_frac", 0.2)),
        seed=seed,
    )
    lambda_cost = float(config.get("utility", {}).get("lambda_cost", 0.0))
    matrices = {
        split: build_matrices(outcomes[outcomes["split"] == split], lambda_cost=lambda_cost)
        for split in ["train", "val", "test"]
    }
    all_query_info = outcomes.drop_duplicates("query_id").set_index("query_id")
    embeddings = build_hashing_embeddings(
        all_query_info,
        n_features=int(data_config.get("hashing_features", 128)),
    )
    return PreparedData(
        config=config,
        data=None,
        outcomes=outcomes,
        embeddings=embeddings,
        matrices=matrices,
    )


def _apply_domain_map(outcomes: pd.DataFrame, domain_map: dict[str, str] | None) -> pd.DataFrame:
    if not domain_map:
        return outcomes
    mapped = outcomes.copy()
    dataset_domains = mapped["dataset"].astype(str).map({str(k): str(v) for k, v in domain_map.items()})
    mapped["domain"] = dataset_domains.fillna(mapped["domain"].astype(str))
    return mapped


def _apply_task_taxonomy_map(outcomes: pd.DataFrame, taxonomy_map: dict[str, dict[str, str]] | None) -> pd.DataFrame:
    if not taxonomy_map:
        return outcomes
    mapped = outcomes.copy()
    normalized = {
        str(dataset): {str(column): str(value) for column, value in values.items()}
        for dataset, values in taxonomy_map.items()
    }
    taxonomy_columns = sorted({column for values in normalized.values() for column in values})
    dataset_names = mapped["dataset"].astype(str)
    for column in taxonomy_columns:
        values = dataset_names.map({dataset: labels.get(column, "unmapped") for dataset, labels in normalized.items()})
        if column in mapped.columns:
            mapped[column] = values.fillna(mapped[column].fillna("unmapped").astype(str))
        else:
            mapped[column] = values.fillna("unmapped")
    return mapped
