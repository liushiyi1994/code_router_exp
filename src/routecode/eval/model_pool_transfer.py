from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from routecode.eval.model_pool_scale import build_model_pool_scale_scenarios, model_pool_stats


@dataclass(frozen=True)
class ModelPoolTransferScenario:
    name: str
    source_family: str
    source_models: list[str]
    target_models: list[str]
    stats: dict[str, Any]


def build_model_pool_transfer_scenarios(
    utility: pd.DataFrame,
    source_size: int,
    target_size: int,
    source_sizes: list[int] | None = None,
    target_sizes: list[int] | None = None,
) -> list[ModelPoolTransferScenario]:
    """Build deterministic disjoint source/target pool transfer scenarios.

    Source pools are fit using train utility only. Target pools are selected
    from models not in the source pool, also using train utility only.
    """

    if utility.shape[1] < 4:
        raise ValueError("Model-pool transfer scenarios require at least four models")
    source_values = _normalized_source_sizes(utility, source_size, source_sizes)
    target_values = _normalized_target_sizes(target_size, target_sizes)
    mean_order = [str(model) for model in utility.mean(axis=0).sort_values(ascending=False).index]

    scenarios: list[ModelPoolTransferScenario] = []
    include_size_suffix = len(source_values) > 1 or len(target_values) > 1
    for source_n in source_values:
        source_candidates = build_model_pool_scale_scenarios(utility, [source_n])
        selected_sources = []
        wanted = {
            ("top", f"top_{source_n}"): "top_to_next",
            ("complementary", f"complementary_{source_n}"): "complementary_to_remaining_top",
            ("dominated", f"dominated_{source_n}"): "dominated_to_remaining_top",
        }
        for candidate in source_candidates:
            key = (candidate.family, candidate.name)
            if key in wanted:
                selected_sources.append((wanted[key], candidate.family, candidate.models))
        for target_n in target_values:
            if source_n + target_n > utility.shape[1]:
                continue
            for base_name, family, source_models in selected_sources:
                name = f"{base_name}_s{source_n}_t{target_n}" if include_size_suffix else base_name
                target_models = _top_remaining(mean_order, source_models, target_n)
                scenarios.append(_scenario(name, family, source_models, target_models, utility))
    if not scenarios:
        raise ValueError("No valid disjoint source/target model-pool transfer scenarios")
    return scenarios


def fit_label_to_target_model(
    train_labels: pd.Series,
    target_utility: pd.DataFrame,
    labels: list[int] | range | None = None,
) -> tuple[dict[int, str], str]:
    """Map transferred route labels to target-pool models using train utility."""

    aligned_labels = train_labels.loc[target_utility.index]
    fallback = str(target_utility.mean(axis=0).idxmax())
    label_values = labels if labels is not None else sorted(int(label) for label in aligned_labels.dropna().unique())
    mapping: dict[int, str] = {}
    for raw_label in label_values:
        label = int(raw_label)
        query_ids = aligned_labels.index[aligned_labels.astype(int) == label]
        if len(query_ids) == 0:
            mapping[label] = fallback
        else:
            mapping[label] = str(target_utility.loc[query_ids].mean(axis=0).idxmax())
    return mapping, fallback


def select_from_label_to_model(labels: pd.Series, mapping: dict[int, str], fallback: str) -> pd.Series:
    selected = [mapping.get(int(label), fallback) for label in labels]
    return pd.Series(selected, index=labels.index, name="selected_model")


def _top_remaining(mean_order: list[str], source_models: list[str], target_size: int) -> list[str]:
    source_set = set(source_models)
    remaining = [model for model in mean_order if model not in source_set]
    return remaining[:target_size]


def _normalized_source_sizes(
    utility: pd.DataFrame,
    source_size: int,
    source_sizes: list[int] | None,
) -> list[int]:
    raw_sizes = source_sizes if source_sizes is not None else [source_size]
    sizes = sorted({min(max(2, int(size)), utility.shape[1] - 2) for size in raw_sizes})
    return [size for size in sizes if size < utility.shape[1]]


def _normalized_target_sizes(target_size: int, target_sizes: list[int] | None) -> list[int]:
    raw_sizes = target_sizes if target_sizes is not None else [target_size]
    return sorted({max(1, int(size)) for size in raw_sizes})


def _scenario(
    name: str,
    source_family: str,
    source_models: list[str],
    target_models: list[str],
    utility: pd.DataFrame,
) -> ModelPoolTransferScenario:
    source_stats = model_pool_stats(utility, source_models)
    target_stats = model_pool_stats(utility, target_models)
    return ModelPoolTransferScenario(
        name=name,
        source_family=source_family,
        source_models=[str(model) for model in source_models],
        target_models=[str(model) for model in target_models],
        stats={
            "source_best_single_model": source_stats["best_single_model"],
            "source_oracle_gap": source_stats["oracle_gap"],
            "source_dominance_ratio": source_stats["dominance_ratio"],
            "source_winner_entropy": source_stats["winner_entropy"],
            "target_best_single_model": target_stats["best_single_model"],
            "target_oracle_gap": target_stats["oracle_gap"],
            "target_dominance_ratio": target_stats["dominance_ratio"],
            "target_winner_entropy": target_stats["winner_entropy"],
        },
    )
