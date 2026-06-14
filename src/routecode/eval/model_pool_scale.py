from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelPoolScenario:
    family: str
    name: str
    models: list[str]
    stats: dict[str, Any]


def build_model_pool_scale_scenarios(
    utility: pd.DataFrame,
    sizes: list[int],
) -> list[ModelPoolScenario]:
    """Build deterministic model-pool scale/composition scenarios.

    `top` pools follow train mean utility. `complementary` pools greedily add
    models that increase oracle gap and winner diversity. `dominated` pools
    greedily add models that add the least routing value.
    """

    if utility.shape[1] < 2:
        raise ValueError("Model-pool scale scenarios require at least two models")
    normalized_sizes = sorted({min(max(2, int(size)), utility.shape[1]) for size in sizes})
    if utility.shape[1] not in normalized_sizes:
        normalized_sizes.append(utility.shape[1])

    mean_order = [str(model) for model in utility.mean(axis=0).sort_values(ascending=False).index]
    original_order = [str(model) for model in utility.columns]
    scenarios: list[ModelPoolScenario] = []
    for size in normalized_sizes:
        if size == utility.shape[1]:
            scenarios.append(_scenario("full", f"full_{size}", original_order, utility))
            continue
        scenarios.append(_scenario("top", f"top_{size}", mean_order[:size], utility))
        scenarios.append(
            _scenario(
                "complementary",
                f"complementary_{size}",
                _greedy_pool(utility, size, mode="complementary"),
                utility,
            )
        )
        scenarios.append(
            _scenario(
                "dominated",
                f"dominated_{size}",
                _greedy_pool(utility, size, mode="dominated"),
                utility,
            )
        )
    return _deduplicate_scenarios(scenarios)


def model_pool_stats(utility: pd.DataFrame, models: list[str]) -> dict[str, Any]:
    subset = utility.loc[:, models]
    means = subset.mean(axis=0).sort_values(ascending=False)
    best_single_model = str(means.index[0])
    best_single_utility = float(means.iloc[0])
    oracle = subset.max(axis=1)
    winners = subset.idxmax(axis=1).astype(str)
    winner_share = winners.value_counts(normalize=True)
    return {
        "best_single_model": best_single_model,
        "best_single_utility": best_single_utility,
        "oracle_mean": float(oracle.mean()),
        "oracle_gap": float(oracle.mean() - best_single_utility),
        "dominance_ratio": float(winner_share.max()) if not winner_share.empty else 0.0,
        "winner_entropy": _winner_entropy(winner_share),
    }


def _scenario(family: str, name: str, models: list[str], utility: pd.DataFrame) -> ModelPoolScenario:
    model_list = [str(model) for model in models]
    return ModelPoolScenario(
        family=family,
        name=name,
        models=model_list,
        stats=model_pool_stats(utility, model_list),
    )


def _greedy_pool(utility: pd.DataFrame, size: int, mode: str) -> list[str]:
    means = utility.mean(axis=0).sort_values(ascending=False)
    selected = [str(means.index[0])]
    remaining = [str(model) for model in means.index[1:]]
    while len(selected) < size and remaining:
        scored = []
        for model in remaining:
            candidate = selected + [model]
            stats = model_pool_stats(utility, candidate)
            score = (
                stats["oracle_gap"],
                stats["winner_entropy"],
                -stats["dominance_ratio"],
                stats["oracle_mean"],
                -float(means[model]),
                model,
            )
            scored.append((score, model))
        if mode == "complementary":
            chosen = max(scored, key=lambda item: item[0])[1]
        elif mode == "dominated":
            chosen = min(scored, key=lambda item: item[0])[1]
        else:
            raise ValueError(f"Unknown greedy pool mode: {mode}")
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def _deduplicate_scenarios(scenarios: list[ModelPoolScenario]) -> list[ModelPoolScenario]:
    unique: list[ModelPoolScenario] = []
    seen = set()
    for scenario in scenarios:
        key = (scenario.family, scenario.name, tuple(scenario.models))
        if key in seen:
            continue
        seen.add(key)
        unique.append(scenario)
    return unique


def _winner_entropy(winner_share: pd.Series) -> float:
    if winner_share.empty:
        return 0.0
    return float(-(winner_share * np.log2(winner_share)).sum())


def exhaustive_extreme_pools(
    utility: pd.DataFrame,
    size: int,
) -> tuple[ModelPoolScenario, ModelPoolScenario]:
    """Return exact complementary and dominated pools for a small pool size."""

    size = min(max(2, int(size)), utility.shape[1])
    scored = [_scenario("exhaustive", f"candidate_{idx}", list(models), utility) for idx, models in enumerate(combinations(utility.columns, size))]
    if not scored:
        raise ValueError("No model-pool combinations available")
    complementary = max(
        scored,
        key=lambda scenario: (
            scenario.stats["oracle_gap"],
            scenario.stats["winner_entropy"],
            -scenario.stats["dominance_ratio"],
            scenario.stats["oracle_mean"],
        ),
    )
    dominated = min(
        scored,
        key=lambda scenario: (
            scenario.stats["oracle_gap"],
            scenario.stats["winner_entropy"],
            -scenario.stats["dominance_ratio"],
            -scenario.stats["oracle_mean"],
        ),
    )
    return (
        ModelPoolScenario("complementary", f"complementary_exact_{size}", complementary.models, complementary.stats),
        ModelPoolScenario("dominated", f"dominated_exact_{size}", dominated.models, dominated.stats),
    )
