from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def selected_values(matrix: pd.DataFrame, selected_models: pd.Series | dict[str, str]) -> pd.Series:
    selected = pd.Series(selected_models)
    values = []
    index = []
    for query_id, model_id in selected.items():
        values.append(float(matrix.loc[query_id, model_id]))
        index.append(query_id)
    return pd.Series(values, index=pd.Index(index, name=matrix.index.name))


def router_summary(
    selected_utility: Iterable[float],
    oracle_utility: Iterable[float],
    selected_quality: Iterable[float] | None = None,
    selected_cost: Iterable[float] | None = None,
    max_cost: float | None = None,
) -> dict[str, float]:
    selected_arr = np.asarray(list(selected_utility), dtype=float)
    oracle_arr = np.asarray(list(oracle_utility), dtype=float)
    summary = {
        "mean_utility": float(np.mean(selected_arr)),
        "oracle_regret": float(np.mean(oracle_arr - selected_arr)),
    }
    if selected_quality is not None:
        summary["mean_quality"] = float(np.mean(np.asarray(list(selected_quality), dtype=float)))
    if selected_cost is not None:
        costs = np.asarray(list(selected_cost), dtype=float)
        denominator = float(max_cost) if max_cost and max_cost > 0 else float(np.max(costs) or 1.0)
        summary["normalized_cost"] = float(np.mean(costs / denominator))
    return summary


def recovered_gap(method: float, baseline: float, reference: float) -> float:
    denominator = reference - baseline
    if denominator <= 1e-12:
        return 0.0
    return float((method - baseline) / denominator)


def model_win_entropy(winners: Iterable[str]) -> float:
    labels = list(winners)
    if not labels:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(probs)).sum())
    return 0.0 if abs(entropy) < 1e-12 else entropy


def empirical_entropy(labels: Iterable[int | str]) -> float:
    return model_win_entropy([str(label) for label in labels])


def dominance_ratio(winners: Iterable[str]) -> float:
    labels = list(winners)
    if not labels:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def bootstrap_mean_ci(
    values: Iterable[float],
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        raise ValueError("Cannot bootstrap an empty array")
    if len(arr) == 1 or n_bootstrap <= 1:
        mean = float(arr.mean())
        return mean, mean
    rng = np.random.default_rng(seed)
    samples = rng.choice(arr, size=(int(n_bootstrap), len(arr)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)
