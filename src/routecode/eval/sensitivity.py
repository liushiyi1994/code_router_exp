from __future__ import annotations

import numpy as np
import pandas as pd


def inject_label_noise(
    labels: pd.Series,
    choices: list[str],
    noise_rate: float,
    seed: int = 0,
) -> pd.Series:
    noisy = labels.astype(str).copy()
    if noisy.empty or noise_rate <= 0:
        return noisy
    unique_choices = [str(choice) for choice in choices]
    if len(unique_choices) < 2:
        return noisy
    rng = np.random.default_rng(int(seed))
    n_noisy = min(len(noisy), int(round(float(noise_rate) * len(noisy))))
    if n_noisy <= 0:
        return noisy
    selected_positions = rng.choice(np.arange(len(noisy)), size=n_noisy, replace=False)
    for position in selected_positions:
        current = str(noisy.iloc[position])
        alternatives = [choice for choice in unique_choices if choice != current]
        if alternatives:
            noisy.iloc[position] = str(rng.choice(alternatives))
    return noisy.rename(labels.name)


def misestimate_cost_utility(
    quality: pd.DataFrame,
    cost: pd.DataFrame,
    lambda_cost: float,
    cost_multiplier: float,
) -> pd.DataFrame:
    return quality - float(lambda_cost) * float(cost_multiplier) * cost


def query_length_buckets(query_info: pd.DataFrame, n_bins: int = 3) -> pd.Series:
    if "query_text" not in query_info.columns:
        lengths = pd.Series(0, index=query_info.index)
    else:
        lengths = query_info["query_text"].fillna("").astype(str).str.split().str.len()
    labels = _bucket_labels(int(n_bins))
    try:
        buckets = pd.qcut(lengths.rank(method="first"), q=len(labels), labels=labels)
    except ValueError:
        buckets = pd.Series(labels[0], index=query_info.index)
    return pd.Series(buckets.astype(str), index=query_info.index, name="query_length_bucket")


def _bucket_labels(n_bins: int) -> list[str]:
    if n_bins <= 1:
        return ["all"]
    if n_bins == 2:
        return ["short", "long"]
    if n_bins == 3:
        return ["short", "medium", "long"]
    return [f"bucket_{idx}" for idx in range(n_bins)]
