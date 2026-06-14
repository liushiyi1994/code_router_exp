from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from routecode.matrix import Matrices
from routecode.metrics import (
    bootstrap_mean_ci,
    empirical_entropy,
    model_win_entropy,
    recovered_gap,
    router_summary,
    selected_values,
)


def evaluate_selection(
    method: str,
    selected_models: pd.Series,
    matrices: Matrices,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    k: int | None = None,
    labels: pd.Series | None = None,
) -> dict[str, Any]:
    selected_models = selected_models.reindex(matrices.utility.index)
    selected_utility = selected_values(matrices.utility, selected_models)
    selected_quality = selected_values(matrices.quality, selected_models)
    selected_cost = selected_values(matrices.cost, selected_models)
    oracle_utility = matrices.utility.max(axis=1)
    low, high = bootstrap_mean_ci(selected_utility, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
    summary = router_summary(
        selected_utility,
        oracle_utility,
        selected_quality=selected_quality,
        selected_cost=selected_cost,
        max_cost=float(matrices.cost.max().max()),
    )
    summary.update(
        {
            "method": method,
            "K": k if k is not None else "",
            "utility_ci_low": low,
            "utility_ci_high": high,
            "recovered_gap_vs_learned": recovered_gap(
                summary["mean_utility"],
                baseline_mean,
                learned_reference_mean,
            ),
            "recovered_gap_vs_oracle": recovered_gap(summary["mean_utility"], baseline_mean, oracle_mean),
            "selected_model_entropy": model_win_entropy(selected_models.astype(str).tolist()),
            "rate_log2K": float(np.log2(k)) if k and k > 0 else 0.0,
            "empirical_H_Z": empirical_entropy(labels.tolist()) if labels is not None else "",
        }
    )
    return summary
