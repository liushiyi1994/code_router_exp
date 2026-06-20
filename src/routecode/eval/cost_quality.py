from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from routecode.matrix import Matrices
from routecode.metrics import model_win_entropy, selected_values


def summarize_method_cost_quality(
    matrices: Matrices,
    selections: dict[str, pd.Series],
    lambda_cost: float,
) -> pd.DataFrame:
    rows = []
    max_cost = float(matrices.cost.max().max())
    if max_cost <= 0:
        max_cost = 1.0
    for method, selected in selections.items():
        selected = selected.reindex(matrices.utility.index)
        utility = selected_values(matrices.utility, selected)
        quality = selected_values(matrices.quality, selected)
        cost = selected_values(matrices.cost, selected)
        rows.append(
            {
                "lambda_cost": float(lambda_cost),
                "method": method,
                "mean_utility": float(utility.mean()),
                "mean_quality": float(quality.mean()),
                "mean_cost": float(cost.mean()),
                "normalized_cost": float((cost / max_cost).mean()),
                "selected_model_entropy": model_win_entropy(selected.astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def cost_quality_frontier(
    summary: pd.DataFrame,
    quality_targets: Iterable[float],
    cost_budgets: Iterable[float],
    lambda_cost: float,
) -> pd.DataFrame:
    rows = []
    for target in quality_targets:
        rows.append(_cost_at_fixed_quality(summary, float(target), float(lambda_cost)))
    for budget in cost_budgets:
        rows.append(_quality_at_fixed_cost(summary, float(budget), float(lambda_cost)))
    return pd.DataFrame(rows)


def default_quality_targets(summary: pd.DataFrame, fractions: Iterable[float]) -> list[float]:
    if summary.empty:
        return []
    max_quality = float(summary["mean_quality"].max())
    return [float(fraction) * max_quality for fraction in fractions]


def default_cost_budgets(summary: pd.DataFrame, fractions: Iterable[float]) -> list[float]:
    if summary.empty:
        return []
    max_cost = float(summary["mean_cost"].max())
    return [float(fraction) * max_cost for fraction in fractions]


def _cost_at_fixed_quality(summary: pd.DataFrame, target: float, lambda_cost: float) -> dict[str, object]:
    eligible = summary[summary["mean_quality"] >= target].copy()
    if eligible.empty:
        return _empty_frontier_row("cost_at_fixed_quality", target, lambda_cost)
    best = eligible.sort_values(["mean_cost", "mean_quality", "method"], ascending=[True, False, True]).iloc[0]
    return _frontier_row("cost_at_fixed_quality", target, lambda_cost, best)


def _quality_at_fixed_cost(summary: pd.DataFrame, budget: float, lambda_cost: float) -> dict[str, object]:
    eligible = summary[summary["mean_cost"] <= budget].copy()
    if eligible.empty:
        return _empty_frontier_row("quality_at_fixed_cost", budget, lambda_cost)
    best = eligible.sort_values(["mean_quality", "mean_cost", "method"], ascending=[False, True, True]).iloc[0]
    return _frontier_row("quality_at_fixed_cost", budget, lambda_cost, best)


def _frontier_row(target_type: str, target_value: float, lambda_cost: float, row: pd.Series) -> dict[str, object]:
    return {
        "lambda_cost": float(lambda_cost),
        "target_type": target_type,
        "target_value": float(target_value),
        "selected_method": str(row["method"]),
        "achieved_quality": float(row["mean_quality"]),
        "achieved_cost": float(row["mean_cost"]),
        "achieved_utility": float(row["mean_utility"]) if "mean_utility" in row else np.nan,
    }


def _empty_frontier_row(target_type: str, target_value: float, lambda_cost: float) -> dict[str, object]:
    return {
        "lambda_cost": float(lambda_cost),
        "target_type": target_type,
        "target_value": float(target_value),
        "selected_method": np.nan,
        "achieved_quality": np.nan,
        "achieved_cost": np.nan,
        "achieved_utility": np.nan,
    }
