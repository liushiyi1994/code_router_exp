from __future__ import annotations

import pandas as pd
import pytest

from routecode.eval.cost_quality import (
    cost_quality_frontier,
    summarize_method_cost_quality,
)
from routecode.matrix import Matrices


def _matrices() -> Matrices:
    query_ids = pd.Index(["q0", "q1"], name="query_id")
    models = ["cheap", "strong"]
    quality = pd.DataFrame({"cheap": [0.6, 0.6], "strong": [1.0, 0.8]}, index=query_ids)
    cost = pd.DataFrame({"cheap": [0.1, 0.1], "strong": [0.4, 0.5]}, index=query_ids)
    utility = quality - 0.5 * cost
    query_info = pd.DataFrame({"dataset": ["d0", "d0"]}, index=query_ids)
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=models)


def test_summarize_method_cost_quality_uses_selected_quality_cost_and_utility():
    matrices = _matrices()
    selections = {
        "cheap_only": pd.Series("cheap", index=matrices.utility.index),
        "strong_only": pd.Series("strong", index=matrices.utility.index),
    }

    table = summarize_method_cost_quality(matrices, selections, lambda_cost=0.5)
    by_method = table.set_index("method")

    assert by_method.loc["cheap_only", "mean_quality"] == pytest.approx(0.6)
    assert by_method.loc["cheap_only", "mean_cost"] == pytest.approx(0.1)
    assert by_method.loc["cheap_only", "mean_utility"] == pytest.approx(0.55)
    assert by_method.loc["strong_only", "mean_quality"] == pytest.approx(0.9)
    assert by_method.loc["strong_only", "mean_cost"] == pytest.approx(0.45)
    assert by_method.loc["strong_only", "lambda_cost"] == 0.5


def test_cost_quality_frontier_selects_min_cost_for_quality_and_max_quality_for_budget():
    summary = pd.DataFrame(
        [
            {"method": "cheap", "mean_quality": 0.6, "mean_cost": 0.1},
            {"method": "balanced", "mean_quality": 0.8, "mean_cost": 0.2},
            {"method": "strong", "mean_quality": 0.9, "mean_cost": 0.5},
        ]
    )

    frontier = cost_quality_frontier(
        summary,
        quality_targets=[0.75, 0.95],
        cost_budgets=[0.15, 0.25],
        lambda_cost=0.0,
    )

    quality_rows = frontier[frontier["target_type"] == "cost_at_fixed_quality"].set_index("target_value")
    assert quality_rows.loc[0.75, "selected_method"] == "balanced"
    assert quality_rows.loc[0.75, "achieved_cost"] == 0.2
    assert pd.isna(quality_rows.loc[0.95, "selected_method"])

    cost_rows = frontier[frontier["target_type"] == "quality_at_fixed_cost"].set_index("target_value")
    assert cost_rows.loc[0.15, "selected_method"] == "cheap"
    assert cost_rows.loc[0.15, "achieved_quality"] == 0.6
    assert cost_rows.loc[0.25, "selected_method"] == "balanced"
