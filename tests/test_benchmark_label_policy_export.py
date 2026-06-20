from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "68_benchmark_label_policy_export.py"
    spec = importlib.util.spec_from_file_location("benchmark_label_policy_export", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evaluate_dataset_model_rule_exports_query_level_regret():
    module = _load_script()
    query_info = pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2"],
            "dataset": ["math500", "aime", "math500"],
        }
    ).set_index("query_id")
    utility = pd.DataFrame(
        {
            "Qwen3-8B": [1.0, 0.0, 1.0],
            "Intern-S1-mini": [0.0, 1.0, 0.0],
        },
        index=["q0", "q1", "q2"],
    )
    selections, summary = module.evaluate_dataset_model_rule(
        query_info=query_info,
        query_model_utility=utility,
        mapping={"math500": "Qwen3-8B", "aime": "Intern-S1-mini"},
        name="exact_math_qwen_intern",
        threshold=0.03,
        selection_basis="test_rule",
    )

    assert selections["selected_model"].tolist() == ["Qwen3-8B", "Intern-S1-mini", "Qwen3-8B"]
    assert selections["oracle_regret"].tolist() == [0.0, 0.0, 0.0]
    assert bool(summary.loc[0, "within_threshold"]) is True
    assert summary.loc[0, "relative_gap_to_oracle"] == pytest.approx(0.0)
    assert summary.loc[0, "route_labels"] == "aime,math500"
    assert "not the core latent-state" in summary.loc[0, "method_caveat"]


def test_evaluate_dataset_model_rule_rejects_unmapped_dataset():
    module = _load_script()
    query_info = pd.DataFrame({"query_id": ["q0"], "dataset": ["gpqa"]}).set_index("query_id")
    utility = pd.DataFrame({"Qwen3-8B": [1.0]}, index=["q0"])

    with pytest.raises(ValueError, match="missing datasets"):
        module.evaluate_dataset_model_rule(
            query_info=query_info,
            query_model_utility=utility,
            mapping={"math500": "Qwen3-8B"},
            name="bad_rule",
            threshold=0.03,
            selection_basis="test_rule",
        )
