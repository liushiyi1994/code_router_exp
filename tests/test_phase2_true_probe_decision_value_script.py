from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "65_true_probe_decision_value.py"
    spec = importlib.util.spec_from_file_location("phase2_true_probe_decision_value", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_true_probe_decision_value_reports_model_changes_and_utility_delta(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "decision_value"
    before_path = tmp_path / "before.csv"
    after_path = tmp_path / "after.csv"
    state_utility_path = tmp_path / "state_model_utility.csv"
    query_utility_path = tmp_path / "query_model_utility.csv"
    predicted_gain_path = tmp_path / "predicted_gain.csv"
    probe_cost_path = tmp_path / "probe_cost.csv"

    pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2"],
            "z0": [0.9, 0.6, 0.7],
            "z1": [0.1, 0.4, 0.3],
        }
    ).to_csv(before_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2"],
            "z0": [0.2, 0.4, 0.8],
            "z1": [0.8, 0.6, 0.2],
        }
    ).to_csv(after_path, index=False)
    pd.DataFrame(
        {
            "state_label": ["z0", "z1"],
            "cheap": [1.0, 0.0],
            "strong": [0.0, 1.0],
        }
    ).to_csv(state_utility_path, index=False)
    pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2"],
            "cheap": [0.2, 0.0, 1.0],
            "strong": [1.0, 1.0, 0.0],
        }
    ).to_csv(query_utility_path, index=False)
    pd.DataFrame({"query_id": ["q0", "q1", "q2"], "predicted_gain": [0.8, 0.1, 0.0]}).to_csv(
        predicted_gain_path,
        index=False,
    )
    pd.DataFrame({"query_id": ["q0", "q1", "q2"], "probe_cost": [0.1, 0.2, 0.3]}).to_csv(
        probe_cost_path,
        index=False,
    )

    paths = module.run(
        before_beliefs_path=str(before_path),
        after_beliefs_path=str(after_path),
        state_model_utility_path=str(state_utility_path),
        query_model_utility_path=str(query_utility_path),
        output_dir=str(out_dir),
        predicted_gain_path=str(predicted_gain_path),
        probe_cost_path=str(probe_cost_path),
    )

    summary = pd.read_csv(paths["summary"])
    by_query = pd.read_csv(paths["by_query"])
    assert int(summary.loc[0, "n_queries"]) == 3
    assert int(summary.loc[0, "selected_model_changes"]) == 2
    assert summary.loc[0, "selected_model_change_rate"] == 2 / 3
    assert summary.loc[0, "mean_before_utility"] == pytest.approx(0.4)
    assert summary.loc[0, "mean_after_utility"] == pytest.approx(1.0)
    assert summary.loc[0, "mean_utility_delta"] == pytest.approx(0.6)
    assert int(summary.loc[0, "nonzero_predicted_gain_rows"]) == 2
    assert summary.loc[0, "mean_probe_cost"] == pytest.approx(0.2)
    assert by_query["query_id"].tolist() == ["q0", "q1", "q2"]
    assert by_query["selected_changed"].tolist() == [True, True, False]
    memo = (out_dir / "m13_true_probe_decision_value_memo.md").read_text(encoding="utf-8")
    assert "decision value" in memo
    assert "selected model changed" in memo
