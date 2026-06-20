from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "70_local_outcomes_policy_matrices.py"
    spec = importlib.util.spec_from_file_location("local_outcomes_policy_matrices", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_outcomes_policy_matrices_script_writes_policy_matrices(tmp_path):
    module = _load_script()
    outcomes_path = tmp_path / "local_model_outcomes.parquet"
    state_targets_path = tmp_path / "state_targets.csv"
    out_dir = tmp_path / "local_policy_matrices"
    pd.DataFrame(
        [
            {"query_id": "train_0", "model_id": "m0", "quality": 1.0, "cost_proxy": 0.2},
            {"query_id": "train_0", "model_id": "m1", "quality": 0.0, "cost_proxy": 0.1},
            {"query_id": "train_1", "model_id": "m0", "quality": 0.0, "cost_proxy": 0.2},
            {"query_id": "train_1", "model_id": "m1", "quality": 1.0, "cost_proxy": 0.4},
            {"query_id": "test_0", "model_id": "m0", "quality": 1.0, "cost_proxy": 0.5},
            {"query_id": "test_0", "model_id": "m1", "quality": 0.5, "cost_proxy": 0.1},
        ]
    ).to_parquet(outcomes_path)
    pd.DataFrame(
        {
            "query_id": ["train_0", "train_1", "test_0"],
            "state_label": [0, 1, 0],
            "split": ["train", "train", "test"],
        }
    ).to_csv(state_targets_path, index=False)

    paths = module.run(
        local_outcomes_path=str(outcomes_path),
        state_targets_path=str(state_targets_path),
        output_dir=str(out_dir),
        lambda_cost=0.1,
    )

    expected = {
        "query_model_utility",
        "query_model_quality",
        "query_model_cost",
        "state_model_utility",
        "state_model_quality",
        "state_model_cost",
        "metadata",
    }
    assert expected == set(paths)
    for path in paths.values():
        assert Path(path).exists()
    query_utility = pd.read_csv(paths["query_model_utility"])
    state_utility = pd.read_csv(paths["state_model_utility"])
    assert query_utility["query_id"].tolist() == ["test_0"]
    assert state_utility["state_label"].tolist() == ["z0", "z1"]
    memo = (out_dir / "m15_local_policy_matrices_memo.md").read_text(encoding="utf-8")
    assert "quality - lambda_cost * cost_proxy" in memo
    assert "does not introduce human route labels" in memo
