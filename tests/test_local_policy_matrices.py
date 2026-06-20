from __future__ import annotations

import pandas as pd
import pytest

from routecode.local_eval.policy_matrices import build_local_policy_matrices


def test_build_local_policy_matrices_uses_train_states_and_test_queries_with_cost():
    outcomes = pd.DataFrame(
        [
            {"query_id": "train_0", "model_id": "m0", "quality": 1.0, "cost_proxy": 0.2},
            {"query_id": "train_0", "model_id": "m1", "quality": 0.0, "cost_proxy": 0.1},
            {"query_id": "train_1", "model_id": "m0", "quality": 0.0, "cost_proxy": 0.2},
            {"query_id": "train_1", "model_id": "m1", "quality": 1.0, "cost_proxy": 0.4},
            {"query_id": "test_0", "model_id": "m0", "quality": 1.0, "cost_proxy": 0.5},
            {"query_id": "test_0", "model_id": "m1", "quality": 0.5, "cost_proxy": 0.1},
        ]
    )
    state_targets = pd.DataFrame(
        {
            "query_id": ["train_0", "train_1", "test_0"],
            "state_label": [0, 1, 0],
            "split": ["train", "train", "test"],
        }
    )

    matrices = build_local_policy_matrices(
        local_outcomes=outcomes,
        state_targets=state_targets,
        lambda_cost=0.1,
    )

    query_utility = matrices.query_model_utility.set_index("query_id")
    assert query_utility.index.tolist() == ["test_0"]
    assert query_utility.loc["test_0", "m0"] == pytest.approx(0.95)
    assert query_utility.loc["test_0", "m1"] == pytest.approx(0.49)
    state_utility = matrices.state_model_utility.set_index("state_label")
    assert state_utility.loc["z0", "m0"] == pytest.approx(0.98)
    assert state_utility.loc["z1", "m1"] == pytest.approx(0.96)
    metadata = matrices.metadata.iloc[0]
    assert int(metadata["policy_queries"]) == 1
    assert int(metadata["train_queries"]) == 2
    assert int(metadata["model_count"]) == 2
    assert float(metadata["lambda_cost"]) == pytest.approx(0.1)


def test_build_local_policy_matrices_rejects_missing_overlap():
    outcomes = pd.DataFrame([{"query_id": "q0", "model_id": "m0", "quality": 1.0, "cost_proxy": 0.0}])
    state_targets = pd.DataFrame({"query_id": ["q1"], "state_label": [0], "split": ["train"]})

    with pytest.raises(ValueError, match="No local outcomes overlap state targets"):
        build_local_policy_matrices(local_outcomes=outcomes, state_targets=state_targets)
