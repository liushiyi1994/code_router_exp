import pandas as pd
import pytest

from routecode.states.action_states import (
    FRONTIER_NEEDED,
    LOCAL_ENOUGH,
    fit_action_state_policy,
    local_frontier_action_labels,
    selected_utility,
)


def test_local_frontier_action_labels_use_cost_aware_utility_matrix():
    utility = pd.DataFrame(
        {
            "small-local": [0.8, 0.2, 0.6],
            "big-local": [0.7, 0.4, 0.6],
            "gpt-frontier": [0.5, 0.9, 0.6],
        },
        index=["q0", "q1", "q2"],
    )

    labels = local_frontier_action_labels(
        utility,
        local_models=["small-local", "big-local"],
        frontier_models=["gpt-frontier"],
    )

    assert labels.to_dict() == {
        "q0": LOCAL_ENOUGH,
        "q1": FRONTIER_NEEDED,
        "q2": LOCAL_ENOUGH,
    }


def test_action_state_policy_selects_train_best_model_per_state():
    utility = pd.DataFrame(
        {
            "small-local": [0.9, 0.8, 0.2, 0.1],
            "big-local": [0.7, 0.6, 0.3, 0.2],
            "gpt-frontier": [0.5, 0.4, 0.9, 0.8],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    labels = pd.Series(
        [LOCAL_ENOUGH, LOCAL_ENOUGH, FRONTIER_NEEDED, FRONTIER_NEEDED],
        index=utility.index,
    )

    policy = fit_action_state_policy(utility, labels)
    selected = policy.select(labels)

    assert selected.to_dict() == {
        "q0": "small-local",
        "q1": "small-local",
        "q2": "gpt-frontier",
        "q3": "gpt-frontier",
    }
    assert selected_utility(utility, selected).mean() == pytest.approx(0.85)
