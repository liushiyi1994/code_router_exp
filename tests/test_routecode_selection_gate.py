from __future__ import annotations

import pandas as pd

from routecode.eval.routecode_selection_gate import RouteCodeSelectionConfig, rank_selection_candidates


def test_rank_selection_candidates_marks_val_selected_and_policy_threshold():
    table = pd.DataFrame(
        {
            "candidate": ["a", "b", "c"],
            "k": [4, 32, 64],
            "alpha": [0.0, 0.0, 0.05],
            "val_relative_gap_to_oracle": [0.04, 0.06, 0.06],
            "policy_slice_relative_gap_to_oracle": [0.05, 0.02, 0.03],
        }
    )

    ranked = rank_selection_candidates(table, threshold=0.03, target_k=32)
    rows = ranked.set_index("candidate")

    assert rows.loc["a", "selected_by_val"]
    assert not rows.loc["a", "policy_slice_within_threshold"]
    assert rows.loc["b", "policy_slice_within_threshold"]
    assert not rows.loc["b", "selected_by_val"]
    assert rows.loc["b", "selected_by_target_rate"]
    assert rows.loc["b", "target_rate_policy_slice_within_threshold"]
    assert not ranked["val_selected_policy_slice_within_threshold"].any()


def test_routecode_selection_config_can_scope_training_datasets():
    config = RouteCodeSelectionConfig(
        k_values=(4,),
        alpha_values=(0.3,),
        training_datasets=("aime", "math500"),
        validation_datasets=("aime", "math500"),
        target_k=16,
    )

    assert config.training_datasets == ("aime", "math500")
    assert config.target_k == 16
