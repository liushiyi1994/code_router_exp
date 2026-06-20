from __future__ import annotations

import pandas as pd

from routecode.probes.signal_analysis import (
    PROBE_SIGNAL_COLUMNS,
    analyze_probe_signal,
)


def _probe_features(query_ids: list[str], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "query_id": query_id,
                "self_confidence": 0.1 if label == 0 else 0.9,
                "agreement_score": 1.0,
                "knn_label_entropy": 0.0 if label == 0 else 1.0,
                "knn_winner_entropy": 0.0,
                "latency_sec": 0.01,
                "input_tokens": 10 + idx,
                "output_tokens": 1 + label,
                "probe_cost_proxy": 0.001 + 0.001 * label,
                "error_type": "",
            }
            for idx, (query_id, label) in enumerate(zip(query_ids, labels, strict=True))
        ]
    )


def test_probe_signal_analysis_scores_state_prediction_when_targets_align():
    query_ids = [f"q{i}" for i in range(12)]
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    probe_features = _probe_features(query_ids, labels)
    state_targets = pd.DataFrame({"query_id": query_ids, "state_label": labels})
    query_features = pd.DataFrame(
        {
            "query_id": query_ids,
            "query_len_feature": [0.0 if label == 0 else 1.0 for label in labels],
        }
    )

    table = analyze_probe_signal(
        probe_features=probe_features,
        state_targets=state_targets,
        query_features=query_features,
        random_state=0,
    )

    assert list(table.columns) == PROBE_SIGNAL_COLUMNS
    executed = table[table["status"] == "executed"]
    assert {"query_only_state_predictor", "probe_only_state_predictor", "query_plus_probe_state_predictor"}.issubset(
        set(executed["method"])
    )
    assert executed["state_prediction_accuracy"].between(0.0, 1.0).all()
    assert executed["state_prediction_accuracy_ci_low"].between(0.0, 1.0).all()
    assert executed["state_prediction_accuracy_ci_high"].between(0.0, 1.0).all()
    assert (executed["state_prediction_accuracy_ci_low"] <= executed["state_prediction_accuracy"]).all()
    assert (executed["state_prediction_accuracy"] <= executed["state_prediction_accuracy_ci_high"]).all()
    assert executed["n_train"].min() > 0
    assert executed["n_test"].min() > 0
    probe_only = executed[executed["method"] == "probe_only_state_predictor"].iloc[0]
    assert probe_only["mean_probe_cost_proxy"] > 0.0


def test_probe_signal_analysis_uses_explicit_train_test_split_when_present():
    train_ids = [f"train_{idx}" for idx in range(8)]
    test_ids = [f"test_{idx}" for idx in range(2)]
    query_ids = train_ids + test_ids
    labels = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]
    probe_features = _probe_features(query_ids, labels)
    state_targets = pd.DataFrame(
        {
            "query_id": query_ids,
            "state_label": labels,
            "split": ["train"] * len(train_ids) + ["test"] * len(test_ids),
        }
    )
    query_features = pd.DataFrame(
        {
            "query_id": query_ids,
            "query_len_feature": [float(label) for label in labels],
        }
    )

    table = analyze_probe_signal(
        probe_features=probe_features,
        state_targets=state_targets,
        query_features=query_features,
        random_state=0,
    )

    executed = table[table["status"] == "executed"]
    assert not executed.empty
    assert executed["n_train"].eq(8).all()
    assert executed["n_test"].eq(2).all()


def test_probe_signal_analysis_blocks_without_aligned_state_targets():
    probe_features = _probe_features(["probe_q0", "probe_q1"], [0, 1])
    state_targets = pd.DataFrame({"query_id": ["state_q0", "state_q1"], "state_label": [0, 1]})

    table = analyze_probe_signal(probe_features=probe_features, state_targets=state_targets)

    assert list(table.columns) == PROBE_SIGNAL_COLUMNS
    assert set(table["status"]) == {"blocked_no_aligned_state_targets"}
    assert table["n_queries"].eq(0).all()
    assert table["state_prediction_accuracy"].isna().all()
