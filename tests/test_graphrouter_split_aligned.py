from __future__ import annotations

import pandas as pd

from routecode.eval.graphrouter_split_aligned import (
    build_routecode_split_masks,
    evaluate_graphrouter_selected_models,
)
from routecode.matrix import Matrices


def _matrices(query_ids: list[str], utility_rows: list[list[float]]) -> Matrices:
    models = ["m0", "m1"]
    utility = pd.DataFrame(utility_rows, index=pd.Index(query_ids, name="query_id"), columns=models)
    quality = utility.copy()
    cost = pd.DataFrame(0.0, index=utility.index, columns=models)
    query_info = pd.DataFrame(
        {
            "query_text": [f"prompt {query_id}" for query_id in query_ids],
            "dataset": ["demo"] * len(query_ids),
            "domain": ["demo"] * len(query_ids),
        },
        index=utility.index,
    )
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=models)


def test_build_routecode_split_masks_preserves_query_level_masks():
    router_data = pd.DataFrame(
        [
            {"query_id": "q0", "routecode_split": "train", "llm": "m0"},
            {"query_id": "q0", "routecode_split": "train", "llm": "m1"},
            {"query_id": "q1", "routecode_split": "val", "llm": "m0"},
            {"query_id": "q1", "routecode_split": "val", "llm": "m1"},
            {"query_id": "q2", "routecode_split": "test", "llm": "m0"},
            {"query_id": "q2", "routecode_split": "test", "llm": "m1"},
        ]
    )

    masks = build_routecode_split_masks(router_data, num_llms=2)

    assert masks.train_row_indices == [0, 1]
    assert masks.val_row_indices == [2, 3]
    assert masks.test_row_indices == [4, 5]
    assert masks.test_query_ids == ["q2"]
    assert masks.test_query_positions == [2]


def test_build_routecode_split_masks_rejects_split_leakage_within_query():
    router_data = pd.DataFrame(
        [
            {"query_id": "q0", "routecode_split": "train", "llm": "m0"},
            {"query_id": "q0", "routecode_split": "test", "llm": "m1"},
        ]
    )

    try:
        build_routecode_split_masks(router_data, num_llms=2)
    except ValueError as exc:
        assert "multiple routecode_split values" in str(exc)
    else:
        raise AssertionError("split leakage inside one query should fail")


def test_evaluate_graphrouter_selected_models_scores_routecode_utility_rows():
    train = _matrices(
        ["q0", "q1", "q2", "q3"],
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
    )
    test = _matrices(["q4", "q5"], [[0.85, 0.15], [0.15, 0.85]])
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05], [0.05, 0.95]],
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )
    selected = pd.Series(["m0", "m1"], index=pd.Index(["q4", "q5"], name="query_id"))

    row = evaluate_graphrouter_selected_models(
        train,
        test,
        embeddings,
        selected,
        prediction_source="raw_predictions.json",
        seed=3,
        n_bootstrap=5,
        knn_k=1,
    )

    assert row["method"] == "graphrouter_split_aligned_gnn"
    assert row["mean_utility"] == 0.85
    assert bool(row["split_aligned_with_routecode"])
    assert bool(row["routecode_metric_compatible"])
    assert bool(row["upstream_model_code_used"])
    assert not bool(row["exact_upstream_command"])
    assert not bool(row["external_api_calls"])
    assert row["prediction_count"] == 2
    assert row["prediction_source"] == "raw_predictions.json"
