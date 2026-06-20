from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from routecode.eval.avengerspro_upstream_metric import (
    avengerspro_payload_has_routing_details,
    evaluate_avengerspro_routing_details,
    selected_models_from_routing_details,
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


def test_exact_avengerspro_payload_without_routing_details_is_not_postprocessable(tmp_path):
    payload_path = tmp_path / "simple_cluster_full_results.json"
    payload_path.write_text(json.dumps({"results": {"accuracy": 75.0}}), encoding="utf-8")

    assert not avengerspro_payload_has_routing_details(payload_path)


def test_selected_models_from_routing_details_aligns_to_test_records_by_order():
    test_records = [
        {"query_id": "q4", "query": "prompt q4"},
        {"query_id": "q5", "query": "prompt q5"},
    ]
    routing_details = [
        {"selected_models": ["m0"], "is_correct": 1.0},
        {"selected_models": ["m1"], "is_correct": 1.0},
    ]

    selected = selected_models_from_routing_details(routing_details, test_records)

    assert selected.to_dict() == {"q4": "m0", "q5": "m1"}


def test_selected_models_from_routing_details_rejects_count_mismatch():
    try:
        selected_models_from_routing_details([{"selected_models": ["m0"]}], [{"query_id": "q4"}, {"query_id": "q5"}])
    except ValueError as exc:
        assert "prediction count mismatch" in str(exc)
    else:
        raise AssertionError("mismatched routing details should fail")


def test_evaluate_avengerspro_routing_details_scores_routecode_utility():
    train = _matrices(
        ["q0", "q1", "q2", "q3"],
        [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
    )
    test = _matrices(["q4", "q5"], [[0.85, 0.15], [0.15, 0.85]])
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05], [0.05, 0.95]],
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )
    test_records = [
        {"query_id": "q4", "query": "prompt q4"},
        {"query_id": "q5", "query": "prompt q5"},
    ]
    routing_details = [
        {"selected_models": ["m0"], "is_correct": 1.0},
        {"selected_models": ["m1"], "is_correct": 1.0},
    ]

    row = evaluate_avengerspro_routing_details(
        train,
        test,
        embeddings,
        routing_details=routing_details,
        test_records=test_records,
        prediction_source="raw_routing_details.json",
        seed=3,
        n_bootstrap=5,
        knn_k=1,
    )

    assert row["method"] == "avengerspro_upstream_simple_cluster_postprocessed"
    assert row["mean_utility"] == 0.85
    assert bool(row["split_aligned_with_routecode"])
    assert bool(row["routecode_metric_compatible"])
    assert bool(row["upstream_model_code_used"])
    assert not bool(row["exact_upstream_command"])
    assert row["prediction_count"] == 2
    assert row["prediction_source"] == "raw_routing_details.json"
