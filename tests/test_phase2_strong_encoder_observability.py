from __future__ import annotations

import math

import pandas as pd

from routecode.matrix import Matrices
from routecode.states.strong_encoders import evaluate_strong_encoder_state_observability


def _matrices(query_ids: list[str], utility_rows: list[list[float]]) -> Matrices:
    model_ids = ["m0", "m1"]
    utility = pd.DataFrame(utility_rows, index=pd.Index(query_ids, name="query_id"), columns=model_ids)
    quality = utility.copy()
    cost = pd.DataFrame(0.0, index=utility.index, columns=model_ids)
    query_info = pd.DataFrame(
        {
            "query_text": [f"text {query_id}" for query_id in query_ids],
            "dataset": ["synthetic"] * len(query_ids),
            "domain": ["d0"] * len(query_ids),
        },
        index=utility.index,
    )
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=model_ids)


def test_strong_encoder_observability_predicts_states_before_selecting_models():
    train = _matrices(["q0", "q1"], [[1.0, 0.0], [0.0, 1.0]])
    test = _matrices(["q2", "q3"], [[1.0, 0.0], [0.0, 1.0]])
    readiness = pd.DataFrame(
        [
            {
                "model_id": "local/test-encoder",
                "cache_status": "cached",
                "runnable_as_encoder_baseline": True,
                "reason": "cached_encoder_candidate",
                "architecture": "BertModel",
                "model_type": "bert",
                "hidden_size": 2,
                "size_gb": 0.01,
                "local_path": "/tmp/test-encoder",
            }
        ]
    )

    def provider(_row: pd.Series, query_info: pd.DataFrame) -> pd.DataFrame:
        values = {
            "q0": [1.0, 0.0],
            "q1": [0.0, 1.0],
            "q2": [0.95, 0.05],
            "q3": [0.05, 0.95],
        }
        return pd.DataFrame.from_dict(values, orient="index").loc[query_info.index]

    table = evaluate_strong_encoder_state_observability(
        train=train,
        test=test,
        readiness_table=readiness,
        embedding_provider=provider,
        k=2,
        alpha=0.0,
        state_families=["flat_routecode", "d2"],
        predictors=["centroid", "knn"],
        n_neighbors=1,
        n_bootstrap=5,
    )

    executed = table.set_index(["state_family", "state_predictor"])
    flat_knn = executed.loc[("flat_routecode", "knn")]
    assert flat_knn["status"] == "executed"
    assert flat_knn["oracle_state_method"] == "flat_routecode_utility_oracle"
    assert flat_knn["deployable_state_method"] == "strong_encoder_knn"
    assert math.isclose(flat_knn["label_accuracy"], 1.0)
    assert math.isclose(flat_knn["deployable_state_mean_utility"], 1.0)
    assert math.isclose(flat_knn["deployable_state_mean_utility_ci_low"], 1.0)
    assert math.isclose(flat_knn["deployable_state_mean_utility_ci_high"], 1.0)
    assert math.isclose(flat_knn["state_observability_gap_ci_low"], 0.0)
    assert math.isclose(flat_knn["state_observability_gap_ci_high"], 0.0)
    assert math.isclose(flat_knn["query_oracle_gap_ci_low"], 0.0)
    assert math.isclose(flat_knn["query_oracle_gap_ci_high"], 0.0)
    assert math.isclose(flat_knn["state_observability_gap"], 0.0)
    assert flat_knn["routing_invariant"] == "query_to_state_to_model"

    d2_centroid = executed.loc[("d2_predictability_constrained", "centroid")]
    assert d2_centroid["oracle_state_method"] == "d2_joint_oracle_labels"
    assert d2_centroid["deployable_state_method"] == "strong_encoder_centroid"
    assert math.isclose(d2_centroid["label_accuracy"], 1.0)
    assert math.isclose(d2_centroid["state_observability_gap"], 0.0)


def test_strong_encoder_observability_returns_explicit_skipped_rows_without_cached_encoder():
    readiness = pd.DataFrame(
        [
            {
                "model_id": "missing/encoder",
                "cache_status": "missing_local_cache",
                "runnable_as_encoder_baseline": False,
                "reason": "missing_local_cache",
                "local_path": "",
            }
        ]
    )

    table = evaluate_strong_encoder_state_observability(
        train=_matrices(["q0", "q1"], [[1.0, 0.0], [0.0, 1.0]]),
        test=_matrices(["q2"], [[1.0, 0.0]]),
        readiness_table=readiness,
        embedding_provider=None,
        k=2,
        n_bootstrap=5,
    )

    row = table.iloc[0]
    assert row["status"] == "skipped"
    assert row["strong_encoder_status"] == "skipped"
    assert row["reason"] == "no_cached_encoder_candidate"
    assert row["model_id"] == "missing/encoder"
    assert pd.isna(row["deployable_state_mean_utility"])
