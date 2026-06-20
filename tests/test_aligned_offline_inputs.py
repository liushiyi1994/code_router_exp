from __future__ import annotations

import pandas as pd

from routecode.matrix import Matrices
from routecode.probes.aligned_inputs import build_aligned_offline_inputs


def _matrices(query_ids: list[str], utility_rows: list[list[float]]) -> Matrices:
    model_ids = ["cheap", "strong"]
    utility = pd.DataFrame(utility_rows, index=pd.Index(query_ids, name="query_id"), columns=model_ids)
    quality = utility.copy()
    cost = pd.DataFrame(0.0, index=utility.index, columns=model_ids)
    query_info = pd.DataFrame(
        {
            "query_text": [f"query {query_id}" for query_id in query_ids],
            "dataset": ["demo"] * len(query_ids),
            "domain": ["demo"] * len(query_ids),
        },
        index=utility.index,
    )
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=model_ids)


def test_build_aligned_offline_inputs_emits_matching_probe_state_and_policy_frames():
    train = _matrices(
        ["tr0", "tr1", "tr2", "tr3"],
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]],
    )
    test = _matrices(
        ["te0", "te1"],
        [[0.85, 0.15], [0.15, 0.85]],
    )
    embeddings = pd.DataFrame(
        {
            "emb_0": [0.0, 0.1, 1.0, 1.1, 0.05, 1.05],
            "emb_1": [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        },
        index=pd.Index(["tr0", "tr1", "tr2", "tr3", "te0", "te1"], name="query_id"),
    )

    bundle = build_aligned_offline_inputs(
        train=train,
        test=test,
        embeddings=embeddings,
        k=2,
        alpha=1.0,
        random_state=0,
        n_neighbors=2,
    )

    assert set(bundle.state_targets["split"]) == {"train", "test"}
    assert set(bundle.probe_features["query_id"]) == {"tr0", "tr1", "tr2", "tr3", "te0", "te1"}
    assert set(bundle.before_beliefs.index) == {"te0", "te1"}
    assert bundle.before_beliefs.index.equals(bundle.after_beliefs.index)
    assert set(bundle.before_beliefs.columns) == set(bundle.state_model_utility.index)
    assert bundle.query_model_utility.index.equals(test.utility.index)
    assert bundle.probe_cost.index.equals(test.utility.index)
    assert bundle.predicted_gain.index.equals(test.utility.index)
    assert bundle.probe_features["probe_type"].eq("offline_knn_uncertainty").all()
