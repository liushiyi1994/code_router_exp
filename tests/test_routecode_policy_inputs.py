from __future__ import annotations

import pandas as pd

from routecode.matrix import Matrices
from routecode.probes.routecode_policy_inputs import build_routecode_policy_inputs


def test_build_routecode_policy_inputs_exports_normalized_one_hot_beliefs():
    train_ids = pd.Index([f"train-{idx}" for idx in range(8)], name="query_id")
    train_utility = pd.DataFrame(
        {
            "model_a": [1.0, 0.9, 1.0, 0.8, 0.1, 0.0, 0.2, 0.1],
            "model_b": [0.0, 0.1, 0.2, 0.0, 1.0, 0.8, 0.9, 1.0],
        },
        index=train_ids,
    )
    train = Matrices(
        utility=train_utility,
        quality=train_utility.copy(),
        cost=pd.DataFrame(0.0, index=train_ids, columns=train_utility.columns),
        query_info=pd.DataFrame(
            {"query_text": [f"query {idx}" for idx in range(8)], "dataset": ["d"] * 8},
            index=train_ids,
        ),
        model_ids=["model_a", "model_b"],
    )
    eval_ids = pd.Index(["eval-0", "eval-1"], name="query_id")
    embeddings = pd.DataFrame(
        {
            "x0": [0.0, 0.1, 0.0, 0.2, 10.0, 10.2, 9.8, 10.1, 0.05, 10.05],
            "x1": [0.0, 0.1, 0.2, 0.0, 10.0, 9.8, 10.2, 10.1, 0.05, 9.95],
        },
        index=train_ids.append(eval_ids),
    )
    query_model_utility = pd.DataFrame(
        {"model_a": [1.0, 0.0], "model_b": [0.0, 1.0]},
        index=eval_ids,
    )

    bundle = build_routecode_policy_inputs(
        train=train,
        embeddings=embeddings,
        query_model_utility=query_model_utility,
        k=2,
        alpha=0.0,
        random_state=0,
        max_iter=5,
        refinement_iter=2,
    )

    assert bundle.before_beliefs.index.tolist() == ["eval-0", "eval-1"]
    assert bundle.before_beliefs.equals(bundle.after_beliefs)
    assert (bundle.before_beliefs.sum(axis=1) == 1.0).all()
    assert set(bundle.state_model_utility.columns) == {"model_a", "model_b"}
    assert bundle.query_model_utility.equals(query_model_utility)
    assert bundle.probe_cost.eq(0.0).all()
    assert bundle.predicted_gain.eq(0.0).all()
    assert bundle.metadata["belief_type"] == "one_hot_routecode_embedding_predicted"
