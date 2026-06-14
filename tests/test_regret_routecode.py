from __future__ import annotations

import pandas as pd

from routecode.codes.regret import RegretOptimizedRouteCode


def _query_info(index: pd.Index) -> pd.DataFrame:
    domains = ["easy" if idx < len(index) / 2 else "hard" for idx, _ in enumerate(index)]
    return pd.DataFrame(
        {
            "query_id": index,
            "dataset": domains,
            "domain": domains,
        }
    ).set_index("query_id")


def test_regret_optimized_routecode_oracle_labels_minimize_selected_model_regret():
    utility = pd.DataFrame(
        {
            "cheap": [0.9, 0.85, 0.2, 0.25],
            "strong": [0.3, 0.35, 0.95, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        index=utility.index,
    )

    codebook = RegretOptimizedRouteCode(n_labels=2, random_state=0).fit(_query_info(utility.index), utility, embeddings)
    labels = codebook.predict_utility_labels(utility)
    selected = codebook.predict_from_labels(labels)

    assert selected.to_dict() == {
        "q0": "cheap",
        "q1": "cheap",
        "q2": "strong",
        "q3": "strong",
    }


def test_regret_optimized_routecode_predicts_labels_from_embeddings_without_utility():
    train_utility = pd.DataFrame(
        {
            "cheap": [0.9, 0.85, 0.2, 0.25],
            "strong": [0.3, 0.35, 0.95, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    train_embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        index=train_utility.index,
    )
    test_query_info = _query_info(pd.Index(["t0", "t1"]))
    test_embeddings = pd.DataFrame([[0.05, 0.0], [5.05, 5.0]], index=test_query_info.index)

    codebook = RegretOptimizedRouteCode(n_labels=2, random_state=0).fit(
        _query_info(train_utility.index),
        train_utility,
        train_embeddings,
    )
    selected = codebook.predict(test_query_info, test_embeddings)

    assert selected.to_dict() == {"t0": "cheap", "t1": "strong"}
