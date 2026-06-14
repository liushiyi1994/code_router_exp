from __future__ import annotations

import pandas as pd

from routecode.routers.matrix_factorization import BinaryThresholdRouter, MatrixFactorizationRouter


def test_matrix_factorization_router_predicts_from_low_rank_utility_structure():
    train_utility = pd.DataFrame(
        {
            "m0": [0.95, 0.90, 0.10, 0.15],
            "m1": [0.10, 0.15, 0.95, 0.90],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    query_info = pd.DataFrame({"query_text": ["a", "aa", "b", "bb"]}, index=train_utility.index)
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [0.95, 0.05], [0.05, 0.95]],
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )
    test_info = pd.DataFrame({"query_text": ["aaa", "bbb"]}, index=["q4", "q5"])

    selected = MatrixFactorizationRouter(rank=2, alpha=0.01).fit(query_info, train_utility, embeddings).predict(
        test_info,
        embeddings,
    )

    assert selected.loc["q4"] == "m0"
    assert selected.loc["q5"] == "m1"


def test_binary_threshold_router_routes_by_predicted_strong_win_rate():
    train_utility = pd.DataFrame(
        {
            "strong": [0.9, 0.85, 0.1, 0.2],
            "weak": [0.2, 0.25, 0.8, 0.75],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    query_info = pd.DataFrame({"query_text": ["a", "aa", "b", "bb"]}, index=train_utility.index)
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8], [0.9, 0.1], [0.1, 0.9]],
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )
    test_info = pd.DataFrame({"query_text": ["aaa", "bbb"]}, index=["q4", "q5"])

    router = BinaryThresholdRouter("strong", "weak", threshold=0.5, random_state=0).fit(
        query_info,
        train_utility,
        embeddings,
    )
    selected = router.predict(test_info, embeddings)
    win_rates = router.predict_strong_win_rate(embeddings.loc[test_info.index])

    assert selected.loc["q4"] == "strong"
    assert selected.loc["q5"] == "weak"
    assert win_rates.loc["q4"] > win_rates.loc["q5"]
