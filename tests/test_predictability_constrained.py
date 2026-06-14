import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode


def _toy_query_info(index: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "query_id": index,
            "dataset": ["toy"] * len(index),
            "domain": ["toy"] * len(index),
        }
    ).set_index("query_id")


def _within_label_embedding_sse(labels: pd.Series, embeddings: pd.DataFrame) -> float:
    total = 0.0
    for label in sorted(labels.unique()):
        group = embeddings.loc[labels.index[labels == label]]
        centroid = group.mean(axis=0)
        total += float(((group - centroid) ** 2).sum(axis=1).sum())
    return total


def test_predictability_weight_moves_codebook_toward_embedding_predictable_labels():
    query_ids = [f"q{i}" for i in range(8)]
    utility = pd.DataFrame(
        {
            "cheap": [1.0, 0.0, 1.0, 0.0, 0.98, 0.02, 0.97, 0.01],
            "strong": [0.0, 1.0, 0.0, 1.0, 0.02, 0.98, 0.03, 0.99],
        },
        index=query_ids,
    )
    embeddings = pd.DataFrame(
        [
            [0.00, 0.0],
            [0.05, 0.0],
            [8.00, 0.0],
            [8.05, 0.0],
            [0.10, 0.0],
            [0.15, 0.0],
            [8.10, 0.0],
            [8.15, 0.0],
        ],
        index=query_ids,
    )
    query_info = _toy_query_info(query_ids)

    utility_only = PredictabilityConstrainedRouteCode(n_labels=2, alpha=0.0, random_state=0).fit(
        query_info,
        utility,
        embeddings,
    )
    predictable = PredictabilityConstrainedRouteCode(n_labels=2, alpha=25.0, random_state=0).fit(
        query_info,
        utility,
        embeddings,
    )

    utility_sse = _within_label_embedding_sse(utility_only.train_labels_, embeddings)
    predictable_sse = _within_label_embedding_sse(predictable.train_labels_, embeddings)

    assert predictable_sse < utility_sse


def test_predictability_constrained_codebook_matches_routecode_prediction_api():
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
    query_info = _toy_query_info(list(utility.index))

    codebook = PredictabilityConstrainedRouteCode(n_labels=2, alpha=1.0, beta=0.0, random_state=0).fit(
        query_info,
        utility,
        embeddings,
    )

    embedding_labels = codebook.predict_labels(embeddings)
    utility_labels = codebook.predict_utility_labels(utility)
    joint_labels = codebook.predict_joint_labels(utility, embeddings)
    selected = codebook.predict(query_info, embeddings)
    oracle_selected = codebook.predict_from_labels(utility_labels)

    assert set(embedding_labels).issubset({0, 1})
    assert set(joint_labels).issubset({0, 1})
    assert np.isclose(codebook.label_entropy(), 1.0)
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q3"] == "strong"
    assert oracle_selected.loc["q0"] == "cheap"
    assert oracle_selected.loc["q3"] == "strong"
