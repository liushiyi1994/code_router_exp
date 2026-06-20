from __future__ import annotations

import numpy as np
import pandas as pd

import routecode.codes.predictability_constrained as d2_module
import routecode.codes.regret as regret_module
import routecode.routers.cluster_lookup as cluster_module
from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.codes.regret import RegretOptimizedRouteCode
from routecode.routers.cluster_lookup import EmbeddingClusterRouter


def _utility() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "m0": [1.0, 0.9, 0.1, 0.2],
            "m1": [0.1, 0.2, 1.0, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )


def _query_info() -> pd.DataFrame:
    return pd.DataFrame({"dataset": ["a", "a", "b", "b"]}, index=_utility().index)


def _embeddings() -> pd.DataFrame:
    return pd.DataFrame(np.eye(4), index=_utility().index)


def test_embedding_cluster_router_passes_n_init_to_kmeans(monkeypatch):
    seen: dict[str, int] = {}

    class FakeKMeans:
        def __init__(self, n_clusters, random_state, n_init):
            seen["n_clusters"] = n_clusters
            seen["random_state"] = random_state
            seen["n_init"] = n_init

        def fit_predict(self, values):
            assert values.shape == (4, 4)
            return np.array([0, 0, 1, 1])

        def predict(self, values):
            return np.array([0] * len(values))

    monkeypatch.setattr(cluster_module, "KMeans", FakeKMeans)

    EmbeddingClusterRouter(n_clusters=2, random_state=7, n_init=1).fit(_query_info(), _utility(), _embeddings())

    assert seen == {"n_clusters": 2, "random_state": 7, "n_init": 1}


def test_regret_routecode_passes_n_init_to_kmeans(monkeypatch):
    seen: dict[str, int] = {}

    class FakeKMeans:
        def __init__(self, n_clusters, random_state, n_init, max_iter):
            seen["n_clusters"] = n_clusters
            seen["random_state"] = random_state
            seen["n_init"] = n_init
            seen["max_iter"] = max_iter

        def fit_predict(self, values):
            assert values.shape == (4, 2)
            return np.array([0, 0, 1, 1])

    monkeypatch.setattr(regret_module, "KMeans", FakeKMeans)

    RegretOptimizedRouteCode(n_labels=2, random_state=7, max_iter=11, n_init=1).fit(
        _query_info(),
        _utility(),
        _embeddings(),
    )

    assert seen == {"n_clusters": 2, "random_state": 7, "n_init": 1, "max_iter": 11}


def test_predictability_constrained_routecode_passes_n_init_to_kmeans(monkeypatch):
    seen: dict[str, int] = {}

    class FakeKMeans:
        def __init__(self, n_clusters, random_state, n_init, max_iter):
            seen["n_clusters"] = n_clusters
            seen["random_state"] = random_state
            seen["n_init"] = n_init
            seen["max_iter"] = max_iter

        def fit_predict(self, values):
            assert values.shape == (4, 6)
            return np.array([0, 0, 1, 1])

    monkeypatch.setattr(d2_module, "KMeans", FakeKMeans)

    PredictabilityConstrainedRouteCode(n_labels=2, alpha=1.0, random_state=7, max_iter=11, n_init=1).fit(
        _query_info(),
        _utility(),
        _embeddings(),
    )

    assert seen == {"n_clusters": 2, "random_state": 7, "n_init": 1, "max_iter": 11}
