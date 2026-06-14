import pandas as pd

from routecode.eval.sensitivity import (
    inject_label_noise,
    misestimate_cost_utility,
    query_length_buckets,
)
from routecode.routers.cluster_lookup import AgglomerativeClusterRouter


def test_inject_label_noise_changes_seeded_fraction_without_unknown_labels():
    labels = pd.Series(["a", "a", "b", "b", "c"], index=[f"q{i}" for i in range(5)])

    noisy = inject_label_noise(labels, choices=["a", "b", "c"], noise_rate=0.4, seed=3)

    assert (noisy != labels).sum() == 2
    assert set(noisy).issubset({"a", "b", "c"})
    assert noisy.equals(inject_label_noise(labels, choices=["a", "b", "c"], noise_rate=0.4, seed=3))


def test_misestimate_cost_utility_scales_cost_before_lambda():
    quality = pd.DataFrame({"m0": [1.0], "m1": [0.5]}, index=["q0"])
    cost = pd.DataFrame({"m0": [0.4], "m1": [0.1]}, index=["q0"])

    utility = misestimate_cost_utility(quality, cost, lambda_cost=0.5, cost_multiplier=2.0)

    assert utility.loc["q0", "m0"] == 0.6
    assert utility.loc["q0", "m1"] == 0.4


def test_query_length_buckets_assigns_short_medium_long():
    query_info = pd.DataFrame(
        {
            "query_text": [
                "tiny",
                "a few words",
                "this is a considerably longer query text",
                "short too",
            ]
        },
        index=["q0", "q1", "q2", "q3"],
    )

    buckets = query_length_buckets(query_info, n_bins=3)

    assert list(buckets.index) == ["q0", "q1", "q2", "q3"]
    assert buckets.loc["q0"] == "short"
    assert buckets.loc["q2"] == "long"


def test_agglomerative_cluster_router_uses_embedding_clusters_and_centroid_prediction():
    utility = pd.DataFrame(
        {
            "m0": [0.9, 0.85, 0.2, 0.25],
            "m1": [0.3, 0.35, 0.95, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )
    query_info = pd.DataFrame({"dataset": ["a", "a", "b", "b"]}, index=utility.index)
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0]],
        index=utility.index,
    )

    router = AgglomerativeClusterRouter(n_clusters=2).fit(query_info, utility, embeddings)
    selected = router.predict(query_info, embeddings)

    assert selected.loc["q0"] == "m0"
    assert selected.loc["q3"] == "m1"
