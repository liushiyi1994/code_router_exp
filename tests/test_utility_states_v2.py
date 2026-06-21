import pandas as pd
import pytest

from routecode.states.utility_states_v2 import (
    EmbeddingStatePredictor,
    build_relative_utility_features,
    confidence_trigger_mask,
    fit_utility_state_model,
    select_confidence_threshold,
    state_policy,
)


def test_relative_utility_features_remove_absolute_difficulty():
    utility = pd.DataFrame(
        {
            "m1": [0.90, 0.40],
            "m2": [0.80, 0.30],
            "m3": [0.70, 0.20],
        },
        index=["easy", "hard"],
    )

    features = build_relative_utility_features(utility, tau=0.2)

    assert features.loc["easy", "centered::m1"] == pytest.approx(features.loc["hard", "centered::m1"])
    assert features.loc["easy", "regret::m3"] == pytest.approx(features.loc["hard", "regret::m3"])
    assert features.loc["easy", "rank::m2"] == pytest.approx(features.loc["hard", "rank::m2"])
    assert features.loc["easy", "margin::best"] == pytest.approx(features.loc["hard", "margin::best"])


def test_relative_state_model_groups_same_routing_pattern_across_difficulty():
    utility = pd.DataFrame(
        {
            "cheap": [0.90, 0.40, 0.20, 0.25],
            "strong": [0.80, 0.30, 0.95, 0.75],
            "other": [0.70, 0.20, 0.10, 0.05],
        },
        index=["easy_cheap", "hard_cheap", "easy_strong", "hard_strong"],
    )

    model = fit_utility_state_model(
        utility,
        method="relative_kmeans",
        n_states=2,
        random_state=3,
        local_models=("cheap",),
        frontier_models=("strong",),
    )

    assert model.labels.loc["easy_cheap"] == model.labels.loc["hard_cheap"]
    assert model.labels.loc["easy_strong"] == model.labels.loc["hard_strong"]
    assert model.labels.loc["easy_cheap"] != model.labels.loc["easy_strong"]

    selected = state_policy(model.labels, model.label_to_model, model.fallback_model)
    assert selected.loc["easy_cheap"] == "cheap"
    assert selected.loc["easy_strong"] == "strong"


def test_two_stage_state_model_keeps_frontier_and_local_regimes_separate():
    utility = pd.DataFrame(
        {
            "local_a": [0.90, 0.88, 0.35, 0.30, 0.54, 0.53],
            "frontier": [0.40, 0.38, 0.95, 0.92, 0.55, 0.54],
            "local_b": [0.82, 0.80, 0.25, 0.20, 0.53, 0.52],
        },
        index=[f"q{i}" for i in range(6)],
    )

    model = fit_utility_state_model(
        utility,
        method="two_stage_relative_kmeans",
        n_states=3,
        random_state=7,
        local_models=("local_a", "local_b"),
        frontier_models=("frontier",),
    )

    local_states = set(model.labels.loc[["q0", "q1"]])
    frontier_states = set(model.labels.loc[["q2", "q3"]])
    assert local_states.isdisjoint(frontier_states)
    assert model.coarse_allocations is not None
    assert {"local_enough", "frontier_needed"}.issubset(set(model.coarse_allocations))


def test_embedding_state_predictor_returns_confidence_and_probe_mask():
    labels = pd.Series(
        ["z0", "z0", "z0", "z1", "z1", "z1"],
        index=[f"q{i}" for i in range(6)],
    )
    embeddings = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ],
        index=labels.index,
    )

    predictor = EmbeddingStatePredictor(kind="knn", n_neighbors=3).fit(embeddings, labels)
    result = predictor.predict(embeddings)
    threshold = select_confidence_threshold(
        result.confidence,
        result.labels,
        labels,
        max_probe_rate=0.5,
    )
    probe = confidence_trigger_mask(result.confidence, threshold)

    assert set(result.labels) == {"z0", "z1"}
    assert result.confidence.between(0.0, 1.0).all()
    assert probe.index.equals(labels.index)
    assert threshold >= 0.0
