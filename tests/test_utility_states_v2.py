import pandas as pd
import pytest

from routecode.states.utility_states_v2 import (
    EmbeddingStatePredictor,
    FrozenTransformerStatePredictor,
    TextCNNStatePredictor,
    TorchEmbeddingStatePredictor,
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


def test_model_holdout_repair_splits_state_with_hidden_model_conflict():
    utility = pd.DataFrame(
        {
            "local": [0.90, 0.88, 0.86, 0.84, 0.25, 0.23],
            "frontier": [0.30, 0.28, 0.26, 0.24, 0.95, 0.93],
            "new_model": [0.95, 0.93, 0.10, 0.08, 0.20, 0.18],
        },
        index=["easy_a", "easy_b", "easy_c", "easy_d", "hard_a", "hard_b"],
    )

    base = fit_utility_state_model(
        utility,
        method="relative_kmeans",
        n_states=2,
        random_state=13,
        local_models=("local",),
        frontier_models=("frontier", "new_model"),
    )
    repaired = fit_utility_state_model(
        utility,
        method="model_holdout_repaired",
        n_states=3,
        random_state=13,
        local_models=("local",),
        frontier_models=("frontier", "new_model"),
        model_holdout_variance_threshold=0.05,
        model_holdout_min_state_size=2,
    )

    assert base.labels.loc["easy_a"] == base.labels.loc["easy_c"]
    assert repaired.labels.loc["easy_a"] != repaired.labels.loc["easy_c"]
    assert repaired.n_states <= 3


def test_model_holdout_repair_keeps_requested_state_budget_after_splits():
    utility = pd.DataFrame(
        {
            "m0": [0.90, 0.89, 0.88, 0.87, 0.20, 0.21, 0.22, 0.23],
            "m1": [0.20, 0.21, 0.22, 0.23, 0.90, 0.89, 0.88, 0.87],
            "heldout": [0.95, 0.94, 0.10, 0.09, 0.92, 0.91, 0.12, 0.11],
        },
        index=[f"q{i}" for i in range(8)],
    )

    model = fit_utility_state_model(
        utility,
        method="model_holdout_repaired",
        n_states=4,
        random_state=19,
        local_models=("m0", "m1"),
        frontier_models=("heldout",),
        model_holdout_variance_threshold=0.02,
        model_holdout_min_state_size=2,
    )

    assert model.n_states <= 4
    assert model.n_states >= 2
    assert model.labels.index.equals(utility.index)


def test_model_holdout_repair_uses_calibration_aware_features():
    utility = pd.DataFrame(
        {
            "m0": [0.90, 0.89, 0.20, 0.21],
            "m1": [0.20, 0.21, 0.90, 0.89],
            "heldout": [0.95, 0.10, 0.92, 0.11],
        },
        index=[f"q{i}" for i in range(4)],
    )

    model = fit_utility_state_model(
        utility,
        method="model_holdout_repaired",
        n_states=2,
        random_state=23,
        local_models=("m0", "m1"),
        frontier_models=("heldout",),
        model_holdout_variance_threshold=0.02,
        model_holdout_min_state_size=2,
    )

    assert any(column.startswith("holdout_raw::") for column in model.feature_columns)
    assert any(column.startswith("holdout_centered::") for column in model.feature_columns)


def test_model_holdout_repair_can_keep_extra_split_states_for_calibration():
    utility = pd.DataFrame(
        {
            "m0": [0.90, 0.89, 0.88, 0.87, 0.20, 0.21, 0.22, 0.23],
            "m1": [0.20, 0.21, 0.22, 0.23, 0.90, 0.89, 0.88, 0.87],
            "heldout": [0.95, 0.94, 0.10, 0.09, 0.92, 0.91, 0.12, 0.11],
        },
        index=[f"q{i}" for i in range(8)],
    )

    model = fit_utility_state_model(
        utility,
        method="model_holdout_repaired",
        n_states=2,
        random_state=29,
        local_models=("m0", "m1"),
        frontier_models=("heldout",),
        model_holdout_variance_threshold=0.02,
        model_holdout_min_state_size=2,
        model_holdout_preserve_state_budget=False,
    )

    assert model.n_states > 2
    assert model.repair_summary is not None
    assert model.repair_summary["states_merged"] == 0


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


def test_embedding_state_predictor_mlp_handles_tiny_class_counts():
    labels = pd.Series(["z0", "z1"], index=["q0", "q1"])
    embeddings = pd.DataFrame([[0.0, 0.0], [1.0, 1.0]], index=labels.index)

    predictor = EmbeddingStatePredictor(kind="mlp", max_iter=20, random_state=11).fit(embeddings, labels)
    result = predictor.predict(embeddings)

    assert set(result.probabilities.columns) == {"z0", "z1"}
    assert result.confidence.between(0.0, 1.0).all()


def test_torch_embedding_state_predictor_learns_separable_states():
    labels = pd.Series(["z0", "z0", "z1", "z1"], index=["a", "b", "c", "d"])
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]],
        index=labels.index,
    )

    predictor = TorchEmbeddingStatePredictor(epochs=80, batch_size=4, random_state=3).fit(embeddings, labels)
    result = predictor.predict(embeddings)

    assert result.labels.astype(str).eq(labels.astype(str)).all()
    assert set(result.probabilities.columns) == {"z0", "z1"}
    assert result.confidence.min() > 0.50


def test_text_cnn_state_predictor_uses_query_tokens():
    texts = pd.Series(
        [
            "write python function binary search",
            "debug python list index error",
            "prove algebra theorem with equation",
            "solve integral symbolic expression",
        ],
        index=["code1", "code2", "math1", "math2"],
    )
    labels = pd.Series(["code", "code", "math", "math"], index=texts.index)

    predictor = TextCNNStatePredictor(
        epochs=100,
        batch_size=4,
        max_length=8,
        embedding_dim=16,
        channels=8,
        random_state=5,
    ).fit(texts, labels)
    result = predictor.predict(texts)

    assert result.labels.astype(str).eq(labels.astype(str)).all()
    assert result.confidence.between(0.0, 1.0).all()


def test_frozen_transformer_state_predictor_can_use_injected_encoder():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
            rows = []
            for text in texts:
                token = 1 if "code" in text else 2
                rows.append([token, 0, 0])
            return {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.tensor([[1, 0, 0] for _ in rows], dtype=torch.long),
            }

    class FakeEncoder(torch.nn.Module):
        config = type("Config", (), {"hidden_size": 2})()

        def forward(self, input_ids, attention_mask):
            hidden = torch.zeros((input_ids.shape[0], input_ids.shape[1], 2), dtype=torch.float32)
            hidden[:, 0, 0] = (input_ids[:, 0] == 1).float()
            hidden[:, 0, 1] = (input_ids[:, 0] == 2).float()
            return type("Output", (), {"last_hidden_state": hidden})()

    texts = pd.Series(["code task", "more code", "math task", "more math"], index=["c1", "c2", "m1", "m2"])
    labels = pd.Series(["code", "code", "math", "math"], index=texts.index)

    predictor = FrozenTransformerStatePredictor(
        tokenizer=FakeTokenizer(),
        encoder=FakeEncoder(),
        epochs=80,
        batch_size=4,
        random_state=7,
    ).fit(texts, labels)
    result = predictor.predict(texts)

    assert result.labels.astype(str).eq(labels.astype(str)).all()
    assert set(result.probabilities.columns) == {"code", "math"}
