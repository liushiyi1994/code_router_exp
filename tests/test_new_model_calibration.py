import pandas as pd

from routecode.eval.new_model_calibration import (
    budgeted_direct_oracle_labels,
    calibrate_new_model_by_label,
    fit_predict_budgeted_direct_router,
    sample_calibration_queries_per_label,
)


def test_sample_calibration_queries_per_label_caps_each_label():
    labels = pd.Series([0, 0, 0, 1, 1, 2], index=[f"q{i}" for i in range(6)])

    sampled = sample_calibration_queries_per_label(labels, examples_per_label=2, seed=0)

    sampled_labels = labels.loc[sampled]
    assert len(sampled) == 5
    assert sampled_labels.value_counts().to_dict() == {0: 2, 1: 2, 2: 1}


def test_calibrate_new_model_by_label_updates_only_labels_where_sampled_new_model_wins():
    labels = pd.Series([0, 0, 1, 1], index=["q0", "q1", "q2", "q3"])
    base_label_utility = pd.DataFrame(
        {
            "old_a": [0.8, 0.4],
            "old_b": [0.7, 0.5],
        },
        index=[0, 1],
    )
    full_utility = pd.DataFrame(
        {
            "old_a": [0.8, 0.82, 0.4, 0.42],
            "old_b": [0.7, 0.72, 0.5, 0.52],
            "new": [0.1, 0.2, 0.95, 0.97],
        },
        index=labels.index,
    )

    result = calibrate_new_model_by_label(
        labels=labels,
        base_label_utility=base_label_utility,
        full_utility=full_utility,
        new_model_id="new",
        calibration_query_ids=pd.Index(["q0", "q2"]),
    )

    assert result.label_to_model[0] == "old_a"
    assert result.label_to_model[1] == "new"
    assert result.estimated_new_model_utility.loc[0] == 0.1
    assert result.estimated_new_model_utility.loc[1] == 0.95


def test_budgeted_direct_oracle_labels_do_not_use_new_model_outside_calibration_ids():
    base_utility = pd.DataFrame(
        {
            "old_a": [0.8, 0.2, 0.3],
            "old_b": [0.7, 0.4, 0.5],
        },
        index=["q0", "q1", "q2"],
    )
    full_utility = base_utility.assign(new=[0.9, 0.95, 0.99])

    labels = budgeted_direct_oracle_labels(
        base_utility=base_utility,
        full_utility=full_utility,
        new_model_id="new",
        calibration_query_ids=pd.Index(["q1"]),
    )

    assert labels.loc["q0"] == "old_a"
    assert labels.loc["q1"] == "new"
    assert labels.loc["q2"] == "old_b"


def test_fit_predict_budgeted_direct_router_supports_stronger_estimators():
    train_embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]],
        index=["q0", "q1", "q2", "q3"],
    )
    test_embeddings = pd.DataFrame(
        [[0.05, 0.0], [4.05, 4.0]],
        index=["t0", "t1"],
    )
    train_labels = pd.Series(["old", "old", "new", "new"], index=train_embeddings.index)

    for method in ["logistic", "svm", "knn", "mlp", "gradient_boosting"]:
        selected = fit_predict_budgeted_direct_router(
            method=method,
            train_labels=train_labels,
            train_embeddings=train_embeddings,
            test_embeddings=test_embeddings,
            random_state=0,
            max_iter=1000,
            n_neighbors=1,
        )

        assert list(selected.index) == ["t0", "t1"]
        assert selected.loc["t0"] == "old"
        assert selected.loc["t1"] == "new"
