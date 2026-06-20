import pandas as pd

from routecode.eval.new_model_calibration import (
    active_calibration_priority_by_label,
    active_state_calibration_priority,
    budgeted_direct_oracle_labels,
    calibrate_new_model_by_active_state,
    calibrate_new_model_by_label,
    conservative_state_model_update,
    fit_predict_budgeted_direct_router,
    sample_active_calibration_queries_by_label,
    sample_active_state_calibration_queries,
    sample_calibration_queries_per_label,
    sample_dataset_stratified_calibration_queries,
    sample_embedding_cluster_calibration_queries,
    sample_random_calibration_queries,
)


def test_sample_calibration_queries_per_label_caps_each_label():
    labels = pd.Series([0, 0, 0, 1, 1, 2], index=[f"q{i}" for i in range(6)])

    sampled = sample_calibration_queries_per_label(labels, examples_per_label=2, seed=0)

    sampled_labels = labels.loc[sampled]
    assert len(sampled) == 5
    assert sampled_labels.value_counts().to_dict() == {0: 2, 1: 2, 2: 1}


def test_active_calibration_sampling_prioritizes_high_traffic_low_margin_labels():
    labels = pd.Series(
        [0, 0, 0, 0, 1, 1, 2, 2],
        index=[f"q{i}" for i in range(8)],
        name="route_label",
    )
    base_label_utility = pd.DataFrame(
        {
            "old_a": [0.60, 0.90, 0.80],
            "old_b": [0.59, 0.10, 0.20],
        },
        index=[0, 1, 2],
    )

    priority = active_calibration_priority_by_label(labels, base_label_utility)
    sampled = sample_active_calibration_queries_by_label(
        labels,
        base_label_utility,
        total_budget=4,
        seed=0,
    )

    assert priority.loc[0] > priority.loc[1]
    assert len(sampled) == 4
    sampled_labels = labels.loc[sampled]
    assert sampled_labels.value_counts().idxmax() == 0
    assert sampled_labels.value_counts().loc[0] >= 2


def test_active_state_calibration_priority_targets_common_uncertain_high_gain_state():
    labels = pd.Series(
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 2],
        index=[f"q{i}" for i in range(10)],
        name="route_label",
    )
    base_state_utility = pd.DataFrame(
        {
            "cheap_local": [0.55, 0.90, 0.55],
            "old_code": [0.50, 0.20, 0.45],
        },
        index=[0, 1, 2],
    )
    observations = pd.DataFrame(
        [
            {"query_id": "q0", "state_label": 0, "utility": 0.88},
            {"query_id": "q1", "state_label": 0, "utility": 0.82},
            {"query_id": "q5", "state_label": 1, "utility": 0.20},
            {"query_id": "q9", "state_label": 2, "utility": 0.95},
        ]
    )

    priority = active_state_calibration_priority(
        labels,
        base_state_utility,
        observations=observations,
        prior_mean=0.50,
        prior_strength=2.0,
    )

    assert priority.iloc[0]["state_label"] == 0
    assert priority.loc[priority["state_label"].eq(0), "value_of_calibration"].iloc[0] > priority.loc[
        priority["state_label"].eq(1), "value_of_calibration"
    ].iloc[0]
    assert priority.loc[priority["state_label"].eq(0), "traffic_mass"].iloc[0] > priority.loc[
        priority["state_label"].eq(2), "traffic_mass"
    ].iloc[0]
    assert set(
        [
            "posterior_mean",
            "posterior_variance",
            "prob_new_beats_current",
            "expected_positive_gain",
            "value_of_calibration",
        ]
    ).issubset(priority.columns)


def test_active_state_calibration_sampling_uses_scouts_then_value_and_query_scores():
    labels = pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=[f"q{i}" for i in range(6)],
        name="route_label",
    )
    base_state_utility = pd.DataFrame(
        {
            "cheap_local": [0.45, 0.88],
            "old_code": [0.40, 0.30],
        },
        index=[0, 1],
    )
    query_features = pd.DataFrame(
        {
            "representativeness": [0.10, 1.00, 0.80, 0.30, 0.90, 0.20],
            "uncertainty": [0.10, 0.20, 0.90, 0.10, 0.10, 0.10],
            "routing_impact": [0.10, 0.60, 0.40, 0.10, 0.80, 0.10],
        },
        index=labels.index,
    )

    selected = sample_active_state_calibration_queries(
        labels,
        base_state_utility,
        query_features=query_features,
        total_budget=4,
        scout_per_state=1,
        prior_mean=0.50,
        prior_strength=2.0,
    )

    assert len(selected) == 4
    assert selected["query_id"].is_unique
    assert selected[selected["selection_phase"].eq("scout")]["state_label"].tolist() == [0, 1]
    assert selected.loc[selected["state_label"].eq(0), "query_id"].tolist()[0] == "q2"
    assert selected["state_label"].value_counts().loc[0] == 3


def test_conservative_state_model_update_switches_only_with_margin_and_confidence():
    base_state_utility = pd.DataFrame(
        {
            "cheap_local": [0.55, 0.90],
            "old_code": [0.50, 0.20],
        },
        index=[0, 1],
    )
    posterior = pd.DataFrame(
        [
            {
                "state_label": 0,
                "posterior_mean": 0.74,
                "prob_new_beats_current": 0.96,
                "current_best_utility": 0.55,
                "current_best_model": "cheap_local",
            },
            {
                "state_label": 1,
                "posterior_mean": 0.91,
                "prob_new_beats_current": 0.60,
                "current_best_utility": 0.90,
                "current_best_model": "cheap_local",
            },
        ]
    )

    update = conservative_state_model_update(
        base_state_utility,
        posterior,
        new_model_id="new_code_model",
        delta=0.01,
        tau=0.90,
    )

    by_state = update.set_index("state_label")
    assert by_state.loc[0, "selected_model"] == "new_code_model"
    assert by_state.loc[0, "switch_to_new_model"] is True
    assert by_state.loc[1, "selected_model"] == "cheap_local"
    assert by_state.loc[1, "switch_to_new_model"] is False


def test_active_state_calibration_loop_onboards_new_model_with_conservative_update():
    labels = pd.Series(
        [0, 0, 0, 1, 1, 1],
        index=[f"q{i}" for i in range(6)],
        name="route_label",
    )
    base_state_utility = pd.DataFrame(
        {
            "cheap_local": [0.55, 0.90],
            "old_code": [0.50, 0.30],
        },
        index=[0, 1],
    )
    full_utility = pd.DataFrame(
        {
            "cheap_local": [0.50, 0.55, 0.60, 0.90, 0.88, 0.92],
            "old_code": [0.50, 0.45, 0.55, 0.30, 0.25, 0.35],
            "new_code": [1.00, 0.95, 1.00, 0.10, 0.20, 0.30],
        },
        index=labels.index,
    )
    query_features = pd.DataFrame(
        {
            "representativeness": [1.00, 0.90, 0.80, 1.00, 0.90, 0.80],
            "uncertainty": [0.20, 0.30, 0.40, 0.20, 0.30, 0.40],
            "routing_impact": [0.90, 0.80, 0.70, 0.20, 0.20, 0.20],
        },
        index=labels.index,
    )

    result = calibrate_new_model_by_active_state(
        labels=labels,
        base_state_utility=base_state_utility,
        full_utility=full_utility,
        new_model_id="new_code",
        total_budget=4,
        query_features=query_features,
        scout_per_state=1,
        prior_mean=0.50,
        prior_strength=2.0,
        delta=0.01,
        tau=0.80,
    )

    assert result.calibration_query_count == 4
    assert result.label_to_model[0] == "new_code"
    assert result.label_to_model[1] == "cheap_local"
    assert result.selected_queries["query_id"].is_unique
    assert set(result.selected_queries["selection_phase"]) == {"scout", "active"}
    by_state = result.table_update.set_index("state_label")
    assert by_state.loc[0, "switch_to_new_model"] is True
    assert by_state.loc[1, "switch_to_new_model"] is False


def test_random_calibration_sampling_matches_budget_without_label_stratification():
    labels = pd.Series(
        [0, 0, 0, 1, 1, 2],
        index=[f"q{i}" for i in range(6)],
        name="route_label",
    )

    sampled = sample_random_calibration_queries(labels, total_budget=4, seed=3)
    repeated = sample_random_calibration_queries(labels, total_budget=4, seed=3)
    capped = sample_random_calibration_queries(labels, total_budget=100, seed=3)

    assert len(sampled) == 4
    assert sampled.equals(repeated)
    assert len(set(sampled)) == 4
    assert set(sampled).issubset(set(labels.index))
    assert len(capped) == len(labels)


def test_dataset_stratified_calibration_sampling_covers_dataset_groups():
    labels = pd.Series(
        [0, 0, 0, 1, 1, 2],
        index=[f"q{i}" for i in range(6)],
        name="route_label",
    )
    query_info = pd.DataFrame(
        {"dataset": ["a", "a", "a", "b", "b", "c"]},
        index=labels.index,
    )

    sampled = sample_dataset_stratified_calibration_queries(
        labels,
        query_info,
        total_budget=4,
        seed=5,
    )
    repeated = sample_dataset_stratified_calibration_queries(
        labels,
        query_info,
        total_budget=4,
        seed=5,
    )

    assert len(sampled) == 4
    assert sampled.equals(repeated)
    assert len(set(sampled)) == 4
    assert set(query_info.loc[sampled, "dataset"]) == {"a", "b", "c"}


def test_embedding_cluster_calibration_sampling_covers_embedding_clusters():
    labels = pd.Series(
        [0, 0, 0, 1, 1, 2],
        index=[f"q{i}" for i in range(6)],
        name="route_label",
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

    sampled = sample_embedding_cluster_calibration_queries(
        labels,
        embeddings,
        total_budget=4,
        n_clusters=2,
        seed=7,
    )
    repeated = sample_embedding_cluster_calibration_queries(
        labels,
        embeddings,
        total_budget=4,
        n_clusters=2,
        seed=7,
    )

    assert len(sampled) == 4
    assert sampled.equals(repeated)
    assert len(set(sampled)) == 4
    assert set(sampled).intersection({"q0", "q1", "q2"})
    assert set(sampled).intersection({"q3", "q4", "q5"})


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


def test_fit_predict_budgeted_direct_router_supports_fast_logistic_solver():
    train_embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]],
        index=["q0", "q1", "q2", "q3"],
    )
    test_embeddings = pd.DataFrame([[0.05, 0.0], [4.05, 4.0]], index=["t0", "t1"])
    train_labels = pd.Series(["old", "old", "new", "new"], index=train_embeddings.index)

    selected = fit_predict_budgeted_direct_router(
        method="logistic",
        train_labels=train_labels,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        random_state=0,
        max_iter=25,
        logistic_solver="saga",
        tol=0.01,
    )

    assert selected.to_dict() == {"t0": "old", "t1": "new"}


def test_fit_predict_budgeted_direct_router_supports_fast_svm_backend():
    train_embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]],
        index=["q0", "q1", "q2", "q3"],
    )
    test_embeddings = pd.DataFrame([[0.05, 0.0], [4.05, 4.0]], index=["t0", "t1"])
    train_labels = pd.Series(["old", "old", "new", "new"], index=train_embeddings.index)

    selected = fit_predict_budgeted_direct_router(
        method="svm",
        train_labels=train_labels,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        random_state=0,
        max_iter=25,
        svm_backend="sgd",
        tol=0.01,
    )

    assert selected.to_dict() == {"t0": "old", "t1": "new"}
