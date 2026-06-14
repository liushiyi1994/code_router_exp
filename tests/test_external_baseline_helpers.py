from __future__ import annotations

import json

import pandas as pd

from routecode.eval.external_baselines import (
    AvengersProClusterRouter,
    StrongWeakPair,
    build_avengerspro_records,
    build_routellm_mf_assets,
    build_routellm_pairwise_records,
    choose_strong_weak_pair,
    load_official_routellm_artifacts,
)
from routecode.matrix import Matrices


def test_choose_strong_weak_pair_defaults_to_train_best_and_worst_mean_utility():
    utility = pd.DataFrame(
        {
            "a": [0.8, 0.9],
            "b": [0.2, 0.3],
            "c": [0.5, 0.4],
        },
        index=["q0", "q1"],
    )

    pair = choose_strong_weak_pair(utility)

    assert pair.strong_model == "a"
    assert pair.weak_model == "b"


def test_choose_strong_weak_pair_respects_configured_pair():
    utility = pd.DataFrame({"a": [0.8], "b": [0.2], "c": [0.5]}, index=["q0"])

    pair = choose_strong_weak_pair(utility, strong_model="c", weak_model="a")

    assert pair.strong_model == "c"
    assert pair.weak_model == "a"


def test_load_official_routellm_artifacts_parses_json_and_seed_csvs(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "mf_results_seed42.json").write_text(
        json.dumps(
            {
                "total": 10,
                "ties": 1,
                "decisive_total": 9,
                "selection_accuracy": 0.7,
                "routing_accuracy": 0.6,
                "total_cost": 1.25,
                "avg_cost": 0.125,
                "datasets": {
                    "aime": {
                        "total": 2,
                        "ties": 0,
                        "decisive_total": 2,
                        "selection_accuracy": 0.5,
                        "routing_accuracy": 1.0,
                        "total_cost": 0.2,
                        "avg_cost": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (results / "mf_selection_accuracy_by_seed.csv").write_text(
        "seed,aime,sample_avg\n42,50.0,70.0\n",
        encoding="utf-8",
    )
    (results / "mf_total_cost_by_seed.csv").write_text(
        "seed,aime,total_cost\n42,0.2,1.25\n",
        encoding="utf-8",
    )

    table = load_official_routellm_artifacts(results)

    assert set(table["scope"]) == {"overall", "dataset"}
    overall = table[table["scope"].eq("overall")].iloc[0]
    dataset = table[table["dataset"].eq("aime")].iloc[0]
    assert overall["seed"] == 42
    assert overall["selection_accuracy"] == 0.7
    assert overall["csv_selection_accuracy"] == 0.7
    assert overall["csv_total_cost"] == 1.25
    assert dataset["selection_accuracy"] == 0.5
    assert dataset["csv_selection_accuracy"] == 0.5
    assert dataset["csv_total_cost"] == 0.2
    assert not table["split_aligned_with_routecode"].any()
    assert not table["routecode_metric_compatible"].any()


def test_build_routellm_pairwise_records_exports_split_aligned_winners():
    train = _tiny_matrices("train", ["q0", "q1"])
    test = _tiny_matrices("test", ["q2", "q3"])
    pair = StrongWeakPair(strong_model="strong", weak_model="weak")

    records = build_routellm_pairwise_records({"train": train, "test": test}, pair)

    assert set(records) == {"train", "test"}
    assert {row["query_id"] for row in records["train"]}.isdisjoint(
        {row["query_id"] for row in records["test"]}
    )
    assert [row["winner"] for row in records["train"]] == ["model_a", "model_b"]
    assert [row["winner"] for row in records["test"]] == ["model_a", "tie"]
    first = records["train"][0]
    assert first["split"] == "train"
    assert first["model_a"] == "strong"
    assert first["model_b"] == "weak"
    assert first["model_a_utility"] == 0.8
    assert first["model_b_utility"] == 0.2
    assert first["model_a_quality"] == 0.9
    assert first["model_b_quality"] == 0.4
    assert first["model_a_cost"] == 0.1
    assert first["model_b_cost"] == 0.2
    assert first["prompt"] == "Prompt q0"
    assert first["dataset"] == "dataset_train"
    assert first["domain"] == "domain_train"


def test_build_routellm_pairwise_records_rejects_missing_pair_model():
    train = _tiny_matrices("train", ["q0"])
    pair = StrongWeakPair(strong_model="strong", weak_model="missing")

    try:
        build_routellm_pairwise_records({"train": train}, pair)
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected a missing pair model ValueError")


def test_build_routellm_mf_assets_exports_official_trainer_schema():
    train = _tiny_matrices("train", ["q0", "q1"])
    test = _tiny_matrices("test", ["q2", "q3"])
    pair = StrongWeakPair(strong_model="strong", weak_model="weak")
    pairwise = build_routellm_pairwise_records({"train": train, "test": test}, pair)
    embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        index=["q0", "q1", "q2", "q3"],
        columns=["x0", "x1"],
    )

    assets = build_routellm_mf_assets(pairwise, embeddings)

    assert assets.prompt_embeddings.shape == (4, 2)
    assert assets.prompt_index == {"q0": 0, "q1": 1, "q2": 2, "q3": 3}
    assert [row["winner"] for row in assets.train_records] == ["model_a", "model_b"]
    assert [row["winner"] for row in assets.test_records] == ["model_a", "tie"]
    first = assets.train_records[0]
    assert first["idx"] == 0
    assert first["dataset_id"] == "dataset_train"
    assert first["score_model_a"] == 0.9
    assert first["score_model_b"] == 0.4
    assert first["cost_model_a"] == 0.1
    assert first["cost_model_b"] == 0.2
    assert first["utility_model_a"] == 0.8
    assert first["utility_model_b"] == 0.2
    assert first["utility_winner"] == "model_a"
    assert first["winner_objective"] == "quality"


def test_build_routellm_mf_assets_rejects_missing_embedding():
    train = _tiny_matrices("train", ["q0"])
    pair = StrongWeakPair(strong_model="strong", weak_model="weak")
    pairwise = build_routellm_pairwise_records({"train": train}, pair)
    embeddings = pd.DataFrame([[1.0, 0.0]], index=["other"], columns=["x0", "x1"])

    try:
        build_routellm_mf_assets(pairwise, embeddings)
    except ValueError as exc:
        assert "embedding" in str(exc)
    else:
        raise AssertionError("Expected a missing embedding ValueError")


def test_build_avengerspro_records_exports_split_aligned_jsonl_schema():
    train = _tiny_matrices("train", ["q0", "q1"])
    test = _tiny_matrices("test", ["q2", "q3"])

    assets = build_avengerspro_records({"train": train, "test": test})

    assert {row["query_id"] for row in assets.train_records}.isdisjoint(
        {row["query_id"] for row in assets.test_records}
    )
    first = assets.train_records[0]
    assert first["query"] == "Prompt q0"
    assert first["dataset"] == "dataset_train"
    assert first["domain"] == "domain_train"
    assert first["records"] == {"strong": 0.9, "weak": 0.4}
    assert first["utilities"] == {"strong": 0.8, "weak": 0.2}
    assert first["usages"]["strong"]["cost"] == 0.1
    assert first["usages"]["strong"]["prompt_tokens"] == 0
    assert assets.baseline_scores["strong"]["dataset_test"] == 70.0
    assert assets.baseline_scores["weak"]["dataset_test"] == 60.0


def test_avengerspro_cluster_router_supports_quality_and_balance_modes():
    query_info = pd.DataFrame(
        {
            "query_id": ["q0", "q1", "q2", "q3"],
            "dataset": ["d"] * 4,
            "domain": ["d"] * 4,
        }
    ).set_index("query_id")
    quality = pd.DataFrame(
        {
            "accurate_expensive": [0.9, 0.8, 0.9, 0.8],
            "cheap_close": [0.7, 0.7, 0.7, 0.7],
        },
        index=query_info.index,
    )
    cost = pd.DataFrame(
        {
            "accurate_expensive": [10.0, 10.0, 10.0, 10.0],
            "cheap_close": [1.0, 1.0, 1.0, 1.0],
        },
        index=query_info.index,
    )
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]],
        index=query_info.index,
    )

    simple = AvengersProClusterRouter(n_clusters=2, mode="simple", random_state=0).fit(
        query_info.iloc[:2],
        quality.iloc[:2],
        cost.iloc[:2],
        embeddings,
    )
    balanced = AvengersProClusterRouter(
        n_clusters=2,
        mode="balance",
        performance_weight=0.25,
        cost_sensitivity=0.75,
        random_state=0,
    ).fit(query_info.iloc[:2], quality.iloc[:2], cost.iloc[:2], embeddings)

    assert simple.predict(query_info.iloc[2:], embeddings).tolist() == [
        "accurate_expensive",
        "accurate_expensive",
    ]
    assert balanced.predict(query_info.iloc[2:], embeddings).tolist() == ["cheap_close", "cheap_close"]
    assert set(simple.predict_labels(embeddings.iloc[2:])) <= {0, 1}


def _tiny_matrices(split: str, query_ids: list[str]) -> Matrices:
    utility_values = {
        "q0": [0.8, 0.2],
        "q1": [0.1, 0.6],
        "q2": [0.7, 0.4],
        "q3": [0.5, 0.5],
    }
    quality_values = {
        "q0": [0.9, 0.4],
        "q1": [0.2, 0.7],
        "q2": [0.8, 0.6],
        "q3": [0.6, 0.6],
    }
    cost_values = {
        "q0": [0.1, 0.2],
        "q1": [0.1, 0.3],
        "q2": [0.2, 0.2],
        "q3": [0.1, 0.1],
    }
    columns = ["strong", "weak"]
    query_info = pd.DataFrame(
        {
            "query_id": query_ids,
            "prompt": [f"Prompt {query_id}" for query_id in query_ids],
            "dataset": [f"dataset_{split}"] * len(query_ids),
            "domain": [f"domain_{split}"] * len(query_ids),
        }
    ).set_index("query_id")
    return Matrices(
        utility=pd.DataFrame([utility_values[query_id] for query_id in query_ids], index=query_ids, columns=columns),
        quality=pd.DataFrame([quality_values[query_id] for query_id in query_ids], index=query_ids, columns=columns),
        cost=pd.DataFrame([cost_values[query_id] for query_id in query_ids], index=query_ids, columns=columns),
        query_info=query_info,
        model_ids=columns,
    )
