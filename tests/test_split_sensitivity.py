import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path

from routecode.config import load_config
from routecode.eval.split_sensitivity import (
    cluster_heldout_split,
    domain_homogeneous_split,
    leave_one_group_split,
    ranking_correlation,
)


def toy_outcomes():
    rows = []
    for query_idx in range(8):
        dataset = "a" if query_idx < 4 else "b"
        domain = "math" if query_idx % 2 == 0 else "code"
        for model in ["m0", "m1"]:
            rows.append(
                {
                    "query_id": f"q{query_idx}",
                    "query_text": f"query {query_idx}",
                    "dataset": dataset,
                    "domain": domain,
                    "model_id": model,
                    "quality": 1.0 if model == "m0" else 0.0,
                    "cost_total": 0.0,
                    "judge": "toy",
                }
            )
    return pd.DataFrame(rows)


def toy_embeddings():
    return pd.DataFrame(
        [[idx, 0.0] if idx < 4 else [10.0 + idx, 0.0] for idx in range(8)],
        index=[f"q{idx}" for idx in range(8)],
    )


def test_leave_one_group_split_holds_group_out_by_query():
    split = leave_one_group_split(toy_outcomes(), group_column="dataset", holdout_value="b", seed=0)
    assert set(split.loc[split["dataset"] == "b", "split"]) == {"test"}
    assert "test" not in set(split.loc[split["dataset"] == "a", "split"])
    assert split.groupby("query_id")["split"].nunique().max() == 1


def test_domain_homogeneous_split_filters_to_one_domain_and_splits_queries():
    split = domain_homogeneous_split(toy_outcomes(), domain_value="math", seed=0)
    assert set(split["domain"]) == {"math"}
    assert set(split["split"]) == {"train", "val", "test"}
    assert split.groupby("query_id")["split"].nunique().max() == 1


def test_cluster_heldout_split_uses_embedding_cluster_as_test_group():
    outcomes = toy_outcomes()
    embeddings = toy_embeddings()
    split, heldout_cluster = cluster_heldout_split(outcomes, embeddings, n_clusters=2, heldout_cluster=1, seed=0)
    assert heldout_cluster == 1
    assert set(split["split"]).issuperset({"train", "val", "test"})
    assert split.groupby("query_id")["split"].nunique().max() == 1


def test_ranking_correlation_compares_common_methods():
    a = pd.DataFrame({"method": ["x", "y", "z"], "mean_utility": [3.0, 2.0, 1.0]})
    b = pd.DataFrame({"method": ["x", "y", "z"], "mean_utility": [1.0, 2.0, 3.0]})
    corr = ranking_correlation(a, b)
    assert np.isclose(corr, -1.0)


def test_build_scenarios_can_bound_pilot_scope():
    module_path = Path(__file__).resolve().parents[1] / "experiments" / "04_split_sensitivity.py"
    spec = importlib.util.spec_from_file_location("split_sensitivity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    outcomes = toy_outcomes()
    embeddings = toy_embeddings()
    config = {
        "run": {"random_seed": 0},
        "split": {"train_frac": 0.5, "val_frac": 0.25, "test_frac": 0.25},
        "split_sensitivity": {
            "max_group_scenarios": 1,
            "cluster_count": 2,
            "cluster_holdout_count": 1,
            "max_model_pool_scenarios": 1,
        },
    }
    scenarios = module.build_scenarios(outcomes, embeddings, config)
    scenario_types = [scenario_type for _, scenario_type, _, _ in scenarios]
    assert len(scenarios) == 6
    assert scenario_types == [
        "random",
        "leave_one_dataset_out",
        "leave_one_domain_out",
        "domain_homogeneous",
        "cluster_held_out",
        "model_pool_holdout",
    ]


def test_split_rate_k_values_uses_split_specific_override():
    module_path = Path(__file__).resolve().parents[1] / "experiments" / "04_split_sensitivity.py"
    spec = importlib.util.spec_from_file_location("split_sensitivity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {
        "routecode": {"k_values": [1, 2, 4, 8, 16, 32, 64, 128]},
        "split_sensitivity": {"k_values": [1, 4, 16]},
    }
    assert module.split_rate_k_values(config) == [1, 4, 16]


def test_split_rate_sweep_uses_rate_specific_fit_controls():
    module_path = Path(__file__).resolve().parents[1] / "experiments" / "04_split_sensitivity.py"
    spec = importlib.util.spec_from_file_location("split_sensitivity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = {
        "routecode": {"max_iter": 25},
        "split_sensitivity": {
            "classifier_max_iter": 300,
            "rate_classifier_max_iter": 75,
            "rate_codebook_max_iter": 9,
            "rate_codebook_n_init": 1,
        },
    }

    assert module.split_rate_fit_controls(config) == {
        "classifier_max_iter": 75,
        "codebook_max_iter": 9,
        "codebook_n_init": 1,
    }


def test_split_sensitivity_partial_tables_make_completed_scenarios_resumable(tmp_path):
    module_path = Path(__file__).resolve().parents[1] / "experiments" / "04_split_sensitivity.py"
    spec = importlib.util.spec_from_file_location("split_sensitivity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sensitivity_rows = [
        {"scenario": "random", "scenario_type": "random", "method": "best_single"},
        {"scenario": "leave_dataset_out:a", "scenario_type": "leave_one_dataset_out", "method": "SKIPPED"},
    ]
    rate_rows = [
        {"scenario": "random", "scenario_type": "random", "rate_log2K_to_80pct_learned_gain": float("nan")},
    ]

    module.write_partial_tables(tmp_path, sensitivity_rows, rate_rows)
    loaded_sensitivity, loaded_rate, completed = module.load_partial_tables(tmp_path)

    assert loaded_sensitivity == sensitivity_rows
    assert loaded_rate[0]["scenario"] == "random"
    assert loaded_rate[0]["scenario_type"] == "random"
    assert pd.isna(loaded_rate[0]["rate_log2K_to_80pct_learned_gain"])
    assert completed == {"random", "leave_dataset_out:a"}


def test_reference_values_for_scenario_come_from_completed_sensitivity_table():
    module_path = Path(__file__).resolve().parents[1] / "experiments" / "04_split_sensitivity.py"
    spec = importlib.util.spec_from_file_location("split_sensitivity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    sensitivity = pd.DataFrame(
        [
            {"scenario": "s0", "method": "best_single", "mean_utility": 0.5},
            {"scenario": "s0", "method": "kNN", "mean_utility": 0.7},
            {"scenario": "s0", "method": "logistic_embedding_router", "mean_utility": 0.6},
            {"scenario": "s0", "method": "routecode_predicted_labels", "mean_utility": 0.65},
            {"scenario": "s0", "method": "query_oracle", "mean_utility": 0.9},
        ]
    )

    values = module.reference_values_for_scenario(sensitivity, "s0")

    assert values == {
        "baseline_mean": 0.5,
        "learned_reference_mean": 0.7,
        "oracle_mean": 0.9,
    }


def test_broad20_split_sensitivity_uses_full_routecode_k_ladder():
    module_path = Path(__file__).resolve().parents[1] / "experiments" / "04_split_sensitivity.py"
    spec = importlib.util.spec_from_file_location("split_sensitivity_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "llmrouterbench_broad20.yaml")

    assert module.split_rate_k_values(config) == config["routecode"]["k_values"]
