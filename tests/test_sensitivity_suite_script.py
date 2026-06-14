from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from routecode.matrix import Matrices


def _load_sensitivity_suite():
    path = Path(__file__).resolve().parents[1] / "experiments" / "09_sensitivity_suite.py"
    spec = importlib.util.spec_from_file_location("sensitivity_suite", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_query_length_rows_use_bucket_local_references(monkeypatch):
    module = _load_sensitivity_suite()
    query_ids = ["q0", "q1", "q2", "q3", "q4", "q5"]
    train = Matrices(
        quality=pd.DataFrame({"m0": [0.9, 0.8, 0.4, 0.3], "m1": [0.2, 0.3, 0.8, 0.9]}, index=query_ids[:4]),
        cost=pd.DataFrame(0.0, index=query_ids[:4], columns=["m0", "m1"]),
        utility=pd.DataFrame({"m0": [0.9, 0.8, 0.4, 0.3], "m1": [0.2, 0.3, 0.8, 0.9]}, index=query_ids[:4]),
        query_info=pd.DataFrame({"query_text": ["a", "two words", "three text words", "four more text words"]}, index=query_ids[:4]),
        model_ids=["m0", "m1"],
    )
    test = Matrices(
        quality=pd.DataFrame({"m0": [0.9, 0.2, 0.1], "m1": [0.1, 0.8, 0.4]}, index=query_ids[3:]),
        cost=pd.DataFrame(0.0, index=query_ids[3:], columns=["m0", "m1"]),
        utility=pd.DataFrame({"m0": [0.9, 0.2, 0.1], "m1": [0.1, 0.8, 0.4]}, index=query_ids[3:]),
        query_info=pd.DataFrame({"query_text": ["tiny", "medium sized query", "a much longer query text here"]}, index=query_ids[3:]),
        model_ids=["m0", "m1"],
    )
    embeddings = pd.DataFrame(
        [[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [5.1, 5.0], [5.2, 5.0], [5.3, 5.0]],
        index=query_ids,
    )
    captured = []

    def capture_row(
        sensitivity,
        variant,
        method,
        selected_models,
        bucket_test,
        baseline_mean,
        learned_reference_mean,
        oracle_mean,
        n_bootstrap,
        ci,
        seed,
        k=None,
        labels=None,
    ):
        captured.append(
            {
                "variant": variant,
                "query_ids": list(bucket_test.utility.index),
                "baseline_mean": baseline_mean,
                "learned_reference_mean": learned_reference_mean,
                "oracle_mean": oracle_mean,
            }
        )
        return {"sensitivity": sensitivity, "variant": variant, "method": method}

    monkeypatch.setattr(module, "_row", capture_row)

    module._query_length_rows(train, test, embeddings, seed=7, k=2, alpha=3.0, beta=0.0, n_bootstrap=10, ci=0.95)

    assert captured
    for row in captured:
        bucket_utility = test.utility.loc[row["query_ids"]]
        assert row["oracle_mean"] == bucket_utility.max(axis=1).mean()


def test_domain_granularity_rows_use_bucket_local_references(monkeypatch):
    module = _load_sensitivity_suite()
    query_ids = ["q0", "q1", "q2", "q3", "q4", "q5"]
    train = Matrices(
        quality=pd.DataFrame({"m0": [0.9, 0.8, 0.4, 0.3], "m1": [0.2, 0.3, 0.8, 0.9]}, index=query_ids[:4]),
        cost=pd.DataFrame(0.0, index=query_ids[:4], columns=["m0", "m1"]),
        utility=pd.DataFrame({"m0": [0.9, 0.8, 0.4, 0.3], "m1": [0.2, 0.3, 0.8, 0.9]}, index=query_ids[:4]),
        query_info=pd.DataFrame(
            {
                "query_text": ["train a", "train b", "train c", "train d"],
                "domain": ["stem", "stem", "code", "code"],
                "dataset": ["math", "math", "humaneval", "humaneval"],
            },
            index=query_ids[:4],
        ),
        model_ids=["m0", "m1"],
    )
    test = Matrices(
        quality=pd.DataFrame({"m0": [0.9, 0.2, 0.1], "m1": [0.1, 0.8, 0.4]}, index=query_ids[3:]),
        cost=pd.DataFrame(0.0, index=query_ids[3:], columns=["m0", "m1"]),
        utility=pd.DataFrame({"m0": [0.9, 0.2, 0.1], "m1": [0.1, 0.8, 0.4]}, index=query_ids[3:]),
        query_info=pd.DataFrame(
            {
                "query_text": ["test a", "test b", "test c"],
                "domain": ["stem", "code", "code"],
                "dataset": ["math", "humaneval", "mbpp"],
            },
            index=query_ids[3:],
        ),
        model_ids=["m0", "m1"],
    )
    embeddings = pd.DataFrame([[0.0], [0.1], [0.2], [0.3], [0.4], [0.5]], index=query_ids)
    captured = []

    def capture_row(
        sensitivity,
        variant,
        method,
        selected_models,
        bucket_test,
        baseline_mean,
        learned_reference_mean,
        oracle_mean,
        n_bootstrap,
        ci,
        seed,
        k=None,
        labels=None,
    ):
        captured.append(
            {
                "sensitivity": sensitivity,
                "variant": variant,
                "method": method,
                "query_ids": list(bucket_test.utility.index),
                "oracle_mean": oracle_mean,
            }
        )
        return {"sensitivity": sensitivity, "variant": variant, "method": method}

    monkeypatch.setattr(module, "_row", capture_row)

    module._domain_granularity_rows(
        train,
        test,
        embeddings,
        seed=7,
        k=2,
        alpha=3.0,
        beta=0.0,
        n_bootstrap=10,
        ci=0.95,
        columns=["domain", "dataset"],
        text_cluster_counts=[],
        min_queries=1,
    )

    variants = {row["variant"] for row in captured}
    assert {"domain:stem", "domain:code", "dataset:math", "dataset:humaneval", "dataset:mbpp"}.issubset(variants)
    for row in captured:
        bucket_utility = test.utility.loc[row["query_ids"]]
        assert row["oracle_mean"] == bucket_utility.max(axis=1).mean()


def test_cost_misestimation_rows_can_use_sensitivity_local_cost_lambda(monkeypatch):
    module = _load_sensitivity_suite()
    train = Matrices(
        quality=pd.DataFrame({"m0": [1.0], "m1": [0.5]}, index=["q0"]),
        cost=pd.DataFrame({"m0": [0.2], "m1": [0.1]}, index=["q0"]),
        utility=pd.DataFrame({"m0": [1.0], "m1": [0.5]}, index=["q0"]),
        query_info=pd.DataFrame({"query_text": ["train"]}, index=["q0"]),
        model_ids=["m0", "m1"],
    )
    test = Matrices(
        quality=pd.DataFrame({"m0": [0.8], "m1": [0.7]}, index=["q1"]),
        cost=pd.DataFrame({"m0": [0.2], "m1": [0.1]}, index=["q1"]),
        utility=pd.DataFrame({"m0": [0.8], "m1": [0.7]}, index=["q1"]),
        query_info=pd.DataFrame({"query_text": ["test"]}, index=["q1"]),
        model_ids=["m0", "m1"],
    )
    embeddings = pd.DataFrame([[0.0], [1.0]], index=["q0", "q1"])
    captured = {}

    monkeypatch.setattr(module, "_references", lambda train_arg, test_arg, embeddings_arg, seed: (0.1, 0.2, 0.3))

    def capture_key_rows(
        sensitivity,
        variant,
        train_arg,
        test_arg,
        embeddings_arg,
        seed,
        k,
        alpha,
        beta,
        n_bootstrap,
        ci,
        baseline_mean=None,
        learned_reference_mean=None,
        oracle_mean=None,
    ):
        captured["train_utility"] = train_arg.utility.copy()
        captured["test_utility"] = test_arg.utility.copy()
        captured["refs"] = (baseline_mean, learned_reference_mean, oracle_mean)
        return [{"sensitivity": sensitivity, "variant": variant, "method": "stub"}]

    monkeypatch.setattr(module, "_key_method_rows", capture_key_rows)

    rows = module._cost_misestimation_rows(
        {"utility": {"lambda_cost": 0.0}, "sensitivity": {"cost_lambda": 2.0, "cost_multipliers": [0.5]}},
        train,
        test,
        embeddings,
        seed=7,
        k=2,
        alpha=3.0,
        beta=0.0,
        n_bootstrap=10,
        ci=0.95,
    )

    assert rows == [{"sensitivity": "cost_misestimation", "variant": "cost_multiplier_0.5", "method": "stub"}]
    assert captured["train_utility"].loc["q0", "m0"] == 0.8
    assert captured["test_utility"].loc["q1", "m0"] == 0.4
    assert captured["refs"] == (0.1, 0.2, 0.3)


def test_sensitivity_memo_describes_curated_taxonomy_layer(tmp_path):
    module = _load_sensitivity_suite()
    table = pd.DataFrame(
        [
            {
                "sensitivity": "domain_granularity",
                "method": "d2_embedding_centroid",
                "recovered_gap_vs_oracle": 0.4,
            }
        ]
    )

    module.write_memo(tmp_path, "configs/llmrouterbench_pilot.yaml", table)

    memo = (tmp_path / "phase_g_sensitivity_memo.md").read_text(encoding="utf-8")
    assert "curated task-family/task-subtype taxonomy" in memo
    assert "still missing or shallow are external embedding backbones and larger benchmark-scale taxonomy coverage" in memo


def test_price_ratio_rows_scale_model_cost_ratios_and_use_shifted_objective(monkeypatch):
    module = _load_sensitivity_suite()
    train = Matrices(
        quality=pd.DataFrame({"cheap": [1.0], "expensive": [1.0]}, index=["q0"]),
        cost=pd.DataFrame({"cheap": [0.1], "expensive": [0.4]}, index=["q0"]),
        utility=pd.DataFrame({"cheap": [1.0], "expensive": [1.0]}, index=["q0"]),
        query_info=pd.DataFrame({"query_text": ["train"]}, index=["q0"]),
        model_ids=["cheap", "expensive"],
    )
    test = Matrices(
        quality=pd.DataFrame({"cheap": [1.0], "expensive": [1.0]}, index=["q1"]),
        cost=pd.DataFrame({"cheap": [0.2], "expensive": [0.8]}, index=["q1"]),
        utility=pd.DataFrame({"cheap": [1.0], "expensive": [1.0]}, index=["q1"]),
        query_info=pd.DataFrame({"query_text": ["test"]}, index=["q1"]),
        model_ids=["cheap", "expensive"],
    )
    embeddings = pd.DataFrame([[0.0], [1.0]], index=["q0", "q1"])
    captured = []

    def capture_key_rows(
        sensitivity,
        variant,
        train_arg,
        test_arg,
        embeddings_arg,
        seed,
        k,
        alpha,
        beta,
        n_bootstrap,
        ci,
        baseline_mean=None,
        learned_reference_mean=None,
        oracle_mean=None,
    ):
        captured.append(
            {
                "sensitivity": sensitivity,
                "variant": variant,
                "train_cost": train_arg.cost.copy(),
                "test_utility": test_arg.utility.copy(),
            }
        )
        return [{"sensitivity": sensitivity, "variant": variant, "method": "stub"}]

    monkeypatch.setattr(module, "_key_method_rows", capture_key_rows)
    monkeypatch.setattr(module, "_references", lambda train_arg, test_arg, embeddings_arg, seed: (0.1, 0.2, 0.3))

    rows = module._price_ratio_rows(
        {"utility": {"lambda_cost": 0.0}, "sensitivity": {"cost_lambda": 1.0, "price_ratio_exponents": [0.0]}},
        train,
        test,
        embeddings,
        seed=7,
        k=2,
        alpha=3.0,
        beta=0.0,
        n_bootstrap=10,
        ci=0.95,
    )

    assert rows == [{"sensitivity": "price_ratio", "variant": "price_ratio_exponent_0", "method": "stub"}]
    shifted = captured[0]
    assert shifted["train_cost"].loc["q0", "cheap"] == shifted["train_cost"].loc["q0", "expensive"]
    assert shifted["test_utility"].loc["q1", "cheap"] == shifted["test_utility"].loc["q1", "expensive"]


def test_scale_price_ratios_keeps_zero_cost_columns_finite():
    module = _load_sensitivity_suite()
    cost = pd.DataFrame(
        {
            "free": [0.0, 0.0],
            "paid": [0.2, 0.4],
        },
        index=["q0", "q1"],
    )

    shifted = module._scale_price_ratios(cost, exponent=0.0)

    assert shifted.notna().all().all()
    assert shifted["free"].sum() == 0.0
    assert shifted["paid"].mean() > 0.0


def test_bootstrap_sampling_rows_emit_configured_bootstrap_counts(monkeypatch):
    module = _load_sensitivity_suite()
    query_ids = ["q0", "q1", "q2", "q3"]
    train = Matrices(
        quality=pd.DataFrame({"m0": [0.9, 0.8], "m1": [0.2, 0.3]}, index=query_ids[:2]),
        cost=pd.DataFrame(0.0, index=query_ids[:2], columns=["m0", "m1"]),
        utility=pd.DataFrame({"m0": [0.9, 0.8], "m1": [0.2, 0.3]}, index=query_ids[:2]),
        query_info=pd.DataFrame({"query_text": ["one", "two words"]}, index=query_ids[:2]),
        model_ids=["m0", "m1"],
    )
    test = Matrices(
        quality=pd.DataFrame({"m0": [0.7, 0.1], "m1": [0.4, 0.9]}, index=query_ids[2:]),
        cost=pd.DataFrame(0.0, index=query_ids[2:], columns=["m0", "m1"]),
        utility=pd.DataFrame({"m0": [0.7, 0.1], "m1": [0.4, 0.9]}, index=query_ids[2:]),
        query_info=pd.DataFrame({"query_text": ["three words here", "four words right here"]}, index=query_ids[2:]),
        model_ids=["m0", "m1"],
    )
    embeddings = pd.DataFrame([[0.0], [0.1], [0.2], [0.3]], index=query_ids)
    captured = []

    monkeypatch.setattr(module, "_references", lambda train_arg, test_arg, embeddings_arg, seed: (0.1, 0.2, 0.3))

    def capture_row(
        sensitivity,
        variant,
        method,
        selected_models,
        test_arg,
        baseline_mean,
        learned_reference_mean,
        oracle_mean,
        n_bootstrap,
        ci,
        seed,
        k=None,
        labels=None,
    ):
        captured.append((sensitivity, variant, method, n_bootstrap))
        return {"sensitivity": sensitivity, "variant": variant, "method": method, "n_bootstrap": n_bootstrap}

    monkeypatch.setattr(module, "_row", capture_row)

    rows = module._bootstrap_sampling_rows(train, test, embeddings, seed=7, k=2, alpha=3.0, beta=0.0, bootstrap_counts=[5, 17], ci=0.95)

    assert [row["variant"] for row in rows] == ["n_bootstrap_5", "n_bootstrap_17"]
    assert captured == [
        ("bootstrap_sampling", "n_bootstrap_5", "d2_embedding_centroid", 5),
        ("bootstrap_sampling", "n_bootstrap_17", "d2_embedding_centroid", 17),
    ]


def test_model_pool_scenarios_include_configured_composition_pools():
    module = _load_sensitivity_suite()
    utility = pd.DataFrame(
        {
            "m0": [0.9, 0.8],
            "m1": [0.7, 0.6],
            "m2": [0.5, 0.4],
            "m3": [0.3, 0.2],
        },
        index=["q0", "q1"],
    )

    scenarios = module._model_pool_scenarios(
        utility,
        sizes=[2],
        configured_pools=[
            {"name": "complementary_pair", "models": ["m0", "m2"]},
            {"name": "too_small", "models": ["m0"]},
            {"name": "unknown_model", "models": ["m0", "missing"]},
        ],
    )

    assert ("model_pool", "full", ["m0", "m1", "m2", "m3"]) in scenarios
    assert ("model_pool", "top_2", ["m0", "m1"]) in scenarios
    assert ("model_pool_composition", "complementary_pair", ["m0", "m2"]) in scenarios
    assert all(scenario[1] != "too_small" for scenario in scenarios)
    assert all(scenario[1] != "unknown_model" for scenario in scenarios)


def test_model_pool_scenarios_include_automatic_dominated_and_complementary_pools():
    module = _load_sensitivity_suite()
    utility = pd.DataFrame(
        {
            "m0": [1.0, 0.0, 0.0, 0.0],
            "m1": [0.0, 1.0, 0.0, 0.0],
            "m2": [0.45, 0.45, 0.45, 0.45],
            "m3": [0.10, 0.10, 0.10, 0.10],
        },
        index=["q0", "q1", "q2", "q3"],
    )

    scenarios = module._model_pool_scenarios(
        utility,
        sizes=[2],
        auto_sizes=[2],
    )

    assert ("model_pool_auto", "complementary_size_2", ["m0", "m1"]) in scenarios
    assert ("model_pool_auto", "dominated_size_2", ["m2", "m3"]) in scenarios


def test_sensitivity_append_readme_preserves_following_sections(tmp_path):
    module = _load_sensitivity_suite()
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Demo",
                "",
                "## Sensitivity Suite",
                "",
                "old content",
                "",
                "## Code-Card Interpretability",
                "",
                "keep this section",
            ]
        ),
        encoding="utf-8",
    )
    table = pd.DataFrame(
        {
            "sensitivity": ["model_pool_auto"],
            "method": ["d2_embedding_centroid"],
            "recovered_gap_vs_oracle": [0.5],
        }
    )

    module.append_readme(tmp_path, "configs/example.yaml", table)

    updated = readme.read_text(encoding="utf-8")
    assert "python experiments/09_sensitivity_suite.py --config configs/example.yaml" in updated
    assert "## Code-Card Interpretability" in updated
    assert "keep this section" in updated
