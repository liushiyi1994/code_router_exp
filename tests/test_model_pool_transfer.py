from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from routecode.eval.model_pool_transfer import (
    build_model_pool_transfer_scenarios,
    fit_label_to_target_model,
)


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "19_model_pool_transfer.py"
    spec = importlib.util.spec_from_file_location("model_pool_transfer", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_model_pool_transfer_scenarios_uses_disjoint_source_and_target_pools():
    utility = pd.DataFrame(
        {
            "m0": [0.90, 0.90, 0.90, 0.90],
            "m1": [0.85, 0.80, 0.85, 0.80],
            "m2": [0.10, 0.10, 1.00, 1.00],
            "m3": [0.70, 0.70, 0.70, 0.70],
            "m4": [0.60, 0.60, 0.60, 0.60],
            "m5": [0.50, 0.50, 0.50, 0.50],
        },
        index=["q0", "q1", "q2", "q3"],
    )

    scenarios = build_model_pool_transfer_scenarios(utility, source_size=3, target_size=2)

    assert scenarios
    assert {scenario.name for scenario in scenarios} >= {
        "top_to_next",
        "complementary_to_remaining_top",
        "dominated_to_remaining_top",
    }
    for scenario in scenarios:
        assert len(scenario.source_models) == 3
        assert len(scenario.target_models) == 2
        assert set(scenario.source_models).isdisjoint(scenario.target_models)
        assert scenario.stats["source_oracle_gap"] >= 0.0
        assert scenario.stats["target_oracle_gap"] >= 0.0


def test_fit_label_to_target_model_uses_train_labels_and_target_utilities_only():
    labels = pd.Series([0, 0, 1, 1], index=["q0", "q1", "q2", "q3"], name="route_label")
    target_utility = pd.DataFrame(
        {
            "target_a": [0.9, 0.8, 0.1, 0.2],
            "target_b": [0.1, 0.2, 0.8, 0.9],
        },
        index=labels.index,
    )

    mapping, fallback = fit_label_to_target_model(labels, target_utility, labels=[0, 1, 2])

    assert mapping == {0: "target_a", 1: "target_b", 2: "target_a"}
    assert fallback == "target_a"


def test_model_pool_transfer_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 7",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 120",
                "  n_models: 8",
                "  n_domains: 4",
                "  n_route_labels: 4",
                "  embedding_dim: 10",
                "  model_ids: [m0, m1, m2, m3, m4, m5, m6, m7]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.06",
                "    m2: 0.07",
                "    m3: 0.08",
                "    m4: 0.09",
                "    m5: 0.10",
                "    m6: 0.11",
                "    m7: 0.12",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "predictability_constrained:",
                "  k: 4",
                "  selected_alpha: 1.0",
                "  beta: 0.0",
                "model_pool_transfer:",
                "  source_size: 4",
                "  target_size: 2",
                "  k: 4",
                "  d2_alpha: 1.0",
                "  d2_beta: 0.0",
                "  direct_router_methods: [logistic, svm]",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_model_pool_transfer.csv"
    memo_path = out_dir / "phase_f_g_model_pool_transfer_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert {
        "target_best_single",
        "target_kNN",
        "target_direct_logistic",
        "target_direct_svm",
        "target_d2_native",
        "source_d2_label_transfer",
    }.issubset(set(table["method"]))
    assert table["source_target_overlap"].eq(0).all()
    assert table["target_model_count"].eq(2).all()
    assert table["source_model_count"].eq(4).all()
    assert table["target_oracle_gap"].ge(0.0).all()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Held-Out Model-Pool Transfer" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "disjoint source and target model pools" in memo
