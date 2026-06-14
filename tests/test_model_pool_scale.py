from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from routecode.eval.model_pool_scale import build_model_pool_scale_scenarios, model_pool_stats


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "18_model_pool_scale.py"
    spec = importlib.util.spec_from_file_location("model_pool_scale", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_model_pool_scale_scenarios_separates_top_complementary_and_dominated():
    utility = pd.DataFrame(
        {
            "dominant": [0.8, 0.8, 0.8, 0.8],
            "complementary": [0.1, 0.1, 1.0, 1.0],
            "near_duplicate": [0.7, 0.7, 0.7, 0.7],
        },
        index=["q0", "q1", "q2", "q3"],
    )

    scenarios = build_model_pool_scale_scenarios(utility, sizes=[2, 3])
    by_name = {scenario.name: scenario for scenario in scenarios}

    assert by_name["top_2"].models == ["dominant", "near_duplicate"]
    assert by_name["complementary_2"].models == ["dominant", "complementary"]
    assert by_name["dominated_2"].models == ["dominant", "near_duplicate"]
    assert by_name["full_3"].models == ["dominant", "complementary", "near_duplicate"]
    assert by_name["complementary_2"].stats["oracle_gap"] > by_name["dominated_2"].stats["oracle_gap"]


def test_model_pool_stats_reports_gap_dominance_and_entropy():
    utility = pd.DataFrame(
        {"m0": [0.9, 0.8, 0.7], "m1": [0.1, 0.95, 0.2]},
        index=["q0", "q1", "q2"],
    )

    stats = model_pool_stats(utility, ["m0", "m1"])

    assert stats["best_single_model"] == "m0"
    assert stats["oracle_gap"] > 0.0
    assert 0.0 <= stats["dominance_ratio"] <= 1.0
    assert stats["winner_entropy"] > 0.0


def test_model_pool_scale_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 5",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 96",
                "  n_models: 4",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  embedding_dim: 8",
                "  model_ids: [m0, m1, m2, m3]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.10",
                "    m2: 0.20",
                "    m3: 0.30",
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
                "model_pool_scale:",
                "  sizes: [2, 4]",
                "  k: 4",
                "  d2_alpha: 1.0",
                "  d2_beta: 0.0",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_model_pool_scale.csv"
    memo_path = out_dir / "phase_f_g_model_pool_scale_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert {"best_single", "kNN", "d2_embedding_centroid"}.issubset(set(table["method"]))
    assert {"top", "complementary", "dominated", "full"}.issubset(set(table["pool_family"]))
    assert table["model_count"].min() >= 2
    assert table["test_oracle_gap"].ge(0.0).all()
    assert table["models"].str.len().gt(0).all()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Model-Pool Scale Robustness" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "top, complementary, and dominated model-pool scenarios" in memo
