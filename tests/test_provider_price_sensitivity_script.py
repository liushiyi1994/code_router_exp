from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "25_provider_price_sensitivity.py"
    spec = importlib.util.spec_from_file_location("provider_price_sensitivity", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_provider_price_sensitivity_script_writes_tables_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
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
                "  n_queries: 48",
                "  n_models: 3",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  embedding_dim: 8",
                "  model_ids: [m0, m1, m2]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.10",
                "    m2: 0.20",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  embedding_clusters: 4",
                "  knn_k: 3",
                "predictability_constrained:",
                "  k: 4",
                "  selected_alpha: 1.0",
                "  beta: 0.0",
                "provider_pricing:",
                "  lambda_values: [0.0, 0.2]",
                "  quality_target_fractions: [0.8]",
                "  cost_budget_fractions: [1.0]",
                "  schedules:",
                "    - name: demo_provider",
                "      provider: ExampleProvider",
                "      source_checked_date: '2026-06-15'",
                "      unmapped_policy: drop",
                "      prices_per_million_tokens:",
                "        m0:",
                "          provider_model_id: provider/m0",
                "          input: 0.10",
                "          output: 0.20",
                "          source_url: https://example.com/m0",
                "        m1:",
                "          provider_model_id: provider/m1",
                "          input: 1.00",
                "          output: 2.00",
                "          source_url: https://example.com/m1",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    coverage_path = out_dir / "table_provider_price_schedule.csv"
    summary_path = out_dir / "table_provider_cost_quality_summary.csv"
    frontier_path = out_dir / "table_provider_cost_quality_frontier.csv"
    memo_path = out_dir / "phase_g_provider_pricing_memo.md"
    assert coverage_path.exists()
    assert summary_path.exists()
    assert frontier_path.exists()
    assert memo_path.exists()

    coverage = pd.read_csv(coverage_path)
    summary = pd.read_csv(summary_path)
    frontier = pd.read_csv(frontier_path)
    assert set(coverage["model_id"]) == {"m0", "m1", "m2"}
    assert coverage["mapped"].sum() == 2
    assert set(summary["schedule"]) == {"demo_provider"}
    assert set(summary["lambda_cost"]) == {0.0, 0.2}
    assert summary["model_coverage_count"].eq(2).all()
    assert {"best_single", "cheapest", "kNN", "d2_embedding_centroid", "quality_oracle"}.issubset(
        set(summary["method"])
    )
    assert {"cost_at_fixed_quality", "quality_at_fixed_cost"} == set(frontier["target_type"])

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Provider-Price Sensitivity" in readme
    assert "table_provider_cost_quality_summary.csv" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "partial provider-price schedule" in memo
    assert "Mapped schedule-model rows" in memo
    assert "`demo_provider` maps `2` of `3` models" in memo
