from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "22_cost_quality_frontier.py"
    spec = importlib.util.spec_from_file_location("cost_quality_frontier", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cost_quality_frontier_script_writes_summary_frontier_memo_and_readme(tmp_path):
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
                "  embedding_clusters: 4",
                "  knn_k: 3",
                "predictability_constrained:",
                "  k: 4",
                "  selected_alpha: 1.0",
                "  beta: 0.0",
                "cost_quality:",
                "  lambda_values: [0.0, 0.2]",
                "  quality_target_fractions: [0.8, 0.95]",
                "  cost_budget_fractions: [0.5, 1.0]",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    summary_path = out_dir / "table_cost_quality_summary.csv"
    frontier_path = out_dir / "table_cost_quality_frontier.csv"
    memo_path = out_dir / "phase_e_cost_quality_memo.md"
    fig_path = out_dir / "fig_cost_quality_frontier.pdf"
    assert summary_path.exists()
    assert frontier_path.exists()
    assert memo_path.exists()
    assert fig_path.exists()

    summary = pd.read_csv(summary_path)
    frontier = pd.read_csv(frontier_path)
    assert {"best_single", "cheapest", "kNN", "d2_embedding_centroid", "quality_oracle"}.issubset(
        set(summary["method"])
    )
    assert set(summary["lambda_cost"]) == {0.0, 0.2}
    assert {"cost_at_fixed_quality", "quality_at_fixed_cost"} == set(frontier["target_type"])
    assert {"all_methods", "deployable_methods"} == set(frontier["frontier_family"])
    assert frontier["selected_method"].notna().any()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Cost-Quality Operating Points" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "fixed-quality and fixed-cost" in memo
