from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "31_llmrouter_cli_metrics.py"
    spec = importlib.util.spec_from_file_location("llmrouter_cli_metrics_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_llmrouter_cli_metrics_script_scores_saved_exact_cli_predictions(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    asset_dir = out_dir / "llmrouter_library_adapters"
    asset_dir.mkdir(parents=True)
    (out_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 3",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 36",
                "  n_models: 2",
                "  n_domains: 2",
                "  n_route_labels: 4",
                "  embedding_dim: 5",
                "  model_ids: [m0, m1]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.06",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    # The deterministic synthetic split has 7 test queries at n_queries=36 with 0.6/0.2/0.2.
    predictions = [{"success": True, "query": f"q{i}", "model_name": "m0"} for i in range(7)]
    (asset_dir / "llmrouter_knn_full_predictions.json").write_text(json.dumps(predictions), encoding="utf-8")

    module.run(str(config_path))

    table_path = out_dir / "table_llmrouter_cli_metrics.csv"
    memo_path = out_dir / "phase_e_llmrouter_cli_metrics_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert list(table["method"]) == ["llmrouter_cli_knn"]
    assert table["routecode_metric_compatible"].all()
    assert table["exact_upstream_command"].all()
    assert int(table.iloc[0]["prediction_count"]) == 7
    assert "llmrouter_knn_full_predictions.json" in table.iloc[0]["prediction_source"]
    assert "Exact LLMRouter route-only CLI predictions" in memo_path.read_text(encoding="utf-8")
    assert "table_llmrouter_cli_metrics.csv" in (out_dir / "README.md").read_text(encoding="utf-8")
