from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "40_avengerspro_upstream_metric.py"
    spec = importlib.util.spec_from_file_location("avengerspro_upstream_metric_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_avengerspro_upstream_metric_script_scores_captured_routing_details(tmp_path, monkeypatch):
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
                "  n_queries: 40",
                "  n_models: 3",
                "  n_domains: 2",
                "  n_route_labels: 3",
                "  embedding_dim: 8",
                "  model_ids: [m0, m1, m2]",
                "  model_costs:",
                "    m0: 0.1",
                "    m1: 0.2",
                "    m2: 0.3",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "external_baselines:",
                "  avengerspro_clusters: [4]",
                "  avengerspro_top_k: 1",
                "  avengerspro_beta: 7.0",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_upstream_router(command_config):
        del command_config
        test_records = [
            json.loads(line)
            for line in (out_dir / "avengerspro_split_aligned" / "test.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        routing_details = []
        for record in test_records:
            best_model = max(record["utilities"], key=record["utilities"].get)
            routing_details.append({"selected_models": [best_model], "is_correct": record["records"][best_model]})
        return {
            "routing_details": routing_details,
            "accuracy": 100.0,
            "correct_routes": len(routing_details),
            "total_queries": len(routing_details),
            "cost_analysis": {"total_cost": 0.0},
        }

    monkeypatch.setattr(module, "_run_upstream_router", fake_run_upstream_router)

    module.run(str(config_path))

    table_path = out_dir / "table_avengerspro_upstream_metric.csv"
    run_dir = out_dir / "avengerspro_upstream_metric"
    assert table_path.exists()
    assert (run_dir / "raw_routing_details.json").exists()
    assert (run_dir / "run_config.json").exists()
    assert (out_dir / "phase_e_avengerspro_upstream_metric_memo.md").exists()

    table = pd.read_csv(table_path)
    row = table.iloc[0]
    assert row["method"] == "avengerspro_upstream_simple_cluster_postprocessed"
    assert bool(row["routecode_metric_compatible"])
    assert bool(row["upstream_model_code_used"])
    assert not bool(row["exact_upstream_command"])
    assert row["prediction_count"] == len(
        (out_dir / "avengerspro_split_aligned" / "test.jsonl").read_text(encoding="utf-8").splitlines()
    )

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Avengers-Pro Upstream Metric" in readme
    assert "not an exact upstream command" in readme
