from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "37_avengerspro_cli_metrics.py"
    spec = importlib.util.spec_from_file_location("avengerspro_cli_metrics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_avengerspro_cli_metrics_writes_full_split_upstream_metrics(tmp_path, monkeypatch):
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
                "external_baselines:",
                "  avengerspro_clusters: [4]",
                "  avengerspro_top_k: 1",
                "  avengerspro_beta: 7.0",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, cwd, stdout_path):
        commands.append(command)
        stdout_path.write_text("Routing evaluation completed successfully\n", encoding="utf-8")
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "results": {
                        "accuracy": 62.5,
                        "correct_routes": 5,
                        "total_queries": 8,
                        "all_sample_avg": 60.0,
                        "cost_analysis": {
                            "total_cost": 1.25,
                            "avg_cost_per_query": 0.15625,
                        },
                        "model_selection_stats": {"m0": 5, "m1": 3},
                    }
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run_upstream_command", fake_run)
    module.run(str(config_path))

    table_path = out_dir / "table_avengerspro_cli_metrics.csv"
    memo_path = out_dir / "phase_e_avengerspro_cli_metrics_memo.md"
    run_dir = out_dir / "avengerspro_cli_metrics"
    assert table_path.exists()
    assert memo_path.exists()
    assert (run_dir / "simple_cluster_full_results.json").exists()
    assert (run_dir / "avengerspro_simple_cluster_stdout.log").exists()
    assert (run_dir / "simple_cluster_config.full.json").exists()
    assert commands
    command_text = " ".join(commands[0])
    assert "--config" in command_text
    assert "--output" in command_text
    assert "simple_cluster_config.full.json" in command_text

    config = json.loads((run_dir / "simple_cluster_config.full.json").read_text(encoding="utf-8"))
    assert config["train_data_path"].endswith("avengerspro_split_aligned/train.jsonl")
    assert config["test_data_path"].endswith("avengerspro_split_aligned/test.jsonl")
    assert "smoke_train.jsonl" not in config["train_data_path"]
    assert "smoke_test.jsonl" not in config["test_data_path"]
    assert config["embedding_cache_path"].endswith("full_embedding_cache.jsonl")
    assert config["embedding_api_key"] == ""

    table = pd.read_csv(table_path)
    row = table.iloc[0]
    assert row["method"] == "avengerspro_cli_simple_cluster_k4"
    assert row["dataset_level_accuracy"] == 0.625
    assert row["sample_level_accuracy"] == 0.6
    assert row["total_queries"] == 8
    assert row["total_cost"] == 1.25
    assert bool(row["exact_upstream_command"])
    assert not bool(row["routecode_metric_compatible"])

    memo = memo_path.read_text(encoding="utf-8")
    assert "exact upstream Avengers-Pro simple-cluster command" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Avengers-Pro CLI Metrics" in readme
