from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "36_embedllm_mf_cli_metrics.py"
    spec = importlib.util.spec_from_file_location("embedllm_mf_cli_metrics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_embedllm_mf_cli_metrics_writes_full_split_upstream_metrics(tmp_path, monkeypatch):
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
                "  embedllm_mf_num_epochs: 2",
                "  embedllm_mf_embedding_dim: 7",
                "  embedllm_mf_batch_size: 64",
                "  embedllm_mf_learning_rate: 0.001",
            ]
        ),
        encoding="utf-8",
    )
    commands: list[list[str]] = []

    def fake_run(command, cwd, stdout_path):
        commands.append(command)
        stdout_path.write_text(
            "\n".join(
                [
                    "Dataset-Level Average Accuracy: 0.6100",
                    "Sample-Level Average Accuracy:  0.7200",
                    "Best Dataset-Level Accuracy: 0.6500 at Epoch 2",
                    "Model saved to saved_model.pth",
                ]
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run_upstream_command", fake_run)

    module.run(str(config_path))

    table_path = out_dir / "table_embedllm_mf_cli_metrics.csv"
    memo_path = out_dir / "phase_e_embedllm_mf_cli_metrics_memo.md"
    log_path = out_dir / "embedllm_mf_cli_metrics/embedllm_mf_stdout.log"
    assert table_path.exists()
    assert memo_path.exists()
    assert log_path.exists()
    assert commands
    command_text = " ".join(commands[0])
    assert "--train-data-path" in command_text
    assert "embedllm_assets/train.csv" in command_text
    assert "--test-data-path" in command_text
    assert "embedllm_assets/test.csv" in command_text
    assert "--eval-mode" in command_text
    assert "router" in command_text

    table = pd.read_csv(table_path)
    row = table.iloc[0]
    assert row["method"] == "embedllm_mf_cli_full_split"
    assert row["best_dataset_level_accuracy"] == 0.65
    assert row["final_sample_level_accuracy"] == 0.72
    assert bool(row["exact_upstream_command"])
    assert not bool(row["routecode_metric_compatible"])

    memo = memo_path.read_text(encoding="utf-8")
    assert "exact upstream EmbedLLM MF command" in memo
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## EmbedLLM MF CLI Metrics" in readme


def test_embedllm_mf_cli_metrics_parser_prefers_final_test_section():
    module = _load_script()
    parsed = module._parse_router_metrics(
        "\n".join(
            [
                "Dataset-Level Average Accuracy: 0.6100",
                "Sample-Level Average Accuracy:  0.6200",
                "Best Dataset-Level Accuracy: 0.6500 at Epoch 2",
                "FINAL TEST SET RESULTS",
                "Dataset-Level Average Accuracy: 0.6300",
                "Sample-Level Average Accuracy:  0.6400",
                "FINAL TRAINING SET RESULTS",
                "Dataset-Level Average Accuracy: 0.7100",
                "Sample-Level Average Accuracy:  0.7200",
            ]
        )
    )

    assert parsed["best_dataset_level_accuracy"] == 0.65
    assert parsed["final_dataset_level_accuracy"] == 0.63
    assert parsed["final_sample_level_accuracy"] == 0.64
