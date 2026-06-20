from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import yaml


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "38_graphrouter_cli_metrics.py"
    spec = importlib.util.spec_from_file_location("graphrouter_cli_metrics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graphrouter_cli_metrics_writes_smoke_config_and_metrics(tmp_path, monkeypatch):
    module = _load_script()
    out_dir = tmp_path / "out"
    asset_dir = out_dir / "graphrouter_assets"
    asset_dir.mkdir(parents=True)
    (out_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    (asset_dir / "router_data.csv").write_text("query_id,llm,effect,cost\nq0,m0,1.0,0.0\n", encoding="utf-8")
    (asset_dir / "LLM_Descriptions.json").write_text('{"m0": {"description": "model"}}\n', encoding="utf-8")
    (asset_dir / "llm_description_embedding.pkl").write_bytes(b"pickle")
    (asset_dir / "config.local.yaml").write_text(
        yaml.safe_dump(
            {
                "saved_router_data_path": str(asset_dir / "router_data.csv"),
                "llm_description_path": str(asset_dir / "LLM_Descriptions.json"),
                "llm_embedding_path": str(asset_dir / "llm_description_embedding.pkl"),
                "model_path": str(asset_dir / "model_path/best_model.pth"),
                "train_epoch": 2000,
                "llm_num": 1,
                "wandb_key": "",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    graphrouter_root = tmp_path / "data/raw/external/LLMRouterBench/baselines/GraphRouter"
    graphrouter_root.mkdir(parents=True)
    (graphrouter_root / "run_exp.py").write_text("# graph\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                f"  output_dir: {out_dir}",
                "  random_seed: 7",
                "data:",
                "  source: synthetic",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "GRAPHROUTER_SOURCE", graphrouter_root)
    commands: list[tuple[list[str], Path]] = []

    def fake_run(command, cwd, stdout_path, env):
        commands.append((command, cwd))
        stdout_path.write_text(
            "\n".join(
                [
                    "BEST TEST CHECKPOINT METRICS (used for model selection)",
                    "Dataset-Level Average Accuracy: 0.5000",
                    "Sample-Level Average Accuracy:  0.6250",
                    "Total Cost:                     1.2500",
                    "Cost Source:                    usd",
                ]
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(module, "_run_upstream_command", fake_run)

    module.run(str(config_path), epochs=3)

    metric_dir = out_dir / "graphrouter_cli_metrics"
    table = pd.read_csv(out_dir / "table_graphrouter_cli_metrics.csv")
    row = table.iloc[0]
    assert row["method"] == "graphrouter_cli_smoke"
    assert row["dataset_level_accuracy"] == 0.5
    assert row["sample_level_accuracy"] == 0.625
    assert row["total_cost"] == 1.25
    assert row["cost_source"] == "usd"
    assert row["epochs"] == 3
    assert bool(row["exact_upstream_command"])
    assert not bool(row["routecode_metric_compatible"])
    assert (metric_dir / "config.smoke.yaml").exists()
    smoke_config = yaml.safe_load((metric_dir / "config.smoke.yaml").read_text(encoding="utf-8"))
    assert smoke_config["train_epoch"] == 3
    assert Path(smoke_config["saved_router_data_path"]).is_absolute()
    assert smoke_config["model_path"].endswith("graphrouter_cli_metrics/model_path/best_model.pth")
    assert commands
    command, cwd = commands[0]
    assert cwd == graphrouter_root
    assert command == ["python", "run_exp.py", "--config_file", str((metric_dir / "config.smoke.yaml").resolve())]
    assert "## GraphRouter CLI Metrics" in (out_dir / "README.md").read_text(encoding="utf-8")


def test_graphrouter_cli_metrics_parser_prefers_best_checkpoint_section():
    module = _load_script()
    parsed = module._parse_graphrouter_metrics(
        "\n".join(
            [
                "LAST EPOCH METRICS",
                "Dataset-Level Average Accuracy: 0.4000",
                "Sample-Level Average Accuracy:  0.4100",
                "BEST TEST CHECKPOINT METRICS (used for model selection)",
                "Dataset-Level Average Accuracy: 0.5000",
                "Sample-Level Average Accuracy:  0.6250",
                "Total Cost:                     1.2500",
                "Cost Source:                    usd",
            ]
        )
    )

    assert parsed == {
        "dataset_level_accuracy": 0.5,
        "sample_level_accuracy": 0.625,
        "total_cost": 1.25,
        "cost_source": "usd",
    }
