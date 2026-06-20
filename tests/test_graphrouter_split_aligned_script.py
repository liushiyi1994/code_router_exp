from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "39_graphrouter_split_aligned.py"
    spec = importlib.util.spec_from_file_location("graphrouter_split_aligned_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_split_aligned_config_uses_absolute_local_paths(tmp_path):
    module = _load_script()
    asset_dir = tmp_path / "graphrouter_assets"
    run_dir = tmp_path / "graphrouter_split_aligned"
    asset_dir.mkdir()
    run_dir.mkdir()

    config = module._split_aligned_config(
        {
            "saved_router_data_path": "old.csv",
            "llm_description_path": "old.json",
            "llm_embedding_path": "old.pkl",
            "model_path": "old.pth",
            "embedding_dim": 8,
            "batch_size": 32,
        },
        asset_dir=asset_dir,
        run_dir=run_dir,
        epochs=3,
        baseline_config={"graphrouter_embedding_dim": 4, "graphrouter_batch_size": 2},
    )

    assert config["saved_router_data_path"] == str((asset_dir / "router_data.csv").resolve())
    assert config["llm_description_path"] == str((asset_dir / "LLM_Descriptions.json").resolve())
    assert config["llm_embedding_path"] == str((asset_dir / "llm_description_embedding.pkl").resolve())
    assert config["model_path"] == str((run_dir / "model_path/best_model.pth").resolve())
    assert config["train_epoch"] == 3
    assert config["embedding_dim"] == 4
    assert config["batch_size"] == 2
    assert config["wandb_key"] == ""


def test_graphrouter_split_aligned_memo_and_readme_include_leakage_boundary(tmp_path):
    module = _load_script()
    out_dir = tmp_path
    (out_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    raw_path = out_dir / "graphrouter_split_aligned/raw_predictions.json"
    table = pd.DataFrame(
        [
            {
                "method": "graphrouter_split_aligned_gnn",
                "mean_utility": 0.75,
                "recovered_gap_vs_oracle": 0.25,
                "prediction_count": 2,
                "checkpoint_selection_split": "val",
                "routecode_metric_compatible": True,
                "selected_models": "m0,m1",
            }
        ]
    )

    module.write_memo(out_dir, "configs/demo.yaml", table, raw_path)
    module.append_readme(out_dir, "configs/demo.yaml", table, raw_path)

    memo = (out_dir / "phase_e_graphrouter_split_aligned_memo.md").read_text(encoding="utf-8")
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "checkpoint is selected on the RouteCode validation split" in memo
    assert "not an exact upstream command" in memo
    assert "table_graphrouter_split_aligned.csv" in readme
    assert "graphrouter_split_aligned_gnn" in readme
