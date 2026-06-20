from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "24_graphrouter_assets.py"
    spec = importlib.util.spec_from_file_location("graphrouter_assets_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_graphrouter_assets_script_writes_assets_summary_memo_and_readme(tmp_path):
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
                "  n_queries: 36",
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
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    asset_dir = out_dir / "graphrouter_assets"
    table_path = out_dir / "table_graphrouter_assets.csv"
    memo_path = out_dir / "phase_e_graphrouter_assets_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    assert (asset_dir / "router_data.csv").exists()
    assert (asset_dir / "LLM_Descriptions.json").exists()
    assert (asset_dir / "llm_description_embedding.pkl").exists()
    assert (asset_dir / "config.local.yaml").exists()

    table = pd.read_csv(table_path)
    assert set(table["split"]) == {"train", "val", "test", "overall"}
    assert table.loc[table["split"] == "overall", "query_count"].item() == 36
    assert table.loc[table["split"] == "overall", "row_count"].item() == 108
    assert table["split_aligned_with_routecode"].all()
    assert not table["routecode_metric_compatible"].all()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## GraphRouter Assets" in readme
    assert "python experiments/24_graphrouter_assets.py" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "GraphRouter-compatible data-contract assets" in memo
