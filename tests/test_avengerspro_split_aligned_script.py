from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "17_avengerspro_split_aligned.py"
    spec = importlib.util.spec_from_file_location("avengerspro_split_aligned", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_avengerspro_split_aligned_script_writes_assets_table_and_memo(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n\n## Next Steps\n\n- old\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 9",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 72",
                "  n_models: 3",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  embedding_dim: 8",
                "  model_ids: [cheap, balanced, expensive]",
                "  model_costs:",
                "    cheap: 0.05",
                "    balanced: 0.20",
                "    expensive: 0.80",
                "utility:",
                "  lambda_cost: 0.35",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "external_baselines:",
                "  avengerspro_clusters: [4]",
                "  avengerspro_top_k: 1",
                "  avengerspro_performance_weight: 0.25",
                "  avengerspro_cost_sensitivity: 0.75",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    run_dir = out_dir / "avengerspro_split_aligned"
    table_path = out_dir / "table_avengerspro_split_aligned.csv"
    memo_path = out_dir / "phase_e_avengerspro_split_aligned_memo.md"
    assert (run_dir / "train.jsonl").exists()
    assert (run_dir / "test.jsonl").exists()
    assert (run_dir / "smoke_train.jsonl").exists()
    assert (run_dir / "smoke_test.jsonl").exists()
    assert (run_dir / "baseline_scores.json").exists()
    assert (run_dir / "embedding_cache.jsonl").exists()
    assert (run_dir / "simple_cluster_config.local.json").exists()
    assert (run_dir / "metadata.json").exists()
    assert table_path.exists()
    assert memo_path.exists()

    table = pd.read_csv(table_path)
    assert table["method"].tolist() == [
        "avengerspro_simple_cluster_k4",
        "avengerspro_balance_cluster_k4_w0.25_c0.75",
    ]
    assert set(table["baseline_family"]) == {"official_algorithm_local_embedding"}
    assert table["split_aligned_with_routecode"].all()
    assert table["routecode_metric_compatible"].all()
    assert table["no_api_calls"].all()
    assert not table["official_command_path"].any()
    assert table["selected_model_entropy"].ge(0.0).all()

    config = (run_dir / "simple_cluster_config.local.json").read_text(encoding="utf-8")
    assert "embedding_cache.jsonl" in config
    assert "smoke_train.jsonl" in config
    cache_line = (run_dir / "embedding_cache.jsonl").read_text(encoding="utf-8").splitlines()[0]
    assert '"embedding"' in cache_line
    assert '"query"' in cache_line

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Avengers-Pro Split-Aligned Evaluation" in readme
    assert "local implementation of the Avengers-Pro cluster-routing contract" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "not an official upstream command-path run" in memo
