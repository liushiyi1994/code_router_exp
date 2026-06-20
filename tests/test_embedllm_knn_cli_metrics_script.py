from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "33_embedllm_knn_cli_metrics.py"
    spec = importlib.util.spec_from_file_location("embedllm_knn_cli_metrics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_embedllm_knn_cli_metrics_runs_exact_tensor_command(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 13",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 28",
                "  n_models: 2",
                "  n_domains: 2",
                "  n_route_labels: 4",
                "  embedding_dim: 6",
                "  model_ids: [m0, m1]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.10",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "external_baselines:",
                "  embedllm_knn_cli_embedding_backend: routecode_embeddings",
                "  embedllm_knn_neighbors: [1, 3]",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    asset_dir = out_dir / "embedllm_assets"
    table_path = out_dir / "table_embedllm_knn_cli_metrics.csv"
    memo_path = out_dir / "phase_e_embedllm_knn_cli_metrics_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    for name in ["knn_train_x.pth", "knn_train_y.pth", "knn_test_x.pth", "knn_test_y.pth"]:
        assert (asset_dir / name).exists()
    train_x = torch.load(asset_dir / "knn_train_x.pth", map_location="cpu")
    train_y = torch.load(asset_dir / "knn_train_y.pth", map_location="cpu")
    assert train_x.shape[0] == train_y.shape[0] == 2
    assert train_x.shape[2] == 6

    table = pd.read_csv(table_path)
    assert table["method"].tolist() == ["embedllm_knn_cli_k1", "embedllm_knn_cli_k3"]
    assert table["baseline_family"].eq("embedllm_knn_exact_cli_correctness").all()
    assert table["exact_upstream_command"].eq(True).all()
    assert table["routecode_metric_compatible"].eq(False).all()
    assert table["split_aligned_with_routecode"].eq(True).all()
    assert table["mean_correctness_accuracy"].between(0.0, 1.0).all()
    assert table["execution_evidence"].map(lambda value: Path(value).exists()).all()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## EmbedLLM KNN CLI Metrics" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "exact upstream EmbedLLM KNN command" in memo
