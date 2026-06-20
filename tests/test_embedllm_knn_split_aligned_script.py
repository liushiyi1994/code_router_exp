from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "29_embedllm_knn_split_aligned.py"
    spec = importlib.util.spec_from_file_location("embedllm_knn_split_aligned", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_embedllm_knn_split_aligned_writes_routecode_metric_outputs(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 11",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 48",
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
                "external_baselines:",
                "  embedllm_knn_neighbors: [1, 3]",
                "  embedllm_knn_embedding_backend: routecode_embeddings",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    run_dir = out_dir / "embedllm_knn_split_aligned"
    table_path = out_dir / "table_embedllm_knn_split_aligned.csv"
    memo_path = out_dir / "phase_e_embedllm_knn_split_aligned_memo.md"
    raw_path = run_dir / "raw_predictions.json"
    assert table_path.exists()
    assert memo_path.exists()
    assert raw_path.exists()
    assert (out_dir / "embedllm_assets/train.csv").exists()
    assert (out_dir / "embedllm_assets/test.csv").exists()

    table = pd.read_csv(table_path)
    assert table["method"].tolist() == [
        "embedllm_knn_split_aligned_k1",
        "embedllm_knn_split_aligned_k3",
    ]
    assert table["baseline_family"].eq("embedllm_knn_local_metric_adapter").all()
    assert table["embedding_backend"].eq("routecode_embeddings").all()
    assert table["split_aligned_with_routecode"].all()
    assert table["routecode_metric_compatible"].all()
    assert not table["official_upstream_checkpoint"].all()
    assert not table["exact_upstream_command"].all()
    assert table["mean_correctness_accuracy"].between(0.0, 1.0).all()
    assert table["mean_selected_correctness_probability"].between(0.0, 1.0).all()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## EmbedLLM KNN Split-Aligned Evaluation" in readme
    assert "local metric adapter" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "same per-model kNN correctness idea" in memo
