from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "16_routellm_mf_split_aligned.py"
    spec = importlib.util.spec_from_file_location("routellm_mf_split_aligned", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_mf_split_aligned_script_trains_evaluates_and_writes_outputs(tmp_path):
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
                "  n_queries: 48",
                "  n_models: 2",
                "  n_domains: 2",
                "  n_route_labels: 4",
                "  embedding_dim: 6",
                "  model_ids: [Qwen3-8B, Qwen2.5-Coder-7B-Instruct]",
                "  model_costs:",
                "    Qwen3-8B: 0.20",
                "    Qwen2.5-Coder-7B-Instruct: 0.10",
                "utility:",
                "  lambda_cost: 0.35",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "external_baselines:",
                "  strong_model: Qwen3-8B",
                "  weak_model: Qwen2.5-Coder-7B-Instruct",
                "  mf_num_epochs: 1",
                "  mf_batch_size: 8",
                "  thresholds: [0.5]",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    run_dir = out_dir / "routellm_mf_split_aligned"
    checkpoint = run_dir / "mf_model.pt"
    metrics_json = run_dir / "raw_metrics.json"
    table_path = out_dir / "table_routellm_mf_split_aligned.csv"
    memo_path = out_dir / "phase_e_routellm_mf_split_aligned_memo.md"
    assert checkpoint.exists()
    assert metrics_json.exists()
    assert table_path.exists()
    assert memo_path.exists()

    table = pd.read_csv(table_path)
    assert table["method"].tolist() == ["routellm_mf_split_aligned_t0.5"]
    row = table.iloc[0]
    assert row["baseline_family"] == "official_code_local_embedding"
    assert bool(row["split_aligned_with_routecode"])
    assert bool(row["official_training_code_used"])
    assert not bool(row["official_upstream_checkpoint"])
    assert bool(row["routecode_metric_compatible"])
    assert row["threshold"] == 0.5
    assert 0.0 <= row["strong_selection_rate"] <= 1.0
    assert 0.0 <= row["selection_accuracy"] <= 1.0

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## RouteLLM MF Split-Aligned Evaluation" in readme
    assert "official MF training code with local RouteCode embeddings" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "not the upstream published RouteLLM checkpoint" in memo
