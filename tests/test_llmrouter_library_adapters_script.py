from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "27_llmrouter_library_adapters.py"
    spec = importlib.util.spec_from_file_location("llmrouter_library_adapters_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_llmrouter_library_adapters_script_writes_metrics_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Demo\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 3",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 96",
                "  n_models: 4",
                "  n_domains: 3",
                "  n_route_labels: 4",
                "  embedding_dim: 8",
                "  model_ids: [m0, m1, m2, m3]",
                "  model_costs:",
                "    m0: 0.05",
                "    m1: 0.06",
                "    m2: 0.07",
                "    m3: 0.08",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "routers:",
                "  knn_k: 3",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
                "llmrouter_library_adapters:",
                "  enabled: true",
                "  llmrouter_root: data/raw/external/LLMRouter",
                "  knn_k: 3",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_llmrouter_library_adapters.csv"
    memo_path = out_dir / "phase_e_llmrouter_library_adapters_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    assert (out_dir / "llmrouter_library_adapters" / "knn_model.pkl").exists()
    table = pd.read_csv(table_path)
    assert "llmrouter_library_knn" in set(table["method"])
    assert table["routecode_metric_compatible"].all()
    assert not table["exact_upstream_command"].any()

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## LLMRouter Library Adapters" in readme
    assert "table_llmrouter_library_adapters.csv" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "local LLMRouter trainer classes" in memo
    assert "not exact upstream command-path results" in memo
