from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "28_transformer_embedding_router.py"
    spec = importlib.util.spec_from_file_location("transformer_embedding_router", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transformer_embedding_router_script_writes_skipped_outputs_without_cached_encoder(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n", encoding="utf-8")
    cache = tmp_path / "hub"
    snapshot = cache / "models--Qwen--Qwen3-4B" / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 2560,
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model.safetensors").write_bytes(b"0" * 1024)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  random_seed: 4",
                f"  output_dir: {out_dir}",
                "data:",
                "  source: synthetic",
                "synthetic:",
                "  n_queries: 24",
                "  n_models: 3",
                "  n_domains: 2",
                "  n_route_labels: 2",
                "  embedding_dim: 4",
                "utility:",
                "  lambda_cost: 0.1",
                "split:",
                "  train_frac: 0.6",
                "  val_frac: 0.2",
                "  test_frac: 0.2",
                "transformer_backbones:",
                f"  cache_dir: {cache}",
                "  requested_model_ids:",
                "    - answerdotai/ModernBERT-base",
                "  max_runnable_gb: 0.01",
                "transformer_embedding_router:",
                "  direct_router_methods: [knn]",
                "  knn_k: 1",
                "bootstrap:",
                "  n_bootstrap: 5",
                "  ci: 0.95",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_transformer_embedding_router.csv"
    memo_path = out_dir / "phase_f_g_transformer_embedding_router_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert set(table["status"]) == {"skipped"}
    assert "no_cached_encoder_candidate" in set(table["reason"])
    assert "answerdotai/ModernBERT-base" in set(table["model_id"])
    assert "Qwen/Qwen3-4B" not in set(table["model_id"])

    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Transformer Embedding Router" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "local-files-only" in memo
    assert "No transformer direct-router metric row was executed" in memo
