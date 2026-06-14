from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "13_transformer_backbone_readiness.py"
    spec = importlib.util.spec_from_file_location("transformer_backbone_readiness", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_transformer_readiness_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "README.md").write_text("# Pilot\n\n## Next Steps\n\n- old\n", encoding="utf-8")
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
                f"  output_dir: {out_dir}",
                "data:",
                "  source: llmrouterbench",
                "transformer_backbones:",
                f"  cache_dir: {cache}",
                "  requested_model_ids:",
                "    - answerdotai/ModernBERT-base",
                "    - Qwen/Qwen3-4B",
                "  max_runnable_gb: 0.01",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    table_path = out_dir / "table_transformer_backbone_readiness.csv"
    memo_path = out_dir / "phase_f_g_transformer_backbone_readiness_memo.md"
    assert table_path.exists()
    assert memo_path.exists()
    table = pd.read_csv(table_path)
    assert set(table["model_id"]) == {"answerdotai/ModernBERT-base", "Qwen/Qwen3-4B"}
    assert not table["runnable_as_encoder_baseline"].any()
    readme = (out_dir / "README.md").read_text(encoding="utf-8")
    assert "## Transformer Backbone Readiness" in readme
    assert "no downloads" in readme
    memo = memo_path.read_text(encoding="utf-8")
    assert "No transformer embedding baseline was executed" in memo
