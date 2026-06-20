from __future__ import annotations

import importlib.util
from pathlib import Path
import json

import pandas as pd

from routecode.local_eval.generation_runner import GenerationResult


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "60_exact_manifest_probe_collection.py"
    spec = importlib.util.spec_from_file_location("phase2_exact_manifest_probe_collection", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_exact_manifest_probe_script_writes_features_logs_metadata_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    manifest_path = tmp_path / "local_exact_task_manifest.csv"
    pd.DataFrame(
        [
            {
                "query_id": "aime:test:1",
                "query_text": "Find 20 + 22.",
                "dataset": "aime",
                "domain": "math",
                "source_split": "test",
                "routecode_split": "test",
                "task_type": "math",
                "gold_answer": "42",
                "choices_json": "[]",
                "metadata_json": "{}",
            },
            {
                "query_id": "math500:test:2",
                "query_text": "Find 5 + 7.",
                "dataset": "math500",
                "domain": "math",
                "source_split": "test",
                "routecode_split": "test",
                "task_type": "math",
                "gold_answer": "12",
                "choices_json": "[]",
                "metadata_json": "{}",
            },
        ]
    ).to_csv(manifest_path, index=False)
    config_path = tmp_path / "exact_manifest_probe.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: exact_manifest_probe_smoke",
                f"  output_dir: {out_dir}",
                "phase2_exact_manifest_probe:",
                f"  task_manifest_path: {manifest_path}",
                "  dry_run: true",
                "  model_ids: [dry_probe]",
                "  max_queries: 2",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 32",
            ]
        ),
        encoding="utf-8",
    )

    features = module.run(config_path=str(config_path), output_dir=str(out_dir))

    features_path = out_dir / "exact_manifest_probe_features.parquet"
    raw_path = out_dir / "exact_manifest_probe_raw_outputs.jsonl"
    metadata_path = out_dir / "exact_manifest_probe_run_metadata.json"
    readme_path = out_dir / "README.md"
    assert len(features) == 2
    assert features_path.exists()
    assert raw_path.exists()
    assert metadata_path.exists()
    assert readme_path.exists()
    saved = pd.read_parquet(features_path)
    assert saved["query_id"].tolist() == ["aime:test:1", "math500:test:2"]
    assert set(saved["probe_type"]) == {"aligned_local_confidence_probe"}
    assert "## Phase 2 Exact Manifest Probe Collection" in readme_path.read_text(encoding="utf-8")


def test_exact_manifest_probe_script_supports_transformers_backend(tmp_path, monkeypatch):
    module = _load_script()

    class FakeTransformersProbeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate(self, *, model_id, prompt, generation_params, task=None):
            assert model_id == "qwen-probe"
            assert generation_params["chat_template_kwargs"]["enable_thinking"] is False
            assert task is not None
            assert "Find 20 + 22" in prompt
            return GenerationResult(
                raw_output="Answer: arithmetic\nConfidence: 0.73",
                latency_sec=0.25,
                tokens_input=11,
                tokens_output=4,
            )

    monkeypatch.setattr(module, "TransformersLocalClient", FakeTransformersProbeClient, raising=False)
    out_dir = tmp_path / "phase2"
    manifest_path = tmp_path / "local_exact_task_manifest.csv"
    pd.DataFrame(
        [
            {
                "query_id": "aime:test:1",
                "query_text": "Find 20 + 22.",
                "dataset": "aime",
                "domain": "math",
                "source_split": "test",
                "routecode_split": "test",
                "task_type": "math",
                "gold_answer": "42",
                "choices_json": "[]",
                "metadata_json": "{}",
            }
        ]
    ).to_csv(manifest_path, index=False)
    config_path = tmp_path / "exact_manifest_probe_transformers.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: exact_manifest_probe_transformers",
                f"  output_dir: {out_dir}",
                "phase2_exact_manifest_probe:",
                f"  task_manifest_path: {manifest_path}",
                "  backend: transformers",
                "  dry_run: false",
                "  model_ids: [qwen-probe]",
                "  model_id_or_path: /tmp/local-qwen",
                "  model_revision: Qwen/local@test",
                "  max_queries: 1",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 32",
                "    chat_template_kwargs:",
                "      enable_thinking: false",
            ]
        ),
        encoding="utf-8",
    )

    features = module.run(config_path=str(config_path), output_dir=str(out_dir))

    metadata = json.loads((out_dir / "exact_manifest_probe_run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["dry_run"] is False
    assert metadata["backend"] == "transformers"
    assert metadata["model_revision"] == "Qwen/local@test"
    assert len(features) == 1
    row = features.iloc[0]
    assert row["probe_model_id"] == "qwen-probe"
    assert row["parsed_probe_answer"] == "arithmetic"
    assert row["self_confidence"] == 0.73
    assert row["error_type"] == ""
