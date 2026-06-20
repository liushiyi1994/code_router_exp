from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "51_true_model_generation_matrix.py"
    spec = importlib.util.spec_from_file_location("phase2_true_model_generation", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_true_model_generation_script_writes_dry_run_parquet_logs_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: phase2_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_eval:",
                "  dry_run: true",
                "  model_ids: [dry_run_model]",
                "  model_revision: dry-run",
                "  max_queries: 4",
                "  datasets: [gsm8k_smoke, mmlu_smoke]",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 16",
            ]
        ),
        encoding="utf-8",
    )

    module.run(str(config_path))

    outcomes_path = out_dir / "local_model_outcomes.parquet"
    raw_path = out_dir / "local_model_raw_outputs.jsonl"
    metadata_path = out_dir / "local_model_run_metadata.json"
    readme_path = out_dir / "README.md"
    assert outcomes_path.exists()
    assert raw_path.exists()
    assert metadata_path.exists()
    assert readme_path.exists()

    frame = pd.read_parquet(outcomes_path)
    assert len(frame) == 4
    assert set(frame["model_id"]) == {"dry_run_model"}
    assert set(frame["dataset"]) == {"gsm8k_smoke", "mmlu_smoke"}
    assert frame["quality"].eq(1.0).all()
    assert "## Phase 2 Local Model Outcomes" in readme_path.read_text(encoding="utf-8")
    assert len(raw_path.read_text(encoding="utf-8").strip().splitlines()) == 4


def test_true_model_generation_script_can_use_exact_task_manifest(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    manifest_path = tmp_path / "local_exact_task_manifest.csv"
    pd.DataFrame(
        [
            {
                "query_id": "aime:test:1",
                "query_text": "What is 20 + 22?",
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
                "query_text": "What is 5 + 7?",
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
    config_path = tmp_path / "phase2_local_manifest.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: phase2_manifest_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_eval:",
                "  dry_run: true",
                "  model_ids: [dry_run_model]",
                "  model_revision: dry-run",
                f"  task_manifest_path: {manifest_path}",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 16",
            ]
        ),
        encoding="utf-8",
    )

    frame = module.run(str(config_path))

    assert frame["query_id"].tolist() == ["aime:test:1", "math500:test:2"]
    assert set(frame["dataset"]) == {"aime", "math500"}
    assert frame["quality"].eq(1.0).all()


def test_true_model_generation_script_can_use_transformers_backend(tmp_path, monkeypatch):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local_transformers.yaml"

    class FakeTransformersClient:
        def __init__(self, **kwargs):
            assert kwargs["model_id_or_path"] == "fake-local-model"

        def generate(self, *, model_id, prompt, generation_params, task):
            from routecode.local_eval.generation_runner import GenerationResult

            del model_id, prompt, generation_params
            return GenerationResult(
                raw_output=f"Final answer: {task.gold_answer}",
                latency_sec=0.01,
                tokens_input=3,
                tokens_output=3,
            )

    monkeypatch.setattr(module, "TransformersLocalClient", FakeTransformersClient)
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: phase2_transformers_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_eval:",
                "  backend: transformers",
                "  dry_run: false",
                "  model_ids: [fake-local-model]",
                "  model_id_or_path: fake-local-model",
                "  model_revision: fake-revision",
                "  max_queries: 2",
                "  datasets: [gsm8k_smoke]",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 8",
            ]
        ),
        encoding="utf-8",
    )

    frame = module.run(str(config_path))

    assert len(frame) == 2
    assert set(frame["model_id"]) == {"fake-local-model"}
    assert frame["quality"].eq(1.0).all()
    metadata = (out_dir / "local_model_run_metadata.json").read_text(encoding="utf-8")
    assert '"backend": "transformers"' in metadata
    assert '"dry_run": false' in metadata


def test_true_model_generation_script_can_use_first_listed_openai_model(tmp_path, monkeypatch):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local_openai.yaml"
    seen_model_ids = []

    class FakeOpenAIClient:
        def __init__(self, *, base_url, api_key, timeout_sec):
            assert base_url == "http://localhost:8001/v1"
            assert api_key == "local-routecode"
            assert timeout_sec == 3.0

        def list_models(self):
            return ["served-qwen"]

        def generate(self, *, model_id, prompt, generation_params, task):
            from routecode.local_eval.generation_runner import GenerationResult

            del prompt, generation_params
            seen_model_ids.append(model_id)
            return GenerationResult(
                raw_output=f"Final answer: {task.gold_answer}",
                latency_sec=0.01,
                tokens_input=3,
                tokens_output=3,
            )

    monkeypatch.setattr(module, "OpenAICompatibleLocalClient", FakeOpenAIClient)
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: phase2_openai_server_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_eval:",
                "  backend: openai",
                "  dry_run: false",
                "  base_url: http://localhost:8001/v1",
                "  api_key: local-routecode",
                "  timeout_sec: 3.0",
                "  model_ids: [__first_listed__]",
                "  model_revision: local-vllm",
                "  max_queries: 2",
                "  datasets: [gsm8k_smoke]",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 8",
            ]
        ),
        encoding="utf-8",
    )

    frame = module.run(str(config_path))

    assert len(frame) == 2
    assert seen_model_ids == ["served-qwen", "served-qwen"]
    assert set(frame["model_id"]) == {"served-qwen"}
    metadata = (out_dir / "local_model_run_metadata.json").read_text(encoding="utf-8")
    assert '"backend": "openai"' in metadata
    assert '"model_ids": [\n    "served-qwen"\n  ]' in metadata


def test_true_model_generation_script_can_use_multiple_openai_endpoints(tmp_path, monkeypatch):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local_multi_endpoint.yaml"
    seen = []

    class FakeOpenAIClient:
        def __init__(self, *, base_url, api_key, timeout_sec):
            self.base_url = base_url
            assert api_key == "local-routecode"
            assert timeout_sec in {3.0, 5.0}

        def list_models(self):
            if self.base_url == "http://localhost:8001/v1":
                return ["served-qwen"]
            if self.base_url == "http://localhost:8002/v1":
                return ["served-coder"]
            return []

        def generate(self, *, model_id, prompt, generation_params, task):
            from routecode.local_eval.generation_runner import GenerationResult

            del prompt
            seen.append((self.base_url, model_id, generation_params["max_tokens"]))
            return GenerationResult(
                raw_output=f"Final answer: {task.gold_answer}",
                latency_sec=0.01,
                tokens_input=3,
                tokens_output=3,
            )

    monkeypatch.setattr(module, "OpenAICompatibleLocalClient", FakeOpenAIClient)
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: phase2_openai_multi_endpoint_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_eval:",
                "  backend: openai",
                "  dry_run: false",
                "  api_key: local-routecode",
                "  timeout_sec: 3.0",
                "  max_queries: 2",
                "  datasets: [gsm8k_smoke]",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 8",
                "  openai_endpoints:",
                "    - name: qwen",
                "      base_url: http://localhost:8001/v1",
                "      model_ids: [__first_listed__]",
                "      model_revision: qwen-vllm",
                "    - name: coder",
                "      base_url: http://localhost:8002/v1",
                "      timeout_sec: 5.0",
                "      model_ids: [__first_listed__]",
                "      model_revision: coder-vllm",
                "      generation_params:",
                "        max_tokens: 12",
            ]
        ),
        encoding="utf-8",
    )

    frame = module.run(str(config_path))

    assert len(frame) == 4
    assert set(frame["model_id"]) == {"served-qwen", "served-coder"}
    assert seen == [
        ("http://localhost:8001/v1", "served-qwen", 8),
        ("http://localhost:8001/v1", "served-qwen", 8),
        ("http://localhost:8002/v1", "served-coder", 12),
        ("http://localhost:8002/v1", "served-coder", 12),
    ]
    metadata = (out_dir / "local_model_run_metadata.json").read_text(encoding="utf-8")
    assert '"backend": "openai_multi_endpoint"' in metadata
    assert '"served-qwen"' in metadata
    assert '"served-coder"' in metadata
    assert '"openai_endpoints"' in metadata
