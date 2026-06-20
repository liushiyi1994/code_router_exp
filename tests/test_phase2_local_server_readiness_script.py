from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from routecode.local_eval.generation_runner import GenerationResult


class ReadyClient:
    def __init__(self, model_id: str = "local-qwen"):
        self.model_id = model_id

    def list_models(self) -> list[str]:
        return [self.model_id]

    def generate(self, **_kwargs) -> GenerationResult:
        return GenerationResult(
            raw_output="Answer: ok",
            latency_sec=0.01,
            tokens_input=4,
            tokens_output=2,
        )


def _load_script():
    path = Path(__file__).resolve().parents[1] / "experiments" / "58_local_server_readiness.py"
    spec = importlib.util.spec_from_file_location("phase2_local_server_readiness", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_server_readiness_script_writes_table_memo_and_readme(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local_readiness.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: local_server_readiness_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_server_readiness:",
                "  base_url: http://localhost:8000/v1",
                "  api_key: local-routecode",
                "  model_ids: [local-qwen]",
                "  timeout_sec: 1.0",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 8",
            ]
        ),
        encoding="utf-8",
    )

    table = module.run(config_path=str(config_path), output_dir=str(out_dir), client=ReadyClient())

    table_path = out_dir / "table_local_server_readiness.csv"
    memo_path = out_dir / "m9_local_server_readiness_memo.md"
    readme_path = out_dir / "README.md"
    assert len(table) == 1
    assert table_path.exists()
    assert memo_path.exists()
    assert readme_path.exists()
    saved = pd.read_csv(table_path)
    assert saved.loc[0, "status"] == "ready"
    assert "## Phase 2 Local Server Readiness" in readme_path.read_text(encoding="utf-8")


def test_local_server_readiness_script_can_use_first_listed_model_placeholder(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local_readiness.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: local_server_readiness_smoke",
                f"  output_dir: {out_dir}",
                "phase2_local_server_readiness:",
                "  base_url: http://localhost:8001/v1",
                "  api_key: local-routecode",
                "  model_ids: [__first_listed__]",
                "  timeout_sec: 1.0",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 8",
            ]
        ),
        encoding="utf-8",
    )

    table = module.run(config_path=str(config_path), output_dir=str(out_dir), client=ReadyClient())

    assert table.loc[0, "model_id"] == "local-qwen"
    assert table.loc[0, "status"] == "ready"


def test_local_server_readiness_script_can_check_multiple_openai_endpoints(tmp_path):
    module = _load_script()
    out_dir = tmp_path / "phase2"
    config_path = tmp_path / "phase2_local_multi_endpoint_readiness.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run:",
                "  name: local_server_readiness_multi_endpoint",
                f"  output_dir: {out_dir}",
                "phase2_local_eval:",
                "  api_key: local-routecode",
                "  timeout_sec: 1.0",
                "  generation_params:",
                "    temperature: 0.0",
                "    max_tokens: 8",
                "  openai_endpoints:",
                "    - name: qwen",
                "      base_url: http://localhost:8001/v1",
                "      model_ids: [__first_listed__]",
                "    - name: coder",
                "      base_url: http://localhost:8002/v1",
                "      model_ids: [__first_listed__]",
            ]
        ),
        encoding="utf-8",
    )

    table = module.run(
        config_path=str(config_path),
        output_dir=str(out_dir),
        clients_by_base_url={
            "http://localhost:8001/v1": ReadyClient("served-qwen"),
            "http://localhost:8002/v1": ReadyClient("served-coder"),
        },
    )

    assert len(table) == 2
    assert table["base_url"].tolist() == ["http://localhost:8001/v1", "http://localhost:8002/v1"]
    assert table["model_id"].tolist() == ["served-qwen", "served-coder"]
    assert table["status"].tolist() == ["ready", "ready"]
    saved = pd.read_csv(out_dir / "table_local_server_readiness.csv")
    assert saved["model_id"].tolist() == ["served-qwen", "served-coder"]
