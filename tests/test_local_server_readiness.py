from __future__ import annotations

from routecode.local_eval.generation_runner import GenerationResult
from routecode.local_eval.server_readiness import READINESS_COLUMNS, inspect_local_server_readiness


class ReadyClient:
    def list_models(self) -> list[str]:
        return ["local-qwen"]

    def generate(self, **_kwargs) -> GenerationResult:
        return GenerationResult(
            raw_output="Answer: ok\nConfidence: 0.7",
            latency_sec=0.05,
            tokens_input=6,
            tokens_output=4,
        )


class FailingClient:
    def list_models(self) -> list[str]:
        raise RuntimeError("server unavailable")

    def generate(self, **_kwargs) -> GenerationResult:
        raise AssertionError("generate should not be called when /models fails")


class CompletionFailingClient:
    def list_models(self) -> list[str]:
        return ["local-qwen"]

    def generate(self, **_kwargs) -> GenerationResult:
        raise RuntimeError("completion unavailable")


def test_local_server_readiness_marks_ready_completion_with_model_list():
    table = inspect_local_server_readiness(
        base_url="http://localhost:8000/v1",
        api_key="local-routecode",
        model_ids=["local-qwen"],
        generation_params={"temperature": 0.0, "max_tokens": 8},
        client=ReadyClient(),
    )

    assert list(table.columns) == READINESS_COLUMNS
    assert len(table) == 1
    row = table.iloc[0]
    assert row["status"] == "ready"
    assert row["model_listed"] is True
    assert row["completion_status"] == "ok"
    assert row["blocking_reasons"] == ""
    assert row["tokens_output"] == 4


def test_local_server_readiness_skips_completion_when_models_endpoint_fails():
    table = inspect_local_server_readiness(
        base_url="http://localhost:8000/v1",
        api_key="local-routecode",
        model_ids=["local-qwen"],
        generation_params={"temperature": 0.0, "max_tokens": 8},
        client=FailingClient(),
    )

    row = table.iloc[0]
    assert row["status"] == "blocked"
    assert row["blocking_reasons"] == "models_endpoint_failed"
    assert row["models_endpoint_status"] == "error"
    assert row["completion_status"] == "skipped"
    assert row["error_type"] == "RuntimeError"
    assert "server unavailable" in row["error_message"]


def test_local_server_readiness_marks_blocked_when_completion_fails():
    table = inspect_local_server_readiness(
        base_url="http://localhost:8000/v1",
        api_key="local-routecode",
        model_ids=["local-qwen"],
        generation_params={"temperature": 0.0, "max_tokens": 8},
        client=CompletionFailingClient(),
    )

    row = table.iloc[0]
    assert row["status"] == "blocked"
    assert row["blocking_reasons"] == "completion_failed"
    assert row["models_endpoint_status"] == "ok"
    assert row["completion_status"] == "error"
    assert row["error_type"] == "RuntimeError"
