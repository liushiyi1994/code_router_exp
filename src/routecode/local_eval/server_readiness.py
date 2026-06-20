from __future__ import annotations

from datetime import UTC, datetime
import json
import time
from typing import Any
from urllib import request

import pandas as pd

from routecode.local_eval.generation_runner import GenerationResult, OpenAICompatibleLocalClient


READINESS_COLUMNS = [
    "check_id",
    "status",
    "base_url",
    "model_id",
    "models_endpoint_status",
    "model_listed",
    "completion_status",
    "latency_sec",
    "tokens_input",
    "tokens_output",
    "blocking_reasons",
    "error_type",
    "error_message",
    "created_at",
]


class OpenAICompatibleReadinessClient:
    """OpenAI-compatible local server probe for readiness checks."""

    def __init__(self, *, base_url: str, api_key: str, timeout_sec: float) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout_sec = float(timeout_sec)
        self._generation_client = OpenAICompatibleLocalClient(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout_sec=self.timeout_sec,
        )

    def list_models(self) -> list[str]:
        req = request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
            method="GET",
        )
        with request.urlopen(req, timeout=self.timeout_sec) as response:  # noqa: S310 - local endpoint by config.
            body = json.loads(response.read().decode("utf-8"))
        data = body.get("data", []) if isinstance(body, dict) else []
        return [str(item.get("id")) for item in data if isinstance(item, dict) and item.get("id")]

    def generate(self, **kwargs: Any) -> GenerationResult:
        return self._generation_client.generate(**kwargs)


def inspect_local_server_readiness(
    *,
    base_url: str,
    api_key: str,
    model_ids: list[str],
    generation_params: dict[str, Any],
    timeout_sec: float = 10.0,
    client: Any | None = None,
) -> pd.DataFrame:
    readiness_client = client or OpenAICompatibleReadinessClient(
        base_url=base_url,
        api_key=api_key,
        timeout_sec=timeout_sec,
    )
    created_at = _now()
    try:
        listed_models = readiness_client.list_models()
        models_endpoint_status = "ok"
        model_error_type = ""
        model_error_message = ""
    except Exception as exc:
        listed_models = []
        models_endpoint_status = "error"
        model_error_type = type(exc).__name__
        model_error_message = str(exc)
    model_ids = _resolve_first_listed_model(model_ids, listed_models)

    rows: list[dict[str, Any]] = []
    prompt = "Reply with a short readiness acknowledgement."
    for model_id in model_ids:
        model_listed = bool(str(model_id) in listed_models)
        if models_endpoint_status == "error":
            rows.append(
                {
                    "check_id": f"local_openai_server:{model_id}",
                    "status": "blocked",
                    "base_url": str(base_url).rstrip("/"),
                    "model_id": str(model_id),
                    "models_endpoint_status": models_endpoint_status,
                    "model_listed": model_listed,
                    "completion_status": "skipped",
                    "latency_sec": 0.0,
                    "tokens_input": len(prompt.split()),
                    "tokens_output": 0,
                    "blocking_reasons": "models_endpoint_failed",
                    "error_type": model_error_type,
                    "error_message": model_error_message,
                    "created_at": created_at,
                }
            )
            continue

        started = time.perf_counter()
        try:
            result = readiness_client.generate(
                model_id=str(model_id),
                prompt=prompt,
                generation_params=generation_params,
                task=None,
            )
            completion_status = "ok"
            latency_sec = float(result.latency_sec)
            tokens_input = int(result.tokens_input)
            tokens_output = int(result.tokens_output)
            completion_error_type = ""
            completion_error_message = ""
            blocking_reasons = ""
            status = "ready"
        except Exception as exc:
            completion_status = "error"
            latency_sec = max(time.perf_counter() - started, 0.0)
            tokens_input = len(prompt.split())
            tokens_output = 0
            completion_error_type = type(exc).__name__
            completion_error_message = str(exc)
            blocking_reasons = "completion_failed"
            status = "blocked"

        if status == "ready" and models_endpoint_status == "ok" and not model_listed:
            status = "warning"
        rows.append(
            {
                "check_id": f"local_openai_server:{model_id}",
                "status": status,
                "base_url": str(base_url).rstrip("/"),
                "model_id": str(model_id),
                "models_endpoint_status": models_endpoint_status,
                "model_listed": model_listed,
                "completion_status": completion_status,
                "latency_sec": latency_sec,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "blocking_reasons": blocking_reasons,
                "error_type": completion_error_type or model_error_type,
                "error_message": _join_errors(model_error_message, completion_error_message),
                "created_at": created_at,
            }
        )
    table = pd.DataFrame(rows, columns=READINESS_COLUMNS)
    if not table.empty:
        table["model_listed"] = table["model_listed"].astype(object)
    return table


def _join_errors(*messages: str) -> str:
    return "; ".join(message for message in messages if message)


def _resolve_first_listed_model(model_ids: list[str], listed_models: list[str]) -> list[str]:
    if "__first_listed__" not in model_ids or not listed_models:
        return model_ids
    return [listed_models[0] if model_id == "__first_listed__" else model_id for model_id in model_ids]


def _now() -> str:
    return datetime.now(UTC).isoformat()
