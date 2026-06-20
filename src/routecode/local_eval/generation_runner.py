from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import time
from typing import Any
from urllib import request

import pandas as pd

from routecode.local_eval.evaluators import score_exact
from routecode.local_eval.parsers import parse_math_answer, parse_multiple_choice_answer
from routecode.local_eval.prompt_templates import prompt_for_task


LOCAL_OUTCOME_COLUMNS = [
    "query_id",
    "query_text",
    "dataset",
    "domain",
    "model_id",
    "model_revision",
    "prompt_template",
    "generation_params_json",
    "raw_output",
    "parsed_answer",
    "gold_answer",
    "quality",
    "cost_proxy",
    "latency_sec",
    "tokens_input",
    "tokens_output",
    "error_type",
    "error_message",
    "created_at",
]


@dataclass(frozen=True)
class LocalEvalTask:
    query_id: str
    query_text: str
    dataset: str
    domain: str
    task_type: str
    gold_answer: str
    choices: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GenerationResult:
    raw_output: str
    latency_sec: float
    tokens_input: int
    tokens_output: int


class DryRunLocalClient:
    """Deterministic no-server client for validating Phase 2 logging plumbing."""

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        generation_params: dict[str, Any],
        task: LocalEvalTask,
    ) -> GenerationResult:
        del model_id, generation_params
        started = time.perf_counter()
        if task.task_type == "multiple_choice":
            raw_output = str(task.gold_answer).strip().upper()[:1]
        else:
            raw_output = f"Final answer: {task.gold_answer}"
        return GenerationResult(
            raw_output=raw_output,
            latency_sec=max(time.perf_counter() - started, 0.0),
            tokens_input=_count_tokens(prompt),
            tokens_output=_count_tokens(raw_output),
        )


class OpenAICompatibleLocalClient:
    """Small stdlib OpenAI-compatible client for local vLLM/llama.cpp/SGLang servers."""

    def __init__(self, base_url: str, api_key: str = "local-routecode", timeout_sec: float = 120.0) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key)
        self.timeout_sec = float(timeout_sec)

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

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        generation_params: dict[str, Any],
        task: LocalEvalTask | None = None,
    ) -> GenerationResult:
        del task
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            **_openai_generation_params(generation_params),
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        with request.urlopen(req, timeout=self.timeout_sec) as response:  # noqa: S310 - local endpoint by config.
            body = json.loads(response.read().decode("utf-8"))
        latency = time.perf_counter() - started
        raw_output = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        return GenerationResult(
            raw_output=raw_output,
            latency_sec=float(latency),
            tokens_input=int(usage.get("prompt_tokens") or _count_tokens(prompt)),
            tokens_output=int(usage.get("completion_tokens") or _count_tokens(raw_output)),
        )


class TransformersLocalClient:
    """Direct local Hugging Face Transformers client for smoke runs without a server."""

    def __init__(
        self,
        model_id_or_path: str,
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id_or_path = str(model_id_or_path)
        if tokenizer is None or model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id_or_path,
                local_files_only=bool(local_files_only),
                trust_remote_code=bool(trust_remote_code),
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id_or_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=bool(local_files_only),
                trust_remote_code=bool(trust_remote_code),
            )
        self.tokenizer = tokenizer
        self.model = model.eval() if hasattr(model, "eval") else model

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        generation_params: dict[str, Any],
        task: LocalEvalTask | None = None,
    ) -> GenerationResult:
        del model_id, task
        started = time.perf_counter()
        prompt_text = self._chat_prompt(prompt, generation_params=generation_params)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = _move_inputs_to_device(inputs, getattr(self.model, "device", None))
        input_ids = inputs["input_ids"]
        input_length = int(input_ids.shape[-1])
        generate_kwargs = _transformers_generation_params(generation_params)
        outputs = self.model.generate(**inputs, **generate_kwargs)
        first_output = outputs[0]
        completion_tokens = first_output[input_length:]
        raw_output = str(self.tokenizer.decode(completion_tokens, skip_special_tokens=True)).strip()
        latency = time.perf_counter() - started
        return GenerationResult(
            raw_output=raw_output,
            latency_sec=float(latency),
            tokens_input=input_length,
            tokens_output=int(len(completion_tokens)),
        )

    def _chat_prompt(self, prompt: str, *, generation_params: dict[str, Any]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_template_kwargs = dict(generation_params.get("chat_template_kwargs", {}))
            return str(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    **chat_template_kwargs,
                )
            )
        return str(prompt)


def run_generation_matrix(
    *,
    tasks: list[LocalEvalTask],
    model_ids: list[str],
    client,
    generation_params: dict[str, Any],
    model_revision: str = "",
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        prompt_template = prompt_for_task(task.task_type)
        prompt = prompt_template.render(
            query_text=task.query_text,
            choices="\n".join(task.choices),
        )
        for model_id in model_ids:
            created_at = _now()
            try:
                result = client.generate(
                    model_id=model_id,
                    prompt=prompt,
                    generation_params=generation_params,
                    task=task,
                )
                raw_output = result.raw_output
                parsed_answer = parse_answer(task.task_type, raw_output)
                quality = score_exact(parsed_answer, task.gold_answer)
                error_type = ""
                error_message = ""
                latency_sec = result.latency_sec
                tokens_input = result.tokens_input
                tokens_output = result.tokens_output
            except Exception as exc:
                raw_output = ""
                parsed_answer = ""
                quality = 0.0
                error_type = type(exc).__name__
                error_message = str(exc)
                latency_sec = 0.0
                tokens_input = _count_tokens(prompt)
                tokens_output = 0
                errors.append(
                    {
                        "query_id": task.query_id,
                        "model_id": model_id,
                        "error_type": error_type,
                        "error_message": error_message,
                        "created_at": created_at,
                    }
                )
            row = {
                "query_id": task.query_id,
                "query_text": task.query_text,
                "dataset": task.dataset,
                "domain": task.domain,
                "model_id": model_id,
                "model_revision": model_revision,
                "prompt_template": prompt_template.template_id,
                "generation_params_json": json.dumps(generation_params, sort_keys=True),
                "raw_output": raw_output,
                "parsed_answer": parsed_answer,
                "gold_answer": task.gold_answer,
                "quality": float(quality),
                "cost_proxy": float(latency_sec + 0.001 * tokens_output),
                "latency_sec": float(latency_sec),
                "tokens_input": int(tokens_input),
                "tokens_output": int(tokens_output),
                "error_type": error_type,
                "error_message": error_message,
                "created_at": created_at,
            }
            rows.append(row)
            raw_logs.append(
                {
                    **row,
                    "prompt": prompt,
                    "task_type": task.task_type,
                }
            )
    return pd.DataFrame(rows, columns=LOCAL_OUTCOME_COLUMNS), raw_logs, errors


def parse_answer(task_type: str, raw_output: str) -> str:
    normalized = str(task_type).lower()
    if normalized == "math":
        return parse_math_answer(raw_output)
    if normalized == "multiple_choice":
        return parse_multiple_choice_answer(raw_output)
    raise ValueError(f"Unsupported local-eval task type without sandboxed evaluator: {task_type}")


def _openai_generation_params(params: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(params)
    if "max_new_tokens" in mapped and "max_tokens" not in mapped:
        mapped["max_tokens"] = mapped.pop("max_new_tokens")
    return mapped


def _transformers_generation_params(params: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(params)
    mapped.pop("chat_template_kwargs", None)
    if "max_tokens" in mapped and "max_new_tokens" not in mapped:
        mapped["max_new_tokens"] = mapped.pop("max_tokens")
    temperature = float(mapped.get("temperature", 0.0))
    if temperature <= 0.0:
        mapped["do_sample"] = False
        mapped.pop("temperature", None)
    else:
        mapped["do_sample"] = True
    return mapped


def _move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to") and device is not None:
        return inputs.to(device)
    if isinstance(inputs, dict) and device is not None:
        moved = {}
        for key, value in inputs.items():
            moved[key] = value.to(device) if hasattr(value, "to") else value
        return moved
    return inputs


def _count_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def _now() -> str:
    return datetime.now(UTC).isoformat()
