from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd

from routecode.controlled.code_scoring import score_python_code_output
from routecode.controlled.config import load_controlled_inputs, load_env_values
from routecode.controlled.costing import TokenPrice, enforce_frontier_budget, estimate_token_cost
from routecode.controlled.surrogate import ControlledModel, load_models, load_prices, short_hash, token_counts


@dataclass(frozen=True)
class Stage0Task:
    query_id: str
    query_text: str
    benchmark: str
    domain: str
    metric: str
    gold_answer: str
    max_output_tokens: int
    split: str = "stage0"
    difficulty: float = 0.25
    difficulty_bin: str = "easy"


def generate_stage0_tasks(benchmarks: dict[str, Any], n_per_benchmark: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bench in benchmarks.get("benchmarks", []):
        name = str(bench["name"])
        domain = str(bench["domain"])
        metric = str(bench["metric"])
        max_output_tokens = min(int(bench.get("max_output_tokens", 128)), 128)
        for idx in range(n_per_benchmark):
            prompt, gold = smoke_prompt(name, domain, idx)
            rows.append(
                {
                    "query_id": f"{name}:live_stage0:{idx:03d}",
                    "query_text": prompt,
                    "benchmark": name,
                    "domain": domain,
                    "metric": metric,
                    "gold_answer": gold,
                    "max_output_tokens": max_output_tokens,
                    "split": "stage0",
                    "difficulty": 0.20 + 0.02 * idx,
                    "difficulty_bin": "easy",
                }
            )
    return pd.DataFrame(rows)


def generate_stage0_tasks_from_manifest(
    manifest_path: str | Path,
    *,
    datasets: list[str] | None = None,
    max_tasks: int | None = None,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"query_id", "query_text", "dataset", "domain", "task_type", "gold_answer"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"controlled task manifest missing required columns: {missing}")
    selected = manifest.copy()
    if datasets:
        allowed = {str(dataset) for dataset in datasets}
        selected = selected[selected["dataset"].astype(str).isin(allowed)]
    selected = selected.drop_duplicates("query_id", keep="first")
    if max_tasks is not None and int(max_tasks) > 0:
        selected = selected.head(int(max_tasks))
    rows: list[dict[str, Any]] = []
    for idx, row in selected.reset_index(drop=True).iterrows():
        query_text = str(row["query_text"]).strip()
        task_type = str(row["task_type"])
        if task_type == "multiple_choice":
            metric = "multiple_choice"
        elif task_type == "pass_at_1":
            metric = "pass_at_1"
        else:
            metric = "exact_final_answer"
        if metric == "exact_final_answer":
            query_text = f"{query_text}\n\nReturn only the final answer with no explanation."
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "query_text": query_text,
                "benchmark": str(row["dataset"]),
                "domain": str(row["domain"]),
                "metric": metric,
                "gold_answer": str(row["gold_answer"]),
                "max_output_tokens": int(row.get("max_output_tokens", 128) or 128),
                "split": str(row.get("routecode_split", row.get("source_split", "manifest"))),
                "difficulty": 0.5,
                "difficulty_bin": "manifest",
                "manifest_row_index": int(idx),
            }
        )
    return pd.DataFrame(rows)


def smoke_prompt(name: str, domain: str, idx: int) -> tuple[str, str]:
    a = 3 + idx
    b = 4 + idx
    if name in {"gpqa", "mmlu_pro"}:
        return (
            f"Choose the correct option. What is {a} + {b}? "
            f"A. {a + b - 1} B. {a + b} C. {a + b + 1} D. {a * b}. "
            "Return only the letter.",
            "B",
        )
    if "code" in domain:
        return (
            f"A Python function returns ({a} * 2) + {b}. What integer does it return? "
            "Return only the integer.",
            str(a * 2 + b),
        )
    if "science" in domain:
        return (
            "A controlled science smoke question: water freezes at 0 degrees Celsius. "
            "Return only the number of degrees Celsius.",
            "0",
        )
    return (f"Compute {a} + {b}. Return only the integer.", str(a + b))


def resolve_key(env_values: dict[str, str], names: list[str]) -> str | None:
    for name in names:
        value = env_values.get(name)
        if value:
            return value
    return None


def normalize_answer(text: str) -> str:
    """Canonicalize exact final answers without destroying LaTeX fractions."""
    cleaned = str(text or "").strip().lower().strip("`").strip()
    cleaned = re.sub(r"^\*{0,2}\s*(?:final\s+answer|answer|the\s+answer\s+is)\s*[:：]\s*\*{0,2}", "", cleaned)
    cleaned = cleaned.strip().strip(".")
    if len(cleaned) > 1 and cleaned[0] in "abcd" and (cleaned[1] in ").:" or cleaned[1].isspace()):
        return cleaned[0]

    cleaned = _extract_boxed(cleaned)
    cleaned = cleaned.replace("\\(", "").replace("\\)", "")
    cleaned = cleaned.replace("\\[", "").replace("\\]", "")
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace("\\left", "").replace("\\right", "")
    cleaned = cleaned.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    cleaned = cleaned.replace("π", "pi").replace("\\pi", "pi")
    cleaned = cleaned.replace("√", "\\sqrt")
    cleaned = re.sub(r"\\text\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", cleaned)
    cleaned = _rewrite_latex_fraction(cleaned)
    cleaned = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt\1", cleaned)
    cleaned = re.sub(r"\\sqrt\s+([a-z0-9]+)", r"sqrt\1", cleaned)
    cleaned = cleaned.replace("^\\circ", "").replace("^{\\circ}", "").replace("\\circ", "")
    cleaned = cleaned.replace("{", "").replace("}", "")
    cleaned = cleaned.replace("\\", "")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip().strip(".")
    return cleaned


def _extract_boxed(text: str) -> str:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return text
    idx = start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        idx += 1
    return text[start + len(marker) :].strip()


def _rewrite_latex_fraction(text: str) -> str:
    pattern = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    malformed_pattern = re.compile(r"\\frac\s*\{([^{}]+)\{([^{}]+)")
    previous = None
    rewritten = text
    while previous != rewritten:
        previous = rewritten
        rewritten = pattern.sub(r"\1/\2", rewritten)
        rewritten = malformed_pattern.sub(r"\1/\2", rewritten)
    return rewritten


def _numeric_value(value: str) -> Fraction | Decimal | None:
    cleaned = value.replace(",", "")
    if re.fullmatch(r"[-+]?\d+/\d+", cleaned):
        try:
            return Fraction(cleaned)
        except ZeroDivisionError:
            return None
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", cleaned):
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None
    return None


def _format_numeric(value: Fraction | Decimal) -> str:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return str(value.numerator)
        return f"{value.numerator}/{value.denominator}"
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def _canonical_answer_set(value: str) -> tuple[str, ...] | None:
    """Return a canonical unordered exact-answer set when the answer encodes one."""
    if not value:
        return None

    parts: list[str]
    if "," in value or ";" in value:
        parts = [part for part in re.split(r"[,;]", value) if part]
    else:
        parts = [value]

    expanded: list[str] = []
    for part in parts:
        if "pm" in part:
            left, right = part.split("pm", 1)
            if right:
                expanded.extend([f"{left}-{right}", f"{left}+{right}"])
            else:
                expanded.append(part)
        else:
            expanded.append(part)

    if len(expanded) <= 1:
        return None
    return tuple(sorted(set(expanded)))


def score_output(text: str, gold: str, metric: str) -> tuple[str, float]:
    if metric == "pass_at_1":
        scored = score_python_code_output(text, gold)
        return scored.parsed_answer, scored.quality
    parsed = normalize_answer(text)
    expected = normalize_answer(gold)
    if metric in {"multiple_choice", "exact_or_multiple_choice"} or expected[:1].upper() in {"A", "B", "C", "D"}:
        option_match = re.search(r"\b([abcd])\b", parsed)
        parsed_option = option_match.group(1).upper() if option_match else parsed[:1].upper()
        return parsed_option, 1.0 if parsed_option == expected[:1].upper() else 0.0
    expected_numeric = _numeric_value(expected)
    if expected_numeric is not None:
        numeric_matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?", parsed)
        if numeric_matches:
            parsed_number = numeric_matches[-1]
            parsed_numeric = _numeric_value(parsed_number)
            if parsed_numeric is not None:
                return _format_numeric(parsed_numeric), 1.0 if parsed_numeric == expected_numeric else 0.0
            return normalize_answer(parsed_number), 0.0
    parsed_set = _canonical_answer_set(parsed)
    expected_set = _canonical_answer_set(expected)
    if parsed_set is not None or expected_set is not None:
        return parsed, 1.0 if parsed_set == expected_set else 0.0
    return parsed, 1.0 if parsed == expected else 0.0


def extract_openai_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def extract_gemini_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []
    for candidate in payload.get("candidates", []) or []:
        content = candidate.get("content") or {}
        for part in content.get("parts", []) or []:
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks).strip()


def usage_from_openai(payload: dict[str, Any], fallback_input: int, fallback_output: int) -> tuple[int, int]:
    usage = payload.get("usage") or {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", fallback_input))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", fallback_output))
    return int(input_tokens or fallback_input), int(output_tokens or fallback_output)


def usage_from_gemini(payload: dict[str, Any], fallback_input: int, fallback_output: int) -> tuple[int, int]:
    usage = payload.get("usageMetadata") or {}
    input_tokens = usage.get("promptTokenCount", fallback_input)
    output_tokens = usage.get("candidatesTokenCount", fallback_output)
    return int(input_tokens or fallback_input), int(output_tokens or fallback_output)


def extract_openai_compatible_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", []) or []
    if not choices:
        return ""
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks = [str(part.get("text", "")) for part in content if isinstance(part, dict)]
        return "\n".join(chunk for chunk in chunks if chunk).strip()
    return ""


def usage_from_openai_compatible(
    payload: dict[str, Any], fallback_input: int, fallback_output: int
) -> tuple[int, int]:
    usage = payload.get("usage") or {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", fallback_input))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", fallback_output))
    return int(input_tokens or fallback_input), int(output_tokens or fallback_output)


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def call_openai(model_id: str, prompt: str, api_key: str, max_output_tokens: int, timeout_s: float) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "text": {"verbosity": "low"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            payload.pop("text", None)
            return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
        raise


def call_gemini(model_id: str, prompt: str, api_key: str, max_output_tokens: int, timeout_s: float) -> dict[str, Any]:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": int(max_output_tokens), "temperature": 0},
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    return post_json(url, payload, headers, timeout_s)


def call_openai_compatible_local(
    server: dict[str, Any], prompt: str, max_output_tokens: int, timeout_s: float
) -> dict[str, Any]:
    base_url = str(server.get("base_url", "")).rstrip("/")
    served_model_name = str(server.get("served_model_name") or server.get("id"))
    messages = [{"role": "user", "content": prompt}]
    if bool(server.get("answer_only_system_prompt", False)):
        messages.insert(
            0,
            {
                "role": "system",
                "content": (
                    "Solve the problem privately. Return only the final answer. "
                    "Do not include reasoning, Markdown, or surrounding text."
                ),
            },
        )
    payload = {
        "model": served_model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": int(max_output_tokens),
    }
    if "qwen" in served_model_name.lower():
        payload["chat_template_kwargs"] = {"enable_thinking": bool(server.get("enable_thinking", False))}
    headers = {
        "Authorization": "Bearer local-routecode",
        "Content-Type": "application/json",
    }
    return post_json(f"{base_url}/chat/completions", payload, headers, timeout_s)


def lazy_load_metadata(server: dict[str, Any], *, generation_latency_s: float) -> dict[str, Any]:
    """Return latency fields for a local lazy-loaded model row.

    `latency_s` is the measured request/generation latency. Model load and
    warmup are operational overhead and are reported separately.
    """

    return {
        "load_mode": str(server.get("load_mode", "ready_endpoint")),
        "model_load_time_s": float(server.get("model_load_time_s", 0.0) or 0.0),
        "warmup_time_s": float(server.get("warmup_time_s", 0.0) or 0.0),
        "latency_s": float(generation_latency_s),
        "latency_excludes_load_warmup": True,
        "start_command": str(server.get("start_command", "")),
        "stop_command": str(server.get("stop_command", "")),
    }


def cache_path(cache_dir: Path, run_id: str, model: ControlledModel, query_id: str) -> Path:
    return cache_dir / run_id / model.provider / model.id.replace("/", "_") / f"{query_id.replace(':', '_')}.json"


def local_readiness(servers: dict[str, Any], timeout_s: float = 2.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in servers.get("local_models", []):
        base_url = str(model.get("base_url", "")).rstrip("/")
        url = f"{base_url}/models" if base_url else ""
        start = time.time()
        status = "skipped"
        error = ""
        latency = 0.0
        if url:
            try:
                with urllib.request.urlopen(url, timeout=timeout_s) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    served_model_name = str(model.get("served_model_name") or model.get("id"))
                    served_ids = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
                    if response.status >= 500:
                        status = "error"
                    elif served_model_name in served_ids:
                        status = "ready"
                    else:
                        status = "unavailable"
                        error = f"served_model_not_found:{served_model_name}"
            except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
                status = "unavailable"
                error = type(exc).__name__
            except (json.JSONDecodeError, ValueError) as exc:
                status = "error"
                error = type(exc).__name__
            latency = time.time() - start
        rows.append(
            {
                "model_id": model.get("id"),
                "served_model_name": model.get("served_model_name"),
                "backend": model.get("backend", "vllm"),
                "base_url": base_url,
                "status": status,
                "latency_s": latency,
                "error_type": error,
                "fallback_mode": model.get("fallback_mode", "controlled_surrogate"),
                "load_mode": model.get("load_mode", "ready_endpoint"),
                "start_command": model.get("start_command", ""),
                "stop_command": model.get("stop_command", ""),
            }
        )
    return pd.DataFrame(rows)


def ready_local_servers(servers: dict[str, Any], readiness: pd.DataFrame) -> list[dict[str, Any]]:
    if readiness.empty:
        return []
    ready_ids = set(readiness.loc[readiness["status"].eq("ready"), "model_id"].astype(str))
    return [
        dict(model)
        for model in servers.get("local_models", [])
        if model.get("enabled", True) and str(model.get("id")) in ready_ids
    ]


def local_servers_for_collection(
    servers: dict[str, Any],
    readiness: pd.DataFrame,
    *,
    cache_dir: Path,
    run_id: str,
    tasks: pd.DataFrame,
    force_rerun: bool,
    force_local_rerun: bool,
    local_model_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    ready_ids = set(readiness.loc[readiness["status"].eq("ready"), "model_id"].astype(str)) if not readiness.empty else set()
    allowed_ids = {str(model_id) for model_id in local_model_ids or []}
    selected: list[dict[str, Any]] = []
    for model in servers.get("local_models", []):
        if not model.get("enabled", True):
            continue
        model_id = str(model.get("id"))
        if allowed_ids and model_id not in allowed_ids:
            continue
        server = dict(model)
        server["_ready"] = model_id in ready_ids
        local_model = ControlledModel(
            id=model_id,
            provider="local",
            role=str(server.get("role", "local")),
            is_local=True,
            is_frontier=False,
            server_backend=str(server.get("backend", "vllm")),
        )
        has_cache = any(cache_path(cache_dir, run_id, local_model, str(task.query_id)).exists() for task in tasks.itertuples())
        if server["_ready"] or (has_cache and not force_rerun and not force_local_rerun):
            selected.append(server)
    return selected


def estimate_stage0_costs(
    tasks: pd.DataFrame,
    models: list[ControlledModel],
    prices: dict[str, TokenPrice],
    config: dict[str, Any],
    max_output_tokens: int,
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for task in tasks.itertuples(index=False):
        task_row = pd.Series(task._asdict())
        for model in models:
            if not model.is_frontier:
                continue
            input_tokens, estimated_output_tokens = token_counts(task_row, model, config)
            output_tokens = max(int(estimated_output_tokens), int(max_output_tokens))
            _, _, total = estimate_token_cost(input_tokens, output_tokens, prices.get(model.id))
            totals[model.id] = totals.get(model.id, 0.0) + total
    return totals


def filter_frontier_models(
    models: list[ControlledModel],
    frontier_model_ids: list[str] | None = None,
) -> list[ControlledModel]:
    frontier_models = [model for model in models if model.is_frontier]
    allowed_ids = {str(model_id) for model_id in frontier_model_ids or []}
    if not allowed_ids:
        return frontier_models
    selected = [model for model in frontier_models if model.id in allowed_ids]
    missing = sorted(allowed_ids - {model.id for model in frontier_models})
    if missing:
        raise ValueError(f"Unknown or disabled frontier model id(s): {', '.join(missing)}")
    return selected


def live_routing_summary(outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    successful = outputs[outputs["status"].eq("success")].copy()
    if successful.empty:
        return pd.DataFrame()
    gpt_cost = successful.loc[successful["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()) if not gpt_cost.empty else float(successful["cost_total_usd"].max()), 1e-12)
    successful["normalized_remote_cost"] = successful["cost_total_usd"] / cost_norm
    successful["utility_cost_aware"] = successful["quality_score"] - float(lambda_cost) * successful["normalized_remote_cost"]
    quality_oracle = successful.loc[successful.groupby("query_id")["quality_score"].idxmax()]
    cost_oracle = successful.loc[successful.groupby("query_id")["utility_cost_aware"].idxmax()]

    query_ids = pd.Index(sorted(successful["query_id"].unique()))
    selections: dict[str, pd.Series] = {}
    probe_rates: dict[str, float] = {}
    for model_id in sorted(successful["model_id"].unique()):
        selections[f"all_{model_id}"] = pd.Series(model_id, index=query_ids)
        probe_rates[f"all_{model_id}"] = 0.0
    local_models = sorted(successful.loc[successful["is_local"].astype(bool), "model_id"].unique())
    frontier_models = sorted(successful.loc[successful["is_frontier"].astype(bool), "model_id"].unique())
    if local_models:
        best_local = (
            successful[successful["model_id"].isin(local_models)]
            .groupby("model_id")["utility_cost_aware"]
            .mean()
            .idxmax()
        )
        selections["best_local"] = pd.Series(best_local, index=query_ids)
        probe_rates["best_local"] = 0.0
    if frontier_models:
        best_frontier = (
            successful[successful["model_id"].isin(frontier_models)]
            .groupby("model_id")["utility_cost_aware"]
            .mean()
            .idxmax()
        )
        selections["best_frontier"] = pd.Series(best_frontier, index=query_ids)
        probe_rates["best_frontier"] = 0.0
    if local_models and "gpt-5.5" in set(successful["model_id"]):
        local_id = str(best_local) if "best_local" in locals() else local_models[0]
        task_domains = successful.drop_duplicates("query_id").set_index("query_id")["domain"]
        code_mask = task_domains.astype(str).isin(["code", "code_live"])
        selections["domain_rule_code_to_gpt_else_local"] = pd.Series(local_id, index=query_ids)
        selections["domain_rule_code_to_gpt_else_local"].loc[code_mask[code_mask].index] = "gpt-5.5"
        probe_rates["domain_rule_code_to_gpt_else_local"] = 0.0
        rescue = local_consistency_rescue(successful, local_id=local_id, frontier_id="gpt-5.5", query_ids=query_ids)
        if rescue is not None:
            rescue_selected, rescue_probe_rate = rescue
            selections["local_consistency_rescue_gpt"] = rescue_selected
            probe_rates["local_consistency_rescue_gpt"] = rescue_probe_rate
            selections["selective_code_consistency_rescue_gpt"] = rescue_selected
            probe_rates["selective_code_consistency_rescue_gpt"] = rescue_probe_rate

    rows: list[dict[str, Any]] = []
    for method, selected in selections.items():
        selected_rows = (
            pd.DataFrame({"query_id": selected.index, "model_id": selected.values})
            .merge(successful, on=["query_id", "model_id"], how="left")
            .dropna(subset=["quality_score"])
        )
        if selected_rows.empty:
            continue
        rows.append(
            _routing_row(
                method,
                selected_rows,
                cost_oracle,
                quality_oracle,
                probe_call_rate=probe_rates.get(method, 0.0),
            )
        )

    rows.append(_routing_row("quality_oracle", quality_oracle, cost_oracle, quality_oracle))
    rows.append(_routing_row("cost_aware_oracle", cost_oracle, cost_oracle, quality_oracle))
    return pd.DataFrame(rows).sort_values(["mean_utility", "mean_quality"], ascending=False)


def _routing_row(
    method: str,
    selected_rows: pd.DataFrame,
    cost_oracle: pd.DataFrame,
    quality_oracle: pd.DataFrame,
    *,
    probe_call_rate: float = 0.0,
) -> dict[str, Any]:
    mean_utility = float(selected_rows["utility_cost_aware"].mean())
    oracle_utility = float(cost_oracle["utility_cost_aware"].mean())
    mean_quality = float(selected_rows["quality_score"].mean())
    oracle_quality = float(quality_oracle["quality_score"].mean())
    return {
        "method": method,
        "n_queries": int(selected_rows["query_id"].nunique()),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_oracle_mean_quality": oracle_quality,
        "cost_oracle_mean_utility": oracle_utility,
        "quality_gap_to_oracle": oracle_quality - mean_quality,
        "utility_gap_to_oracle": oracle_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / oracle_utility if abs(oracle_utility) > 1e-12 else float("nan"),
        "remote_cost_total_usd": float(selected_rows["cost_total_usd"].sum()),
        "frontier_call_rate": float(selected_rows["is_frontier"].astype(bool).mean()),
        "probe_call_rate": float(probe_call_rate),
        "mean_latency_s": float(selected_rows["latency_s"].mean()),
        "p95_latency_s": float(selected_rows["latency_s"].quantile(0.95)),
    }


def local_consistency_rescue(
    successful: pd.DataFrame,
    *,
    local_id: str,
    frontier_id: str,
    query_ids: pd.Index,
) -> tuple[pd.Series, float] | None:
    if "query_text" not in successful.columns:
        return None
    available = set(successful["model_id"].unique())
    if local_id not in available or frontier_id not in available:
        return None
    local_rows = successful[successful["model_id"].eq(local_id)].drop_duplicates("query_id").set_index("query_id")
    selected = pd.Series(local_id, index=query_ids)
    probed = pd.Series(False, index=query_ids)
    for query_id, row in local_rows.iterrows():
        query_text = str(row.get("query_text", ""))
        parsed = str(row.get("parsed_answer", "")).strip()
        if str(row.get("domain", "")) not in {"code", "code_live"}:
            continue
        match = re.search(r"returns\s*\(\s*(-?\d+)\s*\*\s*2\s*\)\s*\+\s*(-?\d+)", query_text)
        if not match:
            continue
        probed.loc[query_id] = True
        expected = str(int(match.group(1)) * 2 + int(match.group(2)))
        if parsed != expected:
            selected.loc[query_id] = frontier_id
    return selected, float(probed.mean())


def run_live_stage0(
    config_path: str | Path,
    *,
    allow_frontier_calls: bool = False,
    output_dir: str | Path = "results/controlled/live_stage0",
    examples_per_benchmark: int | None = None,
    run_suffix: str = "live_stage0",
    force_rerun: bool = False,
    force_local_rerun: bool = False,
    retry_errors: bool = False,
    max_calls_per_frontier_model: int | None = None,
    max_calls_per_local_model: int | None = None,
    frontier_concurrency: int = 1,
    task_manifest_path: str | Path | None = None,
    task_datasets: list[str] | None = None,
    max_tasks: int | None = None,
    max_output_tokens_override: int | None = None,
    local_max_output_tokens_override: int | None = None,
    local_model_ids: list[str] | None = None,
    frontier_model_ids: list[str] | None = None,
    request_timeout_s_override: float | None = None,
) -> dict[str, Path]:
    bundle = load_controlled_inputs(config_path)
    config = bundle["config"]
    prices = load_prices(bundle["prices"])
    models = load_models(bundle["servers"])
    frontier_models = filter_frontier_models(models, frontier_model_ids)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_per_benchmark = (
        int(examples_per_benchmark)
        if examples_per_benchmark is not None
        else int(config.get("surrogate", {}).get("dry_run_examples_per_benchmark", 5))
    )
    tasks = (
        generate_stage0_tasks_from_manifest(task_manifest_path, datasets=task_datasets, max_tasks=max_tasks)
        if task_manifest_path
        else generate_stage0_tasks(bundle["benchmarks"], n_per_benchmark)
    )
    max_output_tokens = (
        int(max_output_tokens_override)
        if max_output_tokens_override is not None
        else int(config.get("live_stage0", {}).get("max_output_tokens", 64))
    )
    local_max_output_tokens = (
        int(local_max_output_tokens_override)
        if local_max_output_tokens_override is not None
        else int(config.get("live_stage0", {}).get("local_max_output_tokens", max_output_tokens))
    )
    timeout_s = (
        float(request_timeout_s_override)
        if request_timeout_s_override is not None
        else float(config.get("live_stage0", {}).get("request_timeout_s", 60.0))
    )
    estimate = estimate_stage0_costs(tasks, frontier_models, prices, config, max_output_tokens)
    if allow_frontier_calls:
        enforce_frontier_budget(
            estimate,
            max_total_frontier_spend_usd=float(config.get("budget", {}).get("max_total_frontier_spend_usd", 0.0)),
            max_spend_per_frontier_model_usd=float(config.get("budget", {}).get("max_spend_per_frontier_model_usd", 0.0)),
        )
    readiness_timeout_s = float(config.get("live_stage0", {}).get("local_readiness_timeout_s", 2.0))
    env_values = load_env_values(config.get("budget", {}).get("env_file", ".env"))
    cache_dir = Path(config.get("budget", {}).get("cache_dir", "results/controlled/raw_outputs"))
    run_id = f"{config.get('run_id', 'controlled')}_{run_suffix}"
    readiness = local_readiness(bundle["servers"], timeout_s=readiness_timeout_s)
    readiness.to_csv(out_dir / "local_readiness.csv", index=False)
    local_servers = local_servers_for_collection(
        bundle["servers"],
        readiness,
        cache_dir=cache_dir,
        run_id=run_id,
        tasks=tasks,
        force_rerun=force_rerun,
        force_local_rerun=force_local_rerun,
        local_model_ids=local_model_ids,
    )
    rows: list[dict[str, Any]] = []
    calls_by_model = {model.id: 0 for model in frontier_models}
    local_calls_by_model = {str(server.get("id")): 0 for server in local_servers}
    for task in tasks.itertuples(index=False):
        task_row = pd.Series(task._asdict())
        for server in local_servers:
            model_id = str(server.get("id"))
            server_ready = bool(server.get("_ready", False))
            local_model = ControlledModel(
                id=model_id,
                provider="local",
                role=str(server.get("role", "local")),
                is_local=True,
                is_frontier=False,
                server_backend=str(server.get("backend", "vllm")),
            )
            input_est, output_est = token_counts(task_row, local_model, config)
            output_est = min(output_est, local_max_output_tokens)
            raw_path = cache_path(cache_dir, run_id, local_model, task.query_id)
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            cache_hit = raw_path.exists() and not force_rerun and not force_local_rerun
            start = time.time()
            status = "success"
            error_type = ""
            response_payload: dict[str, Any] = {}
            raw_text = ""
            input_tokens = input_est
            output_tokens = output_est
            if cache_hit:
                response_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                cached_status = response_payload.get("_status")
                if retry_errors and cached_status == "error":
                    cache_hit = False
                    response_payload = {}
            if (
                not cache_hit
                and max_calls_per_local_model is not None
                and local_calls_by_model[model_id] >= max_calls_per_local_model
            ):
                continue
            if not cache_hit and not server_ready:
                continue
            if cache_hit:
                raw_text = str(response_payload.get("_parsed_text", ""))
                input_tokens = int(response_payload.get("_input_tokens", input_est))
                output_tokens = int(response_payload.get("_output_tokens", output_est))
                status = str(response_payload.get("_status", "success"))
                error_type = str(response_payload.get("_error_type", ""))
                cached_latency = response_payload.get("_latency_s")
                if cached_latency is not None:
                    start = time.time() - float(cached_latency)
            else:
                try:
                    response_payload = call_openai_compatible_local(
                        server, task.query_text, local_max_output_tokens, timeout_s
                    )
                    raw_text = extract_openai_compatible_text(response_payload)
                    input_tokens, output_tokens = usage_from_openai_compatible(response_payload, input_est, output_est)
                except Exception as exc:  # Local server errors must be cached and reported, not crash the run.
                    status = "error"
                    error_type = type(exc).__name__
                    response_payload = {"error_type": error_type, "error": str(exc)[:500]}
                response_payload["_parsed_text"] = raw_text
                response_payload["_input_tokens"] = input_tokens
                response_payload["_output_tokens"] = output_tokens
                response_payload["_status"] = status
                response_payload["_error_type"] = error_type
                response_payload["_latency_s"] = time.time() - start
                response_payload["_served_model_name"] = str(server.get("served_model_name") or model_id)
                raw_path.write_text(json.dumps(response_payload, indent=2, sort_keys=True), encoding="utf-8")
                local_calls_by_model[model_id] += 1
            latency_s = time.time() - start
            load_meta = lazy_load_metadata(server, generation_latency_s=latency_s)
            parsed_answer, quality = score_output(raw_text, task.gold_answer, task.metric)
            if status != "success":
                quality = float("nan")
            rows.append(
                {
                    "run_id": run_id,
                    "query_id": task.query_id,
                    "query_text": task.query_text,
                    "benchmark": task.benchmark,
                    "domain": task.domain,
                    "model_id": model_id,
                    "provider": "local",
                    "is_local": True,
                    "is_frontier": False,
                    "is_probe": str(server.get("role")) == "cheap_probe",
                    "prompt_template_version": f"{run_suffix}_v1",
                    "prompt_hash": short_hash(task.query_text),
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "max_output_tokens": local_max_output_tokens,
                    "start_time_unix": start,
                    "end_time_unix": start + latency_s,
                    "latency_s": load_meta["latency_s"],
                    "model_load_time_s": load_meta["model_load_time_s"],
                    "warmup_time_s": load_meta["warmup_time_s"],
                    "latency_excludes_load_warmup": load_meta["latency_excludes_load_warmup"],
                    "load_mode": load_meta["load_mode"],
                    "status": status,
                    "error_type": error_type,
                    "raw_output_path": str(raw_path),
                    "parsed_answer": parsed_answer,
                    "gold_answer": task.gold_answer,
                    "quality_score": quality,
                    "cost_input_usd": 0.0,
                    "cost_output_usd": 0.0,
                    "cost_total_usd": 0.0,
                    "cache_hit": cache_hit,
                    "server_backend": str(server.get("backend", "vllm")),
                    "server_config_json": json.dumps(
                        {
                            "base_url": server.get("base_url"),
                            "served_model_name": server.get("served_model_name"),
                            "stage": run_suffix,
                            "load_mode": load_meta["load_mode"],
                            "start_command": load_meta["start_command"],
                            "stop_command": load_meta["stop_command"],
                        },
                        sort_keys=True,
                    ),
                    "hardware_id": "local_vllm",
                    "metric": task.metric,
                }
            )
    frontier_jobs: list[tuple[dict[str, Any], ControlledModel]] = []
    for task in tasks.itertuples(index=False):
        task_dict = task._asdict()
        for model in frontier_models:
            task_row = pd.Series(task_dict)
            input_est, output_est = token_counts(task_row, model, config)
            output_est = min(output_est, max_output_tokens)
            raw_path = cache_path(cache_dir, run_id, model, task.query_id)
            cache_hit = raw_path.exists() and not force_rerun
            if cache_hit:
                try:
                    response_payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    cached_status = response_payload.get("_status")
                    if allow_frontier_calls and (
                        cached_status == "skipped"
                        or response_payload.get("skipped") is True
                        or (retry_errors and cached_status == "error")
                    ):
                        cache_hit = False
                except json.JSONDecodeError:
                    cache_hit = False
            if (
                not cache_hit
                and max_calls_per_frontier_model is not None
                and calls_by_model[model.id] >= max_calls_per_frontier_model
            ):
                continue
            if not cache_hit:
                calls_by_model[model.id] += 1
            frontier_jobs.append((task_dict, model))

    def frontier_row(task_dict: dict[str, Any], model: ControlledModel) -> dict[str, Any]:
        task_row = pd.Series(task_dict)
        input_est, output_est = token_counts(task_row, model, config)
        output_est = min(output_est, max_output_tokens)
        price = prices.get(model.id)
        raw_path = cache_path(cache_dir, run_id, model, str(task_dict["query_id"]))
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        cache_hit = raw_path.exists() and not force_rerun
        start = time.time()
        status = "success"
        error_type = ""
        response_payload: dict[str, Any] = {}
        raw_text = ""
        input_tokens = input_est
        output_tokens = output_est
        if cache_hit:
            response_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            cached_status = response_payload.get("_status")
            if allow_frontier_calls and (
                cached_status == "skipped"
                or response_payload.get("skipped") is True
                or (retry_errors and cached_status == "error")
            ):
                cache_hit = False
                response_payload = {}
        if cache_hit:
            raw_text = str(response_payload.get("_parsed_text", ""))
            input_tokens = int(response_payload.get("_input_tokens", input_est))
            output_tokens = int(response_payload.get("_output_tokens", output_est))
            status = str(response_payload.get("_status", "success"))
            error_type = str(response_payload.get("_error_type", ""))
            cached_latency = response_payload.get("_latency_s")
            if cached_latency is not None:
                start = time.time() - float(cached_latency)
        elif not allow_frontier_calls:
            status = "skipped"
            error_type = "frontier_calls_disabled"
            response_payload = {
                "skipped": True,
                "reason": error_type,
                "_status": status,
                "_error_type": error_type,
                "_parsed_text": "",
                "_input_tokens": input_tokens,
                "_output_tokens": output_tokens,
            }
            raw_path.write_text(json.dumps(response_payload, indent=2, sort_keys=True), encoding="utf-8")
        else:
            api_key = resolve_key(
                env_values,
                ["OPENAI_API_KEY", "openai_api_key"]
                if model.provider == "openai"
                else ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"],
            )
            if not api_key:
                status = "error"
                error_type = "missing_api_key"
                response_payload = {"error": error_type}
            else:
                try:
                    if model.provider == "openai":
                        response_payload = call_openai(
                            model.id, str(task_dict["query_text"]), api_key, max_output_tokens, timeout_s
                        )
                        raw_text = extract_openai_text(response_payload)
                        input_tokens, output_tokens = usage_from_openai(response_payload, input_est, output_est)
                    elif model.provider == "google":
                        response_payload = call_gemini(
                            model.id, str(task_dict["query_text"]), api_key, max_output_tokens, timeout_s
                        )
                        raw_text = extract_gemini_text(response_payload)
                        input_tokens, output_tokens = usage_from_gemini(response_payload, input_est, output_est)
                    else:
                        status = "error"
                        error_type = "unsupported_provider"
                        response_payload = {"error": error_type}
                except Exception as exc:  # API errors must be cached and reported, not crash the run.
                    status = "error"
                    error_type = type(exc).__name__
                    response_payload = {"error_type": error_type, "error": str(exc)[:500]}
            response_payload["_parsed_text"] = raw_text
            response_payload["_input_tokens"] = input_tokens
            response_payload["_output_tokens"] = output_tokens
            response_payload["_status"] = status
            response_payload["_error_type"] = error_type
            response_payload["_latency_s"] = time.time() - start
            raw_path.write_text(json.dumps(response_payload, indent=2, sort_keys=True), encoding="utf-8")
        latency_s = time.time() - start
        parsed_answer, quality = score_output(raw_text, str(task_dict["gold_answer"]), str(task_dict["metric"]))
        if status != "success":
            quality = float("nan")
        cost_input, cost_output, cost_total = estimate_token_cost(input_tokens, output_tokens, price)
        return {
            "run_id": run_id,
            "query_id": task_dict["query_id"],
            "query_text": task_dict["query_text"],
            "benchmark": task_dict["benchmark"],
            "domain": task_dict["domain"],
            "model_id": model.id,
            "provider": model.provider,
            "is_local": False,
            "is_frontier": True,
            "is_probe": False,
            "prompt_template_version": f"{run_suffix}_v1",
            "prompt_hash": short_hash(str(task_dict["query_text"])),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_output_tokens": max_output_tokens,
            "start_time_unix": start,
            "end_time_unix": start + latency_s,
            "latency_s": latency_s,
            "model_load_time_s": 0.0,
            "warmup_time_s": 0.0,
            "latency_excludes_load_warmup": True,
            "load_mode": "api",
            "status": status,
            "error_type": error_type,
            "raw_output_path": str(raw_path),
            "parsed_answer": parsed_answer,
            "gold_answer": task_dict["gold_answer"],
            "quality_score": quality,
            "cost_input_usd": cost_input,
            "cost_output_usd": cost_output,
            "cost_total_usd": cost_total,
            "cache_hit": cache_hit,
            "server_backend": "api",
            "server_config_json": json.dumps({"stage": run_suffix}, sort_keys=True),
            "hardware_id": "remote_api",
            "metric": task_dict["metric"],
        }

    if frontier_jobs:
        max_workers = max(1, int(frontier_concurrency))
        if max_workers == 1:
            rows.extend(frontier_row(task_dict, model) for task_dict, model in frontier_jobs)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(frontier_row, task_dict, model) for task_dict, model in frontier_jobs]
                for future in as_completed(futures):
                    rows.append(future.result())
    outputs = pd.DataFrame(rows)
    outputs.to_parquet(out_dir / "model_outputs.parquet", index=False)
    outputs.to_parquet(out_dir / "scored_outputs.parquet", index=False)
    summary = (
        outputs.groupby(["model_id", "provider", "status"], dropna=False)
        .agg(
            n_calls=("query_id", "count"),
            mean_quality=("quality_score", "mean"),
            total_cost_usd=("cost_total_usd", "sum"),
            mean_latency_s=("latency_s", "mean"),
            p95_latency_s=("latency_s", lambda s: float(s.quantile(0.95))),
            mean_model_load_time_s=("model_load_time_s", "mean"),
            mean_warmup_time_s=("warmup_time_s", "mean"),
            cache_hits=("cache_hit", "sum"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "cost_latency_summary.csv", index=False)
    routing_summary = live_routing_summary(
        outputs,
        lambda_cost=float(config.get("routing", {}).get("lambda_cost", 0.0)),
    )
    routing_summary.to_csv(out_dir / "table_live_routing.csv", index=False)
    estimate_df = pd.DataFrame(
        [{"model_id": model_id, "estimated_cost_usd": cost} for model_id, cost in sorted(estimate.items())]
    )
    estimate_df.to_csv(out_dir / "frontier_cost_estimate.csv", index=False)
    write_live_stage0_report(
        out_dir,
        config,
        estimate_df,
        readiness,
        outputs,
        summary,
        routing_summary,
        allow_frontier_calls,
        run_suffix=run_suffix,
        examples_per_benchmark=n_per_benchmark,
        task_manifest_path=str(task_manifest_path) if task_manifest_path else "",
        task_datasets=task_datasets or [],
        task_count=len(tasks),
    )
    report_name = "LIVE_STAGE0_REPORT.md" if run_suffix == "live_stage0" else "LIVE_PILOT_REPORT.md"
    return {"output_dir": out_dir, "report": out_dir / report_name}


def write_live_stage0_report(
    output_dir: Path,
    config: dict[str, Any],
    estimate: pd.DataFrame,
    readiness: pd.DataFrame,
    outputs: pd.DataFrame,
    summary: pd.DataFrame,
    routing_summary: pd.DataFrame,
    allow_frontier_calls: bool,
    run_suffix: str = "live_stage0",
    examples_per_benchmark: int = 5,
    task_manifest_path: str = "",
    task_datasets: list[str] | None = None,
    task_count: int | None = None,
) -> None:
    estimate_lines = "\n".join(
        f"| {row.model_id} | {row.estimated_cost_usd:.4f} |" for row in estimate.itertuples(index=False)
    )
    readiness_lines = "\n".join(
        f"| {row.model_id} | {row.backend} | {row.status} | {row.fallback_mode} |"
        for row in readiness.itertuples(index=False)
    )
    summary_lines = "\n".join(
        f"| {row.model_id} | {row.status} | {int(row.n_calls)} | {row.mean_quality if pd.notna(row.mean_quality) else ''} | "
        f"{row.total_cost_usd:.4f} | {row.mean_latency_s:.3f} | "
        f"{row.mean_model_load_time_s:.3f} | {row.mean_warmup_time_s:.3f} |"
        for row in summary.itertuples(index=False)
    )
    routing_lines = ""
    if not routing_summary.empty:
        routing_lines = "\n".join(
        f"| {row.method} | {int(row.n_queries)} | {row.mean_quality:.4f} | {row.mean_utility:.4f} | "
        f"{row.quality_gap_to_oracle:.4f} | {row.utility_gap_to_oracle:.4f} | "
        f"{row.frontier_call_rate:.4f} | {row.probe_call_rate:.4f} | {row.remote_cost_total_usd:.4f} |"
        for row in routing_summary.head(8).itertuples(index=False)
    )
    frontier_pool = ", ".join(estimate["model_id"].astype(str).tolist()) if not estimate.empty else "none"
    title = "Live Stage 0 Controlled Smoke" if run_suffix == "live_stage0" else "Live Controlled Pilot"
    task_source = (
        f"Task manifest: `{task_manifest_path}`; datasets: `{', '.join(task_datasets or []) or 'all'}`; "
        f"task count: `{task_count}`."
        if task_manifest_path
        else f"Examples per benchmark: `{examples_per_benchmark}`."
    )
    report = f"""# {title}

Run id: `{config.get('run_id')}_{run_suffix}`
{task_source}

Fresh frontier calls requested in this invocation: `{allow_frontier_calls}`.
Frontier pool for this invocation: `{frontier_pool}`.
Claude/Anthropic models were not used.
Ready local vLLM endpoints are called and cached with zero remote-dollar cost; cached local rows are reused when the endpoint is offline.
Local readiness is recorded in `local_readiness.csv`; use `--local-model-ids` for one-model-at-a-time vLLM collection and `--frontier-model-ids` for provider-specific paid runs.
Configured launcher commands are listed in `configs/model_servers.yaml`.

## Pre-call Cost Estimate

Caps: total `${float(config.get('budget', {}).get('max_total_frontier_spend_usd', 0.0)):.2f}`, per model `${float(config.get('budget', {}).get('max_spend_per_frontier_model_usd', 0.0)):.2f}`.

| model | estimated cost usd |
| --- | ---: |
{estimate_lines}

## Local vLLM Readiness

| model | backend | status | fallback |
| --- | --- | --- | --- |
{readiness_lines}

## Model Results

| model | status | calls | mean quality | total cost usd | mean generation latency s | mean load time s | mean warmup time s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{summary_lines}

## Live Routing Summary

| method | queries | mean quality | mean utility | quality gap | utility gap | frontier rate | probe rate | remote cost usd |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{routing_lines}

Raw responses are cached under `{config.get('budget', {}).get('cache_dir')}`. Reruns reuse cache unless `--force-rerun` is set.
Rows in this report may be fresh responses or cached live responses; see `cache_hit` in `model_outputs.parquet`.
Generation latency excludes model loading and warmup for lazy local models. Load and warmup are reported separately in `cost_latency_summary.csv`.
Latency for fresh calls is measured end-to-end after the endpoint is ready. Cache hits use the latency saved in the raw cache when available; older cache entries created before latency persistence may show cache-read latency and should not be used for final latency claims.
"""
    report_name = "LIVE_STAGE0_REPORT.md" if run_suffix == "live_stage0" else "LIVE_PILOT_REPORT.md"
    (output_dir / report_name).write_text(report, encoding="utf-8")
    root_report = Path("results/controlled/RUN_REPORT.md")
    if root_report.exists():
        existing = root_report.read_text(encoding="utf-8")
        marker_title = f"\n## {title}\n"
        legacy_markers = [marker_title]
        if run_suffix == "live_stage0":
            legacy_markers.append("\n## Live Stage 0 Frontier Smoke\n")
        for marker in legacy_markers:
            if marker in existing:
                existing = existing.split(marker)[0].rstrip() + "\n"
                break
        root_report.write_text(existing + marker_title + report.split(f"# {title}", 1)[1], encoding="utf-8")
