from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from openai import OpenAI


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
DEFAULT_MODEL_SERVERS = Path("configs/model_servers.yaml")
DEFAULT_HISTORICAL_LIVE = Path("results/controlled/live_broad100_qwen32_thinking_mcq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the Phase 3 real local/frontier new-model calibration status."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-servers", type=Path, default=DEFAULT_MODEL_SERVERS)
    parser.add_argument("--historical-live-dir", type=Path, default=DEFAULT_HISTORICAL_LIVE)
    parser.add_argument("--run-live-smoke", action="store_true")
    parser.add_argument("--smoke-base-url", default="http://localhost:8001/v1")
    parser.add_argument("--smoke-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--smoke-model-id", default="qwen3-0.6b-probe-live-smoke")
    parser.add_argument("--smoke-limit", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "real_new_model_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_servers = load_yaml(args.model_servers)
    local_readiness = check_local_readiness(model_servers)
    provider_readiness = check_provider_readiness(model_servers)
    local_readiness.to_csv(out_dir / "live_readiness.csv", index=False)
    provider_readiness.to_csv(out_dir / "frontier_provider_readiness.csv", index=False)

    smoke_outputs = pd.DataFrame()
    if args.run_live_smoke:
        if is_endpoint_ready(args.smoke_base_url):
            smoke_outputs = run_live_smoke(args, config)
        else:
            smoke_outputs = pd.DataFrame(
                [
                    {
                        "model_id": args.smoke_model_id,
                        "model_name": args.smoke_model,
                        "status": "blocked_endpoint_unavailable",
                        "base_url": args.smoke_base_url,
                    }
                ]
            )
    smoke_outputs.to_csv(out_dir / "table_live_smoke_outputs.csv", index=False)

    historical = summarize_historical_live(args.historical_live_dir)
    summary = build_real_calibration_table(historical, smoke_outputs, provider_readiness)
    summary.to_csv(out_dir / "table_real_new_model_calibration.csv", index=False)
    cost_latency = build_cost_latency_summary(historical, smoke_outputs, provider_readiness)
    cost_latency.to_csv(out_dir / "cost_latency_summary.csv", index=False)
    write_memo(out_dir / "REAL_NEW_MODEL_CALIBRATION_MEMO.md", args, historical, smoke_outputs, summary, cost_latency)
    print(f"Wrote real new-model calibration status to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def check_local_readiness(model_servers: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in model_servers.get("local_models", []):
        base_url = str(item.get("base_url", ""))
        started = time.perf_counter()
        status = "missing_base_url"
        error_type = ""
        if base_url:
            try:
                payload = read_json_url(f"{base_url.rstrip('/')}/models", timeout=2.0)
                status = "ready" if payload.get("data") else "ready_empty_models"
            except Exception as exc:  # noqa: BLE001 - readiness should record exact failure class.
                status = "unavailable"
                error_type = type(exc).__name__
        rows.append(
            {
                "model_id": item.get("id", ""),
                "served_model_name": item.get("served_model_name", ""),
                "backend": item.get("backend", ""),
                "base_url": base_url,
                "status": status,
                "latency_s": time.perf_counter() - started,
                "error_type": error_type,
                "start_command": item.get("start_command", ""),
                "stop_command": item.get("stop_command", ""),
            }
        )
    return pd.DataFrame(rows)


def check_provider_readiness(model_servers: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in model_servers.get("frontier_models", []):
        env_names = [str(x) for x in item.get("env_key_names", [])]
        available = [name for name in env_names if os.environ.get(name)]
        rows.append(
            {
                "model_id": item.get("id", ""),
                "provider": item.get("provider", ""),
                "role": item.get("role", ""),
                "enabled": bool(item.get("enabled", True)),
                "env_key_names": ",".join(env_names),
                "api_key_available": bool(available),
                "status": "ready" if available else "blocked_no_api_key",
            }
        )
    return pd.DataFrame(rows)


def read_json_url(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def is_endpoint_ready(base_url: str) -> bool:
    try:
        payload = read_json_url(f"{base_url.rstrip('/')}/models", timeout=2.0)
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return False
    return bool(payload.get("data"))


def run_live_smoke(args: argparse.Namespace, config: dict[str, Any]) -> pd.DataFrame:
    source = pd.read_parquet(config["inputs"]["broad100_outputs"])
    queries = (
        source[source["split"].astype(str).eq("test") & source["metric"].astype(str).isin(["exact_final_answer", "multiple_choice"])]
        .drop_duplicates("query_id")
        .sort_values(["benchmark", "query_id"])
        .head(max(args.smoke_limit, 0))
    )
    client = OpenAI(api_key="routecode-local-vllm", base_url=args.smoke_base_url)
    rows = []
    for _, row in queries.iterrows():
        started = time.perf_counter()
        status = "success"
        error_type = ""
        raw_text = ""
        prompt_tokens = 0
        completion_tokens = 0
        try:
            response = client.chat.completions.create(
                model=args.smoke_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only the final answer. For multiple-choice questions, "
                            "return only one letter: A, B, C, or D."
                        ),
                    },
                    {"role": "user", "content": str(row["query_text"])},
                ],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            raw_text = response.choices[0].message.content or ""
            if response.usage is not None:
                prompt_tokens = int(response.usage.prompt_tokens or 0)
                completion_tokens = int(response.usage.completion_tokens or 0)
        except Exception as exc:  # noqa: BLE001 - live experiment table should capture failures.
            status = "error"
            error_type = type(exc).__name__
        latency_s = time.perf_counter() - started
        parsed = parse_answer(raw_text, str(row["metric"]))
        gold = str(row["gold_answer"])
        quality = score_answer(parsed, gold, str(row["metric"])) if status == "success" else float("nan")
        rows.append(
            {
                "run_id": "phase3_real_new_model_calibration_live_smoke",
                "query_id": row["query_id"],
                "split": row["split"],
                "benchmark": row["benchmark"],
                "domain": row.get("domain", ""),
                "metric": row["metric"],
                "model_id": args.smoke_model_id,
                "model_name": args.smoke_model,
                "provider": "local",
                "server_backend": "vllm",
                "base_url": args.smoke_base_url,
                "status": status,
                "error_type": error_type,
                "latency_s": latency_s,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "parsed_answer": parsed,
                "gold_answer": gold,
                "quality_score": quality,
                "cost_total_usd": 0.0,
                "raw_output": raw_text,
            }
        )
    return pd.DataFrame(rows)


def parse_answer(text: str, metric: str) -> str:
    cleaned = (text or "").strip()
    if metric == "multiple_choice":
        answer_matches = re.findall(r"(?i)answer\s*[:：]\s*([A-D])\b", cleaned)
        if answer_matches:
            return answer_matches[-1].upper()
        standalone = re.findall(r"\b([A-D])\b", cleaned.upper())
        if standalone:
            return standalone[-1].upper()
        return cleaned[:1].upper()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", cleaned)
    if boxed:
        return normalize_answer(boxed[-1])
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", cleaned.replace(",", ""))
    if numbers:
        return normalize_answer(numbers[-1])
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return normalize_answer(lines[-1] if lines else cleaned)


def normalize_answer(value: str) -> str:
    value = str(value).strip().lower()
    value = value.replace("$", "").replace(",", "")
    value = re.sub(r"^answer\s*[:：]\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .")


def score_answer(parsed: str, gold: str, metric: str) -> float:
    if metric == "multiple_choice":
        return float(parsed.strip().upper() == gold.strip().upper())
    return float(normalize_answer(parsed) == normalize_answer(gold))


def summarize_historical_live(root: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name in ["cost_latency_summary", "table_live_routing", "frontier_cost_estimate", "local_readiness"]:
        path = root / f"{name}.csv"
        tables[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return tables


def build_real_calibration_table(
    historical: dict[str, pd.DataFrame], smoke_outputs: pd.DataFrame, provider_readiness: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cost_latency = historical.get("cost_latency_summary", pd.DataFrame())
    if not cost_latency.empty:
        for row in cost_latency.to_dict("records"):
            rows.append(
                {
                    "experiment": "historical_live_broad100_qwen32_thinking_mcq",
                    "model_id": row.get("model_id", ""),
                    "provider": row.get("provider", ""),
                    "status": row.get("status", ""),
                    "n_calls": int(row.get("n_calls", 0)),
                    "mean_quality": row.get("mean_quality", float("nan")),
                    "mean_utility": row.get("mean_quality", float("nan")),
                    "total_cost_usd": row.get("total_cost_usd", float("nan")),
                    "mean_latency_s": row.get("mean_latency_s", float("nan")),
                    "evidence_type": "historical_live_or_frontier_cost_estimate",
                    "claim_supported": False,
                    "notes": "Existing local live Broad100 artifact; frontier rows were skipped/estimated, not called.",
                }
            )
    if not smoke_outputs.empty:
        success = smoke_outputs[smoke_outputs["status"].astype(str).eq("success")].copy()
        rows.append(
            {
                "experiment": "current_live_qwen06_smoke",
                "model_id": smoke_outputs["model_id"].iloc[0] if "model_id" in smoke_outputs else "qwen3-0.6b-probe-live-smoke",
                "provider": "local",
                "status": "success" if not success.empty else str(smoke_outputs["status"].iloc[0]),
                "n_calls": int(len(success)),
                "mean_quality": float(success["quality_score"].mean()) if not success.empty else float("nan"),
                "mean_utility": float(success["quality_score"].mean()) if not success.empty else float("nan"),
                "total_cost_usd": 0.0 if not success.empty else float("nan"),
                "mean_latency_s": float(success["latency_s"].mean()) if not success.empty else float("nan"),
                "evidence_type": "current_local_live_smoke",
                "claim_supported": False,
                "notes": "Tiny vLLM smoke verifies serving and logging only; it is not a full calibration result.",
            }
        )
    if not provider_readiness.empty:
        for row in provider_readiness.to_dict("records"):
            rows.append(
                {
                    "experiment": "frontier_live_call_readiness",
                    "model_id": row.get("model_id", ""),
                    "provider": row.get("provider", ""),
                    "status": row.get("status", ""),
                    "n_calls": 0,
                    "mean_quality": float("nan"),
                    "mean_utility": float("nan"),
                    "total_cost_usd": float("nan"),
                    "mean_latency_s": float("nan"),
                    "evidence_type": "environment_readiness",
                    "claim_supported": False,
                    "notes": "No closed-source calls were made without provider keys and explicit budget.",
                }
            )
    return pd.DataFrame(rows)


def build_cost_latency_summary(
    historical: dict[str, pd.DataFrame], smoke_outputs: pd.DataFrame, provider_readiness: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    historical_cost = historical.get("cost_latency_summary", pd.DataFrame())
    if not historical_cost.empty:
        for row in historical_cost.to_dict("records"):
            n_calls = float(row.get("n_calls", 0) or 0)
            rows.append(
                {
                    "source": "historical_live_broad100_qwen32_thinking_mcq",
                    "model_id": row.get("model_id", ""),
                    "provider": row.get("provider", ""),
                    "status": row.get("status", ""),
                    "n_calls": int(n_calls),
                    "total_cost_usd": row.get("total_cost_usd", float("nan")),
                    "cost_per_1k_calls_usd": float(row.get("total_cost_usd", 0.0)) / max(n_calls, 1.0) * 1000.0,
                    "mean_latency_s": row.get("mean_latency_s", float("nan")),
                    "p95_latency_s": row.get("p95_latency_s", float("nan")),
                }
            )
    if not smoke_outputs.empty and "status" in smoke_outputs:
        success = smoke_outputs[smoke_outputs["status"].astype(str).eq("success")]
        if not success.empty:
            rows.append(
                {
                    "source": "current_live_qwen06_smoke",
                    "model_id": success["model_id"].iloc[0],
                    "provider": "local",
                    "status": "success",
                    "n_calls": int(len(success)),
                    "total_cost_usd": 0.0,
                    "cost_per_1k_calls_usd": 0.0,
                    "mean_latency_s": float(success["latency_s"].mean()),
                    "p95_latency_s": float(success["latency_s"].quantile(0.95)),
                }
            )
    if not provider_readiness.empty:
        for row in provider_readiness.to_dict("records"):
            if row.get("status") != "ready":
                rows.append(
                    {
                        "source": "current_frontier_readiness",
                        "model_id": row.get("model_id", ""),
                        "provider": row.get("provider", ""),
                        "status": row.get("status", ""),
                        "n_calls": 0,
                        "total_cost_usd": float("nan"),
                        "cost_per_1k_calls_usd": float("nan"),
                        "mean_latency_s": float("nan"),
                        "p95_latency_s": float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    historical: dict[str, pd.DataFrame],
    smoke_outputs: pd.DataFrame,
    summary: pd.DataFrame,
    cost_latency: pd.DataFrame,
) -> None:
    lines = [
        "# Real Local/Frontier New-Model Calibration Status",
        "",
        "This artifact covers the optional live-call Phase 3 item. It is deliberately conservative: local vLLM calls can verify serving, latency, and zero API cost, but frontier calls are not made without provider keys, budget, caching, and pricing approval.",
        "",
        "## Commands",
        "",
        "```bash",
        "bash scripts/start_vllm_qwen3_0_6b.sh",
        (
            "python experiments/239_phase3_real_new_model_calibration.py "
            "--config configs/probecode_final_eval.yaml --run-live-smoke --smoke-limit "
            f"{args.smoke_limit}"
        ),
        "```",
        "",
        "## Current Live Smoke",
        "",
    ]
    if smoke_outputs.empty:
        lines.append("- Live smoke was not requested.")
    elif "status" in smoke_outputs and smoke_outputs["status"].astype(str).eq("success").any():
        success = smoke_outputs[smoke_outputs["status"].astype(str).eq("success")]
        lines.extend(
            [
                f"- Model: `{success['model_id'].iloc[0]}` served as `{args.smoke_model}`",
                f"- Calls: `{len(success)}`",
                f"- Mean quality on the tiny scored smoke: `{float(success['quality_score'].mean()):.4f}`",
                f"- Mean latency: `{float(success['latency_s'].mean()):.4f}` seconds",
                "- API cost: `$0.00`",
            ]
        )
    else:
        lines.append(f"- Live smoke did not run successfully: `{smoke_outputs['status'].iloc[0]}`")
    historical_cost = historical.get("cost_latency_summary", pd.DataFrame())
    if not historical_cost.empty:
        lines.extend(["", "## Existing Live Artifact", ""])
        for row in historical_cost.to_dict("records"):
            quality = row.get("mean_quality", float("nan"))
            lines.append(
                f"- `{row.get('model_id')}`: status `{row.get('status')}`, calls `{int(row.get('n_calls', 0))}`, "
                f"mean quality `{float(quality):.4f}` if scored, total cost `${float(row.get('total_cost_usd', 0.0)):.4f}`, "
                f"mean latency `{float(row.get('mean_latency_s', float('nan'))):.4f}` seconds"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Local vLLM calls are working in the `ml-gpu` environment.",
            "- The existing Qwen32 thinking live run is useful as a serving/cost/latency artifact, but it is not evidence that the live local model is strong; the capped thinking prompt produced many malformed answers.",
            "- GPT/Gemini live calibration is still not run in this package because provider keys are not present in the environment.",
            "- Therefore the real local/frontier onboarding claim remains incomplete. The cached/simulated onboarding tables are complete, but real frontier deployment evidence still needs approved calls.",
            "",
            "## Output Tables",
            "",
            "- `table_real_new_model_calibration.csv`",
            "- `cost_latency_summary.csv`",
            "- `live_readiness.csv`",
            "- `frontier_provider_readiness.csv`",
            "- `table_live_smoke_outputs.csv`",
            "",
        ]
    )
    if not summary.empty:
        lines.extend(["## Summary Table", "", markdown_table(summary), ""])
    if not cost_latency.empty:
        lines.extend(["## Cost And Latency", "", markdown_table(cost_latency), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    display = df.copy()
    for column in display.columns:
        display[column] = display[column].map(format_cell)
    headers = list(display.columns)
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
