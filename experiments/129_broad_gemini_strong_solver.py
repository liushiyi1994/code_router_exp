from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_gemini_text,
    load_env_values,
    normalize_answer,
    resolve_key,
    score_output,
)


GEMINI_MODEL = "gemini-3.5-flash"
STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
INPUT_PER_MTOK = 1.50
OUTPUT_PER_MTOK = 9.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a broad100 Gemini strong-solver probe.")
    parser.add_argument("--outputs", type=Path, default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_gemini_strong_solver"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-api-spend-usd", type=float, default=6.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_broad_package()
    outputs = package.load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    queries = outputs[outputs["split"].isin(splits)].drop_duplicates("query_id").copy()
    api_key = resolve_key(load_env_values(args.env_file), ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise RuntimeError("Missing Gemini API key.")
    strong = collect_rows(
        queries,
        args.output_dir,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
        temperature=args.temperature,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    strong.to_csv(args.output_dir / "table_broad_gemini_strong_outputs.csv", index=False)
    augmented = append_strong_rows(outputs, strong, lambda_cost=args.lambda_cost)
    augmented.to_parquet(args.output_dir / "model_outputs_with_gemini_strong.parquet", index=False)
    eval_table = evaluate_policies(augmented, package=package, lambda_cost=args.lambda_cost)
    eval_table.to_csv(args.output_dir / "table_broad_gemini_strong_eval.csv", index=False)
    write_memo(args.output_dir, args.outputs, strong, eval_table, args)
    print(f"Wrote Gemini strong broad100 results to {args.output_dir}")


def load_broad_package():
    path = Path("experiments/125_phase3_broad_target_method_package.py")
    spec = importlib.util.spec_from_file_location("broad_package", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prompt_for(row: pd.Series) -> str:
    query = str(row["query_text"]).strip()
    metric = str(row.get("metric", ""))
    if metric == "multiple_choice":
        instruction = "Return only the final option letter, with no explanation."
    elif metric == "pass_at_1":
        instruction = "Return only the Python code solution. Do not wrap it in Markdown."
    else:
        instruction = "Return only the final exact answer, with no explanation, no Markdown, and no surrounding text."
    return (
        "Solve the task carefully. You may reason internally, but the visible response must contain only the requested final output.\n"
        f"{instruction}\n\n"
        f"Task:\n{query}"
    )


def cache_name(query_id: str, thinking_budget: int, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{thinking_budget}:{max_output_tokens}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def call_gemini(
    *,
    prompt: str,
    api_key: str,
    max_output_tokens: int,
    thinking_budget: int,
    temperature: float,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "maxOutputTokens": int(max_output_tokens),
        "temperature": float(temperature),
    }
    if int(thinking_budget) >= 0:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt < 4:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Gemini request failed.")


def gemini_cost(payload: dict[str, Any]) -> float:
    usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    candidate_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    thoughts_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)
    return prompt_tokens * (INPUT_PER_MTOK / 1_000_000) + (candidate_tokens + thoughts_tokens) * (
        OUTPUT_PER_MTOK / 1_000_000
    )


def estimate_missing_cost(prompts: list[str], max_output_tokens: int, thinking_budget: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * (int(max_output_tokens) + max(0, int(thinking_budget)))
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def collect_rows(
    queries: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    thinking_budget: int,
    temperature: float,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_gemini_strong_solver" / GEMINI_MODEL / f"think_{thinking_budget}_max_{max_output_tokens}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = [(row, prompt_for(row)) for _, row in queries.iterrows()]
    missing = [
        prompt
        for row, prompt in prompt_rows
        if load_json_or_none(cache_dir / cache_name(str(row["query_id"]), thinking_budget, max_output_tokens)) is None
    ]
    estimated = estimate_missing_cost(missing, max_output_tokens, thinking_budget)
    if estimated > float(max_api_spend_usd) + 1e-12:
        raise RuntimeError(
            f"Estimated uncached Gemini strong-solver spend ${estimated:.4f} exceeds cap ${float(max_api_spend_usd):.4f}."
        )
    print(f"Estimated uncached Gemini strong-solver spend: ${estimated:.4f}")

    def one(row: pd.Series, prompt: str) -> dict[str, Any]:
        query_id = str(row["query_id"])
        raw_path = cache_dir / cache_name(query_id, thinking_budget, max_output_tokens)
        payload = load_json_or_none(raw_path)
        cache_hit = payload is not None
        started = time.time()
        status = "success"
        error_type = ""
        if not cache_hit:
            try:
                payload = call_gemini(
                    prompt=prompt,
                    api_key=api_key,
                    max_output_tokens=max_output_tokens,
                    thinking_budget=thinking_budget,
                    temperature=temperature,
                )
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:1000], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - started
            write_json_atomic(raw_path, payload)
        text = extract_gemini_text(payload) if payload.get("_status", status) == "success" else ""
        metric = str(row.get("metric", "exact_final_answer"))
        score_input = normalize_answer(text) if metric != "pass_at_1" else text
        parsed, quality = score_output(score_input, str(row["gold_answer"]), metric)
        if payload.get("_status", status) != "success":
            quality = np.nan
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        return {
            "query_id": query_id,
            "split": str(row["split"]),
            "benchmark": str(row["benchmark"]),
            "domain": str(row["domain"]),
            "metric": metric,
            "query_text": str(row["query_text"]),
            "gold_answer": str(row["gold_answer"]),
            "status": str(payload.get("_status", status)),
            "error_type": str(payload.get("_error_type", error_type)),
            "raw_text": text,
            "parsed_answer": parsed,
            "quality_score": float(quality) if not pd.isna(quality) else np.nan,
            "input_tokens": int(usage.get("promptTokenCount", max(1, len(prompt) // 4)) or 0),
            "output_tokens": int((usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0)),
            "candidate_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "thoughts_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
            "cost_total_usd": gemini_cost(payload),
            "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
            "cache_hit": bool(cache_hit),
            "raw_output_path": str(raw_path),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row, prompt) for row, prompt in prompt_rows]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"Gemini strong rows {index}/{len(futures)}")
    return pd.DataFrame(rows)


def load_json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_strong_rows(outputs: pd.DataFrame, strong: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    template_cols = list(outputs.columns)
    rows: list[dict[str, Any]] = []
    run_id = "controlled_surrogate_pilot_v1_broad_gemini_strong"
    for row in strong.itertuples(index=False):
        cost = float(row.cost_total_usd)
        rows.append(
            {
                "run_id": run_id,
                "query_id": row.query_id,
                "query_text": row.query_text,
                "benchmark": row.benchmark,
                "domain": row.domain,
                "split": row.split,
                "model_id": STRONG_MODEL_ID,
                "provider": "google",
                "is_local": False,
                "is_frontier": True,
                "is_probe": False,
                "prompt_template_version": "gemini_strong_think_v1",
                "prompt_hash": "",
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
                "max_output_tokens": 0,
                "start_time_unix": 0.0,
                "end_time_unix": 0.0,
                "latency_s": float(row.latency_s),
                "model_load_time_s": 0.0,
                "warmup_time_s": 0.0,
                "latency_excludes_load_warmup": True,
                "load_mode": "api",
                "status": row.status,
                "error_type": row.error_type,
                "raw_output_path": row.raw_output_path,
                "parsed_answer": row.parsed_answer,
                "gold_answer": row.gold_answer,
                "quality_score": float(row.quality_score) if not pd.isna(row.quality_score) else np.nan,
                "cost_input_usd": 0.0,
                "cost_output_usd": cost,
                "cost_total_usd": cost,
                "cache_hit": bool(row.cache_hit),
                "server_backend": "api",
                "server_config_json": json.dumps({"model": GEMINI_MODEL, "strong_solver": True}, sort_keys=True),
                "hardware_id": "remote_api",
                "metric": row.metric,
                "rank_in_benchmark": 0,
                "normalized_remote_cost": 0.0,
                "utility": 0.0,
                "tool_available": False,
            }
        )
    strong_rows = pd.DataFrame(rows)
    for column in template_cols:
        if column not in strong_rows.columns:
            strong_rows[column] = np.nan
    appended = pd.concat([outputs[template_cols], strong_rows[template_cols]], ignore_index=True)
    gpt_norm = max(
        float(appended[appended["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )
    appended["normalized_remote_cost"] = appended["cost_total_usd"].astype(float) / gpt_norm
    appended["quality_score"] = pd.to_numeric(appended["quality_score"], errors="coerce").fillna(0.0)
    appended["utility"] = appended["quality_score"].astype(float) - float(lambda_cost) * appended["normalized_remote_cost"].astype(float)
    return appended


def evaluate_policies(outputs: pd.DataFrame, *, package, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        split_outputs = outputs[outputs["split"].eq(split)]
        cost_oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()]
        quality_oracle = split_outputs.loc[split_outputs.groupby("query_id")["quality_score"].idxmax()]
        for method, selected in policy_grid(outputs, split=split, package=package).items():
            selected_rows = package.selected_to_rows(outputs, selected, split=split)
            if selected_rows.empty:
                continue
            rows.append(package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost))
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def policy_grid(outputs: pd.DataFrame, *, split: str, package) -> dict[str, pd.Series]:
    split_queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    policies: dict[str, pd.Series] = {
        "all_gemini_strong": pd.Series(STRONG_MODEL_ID, index=split_queries.index),
        "observable_local_state_v5": package.observable_local_state_selection(outputs, split=split),
        "tool_probe_profile_v4": package.profile_v4_selection_for_split(outputs, split=split),
    }
    if split == "test":
        selected_benchmarks = validation_selected_strong_benchmarks(outputs, package=package)
        base = package.observable_local_state_selection(outputs, split=split)
        for query_id, row in split_queries.iterrows():
            if str(row["benchmark"]) in selected_benchmarks:
                base.loc[query_id] = STRONG_MODEL_ID
        policies["val_benchmark_strong_else_observable"] = base
    return policies


def validation_selected_strong_benchmarks(outputs: pd.DataFrame, *, package) -> set[str]:
    val_queries = outputs[outputs["split"].eq("val")].drop_duplicates("query_id").set_index("query_id")
    base = package.observable_local_state_selection(outputs, split="val")
    selected: set[str] = set()
    for benchmark, frame in val_queries.groupby("benchmark"):
        benchmark_ids = set(frame.index.astype(str))
        strong = pd.Series(STRONG_MODEL_ID, index=frame.index)
        base_bench = base[base.index.astype(str).isin(benchmark_ids)]
        strong_rows = package.selected_to_rows(outputs, strong, split="val")
        base_rows = package.selected_to_rows(outputs, base_bench, split="val")
        if strong_rows.empty or base_rows.empty:
            continue
        if float(strong_rows["utility"].mean()) >= float(base_rows["utility"].mean()):
            selected.add(str(benchmark))
    return selected


def write_memo(output_dir: Path, outputs_path: Path, strong: pd.DataFrame, eval_table: pd.DataFrame, args: argparse.Namespace) -> None:
    test = eval_table[eval_table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)
    by_benchmark = strong.groupby(["split", "benchmark"])["quality_score"].mean().reset_index()
    lines = [
        "# Broad100 Gemini Strong Solver",
        "",
        f"Source outputs: `{outputs_path}`",
        f"Model: `{GEMINI_MODEL}` as `{STRONG_MODEL_ID}`.",
        f"Thinking budget: `{args.thinking_budget}`. Max output tokens: `{args.max_output_tokens}`.",
        "Claude is not used.",
        f"Rows: `{len(strong)}`. Successful rows: `{int(strong['status'].eq('success').sum())}`.",
        f"Total Gemini strong cost: `${float(strong['cost_total_usd'].sum()):.4f}`.",
        "",
        "## Quality By Benchmark",
        "",
        "```csv",
        by_benchmark.to_csv(index=False).strip(),
        "```",
        "",
        "## Evaluation",
        "",
        "```csv",
        test.to_csv(index=False).strip(),
        "```",
        "",
    ]
    output_dir.joinpath("BROAD_GEMINI_STRONG_SOLVER_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
