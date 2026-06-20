from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    usage_from_openai,
)

ROUTE_MODEL = "gpt-5.5"
LOCAL_CANDIDATES = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
]
FRONTIER_CANDIDATES = ["gemini-3.5-flash", "gpt-5.5"]
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT-5.5 as a broad100 route judge over local probes.")
    parser.add_argument("--outputs", type=Path, default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_gpt_route_judge"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--max-api-spend-usd", type=float, default=4.00)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_broad_package()
    outputs = package.load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    queries = outputs[outputs["split"].isin(splits)].drop_duplicates("query_id").copy()
    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")

    routes = collect_routes(
        outputs,
        queries,
        args.output_dir,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
        package=package,
    )
    grid = evaluate_grid(outputs, routes, lambda_cost=args.lambda_cost, package=package)
    selected = select_val_threshold(grid)
    routes.to_csv(args.output_dir / "table_broad_gpt_route_judge_routes.csv", index=False)
    grid.to_csv(args.output_dir / "table_broad_gpt_route_judge_gate.csv", index=False)
    selected.to_csv(args.output_dir / "table_broad_gpt_route_judge_selected.csv", index=False)
    write_memo(args.output_dir / "BROAD_GPT_ROUTE_JUDGE_MEMO.md", args.outputs, routes, grid, selected)
    print(f"Wrote broad GPT route judge to {args.output_dir}")


def load_broad_package():
    path = Path("experiments/125_phase3_broad_target_method_package.py")
    spec = importlib.util.spec_from_file_location("broad_package", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_routes(
    outputs: pd.DataFrame,
    queries: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
    package,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_route_judge"
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_query = outputs.set_index(["query_id", "model_id"])
    prompts = [(row, prompt_for(row, by_query, package=package)) for _, row in queries.iterrows()]
    missing_prompts = [prompt for row, prompt in prompts if not (cache_dir / cache_name(str(row["query_id"]))).exists()]
    estimated = estimate_prompt_cost(missing_prompts, max_output_tokens)
    if estimated > float(max_api_spend_usd) + 1e-12:
        raise RuntimeError(
            f"Estimated uncached GPT route-judge spend ${estimated:.4f} exceeds cap ${float(max_api_spend_usd):.4f}."
        )
    print(f"Estimated uncached GPT route-judge spend: ${estimated:.4f}")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [
            executor.submit(call_one_route, row, prompt, cache_dir, api_key, max_output_tokens)
            for row, prompt in prompts
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"route judge rows {index}/{len(futures)}")
    return pd.DataFrame(rows)


def call_one_route(
    row: pd.Series,
    prompt: str,
    cache_dir: Path,
    api_key: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    query_id = str(row["query_id"])
    raw_path = cache_dir / cache_name(query_id)
    cache_hit = raw_path.exists()
    start = time.time()
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        request_payload: dict[str, Any] = {
            "model": ROUTE_MODEL,
            "input": prompt,
            "max_output_tokens": int(max_output_tokens),
            "reasoning": {"effort": "none"},
            "text": {"verbosity": "low"},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 90.0)
        except urllib.error.HTTPError:
            request_payload.pop("text", None)
            payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 90.0)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = extract_openai_text(payload)
    parsed = parse_route_text(text)
    input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
    return {
        "query_id": query_id,
        "split": str(row["split"]),
        "benchmark": str(row["benchmark"]),
        "route_action": parsed["action"],
        "route_confidence": float(parsed["confidence"]),
        "route_reason": parsed["reason"],
        "route_text": text,
        "route_input_tokens": int(input_tokens),
        "route_output_tokens": int(output_tokens),
        "route_cost": route_call_cost(int(input_tokens), int(output_tokens)),
        "route_latency_s": float(time.time() - start),
        "route_cache_hit": cache_hit,
        "route_raw_path": str(raw_path),
    }


def prompt_for(row: pd.Series, by_query: pd.DataFrame, *, package) -> str:
    query_id = str(row["query_id"])
    local_lines = []
    for model_id in LOCAL_CANDIDATES:
        try:
            model_row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        if model_id == "deterministic_math_tool" and not package.deterministic_tool_choice(by_query, query_id):
            continue
        answer = compact(str(model_row.get("parsed_answer", "")), 160)
        status = str(model_row.get("status", ""))
        local_lines.append(f"- {model_id}: status={status}; parsed_answer={answer if answer else '[empty]'}")
    if not local_lines:
        local_lines.append("- no local candidate answer")
    query = compact(str(row["query_text"]), 1700)
    return (
        "You are a deployment route judge for an LLM model router.\n"
        "You see the user query and cached zero-dollar local probe answers. You do not see the gold answer.\n"
        "Choose the model/action most likely to maximize exact-scored correctness while considering remote model cost.\n"
        "Use local models when a local candidate looks credible. Use gemini-3.5-flash for cheap broad reasoning when likely enough. "
        "Use gpt-5.5 for harder math, code, or expert questions when local/Gemini look unreliable.\n"
        "Return JSON only with keys action, confidence, reason.\n"
        "action must be one of: "
        + ", ".join([*LOCAL_CANDIDATES, *FRONTIER_CANDIDATES])
        + ". confidence is 0 to 1.\n\n"
        f"Benchmark: {row['benchmark']}\n"
        f"Domain: {row['domain']}\n"
        f"Metric: {row['metric']}\n"
        f"Query:\n{query}\n\n"
        "Local probe answers:\n"
        + "\n".join(local_lines)
        + '\n\nExample: {"action":"qwen3-14b-awq-local","confidence":0.78,"reason":"short"}\n/no_think'
    )


def parse_route_text(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    action = normalize_action(parsed.get("action", ""))
    if action == "PARSE_FAIL":
        loose = re.search(
            r"\b(deterministic_math_tool|qwen3-4b-local|qwen3-8b-local|qwen3-14b-awq-local|qwen3-32b-awq-local|gemini-3.5-flash|gpt-5.5)\b",
            clean,
            flags=re.I,
        )
        action = normalize_action(loose.group(1) if loose else "")
    try:
        confidence = float(parsed.get("confidence", np.nan))
    except (TypeError, ValueError):
        confidence = np.nan
    if np.isnan(confidence):
        confidence = 0.0
    return {
        "action": action,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason": str(parsed.get("reason", ""))[:300],
    }


def normalize_action(value: object) -> str:
    action = str(value).strip()
    mapping = {model.lower(): model for model in [*LOCAL_CANDIDATES, *FRONTIER_CANDIDATES]}
    return mapping.get(action.lower(), "PARSE_FAIL")


def evaluate_grid(outputs: pd.DataFrame, routes: pd.DataFrame, *, lambda_cost: float, package) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in [0.0, 0.4, 0.55, 0.7, 0.85, 0.95]:
        for fallback in ["observable_local_state_v5", "tool_probe_profile_v4", "gpt-5.5", "gemini-3.5-flash"]:
            for split in ["val", "test"]:
                split_routes = routes[routes["split"].eq(split)].copy()
                if split_routes.empty:
                    continue
                selected = route_policy(outputs, split_routes, threshold=threshold, fallback=fallback, package=package)
                selected_rows = package.selected_to_rows(outputs, selected, split=split)
                if selected_rows.empty:
                    continue
                split_outputs = outputs[outputs["split"].eq(split)]
                cost_oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()]
                quality_oracle = split_outputs.loc[split_outputs.groupby("query_id")["quality_score"].idxmax()]
                row = package.evaluation_row(
                    f"gpt_route_judge_t{threshold:g}_fb_{fallback}",
                    selected_rows,
                    cost_oracle,
                    quality_oracle,
                    lambda_cost=lambda_cost,
                )
                route_cost = split_routes.set_index("query_id").loc[selected.index, "route_cost"].fillna(0.0)
                gpt_cost_norm = max(
                    float(outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
                    1e-12,
                )
                route_cost_norm_mean = float((route_cost / gpt_cost_norm).mean())
                row["selector_confidence_threshold"] = threshold
                row["fallback_policy"] = fallback
                row["route_cost_total_usd"] = float(route_cost.sum())
                row["route_cost_norm_mean"] = route_cost_norm_mean
                row["mean_utility_with_route_cost"] = float(row["mean_utility"] - lambda_cost * route_cost_norm_mean)
                row["oracle_utility_ratio_with_route_cost"] = (
                    float(row["mean_utility_with_route_cost"] / row["cost_oracle_mean_utility"])
                    if abs(float(row["cost_oracle_mean_utility"])) > 1e-12
                    else np.nan
                )
                row["route_latency_mean_s"] = float(split_routes["route_latency_s"].mean())
                row["route_latency_p95_s"] = float(split_routes["route_latency_s"].quantile(0.95))
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def route_policy(outputs: pd.DataFrame, routes: pd.DataFrame, *, threshold: float, fallback: str, package) -> pd.Series:
    if fallback == "observable_local_state_v5":
        fallback_selection = package.observable_local_state_selection(outputs, split=str(routes["split"].iloc[0]))
    elif fallback == "tool_probe_profile_v4":
        fallback_selection = package.profile_v4_selection_for_split(outputs, split=str(routes["split"].iloc[0]))
    else:
        fallback_selection = pd.Series(fallback, index=routes["query_id"].astype(str))
    selected: dict[str, str] = {}
    by_query = outputs.set_index(["query_id", "model_id"])
    for _, row in routes.iterrows():
        query_id = str(row["query_id"])
        action = str(row["route_action"])
        confidence = float(row["route_confidence"])
        if action == "PARSE_FAIL" or confidence < threshold or not action_available(by_query, query_id, action, package):
            action = str(fallback_selection.get(query_id, "qwen3-14b-awq-local"))
        selected[query_id] = action
    return pd.Series(selected)


def action_available(by_query: pd.DataFrame, query_id: str, action: str, package) -> bool:
    if action == "deterministic_math_tool":
        return bool(package.deterministic_tool_choice(by_query, query_id))
    try:
        row = by_query.loc[(query_id, action)]
    except KeyError:
        return False
    return str(row.get("status", "success")) == "success"


def select_val_threshold(grid: pd.DataFrame) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    if val.empty:
        return pd.DataFrame()
    picked = val.sort_values(["mean_utility", "mean_quality", "frontier_call_rate"], ascending=[False, False, True]).head(1)
    threshold = float(picked.iloc[0]["selector_confidence_threshold"])
    fallback = str(picked.iloc[0]["fallback_policy"])
    return grid[grid["selector_confidence_threshold"].eq(threshold) & grid["fallback_policy"].eq(fallback)].copy()


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return route_call_cost(input_tokens, output_tokens)


def route_call_cost(input_tokens: int, output_tokens: int) -> float:
    return (float(input_tokens) / 1_000_000.0) * INPUT_PER_MTOK + (
        float(output_tokens) / 1_000_000.0
    ) * OUTPUT_PER_MTOK


def cache_name(query_id: str) -> str:
    digest = hashlib.sha1(query_id.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_memo(path: Path, outputs_path: Path, routes: pd.DataFrame, grid: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Broad100 GPT Route Judge",
        "",
        f"Source outputs: `{outputs_path}`",
        "",
        "The route judge sees query text and cached local probe answers. It does not see gold answers.",
        "It uses GPT-5.5 only; Claude is not used.",
        "",
        f"Route rows: `{len(routes)}`",
        f"Route cost total: `${float(routes['route_cost'].sum()):.4f}`",
        "",
    ]
    if not selected.empty:
        lines.extend(["## Validation-Selected", "", "```csv", selected.to_csv(index=False).strip(), "```", ""])
    test = grid[grid["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)
    if not test.empty:
        lines.extend(["## Top Test Rows", "", "```csv", test.to_csv(index=False).strip(), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
