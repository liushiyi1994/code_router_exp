from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import extract_openai_text, load_env_values, post_json, resolve_key, usage_from_openai


ROUTE_MODEL = "gpt-5.5"
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00
LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
GEMINI_STRONG = "gemini-3.5-flash-strong-solve"
TOOL = "deterministic_math_tool"
ROUTE_ACTIONS = {
    "USE_QWEN06": "qwen3-0.6b-probe",
    "USE_QWEN4": "qwen3-4b-local",
    "USE_QWEN8": "qwen3-8b-local",
    "USE_QWEN14": "qwen3-14b-awq-local",
    "USE_GEMINI": GEMINI,
    "USE_GEMINI_STRONG": GEMINI_STRONG,
    "USE_BASE_GPT": BASE_GPT,
    "USE_STRONG_GPT": STRONG_GPT,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPT-5.5 route judge over all cached exact-math candidate answers.")
    parser.add_argument("--output-dir", default="results/controlled/exact_math_gpt_route_judge")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-output-tokens", type=int, default=180)
    parser.add_argument("--max-api-spend-usd", type=float, default=4.0)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    return parser.parse_args()


def load_tool_module():
    spec = importlib.util.spec_from_file_location("tool_policy", "experiments/112_tool_augmented_aime_policy.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def truncate(text: object, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def prompt_for(row: pd.Series) -> str:
    return f"""You are a strict route judge for exact-answer math. You do not see the gold answer.

Choose the cheapest action that is likely to be exactly correct. Use strong GPT only when the visible candidates look unreliable and the problem likely needs stronger reasoning.

Actions:
- USE_QWEN06: use Qwen0.6 local answer.
- USE_QWEN4: use Qwen4 local answer.
- USE_QWEN8: use Qwen8 local answer.
- USE_QWEN14: use Qwen14 local answer.
- USE_GEMINI: use cheap Gemini Flash answer.
- USE_GEMINI_STRONG: use thinking-enabled Gemini Flash answer.
- USE_BASE_GPT: use base GPT-5.5 answer.
- USE_STRONG_GPT: use stronger GPT-5.5 reasoning answer.

Rules:
- If multiple cheap candidates agree on a plausible final answer, choose the cheapest reliable agreeing action.
- If the problem is AIME/olympiad-style and candidate answers conflict, prefer USE_STRONG_GPT unless one candidate has a clearly checkable answer.
- If all paid candidates look wrong or format-invalid, choose the best local action instead of wasting cost.
- Return JSON only:
{{"action":"USE_QWEN06|USE_QWEN4|USE_QWEN8|USE_QWEN14|USE_GEMINI|USE_GEMINI_STRONG|USE_BASE_GPT|USE_STRONG_GPT","confidence":0.0,"reason":"short"}}

Dataset: {row['dataset']}
Problem:
{truncate(row['query_text'], 1800)}

Candidate final answers:
- Qwen0.6: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 160)}
- Qwen4: {truncate(row.get('qwen3-4b-local_answer', ''), 160)}
- Qwen8: {truncate(row.get('qwen3-8b-local_answer', ''), 160)}
- Qwen14: {truncate(row.get('qwen3-14b-awq-local_answer', ''), 160)}
- Gemini cheap: {truncate(row.get(f'{GEMINI}_answer', ''), 180)}
- Gemini thinking: {truncate(row.get('gemini_strong_answer', ''), 180)}
- Base GPT: {truncate(row.get(f'{BASE_GPT}_answer', ''), 180)}

Cheap metadata:
query_len={row.get('query_len', '')}; number_count={row.get('number_count', '')}; latex_count={row.get('latex_count', '')}; local_votes={row.get('local_ensemble_votes', '')}
/no_think"""


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return route_call_cost(input_tokens, output_tokens)


def route_call_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def cache_name(query_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id) + ".json"


def parse_route_text(text: object) -> tuple[str, float, str]:
    raw = str(text or "")
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            action = str(payload.get("action", "")).strip().upper()
            confidence = float(payload.get("confidence", 0.0) or 0.0)
            reason = str(payload.get("reason", ""))
            if action in ROUTE_ACTIONS:
                return action, float(np.clip(confidence, 0.0, 1.0)), reason
        except Exception:
            pass
    upper = raw.upper()
    for action in ROUTE_ACTIONS:
        if action in upper:
            return action, 0.5, raw[:120]
    return "USE_GEMINI_STRONG", 0.0, "parse_fallback"


def call_one_route(
    idx: int,
    row: pd.Series,
    prompt: str,
    cache_dir: Path,
    api_key: str,
    max_output_tokens: int,
) -> dict[str, object]:
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
            payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 120.0)
        except urllib.error.HTTPError:
            request_payload.pop("text", None)
            payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 120.0)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = extract_openai_text(payload)
    action, confidence, reason = parse_route_text(text)
    input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
    return {
        "row_index": idx,
        "query_id": query_id,
        "dataset": str(row["dataset"]),
        "split": str(row["split"]),
        "route_action": action,
        "route_confidence": confidence,
        "route_reason": reason,
        "route_text": text,
        "route_input_tokens": int(input_tokens),
        "route_output_tokens": int(output_tokens),
        "route_cost": route_call_cost(int(input_tokens), int(output_tokens)),
        "route_latency_s": float(time.time() - start),
        "route_cache_hit": cache_hit,
        "route_raw_path": str(raw_path),
    }


def collect_routes(frame: pd.DataFrame, output_dir: Path, args: argparse.Namespace, api_key: str) -> pd.DataFrame:
    cache_dir = output_dir / "raw_exact_math_route_judge"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = [(idx, row, prompt_for(row)) for idx, row in frame.iterrows()]
    missing = [prompt for _, row, prompt in prompt_rows if not (cache_dir / cache_name(str(row["query_id"]))).exists()]
    estimated = estimate_prompt_cost(missing, args.max_output_tokens)
    if estimated > args.max_api_spend_usd:
        raise RuntimeError(f"Estimated uncached spend ${estimated:.4f} exceeds cap ${args.max_api_spend_usd:.4f}.")
    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = [
            executor.submit(call_one_route, idx, row, prompt, cache_dir, api_key, args.max_output_tokens)
            for idx, row, prompt in prompt_rows
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def map_route_action(route_action: str, row: pd.Series, strong_cost_cap: float, fallback: str) -> str:
    action = ROUTE_ACTIONS.get(route_action, fallback)
    if action == STRONG_GPT and float(row["strong_cost"]) > strong_cost_cap:
        return fallback
    return action


def evaluate(table: pd.DataFrame, routes: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_by_idx = routes.set_index("row_index")
    rows = []
    for threshold in [0.0, 0.4, 0.6, 0.72, 0.8, 0.9, 0.95]:
        for strong_cost_cap in [0.025, 0.035, 0.05, 0.065, 0.08, 0.1]:
            for fallback in ["qwen3-14b-awq-local", GEMINI_STRONG, BASE_GPT, GEMINI, "qwen3-8b-local"]:
                for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
                    actions: list[str] = []
                    qualities: list[float] = []
                    solver_costs: list[float] = []
                    total_costs: list[float] = []
                    route_calls = 0
                    frontier_calls = 0
                    for idx, row in frame.iterrows():
                        route_cost = 0.0
                        if bool(row["tool_available"]):
                            action = TOOL
                        else:
                            route = route_by_idx.loc[idx]
                            route_cost = float(route.get("route_cost", 0.0) or 0.0)
                            route_calls += 1
                            if float(route.get("route_confidence", 0.0) or 0.0) >= threshold:
                                action = map_route_action(str(route["route_action"]), row, strong_cost_cap, fallback)
                            else:
                                action = fallback
                        quality, solver_cost = row_quality_cost(row, action)
                        if solver_cost > 0:
                            frontier_calls += 1
                        actions.append(action)
                        qualities.append(quality)
                        solver_costs.append(solver_cost)
                        total_costs.append(solver_cost + route_cost)
                    rows.append(
                        split_metrics(
                            frame,
                            actions,
                            qualities,
                            solver_costs,
                            total_costs,
                            route_calls,
                            frontier_calls,
                            threshold,
                            strong_cost_cap,
                            fallback,
                            args.lambda_cost,
                        )
                    )
    grid = pd.DataFrame(rows)
    selected = select_rows(grid, args)
    return grid, selected


def row_quality_cost(row: pd.Series, action: str) -> tuple[float, float]:
    if action == TOOL:
        return float(row["tool_quality"]), 0.0
    if action == GEMINI_STRONG:
        return float(row["gemini_strong_quality"]), float(row["gemini_strong_cost"])
    if action == STRONG_GPT:
        return float(row["strong_quality"]), float(row["strong_cost"])
    if action in LOCAL_MODELS:
        return float(row[f"{action}_quality"]), 0.0
    if action == GEMINI:
        return float(row[f"{GEMINI}_quality"]), float(row[f"{GEMINI}_cost"])
    if action == BASE_GPT:
        return float(row[f"{BASE_GPT}_quality"]), float(row[f"{BASE_GPT}_cost"])
    raise ValueError(action)


def split_metrics(
    frame: pd.DataFrame,
    actions: list[str],
    qualities: list[float],
    solver_costs: list[float],
    total_costs: list[float],
    route_calls: int,
    frontier_calls: int,
    threshold: float,
    strong_cost_cap: float,
    fallback: str,
    lambda_cost: float,
) -> dict[str, object]:
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    normalized_cost = float(np.sum(total_costs) / strong_norm)
    mean_utility = mean_quality - float(lambda_cost) * normalized_cost
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    action_counts = pd.Series(actions).value_counts().to_dict()
    return {
        "method": f"exact_gpt_route_t{threshold:g}_cap{strong_cost_cap:g}_fallback{fallback}",
        "split": str(frame["split"].iloc[0]),
        "threshold": threshold,
        "strong_cost_cap": strong_cost_cap,
        "fallback": fallback,
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_strong_inclusive_oracle": float(oracle_quality - mean_quality),
        "normalized_cost_vs_all_strong": normalized_cost,
        "normalized_solver_cost_vs_all_strong": float(np.sum(solver_costs) / strong_norm),
        "utility_ratio_to_strong_inclusive_oracle": float(mean_utility / oracle_utility),
        "route_call_rate": float(route_calls / max(len(frame), 1)),
        "frontier_call_rate": float(frontier_calls / max(len(frame), 1)),
        "action_counts": json.dumps(action_counts, sort_keys=True),
    }


def select_rows(grid: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    rows = []
    feasible = val[
        val["quality_gap_to_strong_inclusive_oracle"].le(args.quality_gap_target)
        & val["normalized_cost_vs_all_strong"].le(args.cost_target)
        & val["utility_ratio_to_strong_inclusive_oracle"].ge(args.utility_ratio_target)
    ]
    if len(feasible):
        picks = [
            (
                "validation_feasible_min_cost",
                feasible.sort_values(["normalized_cost_vs_all_strong", "quality_gap_to_strong_inclusive_oracle"]).head(1),
            ),
            (
                "validation_feasible_quality_conservative",
                feasible.sort_values(
                    [
                        "mean_quality",
                        "quality_gap_to_strong_inclusive_oracle",
                        "utility_ratio_to_strong_inclusive_oracle",
                        "normalized_cost_vs_all_strong",
                        "method",
                    ],
                    ascending=[False, True, False, True, True],
                ).head(1),
            ),
        ]
    else:
        under_cost = val[val["normalized_cost_vs_all_strong"].le(args.cost_target)]
        picks = [
            (
                "no_validation_feasible_best_gap_under_cost",
                under_cost.sort_values(
                    ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
                    ascending=[True, False],
                ).head(1),
            )
        ]
    seen: set[str] = set()
    for rule, picked in picks:
        if not len(picked):
            continue
        method = str(picked.iloc[0]["method"])
        if method in seen:
            continue
        seen.add(method)
        rows.append(picked.assign(selection_rule=rule))
        rows.append(grid[grid["method"].eq(method) & grid["split"].eq("test")].assign(selection_rule=f"{rule}_test"))
    diag = grid[grid["split"].eq("test") & grid["normalized_cost_vs_all_strong"].le(args.cost_target)].sort_values(
        ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
        ascending=[True, False],
    ).head(1)
    if len(diag):
        rows.append(diag.assign(selection_rule="best_heldout_diagnostic_under_cost"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_memo(output_dir: Path, routes: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "quality_gap_to_strong_inclusive_oracle",
        "normalized_cost_vs_all_strong",
        "utility_ratio_to_strong_inclusive_oracle",
        "route_call_rate",
        "frontier_call_rate",
        "action_counts",
    ]
    lines = [
        "# Exact-Math GPT Route Judge",
        "",
        f"Route rows: `{len(routes)}`. Total route-label cost: `${routes['route_cost'].sum():.4f}`.",
        "",
        "Selected rows:",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in selected[cols].iterrows():
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    output_dir.joinpath("EXACT_MATH_GPT_ROUTE_JUDGE_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tool = load_tool_module()
    router = tool.load_router_module()
    table = tool.add_tool_outputs(tool.load_table(router))
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    route_frame = table[table["split"].isin(splits)].copy()
    if args.max_rows is not None:
        route_frame = route_frame.head(int(args.max_rows)).copy()
    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")
    routes = collect_routes(route_frame, output_dir, args, api_key)
    if len(routes) == len(table):
        grid, selected = evaluate(table, routes, args)
    else:
        grid, selected = pd.DataFrame(), pd.DataFrame()
    routes.to_csv(output_dir / "table_exact_math_gpt_route_judge_routes.csv", index=False)
    grid.to_csv(output_dir / "table_exact_math_gpt_route_judge.csv", index=False)
    selected.to_csv(output_dir / "table_exact_math_gpt_route_judge_selected.csv", index=False)
    if len(selected):
        write_memo(output_dir, routes, selected)
    print(f"Wrote exact-math GPT route judge results to {output_dir}")
    if len(selected):
        print(selected.to_string(index=False))
    else:
        print(routes[["query_id", "route_action", "route_confidence", "route_cost", "route_cache_hit"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
