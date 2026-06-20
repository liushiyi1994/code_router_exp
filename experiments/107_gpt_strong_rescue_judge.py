from __future__ import annotations

import argparse
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


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
ROUTE_MODEL = "gpt-5.5"
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT-5.5 as a strong-rescue route judge.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/gpt_strong_rescue_judge")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=160)
    parser.add_argument("--max-api-spend-usd", type=float, default=1.25)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    frame = table[table["split"].astype(str).isin(splits)].copy()
    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")

    routes = collect_routes(
        frame,
        output_dir,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    merged = table.merge(routes, on="query_id", how="left")
    grid = evaluate_grid(merged, args.lambda_cost)
    selected = selected_rows(grid, args.quality_gap_target, args.cost_target)

    routes.to_csv(output_dir / "table_gpt_strong_rescue_judge_routes.csv", index=False)
    merged.to_csv(output_dir / "query_table_with_gpt_strong_rescue_judge.csv", index=False)
    grid.to_csv(output_dir / "table_gpt_strong_rescue_judge_gate.csv", index=False)
    selected.to_csv(output_dir / "table_gpt_strong_rescue_judge_selected.csv", index=False)
    write_memo(output_dir, args, routes, selected, grid)
    print(f"Wrote GPT strong-rescue judge results to {output_dir}")


def collect_routes(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = [(row, prompt_for(row)) for _, row in frame.iterrows()]
    missing = [prompt for row, prompt in prompts if not (cache_dir / cache_name(str(row["query_id"]))).exists()]
    estimated = estimate_prompt_cost(missing, max_output_tokens)
    if estimated > max_api_spend_usd:
        raise RuntimeError(
            f"Estimated uncached GPT route-judge spend ${estimated:.4f} exceeds cap ${max_api_spend_usd:.4f}."
        )

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [
            executor.submit(call_one_route, row, prompt, cache_dir, api_key, max_output_tokens)
            for row, prompt in prompts
        ]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def call_one_route(
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
            payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 90.0)
        except urllib.error.HTTPError:
            request_payload.pop("text", None)
            payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 90.0)
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = extract_openai_text(payload)
    action, confidence, reason = parse_route_text(text)
    input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
    return {
        "query_id": query_id,
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


def prompt_for(row: pd.Series) -> str:
    return f"""You are a strict routing judge for exact-answer math.

You see candidate final answers from local models, Gemini, and a base GPT-5.5 solver. You do not see the gold answer.
Decide whether a deployment should use one existing answer or escalate to a slower stronger GPT-5.5 reasoning call.

Actions:
- USE_LOCAL if a local answer is very likely exactly correct, or if the row looks hopeless and paying more is wasteful.
- USE_GEMINI if Gemini's answer is likely exactly correct and cheaper than GPT.
- USE_BASE_GPT if the base GPT answer is likely exactly correct.
- USE_STRONG_GPT if base GPT/Gemini/local answers look unreliable but a stronger reasoning call has a realistic chance to solve it.

For hard AIME/olympiad rows with inconsistent candidates and no credible base GPT answer, prefer USE_STRONG_GPT.
Do not trust a local answer just because it is short.

Return JSON only:
{{"action":"USE_LOCAL|USE_GEMINI|USE_BASE_GPT|USE_STRONG_GPT","confidence":0.0,"reason":"short"}}

Dataset: {row['dataset']}
Problem: {truncate(row['query_text'], 1600)}
Local answers:
- Qwen0.6: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 180)}
- Qwen4: {truncate(row.get('qwen3-4b-local_answer', ''), 180)}
- Qwen8: {truncate(row.get('qwen3-8b-local_answer', ''), 180)}
- Qwen14: {truncate(row.get('qwen3-14b-awq-local_answer', ''), 180)}
Gemini answer: {truncate(row.get(f'{GEMINI}_answer', ''), 240)}
Base GPT answer: {truncate(row.get(f'{BASE_GPT}_answer', ''), 240)}
Local max vote: {row.get('local_max_vote', '')}
/no_think"""


def evaluate_grid(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for strong_threshold in [0.0, 0.5, 0.7, 0.85, 0.95]:
        for base_threshold in [0.0, 0.5, 0.7, 0.85, 0.95]:
            for gemini_threshold in [0.0, 0.5, 0.7, 0.85, 0.95]:
                for fallback in ["local", GEMINI, BASE_GPT]:
                    method = (
                        f"judge_st{strong_threshold:g}_bt{base_threshold:g}_"
                        f"gt{gemini_threshold:g}_fb{fallback}"
                    )
                    for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
                        if frame["route_action"].isna().all():
                            continue
                        rows.append(
                            evaluate_split(
                                frame,
                                method=method,
                                strong_threshold=strong_threshold,
                                base_threshold=base_threshold,
                                gemini_threshold=gemini_threshold,
                                fallback=fallback,
                                lambda_cost=lambda_cost,
                            )
                        )
    return pd.DataFrame(rows)


def evaluate_split(
    frame: pd.DataFrame,
    *,
    method: str,
    strong_threshold: float,
    base_threshold: float,
    gemini_threshold: float,
    fallback: str,
    lambda_cost: float,
) -> dict[str, object]:
    qualities: list[float] = []
    solver_costs: list[float] = []
    total_costs: list[float] = []
    actions: list[str] = []
    for _, row in frame.iterrows():
        action = action_to_model(row, strong_threshold, base_threshold, gemini_threshold, fallback)
        quality, solver_cost = row_quality_cost(row, action)
        route_cost = float(row.get("route_cost", 0.0) or 0.0)
        qualities.append(quality)
        solver_costs.append(solver_cost)
        total_costs.append(solver_cost + route_cost)
        actions.append(action)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    normalized_total_cost = float(np.sum(total_costs) / strong_norm)
    normalized_solver_cost = float(np.sum(solver_costs) / strong_norm)
    mean_utility = float(mean_quality - lambda_cost * normalized_total_cost)
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    action_series = pd.Series(actions)
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_strong_inclusive_cost_oracle": float(oracle_quality - mean_quality),
        "normalized_remote_cost_vs_all_strong_gpt": normalized_total_cost,
        "normalized_solver_cost_vs_all_strong_gpt": normalized_solver_cost,
        "utility_ratio_to_strong_inclusive_cost_oracle": float(mean_utility / oracle_utility) if oracle_utility else np.nan,
        "strong_rate": float(action_series.eq(STRONG_GPT).mean()),
        "base_rate": float(action_series.eq(BASE_GPT).mean()),
        "gemini_rate": float(action_series.eq(GEMINI).mean()),
        "action_counts": json.dumps({str(key): int(value) for key, value in action_series.value_counts().to_dict().items()}),
    }


def action_to_model(
    row: pd.Series,
    strong_threshold: float,
    base_threshold: float,
    gemini_threshold: float,
    fallback: str,
) -> str:
    route_action = str(row.get("route_action", ""))
    confidence = float(row.get("route_confidence", 0.0) or 0.0)
    if route_action == "USE_STRONG_GPT" and confidence >= strong_threshold:
        return STRONG_GPT
    if route_action == "USE_BASE_GPT" and confidence >= base_threshold:
        return BASE_GPT
    if route_action == "USE_GEMINI" and confidence >= gemini_threshold:
        return GEMINI
    if route_action == "USE_LOCAL":
        return local_choice(row)
    if fallback == "local":
        return local_choice(row)
    return fallback


def local_choice(row: pd.Series) -> str:
    if float(row.get("local_ensemble_votes", 0.0) or 0.0) >= 2:
        source = str(row.get("local_ensemble_source", ""))
        if source in LOCAL_MODELS:
            return source
    return "qwen3-0.6b-probe"


def row_quality_cost(row: pd.Series, model_id: str) -> tuple[float, float]:
    if model_id == STRONG_GPT:
        return float(row["strong_quality"]), float(row["strong_cost"])
    if model_id in LOCAL_MODELS:
        return float(row[f"{model_id}_quality"]), 0.0
    if model_id == GEMINI:
        return float(row[f"{GEMINI}_quality"]), float(row[f"{GEMINI}_cost"])
    if model_id == BASE_GPT:
        return float(row[f"{BASE_GPT}_quality"]), float(row[f"{BASE_GPT}_cost"])
    raise ValueError(model_id)


def selected_rows(table: pd.DataFrame, quality_gap_target: float, cost_target: float) -> pd.DataFrame:
    val = table[table["split"].eq("val")].copy()
    rows = []
    for selection_rule, candidates in [
        ("val_best_utility_under_cost_target", val[val["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)]),
        (
            "val_feasible_quality_cost_target",
            val[
                val["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)
                & val["quality_gap_to_strong_inclusive_cost_oracle"].le(quality_gap_target)
            ],
        ),
        ("val_best_utility_any_cost", val),
    ]:
        if candidates.empty:
            continue
        picked = candidates.sort_values(
            ["utility_ratio_to_strong_inclusive_cost_oracle", "quality_gap_to_strong_inclusive_cost_oracle"],
            ascending=[False, True],
        ).head(1)
        method = str(picked.iloc[0]["method"])
        matches = table[table["method"].eq(method)].copy()
        matches["selection_rule"] = selection_rule
        rows.append(matches)
    test = table[table["split"].eq("test")].copy()
    diagnostic = test[
        test["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)
        & test["quality_gap_to_strong_inclusive_cost_oracle"].le(quality_gap_target)
    ]
    if not diagnostic.empty:
        picked = diagnostic.sort_values("utility_ratio_to_strong_inclusive_cost_oracle", ascending=False).head(1).copy()
        picked["selection_rule"] = "test_diagnostic_feasible_quality_cost"
        rows.append(picked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_memo(
    output_dir: Path,
    args: argparse.Namespace,
    routes: pd.DataFrame,
    selected: pd.DataFrame,
    grid: pd.DataFrame,
) -> None:
    key_cols = [
        "selection_rule",
        "method",
        "split",
        "mean_quality",
        "quality_gap_to_strong_inclusive_cost_oracle",
        "normalized_remote_cost_vs_all_strong_gpt",
        "normalized_solver_cost_vs_all_strong_gpt",
        "utility_ratio_to_strong_inclusive_cost_oracle",
        "strong_rate",
        "base_rate",
        "gemini_rate",
        "action_counts",
    ]
    best_test = grid[grid["split"].eq("test")].sort_values(
        ["utility_ratio_to_strong_inclusive_cost_oracle", "quality_gap_to_strong_inclusive_cost_oracle"],
        ascending=[False, True],
    ).head(10)
    memo = f"""# GPT Strong-Rescue Judge

Input table: `{args.query_table}`

This diagnostic uses GPT-5.5 with `reasoning.effort=none` as a route judge. It sees cached local,
Gemini, and base-GPT candidate answers, but no gold answer and no strong-GPT output. Route calls are
cached under `raw/`; Claude is not used.

Route-label cost: `${routes['route_cost'].sum():.4f}`.

Route action counts:

{markdown_table(routes['route_action'].value_counts().rename_axis('route_action').reset_index(name='count'), ['route_action', 'count'])}

## Selected Policies

{markdown_table(selected, [col for col in key_cols if col in selected.columns])}

## Best Held-Out Rows By Utility

{markdown_table(best_test, [col for col in key_cols if col in best_test.columns])}

## Interpretation

The GPT route judge is not a successful ProbeRoute++ policy on this slice. It adds nontrivial route
cost, over-trusts some wrong base/local answers, and no validation-selected threshold row satisfies
both the 3-point quality target and the normalized cost target.
"""
    (output_dir / "GPT_STRONG_RESCUE_JUDGE_MEMO.md").write_text(memo, encoding="utf-8")


def parse_route_text(text: object) -> tuple[str, float, str]:
    raw = str(text or "")
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            return (
                str(payload.get("action", "")).upper(),
                float(payload.get("confidence", 0.0) or 0.0),
                str(payload.get("reason", "")),
            )
        except Exception:
            pass
    upper = raw.upper()
    for action in ["USE_STRONG_GPT", "USE_BASE_GPT", "USE_GEMINI", "USE_LOCAL"]:
        if action in upper:
            return action, np.nan, raw[:120]
    return "USE_GEMINI", 0.0, "parse_fallback"


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def route_call_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def cache_name(query_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id) + ".json"


def truncate(text: object, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
