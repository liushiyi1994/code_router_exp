from __future__ import annotations

import argparse
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

from routecode.controlled.live_stage0 import extract_openai_text, load_env_values, post_json, resolve_key, usage_from_openai


ROUTE_MODEL = "gpt-5.5"
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00
LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
GEMINI_STRONG = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIME-only GPT judge with Gemini-strong candidate.")
    parser.add_argument("--output-dir", default="results/controlled/aime_gpt_judge_with_gemini_strong")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=220)
    parser.add_argument("--max-api-spend-usd", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    return parser.parse_args()


def load_router_module():
    spec = importlib.util.spec_from_file_location("gemini_router", "experiments/110_gemini_strong_router.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_expanded_table(router) -> pd.DataFrame:
    base = pd.read_csv("results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv")
    table = router.merge_gemini_strong(
        base,
        [
            "results/controlled/gemini_strong_solver_probe_train/table_gemini_strong_solver_outputs.csv",
            "results/controlled/gemini_strong_solver_probe_val/table_gemini_strong_solver_outputs.csv",
            "results/controlled/gemini_strong_solver_probe_test/table_gemini_strong_solver_outputs.csv",
        ],
    )
    table = router.add_features(table)
    table = router.add_oracle_labels(table, 0.35)
    return table


def baseline_actions(router, table: pd.DataFrame) -> pd.Series:
    fs = router.feature_sets(table)["local"]
    train = router.prepare_features(table[table["split"].eq("train")].copy(), list(fs["cat_cols"]), list(fs["num_cols"]))
    eval_frame = router.prepare_features(
        table[table["split"].isin(["val", "test"])].copy(), list(fs["cat_cols"]), list(fs["num_cols"])
    )
    cost_scale = max(float(train["strong_cost"].sum()) / max(len(train), 1), 1e-12)
    pred = router.fit_expected_quality_router(
        train,
        eval_frame,
        feature_spec=fs,
        regressor=router.regressor_specs()["extra_trees_reg"],
        cost_scale=cost_scale,
        lambda_cost=0.35,
    )
    actions = pd.Series(index=table.index, dtype=object)
    actions.loc[eval_frame.index] = pred
    return actions


def prompt_for(row: pd.Series, baseline_action: str) -> str:
    return f"""You are a strict route judge for AIME exact-answer math. You do not see the gold answer.

Choose the cheapest action that is likely to be exactly correct. If all visible candidates are unreliable and the problem looks solvable by stronger reasoning, choose USE_STRONG_GPT.

Actions:
- USE_QWEN14: use the Qwen14 local answer.
- USE_GEMINI: use the cheap Gemini answer.
- USE_GEMINI_STRONG: use the thinking-enabled Gemini Flash answer.
- USE_BASE_GPT: use the base GPT answer.
- USE_STRONG_GPT: run a slower stronger GPT reasoning call.
- USE_BASELINE: keep the current learned router choice.

Return JSON only:
{{"action":"USE_QWEN14|USE_GEMINI|USE_GEMINI_STRONG|USE_BASE_GPT|USE_STRONG_GPT|USE_BASELINE","confidence":0.0,"reason":"short"}}

Problem:
{truncate(row['query_text'], 1800)}

Current learned-router action: {baseline_action}
Candidate final answers:
- Qwen0.6: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 120)}
- Qwen4: {truncate(row.get('qwen3-4b-local_answer', ''), 120)}
- Qwen8: {truncate(row.get('qwen3-8b-local_answer', ''), 120)}
- Qwen14: {truncate(row.get('qwen3-14b-awq-local_answer', ''), 120)}
- Gemini cheap: {truncate(row.get(f'{GEMINI}_answer', ''), 160)}
- Gemini thinking: {truncate(row.get('gemini_strong_answer', ''), 160)}
- Base GPT: {truncate(row.get(f'{BASE_GPT}_answer', ''), 160)}

Cheap metadata:
query_len={row.get('query_len', '')}; number_count={row.get('number_count', '')}; latex_count={row.get('latex_count', '')}; local_votes={row.get('local_ensemble_votes', '')}
/no_think"""


def collect_routes(frame: pd.DataFrame, actions: pd.Series, output_dir: Path, args: argparse.Namespace, api_key: str) -> pd.DataFrame:
    cache_dir = output_dir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = []
    for idx, row in frame.iterrows():
        baseline_action = str(actions.loc[idx])
        prompt_rows.append((idx, row, prompt_for(row, baseline_action)))
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


def evaluate(table: pd.DataFrame, baseline: pd.Series, routes: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    route_by_idx = routes.set_index("row_index")
    rows = []
    for threshold in [0.0, 0.4, 0.6, 0.8, 0.95]:
        for fallback in ["baseline", GEMINI_STRONG, BASE_GPT, STRONG_GPT]:
            for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
                actions = []
                qualities = []
                solver_costs = []
                total_costs = []
                for idx, row in frame.iterrows():
                    action = str(baseline.loc[idx])
                    route_cost = 0.0
                    if row["dataset"] == "aime" and idx in route_by_idx.index:
                        route = route_by_idx.loc[idx]
                        route_cost = float(route.get("route_cost", 0.0) or 0.0)
                        if float(route.get("route_confidence", 0.0) or 0.0) >= threshold:
                            action = map_route_action(str(route["route_action"]), action, fallback)
                        elif fallback != "baseline":
                            action = fallback
                    quality, solver_cost = row_quality_cost(row, action)
                    actions.append(action)
                    qualities.append(quality)
                    solver_costs.append(solver_cost)
                    total_costs.append(solver_cost + route_cost)
                rows.append(split_metrics(frame, actions, qualities, solver_costs, total_costs, threshold, fallback, args.lambda_cost))
    grid = pd.DataFrame(rows)
    selected = select_rows(grid, args.quality_gap_target, args.cost_target)
    return grid, selected


def map_route_action(route_action: str, baseline_action: str, fallback: str) -> str:
    mapping = {
        "USE_QWEN14": "qwen3-14b-awq-local",
        "USE_GEMINI": GEMINI,
        "USE_GEMINI_STRONG": GEMINI_STRONG,
        "USE_BASE_GPT": BASE_GPT,
        "USE_STRONG_GPT": STRONG_GPT,
        "USE_BASELINE": baseline_action,
    }
    return mapping.get(route_action, baseline_action if fallback == "baseline" else fallback)


def row_quality_cost(row: pd.Series, action: str) -> tuple[float, float]:
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
    threshold: float,
    fallback: str,
    lambda_cost: float,
) -> dict[str, object]:
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    normalized_total_cost = float(np.sum(total_costs) / strong_norm)
    mean_utility = float(mean_quality - lambda_cost * normalized_total_cost)
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    action_series = pd.Series(actions)
    return {
        "method": f"aime_gpt_judge_t{threshold:g}_fb{fallback}",
        "split": str(frame["split"].iloc[0]),
        "threshold": threshold,
        "fallback": fallback,
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_strong_inclusive_oracle": float(oracle_quality - mean_quality),
        "normalized_cost_vs_all_strong": normalized_total_cost,
        "normalized_solver_cost_vs_all_strong": float(np.sum(solver_costs) / strong_norm),
        "utility_ratio_to_strong_inclusive_oracle": float(mean_utility / oracle_utility),
        "action_counts": json.dumps(action_series.value_counts().to_dict(), sort_keys=True),
    }


def select_rows(grid: pd.DataFrame, quality_gap_target: float, cost_target: float) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    rows = []
    feasible = val[
        val["quality_gap_to_strong_inclusive_oracle"].le(quality_gap_target)
        & val["normalized_cost_vs_all_strong"].le(cost_target)
        & val["utility_ratio_to_strong_inclusive_oracle"].ge(0.95)
    ]
    if len(feasible):
        picked = feasible.sort_values(["normalized_cost_vs_all_strong", "quality_gap_to_strong_inclusive_oracle"]).head(1)
        rule = "validation_feasible_min_cost"
    else:
        picked = val[val["normalized_cost_vs_all_strong"].le(cost_target)].sort_values(
            ["quality_gap_to_strong_inclusive_oracle", "utility_ratio_to_strong_inclusive_oracle"],
            ascending=[True, False],
        ).head(1)
        rule = "no_validation_feasible_best_gap_under_cost"
    if len(picked):
        rows.append(picked.assign(selection_rule=rule))
        method = str(picked.iloc[0]["method"])
        rows.append(grid[grid["method"].eq(method) & grid["split"].eq("test")].assign(selection_rule="selected_test"))
    diag = grid[grid["split"].eq("test") & grid["normalized_cost_vs_all_strong"].le(cost_target)].sort_values(
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
        "action_counts",
    ]
    lines = [
        "# AIME GPT Judge With Gemini-Strong Candidate",
        "",
        f"Route rows: `{len(routes)}`. Route cost: `${routes['route_cost'].sum():.4f}`.",
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
    output_dir.joinpath("AIME_GPT_JUDGE_WITH_GEMINI_STRONG_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


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
    for action in ["USE_QWEN14", "USE_GEMINI_STRONG", "USE_STRONG_GPT", "USE_BASE_GPT", "USE_GEMINI", "USE_BASELINE"]:
        if action in upper:
            return action, 0.5, raw[:120]
    return "USE_BASELINE", 0.0, "parse_fallback"


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return route_call_cost(input_tokens, output_tokens)


def route_call_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def cache_name(query_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id) + ".json"


def truncate(text: object, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router = load_router_module()
    table = load_expanded_table(router)
    baseline = baseline_actions(router, table)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    route_frame = table[table["split"].isin(splits) & table["dataset"].eq("aime")].copy()
    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")
    routes = collect_routes(route_frame, baseline, output_dir, args, api_key)
    grid, selected = evaluate(table, baseline, routes, args)
    routes.to_csv(output_dir / "table_aime_gpt_judge_routes.csv", index=False)
    grid.to_csv(output_dir / "table_aime_gpt_judge_with_gemini_strong.csv", index=False)
    selected.to_csv(output_dir / "table_aime_gpt_judge_with_gemini_strong_selected.csv", index=False)
    write_memo(output_dir, routes, selected)
    print(f"Wrote AIME GPT judge results to {output_dir}")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
