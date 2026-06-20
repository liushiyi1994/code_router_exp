from __future__ import annotations

import argparse
import hashlib
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
GPT = "gpt-5.5"
ROUTE_MODEL = "gpt-5.5"
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00
CHOICES = dict(zip(["A", "B", "C", "D"], LOCAL_MODELS, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask GPT-5.5 for rescue probabilities, then tune a budgeted gate.")
    parser.add_argument("--query-table", default="results/controlled/expanded_local_pool_qwen14/query_table_expanded_local_pool.csv")
    parser.add_argument("--output-dir", default="results/controlled/gpt_rescue_score_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=120)
    parser.add_argument("--max-api-spend-usd", type=float, default=1.50)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    return parser.parse_args()


def truncate(text: object, limit: int = 900) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def answer_value(row: pd.Series, model_id: str) -> str:
    norm_col = f"{model_id}_answer_norm"
    raw_col = f"{model_id}_answer"
    value = row.get(norm_col, row.get(raw_col, ""))
    return "" if pd.isna(value) else str(value)


def prompt_for(row: pd.Series) -> str:
    local_lines = []
    for letter, model_id in CHOICES.items():
        local_lines.append(f"{letter}. {answer_value(row, model_id)[:180] or '[empty]'}")
    return f"""You are a cautious RouteCode verifier for exact-answer math routing.

You see the query, four local model final answers, and the Gemini 3.5 Flash final answer.
Do not use any gold answer. Do not assume candidate answers are correct.

Return compact JSON only with:
- local_choice: A, B, C, D, or NONE for the most credible local candidate.
- p_local_correct: probability the chosen local candidate is exactly correct.
- p_gemini_correct: probability the Gemini answer is exactly correct.
- p_gpt_correct: probability GPT-5.5 would answer exactly correctly if called fresh.
- p_hopeless: probability none of these solvers would answer correctly.

Calibrate probabilities. For hard AIME/olympiad rows with inconsistent local answers, p_gpt_correct should usually exceed p_local_correct unless the row looks impossible.

Dataset: {row['dataset']}
Problem:
{truncate(row['query_text'], 1500)}

Local candidate final answers:
{chr(10).join(local_lines)}

Gemini final answer:
{truncate(row.get(f'{GEMINI}_answer', ''), 220)}

Local max vote: {row.get('local_max_vote', '')}

JSON only, for example:
{{"local_choice":"C","p_local_correct":0.25,"p_gemini_correct":0.55,"p_gpt_correct":0.72,"p_hopeless":0.10}}
/no_think"""


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * max_output_tokens
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def route_call_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def cache_name(query_id: str) -> str:
    digest = hashlib.sha1(query_id.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def call_openai(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 90.0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": ROUTE_MODEL,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            payload.pop("text", None)
            return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
        raise


def parse_jsonish(text: object) -> dict[str, Any]:
    raw = str(text or "")
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    payload: dict[str, Any] = {}
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
    local_choice = str(payload.get("local_choice", "")).upper().strip()
    if local_choice not in {*CHOICES.keys(), "NONE"}:
        local_choice = "NONE"
    out: dict[str, Any] = {"local_choice": local_choice}
    for key in ["p_local_correct", "p_gemini_correct", "p_gpt_correct", "p_hopeless"]:
        value = payload.get(key, np.nan)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            m = re.search(rf'"?{key}"?\s*[:=]\s*([0-9.]+)', raw, flags=re.I)
            parsed = float(m.group(1)) if m else np.nan
        if np.isnan(parsed):
            parsed = 0.0
        out[key] = float(np.clip(parsed, 0.0, 1.0))
    return out


def collect_scores(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_rescue_scores" / ROUTE_MODEL
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = [prompt_for(row) for _, row in frame.iterrows()]
    missing = [
        prompt
        for prompt, (_, row) in zip(prompts, frame.iterrows())
        if not (cache_dir / cache_name(str(row["query_id"]))).exists()
    ]
    estimated = estimate_prompt_cost(missing, max_output_tokens)
    if estimated > max_api_spend_usd:
        raise RuntimeError(f"Estimated uncached GPT rescue-score spend ${estimated:.4f} exceeds cap ${max_api_spend_usd:.4f}.")

    def one(row: pd.Series, prompt: str) -> dict[str, Any]:
        query_id = str(row["query_id"])
        raw_path = cache_dir / cache_name(query_id)
        cache_hit = raw_path.exists()
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_openai(prompt, api_key, max_output_tokens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - started
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row_status = str(payload.get("_status", status))
        text = extract_openai_text(payload) if row_status == "success" else ""
        parsed = parse_jsonish(text)
        if row_status == "success":
            input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
        else:
            input_tokens, output_tokens = 0, 0
        return {
            "query_id": query_id,
            "rescue_status": row_status,
            "rescue_error_type": str(payload.get("_error_type", error_type)),
            "rescue_text": text,
            "local_choice": parsed["local_choice"],
            "local_choice_model": CHOICES.get(parsed["local_choice"], ""),
            "p_local_correct": parsed["p_local_correct"],
            "p_gemini_correct": parsed["p_gemini_correct"],
            "p_gpt_correct": parsed["p_gpt_correct"],
            "p_hopeless": parsed["p_hopeless"],
            "rescue_input_tokens": int(input_tokens),
            "rescue_output_tokens": int(output_tokens),
            "rescue_latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
            "rescue_cache_hit": cache_hit,
            "rescue_raw_path": str(raw_path),
            "rescue_route_cost": route_call_cost(int(input_tokens), int(output_tokens)),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row, prompt) for (_, row), prompt in zip(frame.iterrows(), prompts)]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def add_oracle_columns(table: pd.DataFrame, lambda_cost: float, cost_norm: float) -> pd.DataFrame:
    table = table.copy()
    quality_cols = [f"{model_id}_quality" for model_id in LOCAL_MODELS + [GEMINI, GPT]]
    utility_cols = []
    for model_id in LOCAL_MODELS:
        col = f"{model_id}_rescue_utility"
        table[col] = table[f"{model_id}_quality"].astype(float)
        utility_cols.append(col)
    for model_id in [GEMINI, GPT]:
        col = f"{model_id}_rescue_utility"
        table[col] = table[f"{model_id}_quality"].astype(float) - lambda_cost * (
            table[f"{model_id}_cost"].astype(float) / cost_norm
        )
        utility_cols.append(col)
    table["rescue_quality_oracle"] = table[quality_cols].max(axis=1)
    table["rescue_cost_oracle_utility"] = table[utility_cols].max(axis=1)
    return table


def fallback_local_by_dataset(train: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for dataset, frame in train.groupby("dataset"):
        qualities = {model_id: float(frame[f"{model_id}_quality"].mean()) for model_id in LOCAL_MODELS}
        out[str(dataset)] = max(qualities, key=qualities.get)
    return out


def build_actions(
    frame: pd.DataFrame,
    *,
    lambda_cost: float,
    cost_norm: float,
    local_fallback: dict[str, str],
    method: str,
    threshold: float | None = None,
    budget_rate: float | None = None,
) -> pd.Series:
    local_action = frame["local_choice_model"].where(frame["local_choice_model"].isin(LOCAL_MODELS), "")
    local_action = local_action.mask(local_action.eq(""), frame["dataset"].map(lambda value: local_fallback.get(str(value), "qwen3-8b-local")))
    predicted_local_u = frame["p_local_correct"].astype(float)
    predicted_gemini_u = frame["p_gemini_correct"].astype(float) - lambda_cost * (
        frame[f"{GEMINI}_cost"].astype(float) / cost_norm
    )
    predicted_gpt_u = frame["p_gpt_correct"].astype(float) - lambda_cost * (
        frame[f"{GPT}_cost"].astype(float) / cost_norm
    )
    actions = local_action.copy()
    if method == "argmax":
        values = pd.concat(
            [
                predicted_local_u.rename("local"),
                predicted_gemini_u.rename(GEMINI),
                predicted_gpt_u.rename(GPT),
            ],
            axis=1,
        )
        best = values.idxmax(axis=1)
        actions = best.where(best.ne("local"), local_action)
    elif method == "remote_budget":
        assert budget_rate is not None
        remote_u = pd.concat([predicted_gemini_u.rename(GEMINI), predicted_gpt_u.rename(GPT)], axis=1)
        best_remote = remote_u.idxmax(axis=1)
        remote_gain = remote_u.max(axis=1) - predicted_local_u
        budget = int(np.floor(float(budget_rate) * len(frame)))
        remote_index = remote_gain.sort_values(ascending=False).head(budget).index if budget > 0 else []
        actions.loc[remote_index] = best_remote.loc[remote_index]
    elif method == "gpt_budget":
        assert budget_rate is not None
        gpt_gain = predicted_gpt_u - pd.concat([predicted_local_u, predicted_gemini_u], axis=1).max(axis=1)
        budget = int(np.floor(float(budget_rate) * len(frame)))
        gpt_index = gpt_gain.sort_values(ascending=False).head(budget).index if budget > 0 else []
        # Non-GPT rows use whichever local/Gemini predicted utility is better.
        actions = local_action.copy()
        gemini_index = predicted_gemini_u.gt(predicted_local_u)
        actions.loc[gemini_index] = GEMINI
        actions.loc[gpt_index] = GPT
    elif method == "threshold":
        assert threshold is not None
        remote_u = pd.concat([predicted_gemini_u.rename(GEMINI), predicted_gpt_u.rename(GPT)], axis=1)
        best_remote = remote_u.idxmax(axis=1)
        remote_gain = remote_u.max(axis=1) - predicted_local_u
        remote_index = remote_gain[remote_gain.ge(float(threshold))].index
        actions.loc[remote_index] = best_remote.loc[remote_index]
    else:
        raise ValueError(method)
    return actions


def evaluate_actions(frame: pd.DataFrame, actions: pd.Series, lambda_cost: float) -> dict[str, Any]:
    qualities: list[float] = []
    solver_costs: list[float] = []
    total_costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    latencies: list[float] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        route_cost = float(row.get("rescue_route_cost", 0.0) or 0.0)
        route_latency = float(row.get("rescue_latency_s", 0.0) or 0.0)
        if action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            cost = 0.0
            gpt = False
            gemini = False
            latency = route_latency
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            cost = float(row[f"{GEMINI}_cost"])
            gpt = False
            gemini = True
            latency = route_latency + float(row[f"{GEMINI}_latency"])
        elif action == GPT:
            quality = float(row[f"{GPT}_quality"])
            cost = float(row[f"{GPT}_cost"])
            gpt = True
            gemini = False
            latency = route_latency + float(row[f"{GPT}_latency"])
        else:
            raise ValueError(action)
        qualities.append(quality)
        solver_costs.append(cost)
        total_costs.append(cost + route_cost)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
        latencies.append(latency)
    all_gpt_cost = max(float(frame[f"{GPT}_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    solver_cost_norm = float(np.sum(solver_costs) / all_gpt_cost)
    total_cost_norm = float(np.sum(total_costs) / all_gpt_cost)
    solver_utility = float(mean_quality - lambda_cost * solver_cost_norm)
    total_utility = float(mean_quality - lambda_cost * total_cost_norm)
    oracle_utility = float(frame["rescue_cost_oracle_utility"].mean())
    return {
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_expanded_oracle": float(frame["rescue_quality_oracle"].mean() - mean_quality),
        "solver_utility": solver_utility,
        "total_utility_with_route_cost": total_utility,
        "solver_utility_ratio_to_expanded_cost_oracle": float(solver_utility / oracle_utility) if oracle_utility else np.nan,
        "total_utility_ratio_to_expanded_cost_oracle": float(total_utility / oracle_utility) if oracle_utility else np.nan,
        "normalized_solver_cost_vs_all_gpt": solver_cost_norm,
        "normalized_total_cost_vs_all_gpt": total_cost_norm,
        "route_cost_total_usd": float(frame["rescue_route_cost"].sum()),
        "solver_frontier_call_rate": float(np.mean([a or b for a, b in zip(gpt_calls, gemini_calls)])),
        "gpt_solver_call_rate": float(np.mean(gpt_calls)),
        "gemini_solver_call_rate": float(np.mean(gemini_calls)),
        "total_remote_call_rate_including_router": 1.0,
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
        "action_counts": json.dumps({str(k): int(v) for k, v in actions.value_counts().to_dict().items()}, sort_keys=True),
    }


def evaluate(table: pd.DataFrame, lambda_cost: float, frontier_rate_target: float) -> pd.DataFrame:
    cost_norm = max(float(table[f"{GPT}_cost"].mean()), 1e-12)
    table = add_oracle_columns(table, lambda_cost, cost_norm)
    local_fallback = fallback_local_by_dataset(table[table["split"].eq("train")])
    rows: list[dict[str, Any]] = []
    for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
        for method in ["argmax"]:
            actions = build_actions(frame, lambda_cost=lambda_cost, cost_norm=cost_norm, local_fallback=local_fallback, method=method)
            row = evaluate_actions(frame, actions, lambda_cost)
            row["method"] = f"gpt_rescue_score_{method}"
            rows.append(row)
        for budget_rate in [0.25, 0.30, 0.35, frontier_rate_target, 0.45, 0.50, 0.60]:
            for method in ["remote_budget", "gpt_budget"]:
                actions = build_actions(
                    frame,
                    lambda_cost=lambda_cost,
                    cost_norm=cost_norm,
                    local_fallback=local_fallback,
                    method=method,
                    budget_rate=float(budget_rate),
                )
                row = evaluate_actions(frame, actions, lambda_cost)
                row["method"] = f"gpt_rescue_score_{method}{budget_rate:.2f}"
                row["budget_rate"] = budget_rate
                rows.append(row)
        for threshold in np.linspace(-0.25, 0.50, 16):
            actions = build_actions(
                frame,
                lambda_cost=lambda_cost,
                cost_norm=cost_norm,
                local_fallback=local_fallback,
                method="threshold",
                threshold=float(threshold),
            )
            row = evaluate_actions(frame, actions, lambda_cost)
            row["method"] = f"gpt_rescue_score_threshold{threshold:.2f}"
            row["threshold"] = threshold
            rows.append(row)
    return pd.DataFrame(rows)


def select_rows(results: pd.DataFrame, quality_gap_target: float, frontier_rate_target: float) -> pd.DataFrame:
    val = results[results["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_expanded_oracle"].le(quality_gap_target)
        & val["solver_frontier_call_rate"].le(frontier_rate_target)
    ].copy()
    if feasible.empty:
        feasible = val[val["solver_frontier_call_rate"].le(frontier_rate_target)].copy()
        feasible["selection_status"] = "no_validation_quality_feasible_under_frontier_cap"
    else:
        feasible["selection_status"] = "validation_feasible"
    if feasible.empty:
        feasible = val.copy()
        feasible["selection_status"] = "no_validation_frontier_feasible"
    selected = feasible.sort_values(["solver_utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False).head(5)
    test = results[results["split"].eq("test")].copy()
    return selected.merge(test, on="method", how="left", suffixes=("_val", "_test"))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        for value in row:
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    eval_table = table[table["split"].astype(str).isin(splits)].copy()
    if args.max_rows:
        eval_table = eval_table.head(int(args.max_rows)).copy()
    api_key = resolve_key(load_env_values(args.env_file), ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("OpenAI API key not found.")
    scores = collect_scores(
        eval_table,
        output_dir,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    scores_path = output_dir / "table_gpt_rescue_scores.csv"
    scores.to_csv(scores_path, index=False)
    merged = table.merge(scores, on="query_id", how="left")
    merged_path = output_dir / "query_table_with_gpt_rescue_scores.csv"
    merged.to_csv(merged_path, index=False)
    results = evaluate(merged[merged["split"].eq("train") | merged["rescue_status"].notna()].copy(), args.lambda_cost, args.frontier_rate_target)
    selected = select_rows(results, args.quality_gap_target, args.frontier_rate_target)
    results_path = output_dir / "table_gpt_rescue_score_gate.csv"
    selected_path = output_dir / "table_gpt_rescue_score_selected.csv"
    results.to_csv(results_path, index=False)
    selected.to_csv(selected_path, index=False)
    best_test = results[results["split"].eq("test")].sort_values(
        ["solver_utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False
    ).head(10)
    score_summary = scores.groupby("rescue_status").agg(
        n=("query_id", "size"),
        route_cost=("rescue_route_cost", "sum"),
        mean_p_local=("p_local_correct", "mean"),
        mean_p_gemini=("p_gemini_correct", "mean"),
        mean_p_gpt=("p_gpt_correct", "mean"),
        mean_p_hopeless=("p_hopeless", "mean"),
    ).reset_index()
    memo_path = output_dir / "GPT_RESCUE_SCORE_GATE_MEMO.md"
    memo = [
        "# GPT Rescue Score Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        "GPT-5.5 is used as a route scorer, not as the final solver during scoring collection. It sees query text, local candidate answers, and Gemini's cached answer, but no gold answer or GPT solver answer.",
        "The report separates solver frontier-call rate from total remote call rate including the GPT route scorer.",
        "",
        "## Route Score Summary",
        "",
        markdown_table(score_summary),
        "",
        "## Validation-Selected Policies",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Test Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Files",
        "",
        f"- `{scores_path}`",
        f"- `{merged_path}`",
        f"- `{results_path}`",
        f"- `{selected_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
