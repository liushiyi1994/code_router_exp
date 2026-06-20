from __future__ import annotations

import argparse
import hashlib
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
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    usage_from_openai,
)


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
GPT = "gpt-5.5"
ALL_MODELS = LOCAL_MODELS + [GEMINI, GPT]
INPUT_PER_MTOK = {
    GPT: 5.00,
    GEMINI: 1.50,
}
OUTPUT_PER_MTOK = {
    GPT: 30.00,
    GEMINI: 9.00,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a compact frontier judge to adjudicate cached exact-math candidate answers."
    )
    parser.add_argument(
        "--query-table",
        default="results/controlled/gpt_solver_cache_repair/query_table_expanded_local_pool_gpt_repaired.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/answer_adjudicator_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--judge-model", choices=[GPT, GEMINI], default=GEMINI)
    parser.add_argument(
        "--candidate-mode",
        choices=["no_gpt_answer", "with_gpt_answer"],
        default="with_gpt_answer",
        help="Whether the judge sees the cached GPT solver answer. with_gpt_answer is diagnostic, not low-cost deployable.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--max-api-spend-usd", type=float, default=2.00)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    return parser.parse_args()


def truncate(text: object, limit: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def answer_value(row: pd.Series, model_id: str) -> str:
    raw_col = f"{model_id}_answer"
    value = row.get(raw_col, "")
    return "" if pd.isna(value) else str(value)


def answer_lines(row: pd.Series, candidate_mode: str) -> list[str]:
    lines: list[str] = []
    for model_id in LOCAL_MODELS + [GEMINI]:
        lines.append(f"- {model_id}: {truncate(answer_value(row, model_id), 220) or '[empty]'}")
    if candidate_mode == "with_gpt_answer":
        lines.append(f"- {GPT}: {truncate(answer_value(row, GPT), 220) or '[empty]'}")
    return lines


def prompt_for(row: pd.Series, candidate_mode: str) -> str:
    allowed = LOCAL_MODELS + [GEMINI]
    if candidate_mode == "with_gpt_answer":
        allowed = allowed + [GPT]
    return f"""You are an exact-answer math answer adjudicator for model routing.

You see one problem and several model final answers. Choose the candidate answer most likely to be exactly correct.
Do not use any gold answer. Do not solve at length. Penalize answers that do not match the requested final-answer format.
If all candidates look wrong or hopeless, choose HOPLESS.

Return compact JSON only:
{{"selected_model":"{('|'.join(allowed))}|HOPLESS","confidence":0.0,"reason":"short phrase"}}

Dataset: {row['dataset']}
Problem:
{truncate(row['query_text'], 1700)}

Candidate final answers:
{chr(10).join(answer_lines(row, candidate_mode))}

JSON only."""


def estimate_cost(prompts: list[str], max_output_tokens: int, judge_model: str) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return input_tokens * (INPUT_PER_MTOK[judge_model] / 1_000_000) + output_tokens * (
        OUTPUT_PER_MTOK[judge_model] / 1_000_000
    )


def call_cost(input_tokens: int, output_tokens: int, judge_model: str) -> float:
    return input_tokens * (INPUT_PER_MTOK[judge_model] / 1_000_000) + output_tokens * (
        OUTPUT_PER_MTOK[judge_model] / 1_000_000
    )


def cache_name(query_id: str, *, judge_model: str, candidate_mode: str) -> str:
    digest = hashlib.sha1(f"{query_id}:{judge_model}:{candidate_mode}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def call_openai_judge(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 90.0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": GPT,
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


def call_gemini_judge(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 90.0) -> dict[str, Any]:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": int(max_output_tokens),
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_selection(text: object, allowed_models: set[str]) -> tuple[str, float, str]:
    raw = str(text or "")
    payload: dict[str, Any] = {}
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = {}
    selected = str(payload.get("selected_model", "")).strip()
    if selected not in allowed_models and selected != "HOPLESS":
        for candidate in sorted(allowed_models, key=len, reverse=True):
            if candidate in raw:
                selected = candidate
                break
    if selected not in allowed_models and selected != "HOPLESS":
        selected = "HOPLESS"
    try:
        confidence = float(payload.get("confidence", np.nan))
    except (TypeError, ValueError):
        confidence = np.nan
    if np.isnan(confidence):
        confidence = 0.0
    reason = truncate(payload.get("reason", ""), 180)
    return selected, float(np.clip(confidence, 0.0, 1.0)), reason


def collect_judgments(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    judge_model: str,
    candidate_mode: str,
    api_key: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_adjudications" / judge_model / candidate_mode
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = [prompt_for(row, candidate_mode) for _, row in frame.iterrows()]
    missing = [
        prompt
        for prompt, (_, row) in zip(prompts, frame.iterrows())
        if not (cache_dir / cache_name(str(row["query_id"]), judge_model=judge_model, candidate_mode=candidate_mode)).exists()
    ]
    estimated = estimate_cost(missing, max_output_tokens, judge_model)
    if estimated > max_api_spend_usd:
        raise RuntimeError(
            f"Estimated uncached {judge_model} adjudication spend ${estimated:.4f} exceeds cap ${max_api_spend_usd:.4f}."
        )

    allowed_models = set(LOCAL_MODELS + [GEMINI])
    if candidate_mode == "with_gpt_answer":
        allowed_models.add(GPT)

    def one(row: pd.Series, prompt: str) -> dict[str, Any]:
        query_id = str(row["query_id"])
        raw_path = cache_dir / cache_name(query_id, judge_model=judge_model, candidate_mode=candidate_mode)
        cache_hit = raw_path.exists()
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            try:
                if judge_model == GPT:
                    payload = call_openai_judge(prompt, api_key, max_output_tokens)
                else:
                    payload = call_gemini_judge(prompt, api_key, max_output_tokens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - started
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row_status = str(payload.get("_status", status))
        if row_status == "success" and judge_model == GPT:
            text = extract_openai_text(payload)
            input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
        elif row_status == "success":
            text = extract_gemini_text(payload)
            usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
            input_tokens = int(usage.get("promptTokenCount", max(1, len(prompt) // 4)) or 0)
            output_tokens = int(usage.get("candidatesTokenCount", max_output_tokens) or 0)
        else:
            text = ""
            input_tokens = 0
            output_tokens = 0
        selected, confidence, reason = parse_selection(text, allowed_models)
        return {
            "query_id": query_id,
            "adjudicator_status": row_status,
            "adjudicator_error_type": str(payload.get("_error_type", error_type)),
            "adjudicator_text": text,
            "selected_model": selected,
            "selected_confidence": confidence,
            "selected_reason": reason,
            "adjudicator_input_tokens": int(input_tokens),
            "adjudicator_output_tokens": int(output_tokens),
            "adjudicator_latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
            "adjudicator_cache_hit": cache_hit,
            "adjudicator_raw_path": str(raw_path),
            "adjudicator_cost": call_cost(int(input_tokens), int(output_tokens), judge_model),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row, prompt) for (_, row), prompt in zip(frame.iterrows(), prompts)]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def add_oracle_columns(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    table = table.copy()
    cost_norm = max(float(pd.to_numeric(table[f"{GPT}_cost"], errors="coerce").fillna(0.0).mean()), 1e-12)
    utility_cols: list[str] = []
    quality_cols: list[str] = []
    for model_id in ALL_MODELS:
        quality_col = f"{model_id}_quality"
        if quality_col not in table.columns:
            continue
        quality_cols.append(quality_col)
        cost_col = f"{model_id}_cost"
        utility_col = f"{model_id}_adjudicator_utility"
        if model_id in LOCAL_MODELS:
            table[utility_col] = table[quality_col].astype(float)
        else:
            table[utility_col] = table[quality_col].astype(float) - lambda_cost * (
                table[cost_col].astype(float) / cost_norm
            )
        utility_cols.append(utility_col)
    table["adjudicator_quality_oracle"] = table[quality_cols].max(axis=1)
    table["adjudicator_cost_oracle_utility"] = table[utility_cols].max(axis=1)
    return table


def dataset_local_fallback(train: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    for dataset, frame in train.groupby("dataset"):
        means = {model_id: float(frame[f"{model_id}_quality"].mean()) for model_id in LOCAL_MODELS}
        out[str(dataset)] = max(means, key=means.get)
    return out


def actions_for(frame: pd.DataFrame, method: str, local_fallback: dict[str, str]) -> pd.Series:
    selected = frame["selected_model"].where(frame["selected_model"].isin(ALL_MODELS), "")
    fallback = frame["dataset"].map(lambda value: local_fallback.get(str(value), "qwen3-8b-local"))
    if method == "argmax":
        return selected.mask(selected.eq(""), fallback)
    if method.startswith("confidence_"):
        threshold = float(method.split("_", 1)[1])
        actions = selected.mask(selected.eq(""), fallback)
        return actions.where(frame["selected_confidence"].astype(float).ge(threshold), fallback)
    raise ValueError(method)


def evaluate_actions(frame: pd.DataFrame, actions: pd.Series, *, lambda_cost: float, include_all_visible_cost: bool) -> dict[str, Any]:
    qualities: list[float] = []
    solver_costs: list[float] = []
    total_costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        if action not in ALL_MODELS:
            action = "qwen3-8b-local"
        quality = float(row[f"{action}_quality"])
        if include_all_visible_cost:
            solver_cost = float(row[f"{GEMINI}_cost"]) + float(row[f"{GPT}_cost"])
            gpt = True
            gemini = True
        elif action in LOCAL_MODELS:
            solver_cost = 0.0
            gpt = False
            gemini = False
        elif action == GEMINI:
            solver_cost = float(row[f"{GEMINI}_cost"])
            gpt = False
            gemini = True
        else:
            solver_cost = float(row[f"{GPT}_cost"])
            gpt = True
            gemini = False
        route_cost = float(row.get("adjudicator_cost", 0.0) or 0.0)
        qualities.append(quality)
        solver_costs.append(solver_cost)
        total_costs.append(solver_cost + route_cost)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
    all_gpt_cost = max(float(frame[f"{GPT}_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    solver_cost_norm = float(np.sum(solver_costs) / all_gpt_cost)
    total_cost_norm = float(np.sum(total_costs) / all_gpt_cost)
    solver_utility = float(mean_quality - lambda_cost * solver_cost_norm)
    total_utility = float(mean_quality - lambda_cost * total_cost_norm)
    oracle_utility = float(frame["adjudicator_cost_oracle_utility"].mean())
    return {
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_expanded_oracle": float(frame["adjudicator_quality_oracle"].mean() - mean_quality),
        "solver_utility": solver_utility,
        "total_utility_with_route_cost": total_utility,
        "solver_utility_ratio_to_expanded_cost_oracle": float(solver_utility / oracle_utility) if oracle_utility else np.nan,
        "total_utility_ratio_to_expanded_cost_oracle": float(total_utility / oracle_utility) if oracle_utility else np.nan,
        "normalized_solver_cost_vs_all_gpt": solver_cost_norm,
        "normalized_total_cost_vs_all_gpt": total_cost_norm,
        "route_cost_total_usd": float(frame["adjudicator_cost"].sum()),
        "solver_frontier_call_rate": float(np.mean([a or b for a, b in zip(gpt_calls, gemini_calls)])),
        "gpt_solver_call_rate": float(np.mean(gpt_calls)),
        "gemini_solver_call_rate": float(np.mean(gemini_calls)),
        "action_counts": json.dumps({str(k): int(v) for k, v in actions.value_counts().to_dict().items()}, sort_keys=True),
    }


def evaluate(table: pd.DataFrame, *, lambda_cost: float, candidate_mode: str) -> pd.DataFrame:
    table = add_oracle_columns(table, lambda_cost)
    local_fallback = dataset_local_fallback(table[table["split"].eq("train")])
    rows: list[dict[str, Any]] = []
    include_all_visible_cost = candidate_mode == "with_gpt_answer"
    for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
        for method in ["argmax", "confidence_0.50", "confidence_0.65", "confidence_0.80"]:
            actions = actions_for(frame, method, local_fallback)
            row = evaluate_actions(
                frame,
                actions,
                lambda_cost=lambda_cost,
                include_all_visible_cost=include_all_visible_cost,
            )
            row["method"] = f"answer_adjudicator_{method}"
            row["candidate_mode"] = candidate_mode
            row["all_visible_solver_cost_accounting"] = include_all_visible_cost
            rows.append(row)
    return pd.DataFrame(rows)


def select_rows(results: pd.DataFrame, *, quality_gap_target: float, frontier_rate_target: float) -> pd.DataFrame:
    val = results[results["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_expanded_oracle"].le(quality_gap_target)
        & val["solver_frontier_call_rate"].le(frontier_rate_target)
    ].copy()
    if feasible.empty:
        feasible = val[val["quality_gap_to_expanded_oracle"].le(quality_gap_target)].copy()
        feasible["selection_status"] = "quality_feasible_but_frontier_over_cap_or_diagnostic"
    else:
        feasible["selection_status"] = "validation_feasible"
    if feasible.empty:
        feasible = val.copy()
        feasible["selection_status"] = "no_validation_quality_feasible"
    selected = feasible.sort_values(["total_utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False).head(5)
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

    env = load_env_values(args.env_file)
    if args.judge_model == GPT:
        api_key = resolve_key(env, ["OPENAI_API_KEY", "openai_api_key"])
    else:
        api_key = resolve_key(env, ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise RuntimeError(f"{args.judge_model} API key not found.")

    judgments = collect_judgments(
        eval_table,
        output_dir,
        judge_model=args.judge_model,
        candidate_mode=args.candidate_mode,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    judgments_path = output_dir / "table_answer_adjudications.csv"
    judgments.to_csv(judgments_path, index=False)

    merged = table.merge(judgments, on="query_id", how="left")
    merged_path = output_dir / "query_table_with_answer_adjudications.csv"
    merged.to_csv(merged_path, index=False)
    results = evaluate(
        merged[merged["split"].eq("train") | merged["adjudicator_status"].notna()].copy(),
        lambda_cost=args.lambda_cost,
        candidate_mode=args.candidate_mode,
    )
    selected = select_rows(
        results,
        quality_gap_target=args.quality_gap_target,
        frontier_rate_target=args.frontier_rate_target,
    )
    results_path = output_dir / "table_answer_adjudicator_gate.csv"
    selected_path = output_dir / "table_answer_adjudicator_selected.csv"
    results.to_csv(results_path, index=False)
    selected.to_csv(selected_path, index=False)

    status_summary = judgments.groupby("adjudicator_status").agg(
        n=("query_id", "size"),
        route_cost=("adjudicator_cost", "sum"),
        mean_confidence=("selected_confidence", "mean"),
    ).reset_index()
    best_test = results[results["split"].eq("test")].sort_values(
        ["mean_quality", "total_utility_ratio_to_expanded_cost_oracle"], ascending=False
    ).head(8)
    note = (
        "Diagnostic: the judge saw the GPT solver answer, so solver-cost accounting charges both Gemini and GPT "
        "for every adjudicated row. This is not a low-frontier deployable router."
        if args.candidate_mode == "with_gpt_answer"
        else "Deployable-style: the judge did not see the GPT solver answer, so selected-solver accounting is meaningful."
    )
    memo_path = output_dir / "ANSWER_ADJUDICATOR_GATE_MEMO.md"
    memo = [
        "# Answer Adjudicator Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Judge model: `{args.judge_model}`.",
        f"Candidate mode: `{args.candidate_mode}`.",
        note,
        "The judge prompt never includes gold answers.",
        "",
        "## Adjudication Summary",
        "",
        markdown_table(status_summary),
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Test Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Files",
        "",
        f"- `{judgments_path}`",
        f"- `{merged_path}`",
        f"- `{results_path}`",
        f"- `{selected_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
