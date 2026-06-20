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
    extract_gemini_text,
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    usage_from_openai,
)

JUDGE_MODEL = "gpt-5.5"
GEMINI_JUDGE_MODEL = "gemini-3.5-flash"
LOCAL_CANDIDATES = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
]
FRONTIER_CANDIDATES = ["gemini-3.5-flash", "gpt-5.5"]
ALL_CANDIDATES = [*LOCAL_CANDIDATES, *FRONTIER_CANDIDATES]
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00
GEMINI_INPUT_PER_MTOK = 1.50
GEMINI_OUTPUT_PER_MTOK = 9.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adjudicate broad100 cached candidate answers with GPT-5.5 or Gemini.")
    parser.add_argument("--outputs", type=Path, default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_answer_adjudicator"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--judge-model", choices=[JUDGE_MODEL, GEMINI_JUDGE_MODEL], default=JUDGE_MODEL)
    parser.add_argument(
        "--candidate-mode",
        choices=["local_only", "with_frontier_answers"],
        default="with_frontier_answers",
    )
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "medium"], default="none")
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
    if args.judge_model == JUDGE_MODEL:
        api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    else:
        api_key = resolve_key(env_values, ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise RuntimeError(f"Missing API key for {args.judge_model}.")

    adjudications = collect_adjudications(
        outputs,
        queries,
        args.output_dir,
        api_key=api_key,
        judge_model=args.judge_model,
        candidate_mode=args.candidate_mode,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
        package=package,
    )
    grid = evaluate_grid(outputs, adjudications, lambda_cost=args.lambda_cost, candidate_mode=args.candidate_mode, package=package)
    selected = select_val_row(grid)

    adjudications.to_csv(args.output_dir / "table_broad_answer_adjudications.csv", index=False)
    grid.to_csv(args.output_dir / "table_broad_answer_adjudicator_gate.csv", index=False)
    selected.to_csv(args.output_dir / "table_broad_answer_adjudicator_selected.csv", index=False)
    write_memo(
        args.output_dir / "BROAD_ANSWER_ADJUDICATOR_MEMO.md",
        args.outputs,
        adjudications,
        grid,
        selected,
        args.candidate_mode,
        args.reasoning_effort,
        args.judge_model,
    )
    print(f"Wrote broad answer adjudicator to {args.output_dir}")


def load_broad_package():
    path = Path("experiments/125_phase3_broad_target_method_package.py")
    spec = importlib.util.spec_from_file_location("broad_package", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_adjudications(
    outputs: pd.DataFrame,
    queries: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    judge_model: str,
    candidate_mode: str,
    max_output_tokens: int,
    reasoning_effort: str,
    max_api_spend_usd: float,
    concurrency: int,
    package,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_adjudicator" / candidate_mode
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_query = outputs.set_index(["query_id", "model_id"])
    prompts = [(row, prompt_for(row, by_query, candidate_mode=candidate_mode, package=package)) for _, row in queries.iterrows()]
    missing = [
        prompt
        for row, prompt in prompts
        if not (
            cache_dir
            / cache_name(
                str(row["query_id"]),
                candidate_mode=candidate_mode,
                reasoning_effort=reasoning_effort,
                judge_model=judge_model,
            )
        ).exists()
    ]
    estimated = estimate_prompt_cost(missing, max_output_tokens, judge_model)
    if estimated > float(max_api_spend_usd) + 1e-12:
        raise RuntimeError(
            f"Estimated uncached {judge_model} adjudicator spend ${estimated:.4f} exceeds cap ${float(max_api_spend_usd):.4f}."
        )
    print(f"Estimated uncached {judge_model} adjudicator spend: ${estimated:.4f}")
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [
            executor.submit(
                call_one,
                row,
                prompt,
                cache_dir,
                api_key,
                judge_model,
                candidate_mode,
                max_output_tokens,
                reasoning_effort,
            )
            for row, prompt in prompts
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"adjudicator rows {index}/{len(futures)}")
    return pd.DataFrame(rows)


def call_one(
    row: pd.Series,
    prompt: str,
    cache_dir: Path,
    api_key: str,
    judge_model: str,
    candidate_mode: str,
    max_output_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    query_id = str(row["query_id"])
    raw_path = cache_dir / cache_name(
        query_id,
        candidate_mode=candidate_mode,
        reasoning_effort=reasoning_effort,
        judge_model=judge_model,
    )
    cache_hit = raw_path.exists()
    started = time.time()
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        if judge_model == JUDGE_MODEL:
            request_payload: dict[str, Any] = {
                "model": JUDGE_MODEL,
                "input": prompt,
                "max_output_tokens": int(max_output_tokens),
                "text": {"verbosity": "low"},
                "reasoning": {"effort": reasoning_effort},
            }
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            try:
                payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 90.0)
            except urllib.error.HTTPError:
                request_payload.pop("text", None)
                payload = post_json("https://api.openai.com/v1/responses", request_payload, headers, 90.0)
        else:
            request_payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "maxOutputTokens": int(max_output_tokens),
                    "temperature": 0,
                    "thinkingConfig": {"thinkingBudget": 0},
                },
            }
            headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
            payload = post_json(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_JUDGE_MODEL}:generateContent",
                request_payload,
                headers,
                90.0,
            )
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = extract_openai_text(payload) if judge_model == JUDGE_MODEL else extract_gemini_text(payload)
    parsed = parse_selection(text, candidate_mode=candidate_mode)
    if judge_model == JUDGE_MODEL:
        input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
    else:
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        input_tokens = int(usage.get("promptTokenCount", max(1, len(prompt) // 4)) or 0)
        output_tokens = int(usage.get("candidatesTokenCount", max_output_tokens) or 0)
    return {
        "query_id": query_id,
        "split": str(row["split"]),
        "benchmark": str(row["benchmark"]),
        "selected_model": parsed["selected_model"],
        "selected_confidence": float(parsed["confidence"]),
        "selected_reason": parsed["reason"],
        "adjudicator_text": text,
        "adjudicator_input_tokens": int(input_tokens),
        "adjudicator_output_tokens": int(output_tokens),
        "adjudicator_cost": call_cost(int(input_tokens), int(output_tokens), judge_model),
        "adjudicator_latency_s": float(time.time() - started),
        "adjudicator_cache_hit": cache_hit,
        "adjudicator_raw_path": str(raw_path),
    }


def prompt_for(row: pd.Series, by_query: pd.DataFrame, *, candidate_mode: str, package) -> str:
    query_id = str(row["query_id"])
    candidate_ids = candidate_ids_for_mode(candidate_mode)
    lines: list[str] = []
    for model_id in candidate_ids:
        try:
            model_row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        if model_id == "deterministic_math_tool" and not package.deterministic_tool_choice(by_query, query_id):
            continue
        status = str(model_row.get("status", ""))
        answer = compact(str(model_row.get("parsed_answer", "")), 180)
        if status != "success":
            answer = "[unavailable]"
        lines.append(f"- {model_id}: status={status}; parsed_answer={answer or '[empty]'}")
    if not lines:
        lines.append("- no usable candidate answers")
    query = compact(str(row["query_text"]), 1700)
    return (
        "You are an answer adjudicator for exact-scored model routing.\n"
        "You see one query and cached candidate final answers. You do not see the gold answer.\n"
        "Choose the candidate model whose parsed answer is most likely to be exactly correct.\n"
        "You may solve briefly in private if that helps distinguish candidates, but return only JSON.\n"
        "For code tasks, parsed_answer=passed means the code passed hidden/unit tests; prefer a passed candidate.\n"
        "For multiple-choice tasks, choose the most credible letter. For math tasks, choose the most credible final value.\n"
        "If every candidate is unavailable or clearly wrong, choose HOPLESS.\n"
        "Return compact JSON only with keys selected_model, confidence, reason.\n"
        "selected_model must be one of: "
        + ", ".join([*candidate_ids, "HOPLESS"])
        + ". confidence is 0 to 1.\n\n"
        f"Benchmark: {row['benchmark']}\n"
        f"Domain: {row['domain']}\n"
        f"Metric: {row['metric']}\n"
        f"Query:\n{query}\n\n"
        "Candidate parsed answers:\n"
        + "\n".join(lines)
        + '\n\nExample: {"selected_model":"qwen3-4b-local","confidence":0.82,"reason":"short"}\n/no_think'
    )


def candidate_ids_for_mode(candidate_mode: str) -> list[str]:
    if candidate_mode == "local_only":
        return LOCAL_CANDIDATES
    if candidate_mode == "with_frontier_answers":
        return ALL_CANDIDATES
    raise ValueError(candidate_mode)


def parse_selection(text: str, *, candidate_mode: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    allowed = {*candidate_ids_for_mode(candidate_mode), "HOPLESS"}
    selected = str(parsed.get("selected_model", "")).strip()
    if selected not in allowed:
        for model_id in sorted(allowed, key=len, reverse=True):
            if model_id in clean:
                selected = model_id
                break
    if selected not in allowed:
        selected = "HOPLESS"
    try:
        confidence = float(parsed.get("confidence", np.nan))
    except (TypeError, ValueError):
        confidence = np.nan
    if np.isnan(confidence):
        confidence = 0.0
    return {
        "selected_model": selected,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason": compact(str(parsed.get("reason", "")), 220),
    }


def evaluate_grid(outputs: pd.DataFrame, adjudications: pd.DataFrame, *, lambda_cost: float, candidate_mode: str, package) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for threshold in [0.0, 0.4, 0.55, 0.7, 0.85, 0.95]:
        for fallback in ["observable_local_state_v5", "tool_probe_profile_v4", "qwen3-14b-awq-local"]:
            for split in ["val", "test"]:
                split_adj = adjudications[adjudications["split"].eq(split)].copy()
                if split_adj.empty:
                    continue
                selected = adjudicator_policy(outputs, split_adj, threshold=threshold, fallback=fallback, package=package)
                selected_rows = package.selected_to_rows(outputs, selected, split=split)
                if selected_rows.empty:
                    continue
                split_outputs = outputs[outputs["split"].eq(split)]
                cost_oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()]
                quality_oracle = split_outputs.loc[split_outputs.groupby("query_id")["quality_score"].idxmax()]
                row = package.evaluation_row(
                    f"answer_adjudicator_{candidate_mode}_t{threshold:g}_fb_{fallback}",
                    selected_rows,
                    cost_oracle,
                    quality_oracle,
                    lambda_cost=lambda_cost,
                )
                cost_columns = extra_cost_columns(outputs, split_adj, selected, selected_rows, lambda_cost=lambda_cost, candidate_mode=candidate_mode)
                row.update(cost_columns)
                row["candidate_mode"] = candidate_mode
                row["selector_confidence_threshold"] = threshold
                row["fallback_policy"] = fallback
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def adjudicator_policy(outputs: pd.DataFrame, adjudications: pd.DataFrame, *, threshold: float, fallback: str, package) -> pd.Series:
    split = str(adjudications["split"].iloc[0])
    if fallback == "observable_local_state_v5":
        fallback_selection = package.observable_local_state_selection(outputs, split=split)
    elif fallback == "tool_probe_profile_v4":
        fallback_selection = package.profile_v4_selection_for_split(outputs, split=split)
    else:
        fallback_selection = pd.Series(fallback, index=adjudications["query_id"].astype(str))
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for _, row in adjudications.iterrows():
        query_id = str(row["query_id"])
        model_id = str(row["selected_model"])
        confidence = float(row["selected_confidence"])
        if model_id == "HOPLESS" or confidence < threshold or not action_available(by_query, query_id, model_id, package):
            model_id = str(fallback_selection.get(query_id, "qwen3-14b-awq-local"))
        selected[query_id] = model_id
    return pd.Series(selected)


def action_available(by_query: pd.DataFrame, query_id: str, model_id: str, package) -> bool:
    if model_id == "deterministic_math_tool":
        return bool(package.deterministic_tool_choice(by_query, query_id))
    try:
        row = by_query.loc[(query_id, model_id)]
    except KeyError:
        return False
    return str(row.get("status", "success")) == "success"


def extra_cost_columns(
    outputs: pd.DataFrame,
    adjudications: pd.DataFrame,
    selected: pd.Series,
    selected_rows: pd.DataFrame,
    *,
    lambda_cost: float,
    candidate_mode: str,
) -> dict[str, Any]:
    gpt_cost_norm = max(
        float(outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )
    route_cost = adjudications.set_index("query_id").loc[selected.index, "adjudicator_cost"].fillna(0.0)
    route_norm_mean = float((route_cost / gpt_cost_norm).mean())
    full_probe_norm_mean = 0.0
    full_probe_frontier_rate = 0.0
    if candidate_mode == "with_frontier_answers":
        frontier_costs = (
            outputs[outputs["model_id"].isin(FRONTIER_CANDIDATES)]
            .pivot_table(index="query_id", values="cost_total_usd", aggfunc="sum")
            .reindex(selected.index)
            .fillna(0.0)["cost_total_usd"]
        )
        full_probe_norm_mean = float((frontier_costs / gpt_cost_norm).mean())
        full_probe_frontier_rate = 1.0
    selected_solver_norm_mean = float(selected_rows["normalized_remote_cost"].mean())
    selected_with_route_norm = selected_solver_norm_mean + route_norm_mean
    full_deployment_norm = full_probe_norm_mean + route_norm_mean if candidate_mode == "with_frontier_answers" else selected_with_route_norm
    mean_quality = float(selected_rows["quality_score"].mean())
    oracle_utility = float(
        outputs[outputs["split"].eq(str(selected_rows["split"].iloc[0]))]
        .loc[lambda frame: frame.groupby("query_id")["utility"].idxmax(), "utility"]
        .mean()
    )
    utility_with_route = mean_quality - lambda_cost * selected_with_route_norm
    utility_full_deployment = mean_quality - lambda_cost * full_deployment_norm
    return {
        "adjudicator_route_cost_total_usd": float(route_cost.sum()),
        "adjudicator_route_cost_norm_mean": route_norm_mean,
        "selected_solver_norm_cost_with_route": selected_with_route_norm,
        "utility_with_selected_solver_and_route_cost": utility_with_route,
        "utility_ratio_with_selected_solver_and_route_cost": utility_with_route / oracle_utility if oracle_utility else np.nan,
        "full_candidate_probe_norm_cost": full_probe_norm_mean,
        "full_candidate_probe_frontier_rate": full_probe_frontier_rate,
        "full_deployment_norm_cost": full_deployment_norm,
        "utility_with_full_candidate_probe_cost": utility_full_deployment,
        "utility_ratio_with_full_candidate_probe_cost": utility_full_deployment / oracle_utility if oracle_utility else np.nan,
    }


def select_val_row(grid: pd.DataFrame) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    if val.empty:
        return pd.DataFrame()
    picked = val.sort_values(["mean_utility", "mean_quality", "frontier_call_rate"], ascending=[False, False, True]).head(1)
    threshold = float(picked.iloc[0]["selector_confidence_threshold"])
    fallback = str(picked.iloc[0]["fallback_policy"])
    mode = str(picked.iloc[0]["candidate_mode"])
    return grid[
        grid["selector_confidence_threshold"].eq(threshold)
        & grid["fallback_policy"].eq(fallback)
        & grid["candidate_mode"].eq(mode)
    ].copy()


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int, judge_model: str) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return call_cost(input_tokens, output_tokens, judge_model)


def call_cost(input_tokens: int, output_tokens: int, judge_model: str) -> float:
    if judge_model == GEMINI_JUDGE_MODEL:
        input_per_mtok = GEMINI_INPUT_PER_MTOK
        output_per_mtok = GEMINI_OUTPUT_PER_MTOK
    else:
        input_per_mtok = INPUT_PER_MTOK
        output_per_mtok = OUTPUT_PER_MTOK
    return (float(input_tokens) / 1_000_000.0) * input_per_mtok + (
        float(output_tokens) / 1_000_000.0
    ) * output_per_mtok


def cache_name(query_id: str, *, candidate_mode: str, reasoning_effort: str, judge_model: str) -> str:
    digest = hashlib.sha1(f"{query_id}:{candidate_mode}:{judge_model}:{reasoning_effort}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_memo(
    path: Path,
    outputs_path: Path,
    adjudications: pd.DataFrame,
    grid: pd.DataFrame,
    selected: pd.DataFrame,
    candidate_mode: str,
    reasoning_effort: str,
    judge_model: str,
) -> None:
    note = (
        "Diagnostic mode: the judge saw cached GPT/Gemini candidate answers, so full deployment cost includes probing both frontier candidates."
        if candidate_mode == "with_frontier_answers"
        else "Local-only mode: the judge saw only local/tool candidate answers."
    )
    lines = [
        "# Broad100 Answer Adjudicator",
        "",
        f"Source outputs: `{outputs_path}`",
        f"Candidate mode: `{candidate_mode}`",
        f"Judge model: `{judge_model}`",
        f"Reasoning effort: `{reasoning_effort}`",
        note,
        "Claude is not used.",
        f"Adjudicator rows: `{len(adjudications)}`",
        f"Adjudicator route cost total: `${float(adjudications['adjudicator_cost'].sum()):.4f}`",
        "",
    ]
    if not selected.empty:
        lines.extend(["## Validation-Selected", "", "```csv", selected.to_csv(index=False).strip(), "```", ""])
    test = grid[grid["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)
    if not test.empty:
        lines.extend(["## Top Test Rows By Selected-Solver Utility", "", "```csv", test.to_csv(index=False).strip(), "```", ""])
    full = grid[grid["split"].eq("test")].sort_values(
        ["utility_with_full_candidate_probe_cost", "mean_quality"], ascending=False
    ).head(10)
    if not full.empty:
        lines.extend(["## Top Test Rows By Full Candidate-Probe Utility", "", "```csv", full.to_csv(index=False).strip(), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
