from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_gemini_text,
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    score_output,
    usage_from_openai,
)


GPT_MODEL = "gpt-5.5"
GEMINI_MODEL = "gemini-3.5-flash"
VERIFIER_MODEL_ID = "gemini-3.5-flash-task-verifier"
GPT_INPUT_PER_MTOK = 5.00
GPT_OUTPUT_PER_MTOK = 30.00
GEMINI_INPUT_PER_MTOK = 1.50
GEMINI_OUTPUT_PER_MTOK = 9.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task-specific verifier-answer action for broad100 hard slices.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--benchmark-composed-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_tool_aware_benchmark_composed_policy/"
            "table_tool_aware_benchmark_composed_choices.csv"
        ),
    )
    parser.add_argument("--benchmark-composed-method", default="tool_aware_benchmark_composed_eps0.01_recall_then_quality")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_task_specific_verifier_action"),
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--provider-model", choices=[GEMINI_MODEL, GPT_MODEL], default=GEMINI_MODEL)
    parser.add_argument("--benchmarks", default="gpqa,mmlupro,aime,livemathbench,math500")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=160)
    parser.add_argument("--max-api-spend-usd", type=float, default=1.50)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_181")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_181")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_181")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_181")
    exp179 = load_module("experiments/179_cached_adjudicator_blend_policy.py", "adjudicator_blend_179_for_181")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(target, args.benchmark_composed_choices, args.benchmark_composed_method, exp171)
    benchmarks = {item.strip().lower() for item in args.benchmarks.split(",") if item.strip()}
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    query_frame = (
        outputs[outputs["split"].astype(str).isin(splits) & outputs["benchmark"].astype(str).str.lower().isin(benchmarks)]
        .drop_duplicates("query_id")
        .copy()
    )
    api_key = resolve_provider_key(args.provider_model, args.env_file)
    verifier = collect_verifier_rows(
        query_frame,
        outputs,
        args.output_dir,
        api_key=api_key,
        provider_model=args.provider_model,
        max_output_tokens=int(args.max_output_tokens),
        max_api_spend_usd=float(args.max_api_spend_usd),
        concurrency=int(args.concurrency),
        force_rerun=bool(args.force_rerun),
    )
    augmented = append_verifier_rows(outputs, verifier, lambda_cost=float(args.lambda_cost))
    base_choices = practical_base_choices(target, outputs, exp172, exp175, exp177, exp179)
    policy_internal, query_choices = evaluate_policies(
        base_choices,
        verifier,
        augmented,
        target,
        exp172,
        lambda_cost=float(args.lambda_cost),
    )
    selected = selected_rows(policy_internal, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    policy_table = exp172.add_bootstrap_ci(policy_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    policy_table = policy_table.drop(columns=["_utility_values"], errors="ignore")
    selected = selected.drop(columns=["_utility_values"], errors="ignore")

    verifier.to_csv(args.output_dir / "table_task_specific_verifier_outputs.csv", index=False)
    augmented.to_parquet(args.output_dir / "model_outputs_with_task_verifier.parquet", index=False)
    policy_table.to_csv(args.output_dir / "table_task_specific_verifier_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_task_specific_verifier_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_task_specific_verifier_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "TASK_SPECIFIC_VERIFIER_ACTION_MEMO.md", args, verifier, policy_table, selected)
    print(f"Wrote task-specific verifier action results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolve_provider_key(provider_model: str, env_file: str) -> str:
    env = load_env_values(env_file)
    if provider_model == GPT_MODEL:
        key = resolve_key(env, ["OPENAI_API_KEY", "openai_api_key"])
    else:
        key = resolve_key(env, ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not key:
        raise RuntimeError(f"Missing API key for {provider_model}.")
    return key


def collect_verifier_rows(
    queries: pd.DataFrame,
    outputs: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    provider_model: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
    force_rerun: bool,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_task_verifier" / provider_model / f"max_{max_output_tokens}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    by_query = outputs.groupby("query_id", sort=False)
    output_map = {str(query_id): group.copy() for query_id, group in by_query}
    prompt_rows = [(row, prompt_for(row, output_map.get(str(row["query_id"]), pd.DataFrame()))) for _, row in queries.iterrows()]
    missing = [
        prompt
        for row, prompt in prompt_rows
        if force_rerun or not (cache_dir / cache_name(str(row["query_id"]), provider_model, max_output_tokens)).exists()
    ]
    estimated = estimate_cost(missing, max_output_tokens, provider_model)
    if estimated > float(max_api_spend_usd) + 1e-12:
        raise RuntimeError(
            f"Estimated uncached {provider_model} verifier spend ${estimated:.4f} exceeds cap ${float(max_api_spend_usd):.4f}."
        )
    print(f"Estimated uncached {provider_model} task-verifier spend: ${estimated:.4f}")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [
            executor.submit(
                call_one,
                row,
                prompt,
                cache_dir,
                api_key,
                provider_model,
                max_output_tokens,
                force_rerun,
            )
            for row, prompt in prompt_rows
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"task verifier rows {index}/{len(futures)}")
    return pd.DataFrame(rows)


def call_one(
    row: pd.Series,
    prompt: str,
    cache_dir: Path,
    api_key: str,
    provider_model: str,
    max_output_tokens: int,
    force_rerun: bool,
) -> dict[str, Any]:
    query_id = str(row["query_id"])
    raw_path = cache_dir / cache_name(query_id, provider_model, max_output_tokens)
    cache_hit = raw_path.exists() and not force_rerun
    started = time.time()
    status = "success"
    error_type = ""
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        try:
            payload = call_provider_with_retries(prompt, api_key, provider_model, max_output_tokens)
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            payload = {"error": str(exc)[:1000], "error_type": error_type}
        payload["_status"] = status
        payload["_error_type"] = error_type
        payload["_latency_s"] = time.time() - started
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = extract_openai_text(payload) if provider_model == GPT_MODEL else extract_gemini_text(payload)
    parsed_payload = parse_verifier_json(text)
    answer_text = parsed_payload.get("answer", text)
    parsed_answer, quality = score_output(str(answer_text), str(row["gold_answer"]), str(row["metric"]))
    if str(payload.get("_status", status)) != "success":
        quality = np.nan
    if str(payload.get("_status", status)) == "success":
        input_tokens, output_tokens = usage_tokens(payload, prompt, max_output_tokens, provider_model)
    else:
        input_tokens, output_tokens = 0, 0
    return {
        "query_id": query_id,
        "split": str(row["split"]),
        "benchmark": str(row["benchmark"]),
        "domain": str(row["domain"]),
        "metric": str(row["metric"]),
        "query_text": str(row["query_text"]),
        "gold_answer": str(row["gold_answer"]),
        "provider_model": provider_model,
        "status": str(payload.get("_status", status)),
        "error_type": str(payload.get("_error_type", error_type)),
        "raw_text": text,
        "verifier_answer": str(answer_text),
        "parsed_answer": parsed_answer,
        "quality_score": float(quality) if not pd.isna(quality) else np.nan,
        "verifier_confidence": float(parsed_payload.get("confidence", 0.0)),
        "supported_model": str(parsed_payload.get("supported_model", "")),
        "verifier_reason": str(parsed_payload.get("reason", ""))[:240],
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_total_usd": call_cost(int(input_tokens), int(output_tokens), provider_model),
        "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
        "cache_hit": bool(cache_hit),
        "raw_output_path": str(raw_path),
    }


def call_provider(prompt: str, api_key: str, provider_model: str, max_output_tokens: int) -> dict[str, Any]:
    if provider_model == GPT_MODEL:
        payload: dict[str, Any] = {
            "model": GPT_MODEL,
            "input": prompt,
            "max_output_tokens": int(max_output_tokens),
            "text": {"verbosity": "low"},
            "reasoning": {"effort": "minimal"},
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            return post_json("https://api.openai.com/v1/responses", payload, headers, 120.0)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            payload.pop("text", None)
        try:
            return post_json("https://api.openai.com/v1/responses", payload, headers, 120.0)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            payload.pop("reasoning", None)
            return post_json("https://api.openai.com/v1/responses", payload, headers, 120.0)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": int(max_output_tokens),
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    return post_json(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent", payload, headers, 120.0)


def call_provider_with_retries(prompt: str, api_key: str, provider_model: str, max_output_tokens: int) -> dict[str, Any]:
    delays = [0.0, 2.0, 5.0, 10.0]
    last_exc: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            return call_provider(prompt, api_key, provider_model, max_output_tokens)
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                raise
        except Exception as exc:
            last_exc = exc
            raise
    assert last_exc is not None
    raise last_exc


def prompt_for(row: pd.Series, candidates: pd.DataFrame) -> str:
    query = compact(str(row["query_text"]), 2300)
    metric = str(row.get("metric", ""))
    benchmark = str(row.get("benchmark", ""))
    if metric == "multiple_choice":
        task_instruction = "Solve the multiple-choice task. The answer must be a single option letter."
    elif benchmark in {"aime", "math500", "livemathbench", "gsm8k"}:
        task_instruction = "Solve the math task. The answer must be the final exact value only."
    else:
        task_instruction = "Solve the exact-scored task. The answer must be the final exact answer only."
    lines: list[str] = []
    keep_models = [
        "deterministic_math_tool",
        "qwen3-4b-local",
        "qwen3-8b-local",
        "qwen3-14b-awq-local",
        "qwen3-32b-awq-local",
        "qwen3-32b-awq-selfconsistency-n3-local",
        "gemini-3.5-flash",
        "gpt-5.5",
        "gemini-3.5-flash-strong-solve",
    ]
    for model_id in keep_models:
        group = candidates[candidates["model_id"].astype(str).eq(model_id)]
        if group.empty:
            continue
        item = group.iloc[0]
        answer = compact(str(item.get("parsed_answer", "")), 160)
        if not answer or answer.lower() in {"nan", "none"}:
            answer = "[empty]"
        lines.append(f"- {model_id}: {answer}")
    return (
        "You are a task-specific verifier for a model router.\n"
        "You must solve independently; candidate answers are hints and may be wrong.\n"
        f"{task_instruction}\n"
        "Return compact JSON only with keys: answer, confidence, supported_model, reason.\n"
        "confidence is 0 to 1. supported_model is the candidate model whose answer you trust most, or NONE.\n\n"
        f"Benchmark: {benchmark}\n"
        f"Metric: {metric}\n"
        f"Task:\n{query}\n\n"
        "Candidate final answers:\n"
        + "\n".join(lines)
        + '\n\nExample: {"answer":"B","confidence":0.74,"supported_model":"qwen3-32b-awq-local","reason":"short"}\n/no_think'
    )


def parse_verifier_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    answer = str(parsed.get("answer", "")).strip() or clean
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer": answer,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "supported_model": str(parsed.get("supported_model", "")),
        "reason": compact(str(parsed.get("reason", "")), 220),
    }


def append_verifier_rows(outputs: pd.DataFrame, verifier: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    gpt_cost = mean_gpt_cost(outputs)
    rows: list[dict[str, Any]] = []
    for row in verifier.itertuples(index=False):
        cost = float(row.cost_total_usd)
        normalized = cost / max(gpt_cost, 1e-12)
        quality = 0.0 if pd.isna(row.quality_score) else float(row.quality_score)
        rows.append(
            {
                "run_id": "task_specific_verifier_action",
                "query_id": str(row.query_id),
                "query_text": str(row.query_text),
                "benchmark": str(row.benchmark),
                "domain": str(row.domain),
                "model_id": VERIFIER_MODEL_ID,
                "provider": str(row.provider_model),
                "is_local": False,
                "is_frontier": True,
                "is_probe": True,
                "prompt_template_version": "task_specific_verifier_v1",
                "prompt_hash": "",
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
                "max_output_tokens": np.nan,
                "start_time_unix": np.nan,
                "end_time_unix": np.nan,
                "latency_s": float(row.latency_s),
                "model_load_time_s": 0.0,
                "warmup_time_s": 0.0,
                "latency_excludes_load_warmup": True,
                "load_mode": "api",
                "status": str(row.status),
                "error_type": str(row.error_type),
                "raw_output_path": str(row.raw_output_path),
                "parsed_answer": str(row.parsed_answer),
                "gold_answer": str(row.gold_answer),
                "quality_score": quality,
                "cost_input_usd": np.nan,
                "cost_output_usd": np.nan,
                "cost_total_usd": cost,
                "cache_hit": bool(row.cache_hit),
                "server_backend": "api",
                "server_config_json": "{}",
                "hardware_id": "",
                "metric": str(row.metric),
                "rank_in_benchmark": np.nan,
                "split": str(row.split),
                "normalized_remote_cost": normalized,
                "utility": quality - float(lambda_cost) * normalized,
                "tool_available": False,
            }
        )
    appended = pd.concat([outputs, pd.DataFrame(rows)], ignore_index=True, sort=False)
    appended["quality_score"] = pd.to_numeric(appended["quality_score"], errors="coerce").fillna(0.0)
    appended["normalized_remote_cost"] = pd.to_numeric(appended["normalized_remote_cost"], errors="coerce").fillna(0.0)
    appended["utility"] = appended["quality_score"].astype(float) - float(lambda_cost) * appended["normalized_remote_cost"].astype(float)
    return appended


def practical_base_choices(target: pd.DataFrame, outputs: pd.DataFrame, exp172, exp175, exp177, exp179) -> pd.DataFrame:
    priors = exp172.fit_train_priors(outputs)
    feature_frame, cat_cols, num_cols = exp177.build_feature_frame(outputs, target)
    choices = exp179.fit_base_choices(exp177, exp172, exp175, feature_frame, target, outputs, priors, cat_cols, num_cols)
    base = choices["hgb_l1_gate_rank_localplus_pen0.25"].copy()
    base = base.rename(columns={"base_model_id": "model_id"})
    return base[["query_id", "split", "model_id"]]


def evaluate_policies(
    base_choices: pd.DataFrame,
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    detail: list[pd.DataFrame] = []
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    verifier_map = verifier.set_index("query_id").to_dict("index")
    gpt_cost = mean_gpt_cost(outputs)
    thresholds = [0.0, 0.5, 0.7, 0.85, 0.95]
    methods: list[tuple[str, float, str]] = [("base_candidate_ranker", np.nan, "base")]
    methods.append(("always_task_verifier_on_covered", 0.0, "always"))
    for threshold in thresholds:
        methods.append((f"task_verifier_conf_ge_{threshold:g}", threshold, "confidence"))
        methods.append((f"task_verifier_disagree_conf_ge_{threshold:g}", threshold, "disagree_confidence"))
    for method, threshold, mode in methods:
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            choice = choose_policy(base_choices[base_choices["split"].eq(split)].copy(), verifier_map, threshold, mode)
            selected_rows = choice[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                method,
                "task_specific_verifier_policy" if mode != "base" else "reference",
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            route_cost_norm = extra_probe_norm(choice, verifier_map, gpt_cost)
            route_utilities = selected_rows["quality_score"].to_numpy(dtype=float) - float(lambda_cost) * (
                selected_rows["normalized_remote_cost"].to_numpy(dtype=float) + route_cost_norm
            )
            row["probe_call_rate"] = float(choice["verifier_probed"].mean()) if not choice.empty else 0.0
            row["verifier_select_rate"] = float(choice["model_id"].astype(str).eq(VERIFIER_MODEL_ID).mean()) if not choice.empty else 0.0
            row["extra_probe_norm_cost_mean"] = float(np.mean(route_cost_norm)) if len(route_cost_norm) else 0.0
            row["mean_utility_with_probe_cost"] = float(np.mean(route_utilities)) if len(route_utilities) else np.nan
            row["oracle_utility_ratio_with_probe_cost"] = float(row["mean_utility_with_probe_cost"] / max(float(row["cost_oracle_mean_utility"]), 1e-12))
            row["_utility_values_with_probe_cost"] = route_utilities.tolist()
            rows.append(row)
            if split == "test":
                detail.append(
                    selected_rows[
                        ["query_id", "query_text", "benchmark", "metric", "model_id", "quality_score", "utility", "normalized_remote_cost", "is_frontier", "parsed_answer"]
                    ].merge(choice[["query_id", "base_model_id", "verifier_probed", "verifier_confidence"]], on="query_id", how="left").assign(method=method)
                )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False]), (
        pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    )


def choose_policy(base: pd.DataFrame, verifier_map: dict[str, dict[str, Any]], threshold: float, mode: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        query_id = str(row.query_id)
        base_model = str(row.model_id)
        item = verifier_map.get(query_id)
        use_verifier = False
        confidence = 0.0
        if item is not None and str(item.get("status", "")) == "success":
            confidence = float(item.get("verifier_confidence", 0.0) or 0.0)
            if mode == "always":
                use_verifier = True
            elif mode == "confidence":
                use_verifier = confidence >= float(threshold)
            elif mode == "disagree_confidence":
                base_answer = ""
                supported = str(item.get("supported_model", ""))
                use_verifier = confidence >= float(threshold) and supported and supported != base_model
        model_id = VERIFIER_MODEL_ID if use_verifier else base_model
        verifier_probed = item is not None and mode != "base"
        rows.append(
            {
                "query_id": query_id,
                "split": str(row.split),
                "model_id": model_id,
                "base_model_id": base_model,
                "verifier_probed": verifier_probed,
                "verifier_confidence": confidence,
            }
        )
    return pd.DataFrame(rows)


def extra_probe_norm(choice: pd.DataFrame, verifier_map: dict[str, dict[str, Any]], gpt_cost: float) -> np.ndarray:
    costs: list[float] = []
    for row in choice.itertuples(index=False):
        item = verifier_map.get(str(row.query_id))
        if item is None or not bool(row.verifier_probed) or str(row.model_id) == VERIFIER_MODEL_ID:
            costs.append(0.0)
        else:
            costs.append(float(item.get("cost_total_usd", 0.0) or 0.0) / max(gpt_cost, 1e-12))
    return np.asarray(costs, dtype=float)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility", "mean_utility_with_probe_cost"]:
        val = table[table["split"].eq("val") & table["family"].ne("reference")].copy()
        if val.empty:
            continue
        best = val.sort_values([objective, "frontier_call_rate"], ascending=[False, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule=f"val_best_{objective}"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule=f"val_best_{objective}_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(
        table[["method", "split", "_utility_values"]],
        on=["method", "split"],
        how="left",
    )
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)


def usage_tokens(payload: dict[str, Any], prompt: str, max_output_tokens: int, provider_model: str) -> tuple[int, int]:
    if provider_model == GPT_MODEL:
        return usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
    usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
    return int(usage.get("promptTokenCount", max(1, len(prompt) // 4)) or 0), int(
        (usage.get("candidatesTokenCount", max_output_tokens) or 0) + (usage.get("thoughtsTokenCount", 0) or 0)
    )


def estimate_cost(prompts: list[str], max_output_tokens: int, provider_model: str) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return call_cost(input_tokens, output_tokens, provider_model)


def call_cost(input_tokens: int, output_tokens: int, provider_model: str) -> float:
    if provider_model == GPT_MODEL:
        return input_tokens * GPT_INPUT_PER_MTOK / 1_000_000 + output_tokens * GPT_OUTPUT_PER_MTOK / 1_000_000
    return input_tokens * GEMINI_INPUT_PER_MTOK / 1_000_000 + output_tokens * GEMINI_OUTPUT_PER_MTOK / 1_000_000


def cache_name(query_id: str, provider_model: str, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{provider_model}:{max_output_tokens}:task_specific_v1".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def mean_gpt_cost(outputs: pd.DataFrame) -> float:
    return max(
        float(outputs[outputs["model_id"].astype(str).eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    labels = plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#7b6857")
    ax.set_xlabel("Held-out test selected-solver utility")
    ax.set_title("Task-Specific Verifier Action")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_task_specific_verifier_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "split",
        "selection_rule",
        "family",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_probe_cost",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "probe_call_rate",
        "verifier_select_rate",
        "extra_probe_norm_cost_mean",
    ]
    lines = [
        "# Task-Specific Verifier Action",
        "",
        f"Provider model: `{args.provider_model}`",
        f"Benchmarks: `{args.benchmarks}`",
        "Claude is not used.",
        f"Verifier rows: `{len(verifier)}`",
        f"Verifier cache-hit rate: `{float(verifier['cache_hit'].mean()) if not verifier.empty else 0.0:.4f}`",
        f"Verifier cost total: `${float(verifier['cost_total_usd'].sum()) if not verifier.empty else 0.0:.4f}`",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/181_task_specific_verifier_action.py",
        (
            "PYTHONPATH=src python experiments/181_task_specific_verifier_action.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--output-dir {args.output_dir} "
            f"--provider-model {args.provider_model} "
            f"--benchmarks {args.benchmarks} "
            f"--max-api-spend-usd {args.max_api_spend_usd}"
        ),
        "```",
        "",
        "## Verifier Accuracy By Benchmark",
        "",
        markdown_table(
            verifier.groupby(["split", "benchmark"], as_index=False).agg(
                n_queries=("query_id", "nunique"),
                mean_quality=("quality_score", "mean"),
                mean_confidence=("verifier_confidence", "mean"),
                cost_total_usd=("cost_total_usd", "sum"),
            )
            if not verifier.empty
            else pd.DataFrame()
        ),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]),
        "",
        "## Best Held-Out Rows",
        "",
        markdown_table(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)[[column for column in cols if column in table.columns]]),
        "",
        "## Interpretation",
        "",
        "- `mean_utility` charges only the selected action. `mean_utility_with_probe_cost` also charges verifier probes when the verifier is used as a gate but not selected as the final answer.",
        "- This is a real new probe/action branch, not a recombination of cached route policies.",
        "- Validation-selected rows remain the only deployable rows; top-test rows are diagnostic.",
        "",
        "## Artifacts",
        "",
        f"- Verifier outputs: `{args.output_dir / 'table_task_specific_verifier_outputs.csv'}`",
        f"- Augmented outputs: `{args.output_dir / 'model_outputs_with_task_verifier.parquet'}`",
        f"- All policy table: `{args.output_dir / 'table_task_specific_verifier_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_task_specific_verifier_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_task_specific_verifier_query_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_task_specific_verifier_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
