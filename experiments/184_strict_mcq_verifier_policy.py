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
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    score_output,
    usage_from_openai,
)


GPT_MODEL = "gpt-5.5"
STRICT_VERIFIER_ID = "gpt-5.5-strict-mcq-verifier"
GPT_INPUT_PER_MTOK = 5.00
GPT_OUTPUT_PER_MTOK = 30.00
THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]
BENCHMARK_SETS: list[tuple[str, ...]] = [("gpqa", "mmlupro"), ("gpqa",), ("mmlupro",)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict GPT MCQ verifier policy for GPQA/MMLUPro.")
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
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_strict_mcq_verifier_policy"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--benchmarks", default="gpqa,mmlupro")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--max-api-spend-usd", type=float, default=2.00)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_184")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_184")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_184")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_184")
    exp179 = load_module("experiments/179_cached_adjudicator_blend_policy.py", "adjudicator_blend_179_for_184")
    exp181 = load_module("experiments/181_task_specific_verifier_action.py", "task_verifier_181_for_184")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    benchmarks = {item.strip().lower() for item in args.benchmarks.split(",") if item.strip()}
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    query_frame = (
        outputs[outputs["split"].astype(str).isin(splits) & outputs["benchmark"].astype(str).str.lower().isin(benchmarks)]
        .drop_duplicates("query_id")
        .copy()
    )
    env = load_env_values(args.env_file)
    api_key = resolve_key(env, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for strict GPT verifier.")

    verifier = collect_rows(
        query_frame,
        outputs,
        args.output_dir,
        api_key=api_key,
        max_output_tokens=int(args.max_output_tokens),
        max_api_spend_usd=float(args.max_api_spend_usd),
        concurrency=int(args.concurrency),
        force_rerun=bool(args.force_rerun),
    )
    augmented = append_verifier_outputs(outputs, verifier, lambda_cost=float(args.lambda_cost))
    base_choices = exp181.practical_base_choices(target, outputs, exp172, exp175, exp177, exp179)
    policy_table, query_choices = evaluate_policies(
        base_choices,
        verifier,
        augmented,
        target,
        exp172,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = selected_rows(policy_table, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))

    verifier.to_csv(args.output_dir / "table_strict_mcq_verifier_outputs.csv", index=False)
    augmented.to_parquet(args.output_dir / "model_outputs_with_strict_mcq_verifier.parquet", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_strict_mcq_verifier_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_strict_mcq_verifier_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_strict_mcq_verifier_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "STRICT_MCQ_VERIFIER_POLICY_MEMO.md", args, verifier, policy_table, selected)
    print(f"Wrote strict MCQ verifier results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def collect_rows(
    queries: pd.DataFrame,
    outputs: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
    force_rerun: bool,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_strict_mcq_verifier" / GPT_MODEL / f"max_{max_output_tokens}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_map = {str(query_id): group.copy() for query_id, group in outputs.groupby("query_id", sort=False)}
    prompt_rows = [(row, prompt_for(row, output_map.get(str(row["query_id"]), pd.DataFrame()))) for _, row in queries.iterrows()]
    missing = [
        prompt
        for row, prompt in prompt_rows
        if force_rerun or not (cache_dir / cache_name(str(row["query_id"]), max_output_tokens)).exists()
    ]
    estimated = estimate_cost(missing, max_output_tokens)
    if estimated > float(max_api_spend_usd) + 1e-12:
        raise RuntimeError(f"Estimated uncached GPT verifier spend ${estimated:.4f} exceeds cap ${float(max_api_spend_usd):.4f}.")
    print(f"Estimated uncached strict GPT MCQ verifier spend: ${estimated:.4f}")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [
            executor.submit(call_one, row, prompt, cache_dir, api_key, max_output_tokens, force_rerun)
            for row, prompt in prompt_rows
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx % 20 == 0 or idx == len(futures):
                print(f"strict verifier rows {idx}/{len(futures)}")
    return pd.DataFrame(rows)


def call_one(row: pd.Series, prompt: str, cache_dir: Path, api_key: str, max_output_tokens: int, force_rerun: bool) -> dict[str, Any]:
    query_id = str(row["query_id"])
    raw_path = cache_dir / cache_name(query_id, max_output_tokens)
    cache_hit = raw_path.exists() and not force_rerun
    started = time.time()
    status = "success"
    error_type = ""
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    else:
        try:
            payload = call_openai_strict(prompt, api_key, max_output_tokens)
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            payload = {"error": str(exc)[:1000], "error_type": error_type}
        payload["_status"] = status
        payload["_error_type"] = error_type
        payload["_latency_s"] = time.time() - started
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text = extract_openai_text(payload)
    parsed = parse_json(text)
    answer_text = parsed.get("answer", text)
    parsed_answer, quality = score_output(str(answer_text), str(row["gold_answer"]), str(row["metric"]))
    if str(payload.get("_status", status)) != "success":
        quality = np.nan
    if str(payload.get("_status", status)) == "success":
        input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
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
        "provider_model": GPT_MODEL,
        "status": str(payload.get("_status", status)),
        "response_status": str(payload.get("status", "")),
        "incomplete_reason": str((payload.get("incomplete_details") or {}).get("reason", "")),
        "error_type": str(payload.get("_error_type", error_type)),
        "raw_text": text,
        "verifier_answer": str(answer_text),
        "parsed_answer": parsed_answer,
        "quality_score": float(quality) if not pd.isna(quality) else np.nan,
        "verifier_confidence": float(parsed.get("confidence", 0.0)),
        "supported_model": str(parsed.get("supported_model", "")),
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost_total_usd": call_cost(int(input_tokens), int(output_tokens)),
        "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
        "cache_hit": bool(cache_hit),
        "raw_output_path": str(raw_path),
    }


def call_openai_strict(prompt: str, api_key: str, max_output_tokens: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": GPT_MODEL,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        return post_json("https://api.openai.com/v1/responses", payload, headers, 120.0)
    except urllib.error.HTTPError as exc:
        if exc.code != 400:
            raise
        payload["reasoning"] = {"effort": "minimal"}
        try:
            return post_json("https://api.openai.com/v1/responses", payload, headers, 120.0)
        except urllib.error.HTTPError as exc2:
            if exc2.code != 400:
                raise
            payload.pop("reasoning", None)
            return post_json("https://api.openai.com/v1/responses", payload, headers, 120.0)


def prompt_for(row: pd.Series, candidates: pd.DataFrame) -> str:
    keep_models = [
        "qwen3-4b-local",
        "qwen3-8b-local",
        "qwen3-14b-awq-local",
        "qwen3-32b-awq-local",
        "qwen3-32b-awq-selfconsistency-n3-local",
        "gemini-3.5-flash",
        "gpt-5.5",
        "gemini-3.5-flash-strong-solve",
    ]
    lines: list[str] = []
    for model_id in keep_models:
        group = candidates[candidates["model_id"].astype(str).eq(model_id)]
        if group.empty:
            continue
        answer = compact(str(group.iloc[0].get("parsed_answer", "")), 80)
        if not answer or answer.lower() in {"nan", "none"}:
            answer = "[empty]"
        lines.append(f"{model_id}: {answer}")
    return (
        "Answer this multiple-choice question. Return JSON only. Do not explain. Do not show reasoning.\n"
        "The JSON schema is exactly: {\"answer\":\"A|B|C|D\",\"confidence\":0.0,\"supported_model\":\"MODEL_OR_NONE\"}.\n"
        "Pick supported_model from the candidate list only if that candidate has the same final option you trust.\n"
        f"Benchmark: {row.get('benchmark')}\n"
        f"Question:\n{compact(str(row['query_text']), 2200)}\n\n"
        "Candidate final options:\n"
        + "\n".join(lines)
        + "\nReturn only the JSON object."
    )


def parse_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I).strip()
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    parsed: dict[str, Any] = {}
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    answer = str(parsed.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        option = re.search(r"\b([ABCD])\b", clean.upper())
        answer = option.group(1) if option else clean
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "answer": answer,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "supported_model": str(parsed.get("supported_model", "")),
    }


def append_verifier_outputs(outputs: pd.DataFrame, verifier: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    gpt_cost = mean_gpt_cost(outputs)
    rows: list[dict[str, Any]] = []
    for row in verifier.itertuples(index=False):
        cost = float(row.cost_total_usd)
        normalized = cost / max(gpt_cost, 1e-12)
        quality = 0.0 if pd.isna(row.quality_score) else float(row.quality_score)
        rows.append(
            {
                "run_id": "strict_mcq_verifier",
                "query_id": str(row.query_id),
                "query_text": str(row.query_text),
                "benchmark": str(row.benchmark),
                "domain": str(row.domain),
                "model_id": STRICT_VERIFIER_ID,
                "provider": GPT_MODEL,
                "is_local": False,
                "is_frontier": True,
                "is_probe": True,
                "prompt_template_version": "strict_mcq_verifier_v1",
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
                "latency_s": float(row.latency_s),
                "status": str(row.status),
                "error_type": str(row.error_type),
                "raw_output_path": str(row.raw_output_path),
                "parsed_answer": str(row.parsed_answer),
                "gold_answer": str(row.gold_answer),
                "quality_score": quality,
                "cost_total_usd": cost,
                "cache_hit": bool(row.cache_hit),
                "metric": str(row.metric),
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


def evaluate_policies(
    base_choices: pd.DataFrame,
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    verifier_map = verifier.set_index("query_id").to_dict("index")
    gpt_cost = mean_gpt_cost(outputs)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    methods: list[tuple[str, str, str, float, tuple[str, ...]]] = [("base_candidate_ranker", "reference", "base", np.nan, tuple())]
    for benchmarks in BENCHMARK_SETS:
        bench_name = "-".join(benchmarks)
        for threshold in THRESHOLDS:
            methods.append((f"strict_verifier_answer_thr{threshold:g}_bench{bench_name}", "strict_mcq_verifier", "answer", threshold, benchmarks))
            methods.append((f"strict_verifier_support_thr{threshold:g}_bench{bench_name}", "strict_mcq_support", "support", threshold, benchmarks))
    for method, family, mode, threshold, benchmarks in methods:
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            split_base = base_choices[base_choices["split"].astype(str).eq(split)].copy()
            choices = choose_policy(split_base, verifier_map, outputs, mode=mode, threshold=threshold, benchmarks=set(benchmarks))
            selected_rows = choices[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                method,
                family,
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            route_cost_norm = probe_norm_cost(choices, verifier_map, gpt_cost)
            route_util = selected_rows["quality_score"].to_numpy(dtype=float) - float(lambda_cost) * (
                selected_rows["normalized_remote_cost"].to_numpy(dtype=float) + route_cost_norm
            )
            row["probe_call_rate"] = float(choices["verifier_probed"].mean()) if not choices.empty else 0.0
            row["override_rate"] = float(choices["overrode_base"].mean()) if not choices.empty else 0.0
            row["extra_probe_norm_cost_mean"] = float(route_cost_norm.mean()) if len(route_cost_norm) else 0.0
            row["mean_utility_with_probe_cost"] = float(route_util.mean()) if len(route_util) else np.nan
            row["oracle_utility_ratio_with_probe_cost"] = float(row["mean_utility_with_probe_cost"] / max(float(row["cost_oracle_mean_utility"]), 1e-12))
            rows.append(row)
            if split == "test":
                details.append(selected_rows[["query_id", "benchmark", "model_id", "quality_score", "utility", "parsed_answer"]].merge(choices, on="query_id", how="left").assign(method=method, family=family))
    return pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False]), (
        pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    )


def choose_policy(
    base: pd.DataFrame,
    verifier_map: dict[str, dict[str, Any]],
    outputs: pd.DataFrame,
    *,
    mode: str,
    threshold: float,
    benchmarks: set[str],
) -> pd.DataFrame:
    available = {str(query_id): set(group["model_id"].astype(str)) for query_id, group in outputs.groupby("query_id", sort=False)}
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        query_id = str(row.query_id)
        base_model = str(row.model_id)
        selected = base_model
        probed = False
        confidence = 0.0
        supported = ""
        item = verifier_map.get(query_id)
        if mode != "base" and item is not None and str(item.get("status", "")) == "success":
            benchmark = str(item.get("benchmark", "")).lower()
            probed = not benchmarks or benchmark in benchmarks
            confidence = float(item.get("verifier_confidence", 0.0) or 0.0)
            supported = str(item.get("supported_model", "") or "")
            if probed and confidence >= float(threshold):
                if mode == "answer":
                    selected = STRICT_VERIFIER_ID
                elif mode == "support" and supported in available.get(query_id, set()) and supported.upper() != "NONE":
                    selected = supported
        rows.append(
            {
                "query_id": query_id,
                "model_id": selected,
                "base_model_id": base_model,
                "verifier_probed": probed,
                "overrode_base": selected != base_model,
                "verifier_confidence": confidence,
                "supported_model": supported,
            }
        )
    return pd.DataFrame(rows)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility", "mean_utility_with_probe_cost"]:
        val = table[table["split"].eq("val") & table["family"].ne("reference")].copy()
        if val.empty:
            continue
        best = val.sort_values([objective, "frontier_call_rate", "override_rate"], ascending=[False, True, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule=f"val_best_{objective}"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule=f"val_best_{objective}_test"))
    reference = table[table["split"].eq("test") & table["family"].eq("reference")]
    if not reference.empty:
        rows.append(reference.assign(selection_rule="reference_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed).drop(columns=["_utility_values"], errors="ignore")


def probe_norm_cost(choice: pd.DataFrame, verifier_map: dict[str, dict[str, Any]], gpt_cost: float) -> np.ndarray:
    costs: list[float] = []
    for row in choice.itertuples(index=False):
        item = verifier_map.get(str(row.query_id))
        if item is None or not bool(row.verifier_probed) or str(row.model_id) == STRICT_VERIFIER_ID:
            costs.append(0.0)
        else:
            costs.append(float(item.get("cost_total_usd", 0.0) or 0.0) / max(gpt_cost, 1e-12))
    return np.asarray(costs, dtype=float)


def estimate_cost(prompts: list[str], max_output_tokens: int) -> float:
    return call_cost(sum(max(1, len(prompt) // 4) for prompt in prompts), len(prompts) * int(max_output_tokens))


def call_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * GPT_INPUT_PER_MTOK / 1_000_000 + output_tokens * GPT_OUTPUT_PER_MTOK / 1_000_000


def mean_gpt_cost(outputs: pd.DataFrame) -> float:
    return max(float(outputs[outputs["model_id"].astype(str).eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()), 1e-12)


def cache_name(query_id: str, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{GPT_MODEL}:{max_output_tokens}:strict_mcq_v1".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#806b55")
    ax.set_xlabel("Held-out test selected-action utility")
    ax.set_title("Strict MCQ Verifier Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_strict_mcq_verifier_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_probe_cost",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "frontier_call_rate",
        "probe_call_rate",
        "override_rate",
        "extra_probe_norm_cost_mean",
    ]
    lines = [
        "# Strict MCQ Verifier Policy",
        "",
        "This branch reruns a stricter GPT-5.5 multiple-choice verifier prompt with `reasoning.effort=none` where supported.",
        "Claude is not used.",
        "",
        f"Verifier rows: `{len(verifier)}`",
        f"Verifier cache-hit rate: `{float(verifier['cache_hit'].mean()) if not verifier.empty else 0.0:.4f}`",
        f"Verifier cost total: `${float(verifier['cost_total_usd'].sum()) if not verifier.empty else 0.0:.4f}`",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/184_strict_mcq_verifier_policy.py",
        (
            "PYTHONPATH=src python experiments/184_strict_mcq_verifier_policy.py "
            f"--target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir} "
            f"--max-output-tokens {args.max_output_tokens} --max-api-spend-usd {args.max_api_spend_usd}"
        ),
        "```",
        "",
        "## Verifier Quality",
        "",
        markdown_table(
            verifier.groupby(["split", "benchmark"], as_index=False).agg(
                n_queries=("query_id", "nunique"),
                mean_quality=("quality_score", "mean"),
                mean_confidence=("verifier_confidence", "mean"),
                incomplete_rate=("incomplete_reason", lambda s: float((s.astype(str) != "").mean())),
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
        markdown_table(
            table[table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)
            .head(12)[[column for column in cols if column in table.columns]]
        ),
        "",
        "## Interpretation",
        "",
        "- `mean_utility` charges only the selected final action.",
        "- `mean_utility_with_probe_cost` also charges strict verifier probes when they are used as support evidence.",
        "",
        "## Artifacts",
        "",
        f"- Verifier outputs: `{args.output_dir / 'table_strict_mcq_verifier_outputs.csv'}`",
        f"- All policy rows: `{args.output_dir / 'table_strict_mcq_verifier_policy_all.csv'}`",
        f"- Selected policy rows: `{args.output_dir / 'table_strict_mcq_verifier_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_strict_mcq_verifier_query_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_strict_mcq_verifier_policy_utility.pdf'}`",
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
