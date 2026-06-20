from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METHODS = "pred_rf_thr-0.0288"
THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]
ACTION_POOL = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gemini-3.5-flash-strong-solve",
    "gpt-5.5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh local vLLM verifier for current RouteCode actions.")
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
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_current_action_verifier_qwen14b"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--model-id", default="qwen3-14b-awq-current-action-verifier")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-query-chars", type=int, default=2200)
    parser.add_argument("--max-answer-chars", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_187")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_187")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_187")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_187")
    exp183 = load_module("experiments/183_local_safe_gain_gate.py", "local_safe_183_for_187")
    exp185 = load_module("experiments/185_probe_fusion_policy.py", "probe_fusion_185_for_187")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    rows_by_query = exp177.rows_by_query_map(outputs)
    base_choices = exp183.build_base_choices(exp177, exp172, exp175, outputs, target, rows_by_query)
    feature_frame = exp183.build_local_safe_features(base_choices, target, outputs, rows_by_query)
    scored = exp185.score_local_specs(feature_frame, exp183)
    local_specs = [parse_method(method) for method in parse_csv(args.methods)]
    local_specs = [spec for spec in local_specs if spec is not None]
    if not local_specs:
        raise ValueError("No valid local-safe methods were provided.")
    probe_inputs = build_probe_inputs(scored, local_specs, rows_by_query, splits=set(parse_csv(args.splits)))
    if args.limit is not None:
        probe_inputs = probe_inputs.head(int(args.limit)).copy()
    probe = collect_probe(args, probe_inputs)
    policy_table, query_choices = evaluate_policies(
        scored,
        local_specs,
        probe,
        outputs,
        target,
        exp172,
        exp183,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = selected_rows(policy_table, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    query_choices = filter_query_choices(query_choices, policy_table, selected)

    scored.to_csv(args.output_dir / "table_current_action_verifier_features.csv", index=False)
    probe.to_csv(args.output_dir / "table_current_action_verifier_probe.csv", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_current_action_verifier_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_current_action_verifier_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_current_action_verifier_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "CURRENT_ACTION_VERIFIER_MEMO.md", args, probe, policy_table, selected)
    print(f"Wrote current-action verifier results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_method(method: str) -> tuple[str, str, float] | None:
    match = re.fullmatch(r"(pred_[A-Za-z0-9]+)_thr(-?[0-9.]+)", method)
    if not match:
        return None
    return method, match.group(1), float(match.group(2))


def build_probe_inputs(
    scored: pd.DataFrame,
    local_specs: list[tuple[str, str, float]],
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    *,
    splits: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, pred_col, threshold in local_specs:
        for split in ["train", "val", "test"]:
            if split not in splits:
                continue
            split_frame = scored[scored["split"].astype(str).eq(split)].copy()
            choices = choose_local_actions(split_frame, pred_col=pred_col, threshold=threshold)
            info = split_frame.drop_duplicates("query_id").set_index("query_id")
            for item in choices.itertuples(index=False):
                query_id = str(item.query_id)
                if query_id not in info.index:
                    continue
                row = info.loc[query_id]
                candidate_model = str(item.model_id)
                actions = rows_by_query.get(query_id, {})
                if candidate_model not in actions:
                    continue
                candidates = candidate_actions(actions)
                rows.append(
                    {
                        "query_id": query_id,
                        "split": split,
                        "benchmark": str(row.get("benchmark", "")),
                        "domain": str(row.get("domain", "")),
                        "metric": str(row.get("metric", "")),
                        "method": method,
                        "current_model_id": candidate_model,
                        "base_model_id": str(item.base_model_id),
                        "consensus_model": str(getattr(item, "consensus_model", "") or ""),
                        "query_text": str(row.get("query_text", "")),
                        "current_answer": str(actions[candidate_model].get("parsed_answer", "") or ""),
                        "candidate_actions_json": json.dumps(candidates, sort_keys=True),
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates(["query_id", "method", "current_model_id"])
    return frame.sort_values(["split", "benchmark", "query_id", "method"])


def choose_local_actions(frame: pd.DataFrame, *, pred_col: str, threshold: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        base_model = str(item.base_model_id)
        consensus_model = str(item.consensus_model)
        selected = base_model
        if consensus_model and float(getattr(item, pred_col)) >= float(threshold):
            selected = consensus_model
        rows.append(
            {
                "query_id": str(item.query_id),
                "model_id": selected,
                "base_model_id": base_model,
                "consensus_model": consensus_model,
            }
        )
    return pd.DataFrame(rows)


def candidate_actions(actions: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for model_id in ACTION_POOL:
        if model_id not in actions:
            continue
        answer = str(actions[model_id].get("parsed_answer", "") or "")
        status = str(actions[model_id].get("status", "") or "")
        if not answer and status != "success":
            continue
        rows.append({"model_id": model_id, "answer": answer, "status": status})
    return rows


def collect_probe(args: argparse.Namespace, probe_inputs: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_current_action_verifier" / safe_part(str(args.model_id))
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(probe_inputs.to_dict("records"))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = {
            executor.submit(call_or_load_one, args, raw_dir, task): task
            for task in tasks
        }
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx % 20 == 0 or idx == len(futures):
                print(f"current-action verifier rows {idx}/{len(futures)}")
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id", "method"])


def call_or_load_one(args: argparse.Namespace, raw_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    raw_path = raw_dir / f"{safe_part(str(task['query_id']))}_{cache_digest(task)}.json"
    cache_hit = raw_path.exists() and not args.force_rerun
    status = "success"
    error_type = ""
    started = time.time()
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        status = str(payload.get("_status", "success"))
        error_type = str(payload.get("_error_type", ""))
    else:
        prompt = verifier_prompt(task, max_query_chars=int(args.max_query_chars), max_answer_chars=int(args.max_answer_chars))
        try:
            payload = call_vllm_chat(
                base_url=str(args.base_url),
                served_model_name=str(args.served_model_name),
                prompt=prompt,
                max_tokens=int(args.max_tokens),
                timeout_s=float(args.timeout_s),
            )
        except Exception as exc:
            status = "error"
            error_type = type(exc).__name__
            payload = {"error_type": error_type, "error": str(exc)[:1000]}
        payload["_status"] = status
        payload["_error_type"] = error_type
        payload["_latency_s"] = time.time() - started
        raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    raw_text = extract_text(payload)
    parsed = parse_verifier_output(raw_text)
    if status != "success":
        parsed = {"verdict": "unknown", "supported_model": "", "confidence": 0.0, "reason": ""}
    return {
        **{key: task[key] for key in ["query_id", "split", "benchmark", "domain", "metric", "method", "current_model_id", "base_model_id", "consensus_model"]},
        "model_id": args.model_id,
        "status": status,
        "error_type": error_type,
        "cache_hit": bool(cache_hit),
        "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
        "raw_output_path": str(raw_path),
        "raw_text": raw_text,
        "verdict": parsed["verdict"],
        "supported_model": parsed["supported_model"],
        "confidence": float(parsed["confidence"]),
        "reason": parsed.get("reason", ""),
    }


def verifier_prompt(task: dict[str, Any], *, max_query_chars: int, max_answer_chars: int) -> str:
    candidates = json.loads(str(task["candidate_actions_json"]))
    candidate_lines = []
    for item in candidates:
        candidate_lines.append(f"- {item['model_id']}: {truncate(str(item.get('answer', '')), max_answer_chars) or '[empty]'}")
    valid_models = ", ".join([item["model_id"] for item in candidates])
    return (
        "You are a local RouteCode action verifier. Decide whether the CURRENT selected answer should be kept, "
        "or whether one of the listed candidate answers is better. Do not solve from scratch unless needed to compare answers.\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"verdict":"accept|switch|escalate|unknown","supported_model":"MODEL_ID_OR_NONE","confidence":0.0,"reason":"brief"}\n'
        "- verdict must be one of: accept, switch, escalate, unknown.\n"
        "- supported_model must be the best model_id from the candidate list, or NONE.\n"
        "- Use accept only if the current selected answer is likely correct.\n"
        "- Use switch if another listed candidate answer is more likely correct.\n"
        "- Use escalate only if the current answer is unreliable and no listed candidate is clearly better.\n"
        "- confidence is between 0 and 1.\n"
        "- reason must be at most 8 words.\n\n"
        f"Benchmark: {task['benchmark']}\n"
        f"Metric: {task['metric']}\n"
        f"Allowed supported_model values: {valid_models}, NONE\n"
        f"Current selected model: {task['current_model_id']}\n"
        f"Current selected answer: {truncate(str(task['current_answer']), max_answer_chars) or '[empty]'}\n\n"
        f"Task:\n{truncate(str(task['query_text']), max_query_chars)}\n\n"
        "Candidate answers:\n"
        + "\n".join(candidate_lines)
    )


def call_vllm_chat(*, base_url: str, served_model_name: str, prompt: str, max_tokens: int, timeout_s: float) -> dict[str, Any]:
    payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": "Bearer local-routecode", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def extract_text(payload: dict[str, Any]) -> str:
    try:
        return str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    except Exception:
        return ""


def parse_verifier_output(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    payload: dict[str, Any] = {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                payload = parsed
        except json.JSONDecodeError:
            payload = {}
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"accept", "switch", "escalate", "unknown"}:
        lowered = text.lower()
        if "switch" in lowered:
            verdict = "switch"
        elif "escalate" in lowered:
            verdict = "escalate"
        elif "accept" in lowered:
            verdict = "accept"
        else:
            verdict = "unknown"
    if not payload:
        fallback_supported = re.search(r'"supported_model"\s*:\s*"([^"]+)"', text)
        fallback_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
        fallback_reason = re.search(r'"reason"\s*:\s*"([^"]*)"', text)
        if fallback_supported:
            payload["supported_model"] = fallback_supported.group(1)
        if fallback_conf:
            payload["confidence"] = fallback_conf.group(1)
        if fallback_reason:
            payload["reason"] = fallback_reason.group(1)
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)
    supported = str(payload.get("supported_model", "") or "").strip()
    if supported.upper() == "NONE":
        supported = ""
    return {
        "verdict": verdict,
        "supported_model": supported,
        "confidence": confidence,
        "reason": str(payload.get("reason", ""))[:240],
    }


def evaluate_policies(
    scored: pd.DataFrame,
    local_specs: list[tuple[str, str, float]],
    probe: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    exp183,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    available = {str(query_id): set(group["model_id"].astype(str)) for query_id, group in outputs.groupby("query_id", sort=False)}
    probe_map = probe.set_index(["query_id", "method"]).to_dict("index")
    methods: list[dict[str, Any]] = [{"method": "base_candidate_ranker", "family": "reference", "kind": "base"}]
    for local_method, pred_col, threshold in local_specs:
        methods.append(
            {
                "method": f"local_{local_method}",
                "family": "local_safe_reference",
                "kind": "local",
                "local_method": local_method,
                "pred_col": pred_col,
                "local_threshold": threshold,
            }
        )
        for confidence in THRESHOLDS:
            methods.extend(
                [
                    {
                        "method": f"current_verifier_switch_conf{confidence:g}_{local_method}",
                        "family": "current_action_verifier_switch",
                        "kind": "switch",
                        "local_method": local_method,
                        "pred_col": pred_col,
                        "local_threshold": threshold,
                        "confidence": confidence,
                    },
                    {
                        "method": f"current_verifier_reject_to_base_conf{confidence:g}_{local_method}",
                        "family": "current_action_verifier_reject_to_base",
                        "kind": "reject_to_base",
                        "local_method": local_method,
                        "pred_col": pred_col,
                        "local_threshold": threshold,
                        "confidence": confidence,
                    },
                    {
                        "method": f"current_verifier_accept_required_conf{confidence:g}_{local_method}",
                        "family": "current_action_verifier_accept_required",
                        "kind": "accept_required",
                        "local_method": local_method,
                        "pred_col": pred_col,
                        "local_threshold": threshold,
                        "confidence": confidence,
                    },
                ]
            )
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for spec in methods:
        for split in ["val", "test"]:
            split_frame = scored[scored["split"].astype(str).eq(split)].copy()
            choices = choose_policy_actions(split_frame, spec, exp183, probe_map, available)
            selected_rows = choices[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            target_split = target[target["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                str(spec["method"]),
                str(spec["family"]),
                split,
                selected_rows,
                outputs,
                target=target_split,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            row.update(
                {
                    "verifier_call_rate": float(choices["verifier_available"].mean()) if not choices.empty else 0.0,
                    "switch_rate": float(choices["switched"].mean()) if not choices.empty else 0.0,
                    "reject_to_base_rate": float(choices["rejected_to_base"].mean()) if not choices.empty else 0.0,
                    "override_rate": float(choices["overrode_base"].mean()) if not choices.empty else 0.0,
                    "mean_utility_with_probe_cost": row["mean_utility"],
                    "oracle_utility_ratio_with_probe_cost": row["oracle_utility_ratio"],
                }
            )
            rows.append(row)
            if split == "test":
                details.append(
                    selected_rows[
                        [
                            "query_id",
                            "query_text",
                            "benchmark",
                            "metric",
                            "model_id",
                            "quality_score",
                            "utility",
                            "normalized_remote_cost",
                            "is_frontier",
                            "parsed_answer",
                        ]
                    ]
                    .merge(choices, on="query_id", how="left")
                    .assign(method=str(spec["method"]), family=str(spec["family"]))
                )
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    return table, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def choose_policy_actions(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    exp183,
    probe_map: dict[tuple[str, str], dict[str, Any]],
    available: dict[str, set[str]],
) -> pd.DataFrame:
    if spec["kind"] in {"local", "switch", "reject_to_base", "accept_required"}:
        local_choices = exp183.choose_actions(
            frame,
            pred_col=str(spec.get("pred_col", "")),
            threshold=float(spec.get("local_threshold", 0.0)),
            family="local_safe_gain_gate",
        )
        local_method = str(spec.get("local_method", ""))
    else:
        local_choices = exp183.choose_actions(frame, pred_col="", threshold=0.0, family="reference")
        local_method = ""
    rows: list[dict[str, Any]] = []
    for item in local_choices.itertuples(index=False):
        query_id = str(item.query_id)
        base_model = str(item.base_model_id)
        selected = str(item.model_id)
        local_selected = selected
        probe = probe_map.get((query_id, local_method), {})
        available_probe = bool(probe)
        verdict = str(probe.get("verdict", "unknown"))
        supported = str(probe.get("supported_model", "") or "")
        confidence = float(probe.get("confidence", 0.0) or 0.0)
        threshold = float(spec.get("confidence", 0.0))
        switched = False
        rejected = False
        if available_probe and confidence >= threshold:
            if spec["kind"] == "switch" and verdict in {"switch", "accept"}:
                if supported and supported in available.get(query_id, set()):
                    selected = supported
                    switched = selected != local_selected
            elif spec["kind"] == "reject_to_base" and verdict in {"switch", "escalate", "unknown"}:
                if supported and supported in available.get(query_id, set()):
                    selected = supported
                    switched = selected != local_selected
                else:
                    selected = base_model
                    rejected = selected != local_selected
            elif spec["kind"] == "accept_required" and verdict != "accept":
                if supported and supported in available.get(query_id, set()):
                    selected = supported
                    switched = selected != local_selected
                else:
                    selected = base_model
                    rejected = selected != local_selected
        rows.append(
            {
                "query_id": query_id,
                "model_id": selected,
                "base_model_id": base_model,
                "local_model_id": local_selected,
                "verifier_available": available_probe,
                "verifier_verdict": verdict,
                "supported_model": supported,
                "verifier_confidence": confidence,
                "switched": switched,
                "rejected_to_base": rejected,
                "overrode_base": selected != base_model,
            }
        )
    return pd.DataFrame(rows)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[
        table["split"].eq("val")
        & table["family"].isin(
            [
                "current_action_verifier_switch",
                "current_action_verifier_reject_to_base",
                "current_action_verifier_accept_required",
            ]
        )
    ].copy()
    if not val.empty:
        best = val.sort_values(["mean_utility", "frontier_call_rate", "override_rate"], ascending=[False, True, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_mean_utility"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule="val_best_mean_utility_test"))
    reference = table[table["split"].eq("test") & table["family"].eq("reference")]
    if not reference.empty:
        rows.append(reference.assign(selection_rule="reference_test"))
    local_ref = table[table["split"].eq("test") & table["family"].eq("local_safe_reference")]
    if not local_ref.empty:
        rows.append(local_ref.assign(selection_rule="local_reference_test"))
    top_test = (
        table[table["split"].eq("test") & table["family"].ne("reference")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(16)
    )
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
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed).drop(columns=["_utility_values"], errors="ignore")


def filter_query_choices(query_choices: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    selected_methods = set(selected.get("method", pd.Series(dtype=str)).astype(str).tolist())
    top_methods = set(
        table[table["split"].eq("test")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(24)["method"]
        .astype(str)
        .tolist()
    )
    return query_choices[query_choices["method"].astype(str).isin(selected_methods | top_methods)].copy()


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:96]


def cache_digest(task: dict[str, Any]) -> str:
    text = f"v2|{task['query_id']}|{task['method']}|{task['current_model_id']}|{task['current_answer']}"
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.70)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4c7773")
    ax.set_xlabel("Held-out test selected-action utility")
    ax.set_title("Current-Action vLLM Verifier Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_current_action_verifier_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, probe: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    selected_cols = [
        "selection_rule",
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "verifier_call_rate",
        "switch_rate",
        "reject_to_base_rate",
        "override_rate",
    ]
    diagnostics = (
        probe.groupby(["split", "verdict"], dropna=False)
        .agg(n=("query_id", "size"), mean_confidence=("confidence", "mean"), mean_latency_s=("latency_s", "mean"))
        .reset_index()
    )
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    text = [
        "# Current-Action vLLM Verifier",
        "",
        "This branch uses local vLLM to judge the current selected action directly.",
        "It makes no GPT, Gemini, or Claude calls.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/187_current_action_verifier_vllm.py",
        f"PYTHONPATH=src python experiments/187_current_action_verifier_vllm.py --methods {args.methods}",
        "```",
        "",
        "## Run Config",
        "",
        f"- Served model: `{args.served_model_name}` via `{args.base_url}`",
        f"- Probe rows: `{len(probe)}`",
        f"- Cache-hit rate: `{float(probe['cache_hit'].mean()) if len(probe) else 0.0:.4f}`",
        "",
        "## Probe Diagnostics",
        "",
        markdown_table(diagnostics),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in selected_cols if column in selected.columns]]) if not selected.empty else "_No selected rows._",
        "",
        "## Best Held-Out Rows",
        "",
        markdown_table(top_test[[column for column in selected_cols if column in top_test.columns]]) if not top_test.empty else "_No held-out rows._",
        "",
        "## Interpretation",
        "",
        "- This is the direct current-action verifier branch requested by the prior negative result.",
        "- Local verifier probe cost is zero remote API cost, but verifier-call rate and latency are reported.",
        "- A deployable win requires validation-selected test utility to beat the local-safe reference.",
        "",
        "## Artifacts",
        "",
        f"- Probe rows: `{path.parent / 'table_current_action_verifier_probe.csv'}`",
        f"- All policy rows: `{path.parent / 'table_current_action_verifier_policy_all.csv'}`",
        f"- Selected policy rows: `{path.parent / 'table_current_action_verifier_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_current_action_verifier_query_choices.csv'}`",
        f"- Figure: `{path.parent / 'fig_current_action_verifier_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
