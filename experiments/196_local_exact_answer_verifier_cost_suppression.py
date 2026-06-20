from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_POLICY = "qwen14_bbh_support2_conf0_nonfrontier"
FRONTIERS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve", "gpt-5.5"}
STRONG_OR_FRONTIER = {"qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local", *FRONTIERS}
CHEAP_LOCAL_ACTIONS = {
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
}
LOCAL_ACTIONS = [
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
]
EXACT_SCOPES = [
    ("gsm8k",),
    ("math500",),
    ("livemathbench",),
    ("gsm8k", "math500"),
    ("math500", "livemathbench"),
    ("gsm8k", "math500", "livemathbench"),
    ("aime", "gsm8k", "math500", "livemathbench"),
]
THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95]
MODEL_STRATEGIES = {
    "qwen4": "qwen3-4b-local",
    "qwen8": "qwen3-8b-local",
    "qwen14": "qwen3-14b-awq-local",
    "qwen32": "qwen3-32b-awq-local",
    "qwen32_sc": "qwen3-32b-awq-selfconsistency-n3-local",
    "tool": "deterministic_math_tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a local vLLM verifier to suppress exact-answer frontier calls."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_conservative_support_abstention_policy/"
            "table_conservative_support_abstention_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default=BASE_POLICY)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_local_exact_answer_verifier_cost_suppression"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--model-id", default="qwen3-14b-awq-exact-answer-verifier")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--benchmarks", default="aime,gsm8k,math500,livemathbench")
    parser.add_argument("--strategies", default="qwen4,qwen8,qwen14,qwen32,qwen32_sc,majority2_cheapest,majority2_strongest,majority3_cheapest,majority3_strongest")
    parser.add_argument("--max-query-chars", type=int, default=1800)
    parser.add_argument("--max-answer-chars", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=64)
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
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    base = pd.read_csv(args.base_query_choices)
    base = base[
        base["policy"].astype(str).eq(str(args.base_policy))
        & base["split"].astype(str).isin(parse_csv(args.splits))
    ].copy()
    if base.empty:
        raise RuntimeError(f"Base policy {args.base_policy!r} not found in {args.base_query_choices}.")
    action_map = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][
        ["query_id", "utility", "quality_score"]
    ].rename(columns={"utility": "oracle_utility", "quality_score": "oracle_quality"})
    base = drop_prefixed(base, ["oracle_utility", "oracle_quality"]).merge(oracle, on="query_id", how="left")
    target_table = build_oracle_target_table(outputs)

    strategies = parse_csv(args.strategies)
    benchmarks = set(parse_csv(args.benchmarks))
    candidate_inputs = build_candidate_inputs(base, action_map, strategies, benchmarks)
    if args.limit is not None:
        candidate_inputs = candidate_inputs.head(int(args.limit)).copy()
    verifier = collect_verifier_rows(args, candidate_inputs)
    rules = enumerate_rules(strategies)
    policy_table, query_choices = evaluate_rules(base, action_map, verifier, rules, args)
    selected = selected_rows(policy_table)
    selected_policies = set(selected["policy"].astype(str)) if not selected.empty else set()
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    target_table.to_csv(args.output_dir / "table_oracle_targets_local_vs_large.csv", index=False)
    candidate_inputs.to_csv(args.output_dir / "table_exact_answer_verifier_candidates.csv", index=False)
    verifier.to_csv(args.output_dir / "table_exact_answer_verifier_outputs.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output_dir / "table_exact_answer_verifier_rule_library.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_exact_answer_verifier_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_exact_answer_verifier_policy_selected.csv", index=False)
    query_choices_to_write.to_csv(args.output_dir / "table_exact_answer_verifier_query_choices.csv", index=False)
    write_memo(args.output_dir / "LOCAL_EXACT_ANSWER_VERIFIER_COST_SUPPRESSION_MEMO.md", args, verifier, selected)
    print(f"Wrote local exact-answer verifier cost-suppression results to {args.output_dir}")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def drop_prefixed(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    cols = [col for col in frame.columns if any(str(col).startswith(prefix) for prefix in prefixes)]
    return frame.drop(columns=cols, errors="ignore")


def build_oracle_target_table(outputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id, group in outputs.groupby("query_id", sort=False):
        cheap_local = group[group["model_id"].astype(str).isin(CHEAP_LOCAL_ACTIONS)].copy()
        large = group[group["model_id"].astype(str).isin(STRONG_OR_FRONTIER)].copy()
        if cheap_local.empty:
            cheap_local = group[group["is_frontier"].astype(bool).eq(False)].copy()
        if large.empty:
            large = group.copy()
        local_row = cheap_local.loc[cheap_local["utility"].astype(float).idxmax()]
        large_row = large.loc[large["utility"].astype(float).idxmax()]
        rows.append(
            {
                "query_id": str(query_id),
                "split": str(group.iloc[0]["split"]),
                "benchmark": str(group.iloc[0]["benchmark"]),
                "domain": str(group.iloc[0]["domain"]),
                "query_text": str(group.iloc[0]["query_text"]),
                "best_local_action": str(local_row["model_id"]),
                "best_large_action": str(large_row["model_id"]),
                "local_utility": float(local_row["utility"]),
                "large_utility": float(large_row["utility"]),
                "delta_large": float(large_row["utility"]) - float(local_row["utility"]),
                "need_large": bool(float(large_row["utility"]) > float(local_row["utility"]) + 1e-12),
                "local_quality": float(local_row["quality_score"]),
                "large_quality": float(large_row["quality_score"]),
                "local_normalized_cost": float(local_row["normalized_remote_cost"]),
                "large_normalized_cost": float(large_row["normalized_remote_cost"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"])


def build_candidate_inputs(
    base: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    strategies: list[str],
    benchmarks: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        if str(row.benchmark) not in benchmarks or not bool(row.fused_frontier):
            continue
        query_id = str(row.query_id)
        actions = action_map.get(query_id, {})
        for strategy in strategies:
            candidate_model = candidate_for_strategy(actions, strategy)
            if not candidate_model:
                continue
            item = actions.get(candidate_model, {})
            answer = normalize_answer(item.get("parsed_answer", ""))
            if not answer:
                continue
            rows.append(
                {
                    "query_id": query_id,
                    "split": str(row.split),
                    "benchmark": str(row.benchmark),
                    "domain": str(row.domain),
                    "metric": str(row.metric),
                    "strategy": strategy,
                    "candidate_model_id": candidate_model,
                    "candidate_answer": str(item.get("parsed_answer", "") or ""),
                    "candidate_quality": float(item.get("quality_score", 0.0) or 0.0),
                    "candidate_utility": float(item.get("utility", 0.0) or 0.0),
                    "base_model_id": str(row.fused_model),
                    "base_quality": float(row.fused_quality),
                    "base_utility": float(row.fused_utility),
                    "query_text": str(row.query_text),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.drop_duplicates(["query_id", "strategy", "candidate_model_id"]).sort_values(
        ["split", "benchmark", "query_id", "strategy"]
    )


def candidate_for_strategy(actions: dict[str, dict[str, Any]], strategy: str) -> str:
    if strategy in MODEL_STRATEGIES:
        model_id = MODEL_STRATEGIES[strategy]
        return model_id if model_id in actions else ""
    match = re.fullmatch(r"majority([23])_(cheapest|strongest)", strategy)
    if not match:
        return ""
    min_votes = int(match.group(1))
    order = match.group(2)
    answer_by_model: dict[str, str] = {}
    answers: list[str] = []
    for model_id in LOCAL_ACTIONS:
        item = actions.get(model_id)
        if not item:
            continue
        answer = normalize_answer(item.get("parsed_answer", ""))
        if not answer:
            continue
        answer_by_model[model_id] = answer
        answers.append(answer)
    if not answers:
        return ""
    answer, count = Counter(answers).most_common(1)[0]
    if count < min_votes:
        return ""
    model_order = LOCAL_ACTIONS if order == "cheapest" else list(reversed(LOCAL_ACTIONS))
    for model_id in model_order:
        if answer_by_model.get(model_id) == answer:
            return model_id
    return ""


def collect_verifier_rows(args: argparse.Namespace, candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    raw_dir = args.output_dir / "raw_exact_answer_verifier" / safe_part(str(args.model_id))
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(candidates.to_dict("records"))
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = {executor.submit(call_or_load_one, args, raw_dir, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"exact-answer verifier rows {index}/{len(futures)}")
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id", "strategy"])


def call_or_load_one(args: argparse.Namespace, raw_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    raw_path = raw_dir / f"{safe_part(str(task['query_id']))}_{safe_part(str(task['strategy']))}_{cache_digest(task)}.json"
    cache_hit = raw_path.exists() and not args.force_rerun
    started = time.time()
    status = "success"
    error_type = ""
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        status = str(payload.get("_status", "success"))
        error_type = str(payload.get("_error_type", ""))
    else:
        prompt = verifier_prompt(
            task,
            max_query_chars=int(args.max_query_chars),
            max_answer_chars=int(args.max_answer_chars),
        )
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
        parsed = {"verdict": "unknown", "confidence": 0.0, "reason": ""}
    return {
        **{
            key: task[key]
            for key in [
                "query_id",
                "split",
                "benchmark",
                "domain",
                "metric",
                "strategy",
                "candidate_model_id",
                "candidate_answer",
                "candidate_quality",
                "candidate_utility",
                "base_model_id",
                "base_quality",
                "base_utility",
            ]
        },
        "model_id": str(args.model_id),
        "status": status,
        "error_type": error_type,
        "cache_hit": bool(cache_hit),
        "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
        "raw_output_path": str(raw_path),
        "raw_text": raw_text,
        "verdict": parsed["verdict"],
        "confidence": float(parsed["confidence"]),
        "reason": parsed.get("reason", ""),
    }


def verifier_prompt(task: dict[str, Any], *, max_query_chars: int, max_answer_chars: int) -> str:
    query = truncate(str(task["query_text"]), max_query_chars)
    answer = truncate(str(task["candidate_answer"]), max_answer_chars) or "[empty]"
    return (
        "You are a strict local verifier for exact-answer routing. Decide whether the candidate final answer is reliable enough "
        "to use instead of paying for a frontier solver. Do not reward plausible answers; accept only when you can verify the answer.\n"
        'Return JSON only: {"verdict":"accept|escalate","confidence":0.0,"reason":"short"}\n'
        "- Use accept only if the candidate answer is very likely correct.\n"
        "- Use escalate if the answer is wrong, malformed, unsupported, or uncertain.\n"
        "- confidence must be between 0 and 1.\n\n"
        f"Benchmark: {task['benchmark']}\n"
        f"Metric: {task['metric']}\n"
        f"Candidate model: {task['candidate_model_id']}\n\n"
        f"Task:\n{query}\n\n"
        f"Candidate final answer:\n{answer}\n"
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
    if verdict not in {"accept", "escalate", "unknown"}:
        lowered = text.lower()
        if "accept" in lowered:
            verdict = "accept"
        elif "escalate" in lowered or "reject" in lowered:
            verdict = "escalate"
        else:
            verdict = "unknown"
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "verdict": verdict,
        "confidence": min(max(confidence, 0.0), 1.0),
        "reason": str(payload.get("reason", ""))[:240],
    }


def enumerate_rules(strategies: list[str]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_exp194",
            "family": "reference",
            "scope_json": "[]",
            "strategy": "none",
            "threshold": np.nan,
        }
    ]
    for scope in EXACT_SCOPES:
        for strategy in strategies:
            for threshold in THRESHOLDS:
                rules.append(
                    {
                        "policy": f"exact_verifier_scope{'+'.join(scope)}_{strategy}_thr{threshold:g}",
                        "family": "local_exact_answer_verifier",
                        "scope_json": json.dumps(scope),
                        "strategy": strategy,
                        "threshold": float(threshold),
                    }
                )
    return rules


def evaluate_rules(
    base: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    verifier: pd.DataFrame,
    rules: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    verifier_map = (
        verifier.set_index(["query_id", "strategy"]).to_dict("index") if not verifier.empty else {}
    )
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for rule in rules:
        choices = apply_rule(base, action_map, verifier_map, rule)
        details.append(choices.assign(policy=str(rule["policy"]), family=str(rule["family"])))
        for split, group in choices.groupby("split", dropna=False):
            values = group["selected_utility"].astype(float).to_numpy()
            ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            oracle_q = float(group["oracle_quality"].astype(float).mean())
            mean_u = float(values.mean())
            mean_q = float(group["selected_quality"].astype(float).mean())
            rows.append(
                {
                    **rule,
                    "split": split,
                    "n_queries": int(len(group)),
                    "mean_quality": mean_q,
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "cost_oracle_mean_utility": oracle_u,
                    "quality_oracle_mean_quality": oracle_q,
                    "oracle_utility_ratio": mean_u / max(oracle_u, 1e-12),
                    "utility_gap_to_oracle": oracle_u - mean_u,
                    "quality_gap_to_oracle": oracle_q - mean_q,
                    "frontier_call_rate": float(group["selected_frontier"].mean()),
                    "probe_call_rate": float(group["probe_used"].mean()),
                    "changed_rate": float(group["changed"].mean()),
                    "accepted_probe_rate": float(group["accepted_probe"].mean()),
                    "mean_probe_latency_s": float(group.loc[group["probe_used"], "probe_latency_s"].mean())
                    if bool(group["probe_used"].any())
                    else 0.0,
                    "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict(), sort_keys=True),
                }
            )
    return pd.DataFrame(rows), pd.concat(details, ignore_index=True)


def apply_rule(
    base: pd.DataFrame,
    action_map: dict[str, dict[str, dict[str, Any]]],
    verifier_map: dict[tuple[str, str], dict[str, Any]],
    rule: dict[str, Any],
) -> pd.DataFrame:
    scope = set(json.loads(str(rule.get("scope_json", "[]"))))
    strategy = str(rule.get("strategy", "none"))
    threshold = 0.0 if pd.isna(rule.get("threshold", np.nan)) else float(rule["threshold"])
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        selected = str(row.fused_model)
        probe_used = False
        accepted_probe = False
        probe_latency = 0.0
        verifier_row: dict[str, Any] = {}
        if strategy != "none" and str(row.benchmark) in scope and bool(row.fused_frontier):
            verifier_row = verifier_map.get((str(row.query_id), strategy), {})
            if verifier_row:
                probe_used = True
                probe_latency = float(verifier_row.get("latency_s", 0.0) or 0.0)
                verdict = str(verifier_row.get("verdict", "unknown"))
                confidence = float(verifier_row.get("confidence", 0.0) or 0.0)
                candidate = str(verifier_row.get("candidate_model_id", "") or "")
                if verdict == "accept" and confidence >= threshold and candidate in action_map.get(str(row.query_id), {}):
                    selected = candidate
                    accepted_probe = True
        action = action_map.get(str(row.query_id), {}).get(selected, {})
        quality = float(action.get("quality_score", row.fused_quality))
        utility = float(action.get("utility", row.fused_utility))
        rows.append(
            {
                **row._asdict(),
                "selected_model": selected,
                "selected_quality": quality,
                "selected_utility": utility,
                "selected_frontier": bool(action.get("is_frontier", False)),
                "changed": selected != str(row.fused_model),
                "probe_used": bool(probe_used),
                "accepted_probe": bool(accepted_probe),
                "probe_latency_s": probe_latency,
                "verifier_verdict": str(verifier_row.get("verdict", "")),
                "verifier_confidence": float(verifier_row.get("confidence", 0.0) or 0.0),
                "candidate_model_id": str(verifier_row.get("candidate_model_id", "")),
            }
        )
    return pd.DataFrame(rows)


def selected_rows(policy_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    val = policy_table[policy_table["split"].astype(str).eq("val")].copy()
    test = policy_table[policy_table["split"].astype(str).eq("test")].copy()
    for split, frame in [("val", val), ("test", test)]:
        base = frame[frame["policy"].astype(str).eq("base_exp194")]
        if not base.empty:
            row = base.iloc[0].copy()
            row["selection_rule"] = f"base_reference_{split}"
            rows.append(row)
    candidates = val[
        val["family"].astype(str).eq("local_exact_answer_verifier")
        & val["changed_rate"].astype(float).gt(0.0)
    ].copy()
    if not candidates.empty:
        best = candidates.sort_values(
            ["mean_utility", "frontier_call_rate", "probe_call_rate", "changed_rate"],
            ascending=[False, True, True, True],
        ).iloc[0].copy()
        best["selection_rule"] = "val_best_local_exact_answer_verifier"
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = "val_best_local_exact_answer_verifier_test"
            rows.append(test_row)
    top_test = (
        test[test["family"].astype(str).eq("local_exact_answer_verifier")]
        .sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True])
        .head(8)
    )
    for _, row in top_test.iterrows():
        diagnostic = row.copy()
        diagnostic["selection_rule"] = "top_test_diagnostic"
        rows.append(diagnostic)
    return pd.DataFrame(rows).drop_duplicates(["selection_rule", "policy", "split"], keep="first")


def normalize_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return ""
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    return text.removeprefix("answer:").strip().strip("$").strip()


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:140].strip("_") or "item"


def cache_digest(task: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "query_id": task.get("query_id"),
            "strategy": task.get("strategy"),
            "candidate_model_id": task.get("candidate_model_id"),
            "candidate_answer": task.get("candidate_answer"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    return text if len(text) <= max_chars else text[: max(0, max_chars - 20)] + "\n[truncated]"


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "policy",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "probe_call_rate",
        "changed_rate",
        "accepted_probe_rate",
        "mean_probe_latency_s",
    ]
    success = int(verifier["status"].astype(str).eq("success").sum()) if not verifier.empty else 0
    total = int(len(verifier))
    lines = [
        "# Local Exact-Answer Verifier Cost-Suppression",
        "",
        "This experiment verifies local exact-answer substitutes before paying for a frontier/API action.",
        "It is deployable in the sense that the verifier sees only the query and a local candidate answer, not the frontier answer.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/196_local_exact_answer_verifier_cost_suppression.py",
        "```",
        "",
        "## Verifier",
        "",
        f"- Verifier model: `{args.model_id}` served as `{args.served_model_name}`",
        f"- Successful rows: `{success}/{total}`",
        "- No GPT, Gemini, or Claude calls are made by this script.",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- Validation-selected rows are deployable threshold policies over a local verifier signal.",
        "- Top-test rows are diagnostics only and must not be used as selected results.",
        "- This tests the pre-call version of the same-answer diagnostic from Experiment 195.",
        "",
        "## Artifacts",
        "",
        f"- Oracle target table: `{path.parent / 'table_oracle_targets_local_vs_large.csv'}`",
        f"- Candidate inputs: `{path.parent / 'table_exact_answer_verifier_candidates.csv'}`",
        f"- Verifier outputs: `{path.parent / 'table_exact_answer_verifier_outputs.csv'}`",
        f"- All policies: `{path.parent / 'table_exact_answer_verifier_policy_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_exact_answer_verifier_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_exact_answer_verifier_query_choices.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
