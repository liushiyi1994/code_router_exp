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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FRONTIERS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve", "gpt-5.5"}
THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]
PROMPT_VERSION = "local_support_v1_no_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-agnostic local support verifier for the current Broad100 base policy. "
            "The verifier sees only the query and cheap local candidate answers."
        )
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
            "results/controlled/broad100_current_policy_variable_verifier_fusion/"
            "table_current_policy_variable_verifier_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default="base_current_policy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_current_base_local_support_verifier"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--model-id", default="qwen3-14b-awq-local-support-verifier")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-query-chars", type=int, default=1600)
    parser.add_argument("--max-answer-chars", type=int, default=160)
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
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    base = load_current_base(args.base_query_choices, args.base_policy)
    probe_inputs = build_probe_inputs(outputs, base, splits=set(parse_csv(args.splits)))
    if args.limit is not None:
        probe_inputs = probe_inputs.head(int(args.limit)).copy()
    probe = collect_probe(args, probe_inputs)
    policy_table, query_choices = evaluate_policies(outputs, base, probe, args)
    selected = select_policies(policy_table, int(args.bootstrap_samples), int(args.seed))

    probe_inputs.to_csv(args.output_dir / "table_local_support_verifier_inputs.csv", index=False)
    probe.to_csv(args.output_dir / "table_local_support_verifier_probe.csv", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_local_support_verifier_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_local_support_verifier_policy_selected.csv", index=False)
    selected_methods = set(selected["policy"].astype(str).tolist()) if not selected.empty else set()
    query_choices[query_choices["policy"].astype(str).isin(selected_methods)].to_csv(
        args.output_dir / "table_local_support_verifier_query_choices.csv", index=False
    )
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "LOCAL_SUPPORT_VERIFIER_MEMO.md", args, probe, policy_table, selected)
    print(f"Wrote local support verifier results to {args.output_dir}")


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_current_base(path: Path, policy: str) -> pd.DataFrame:
    base = pd.read_csv(path)
    base = base[base["policy"].astype(str).eq(str(policy))].copy()
    if base.empty:
        raise RuntimeError(f"Policy {policy!r} not found in {path}")
    selected_col = "selected_model" if "selected_model" in base.columns else "selected_model_id"
    keep = [
        "query_id",
        "query_text",
        "benchmark",
        "domain",
        "metric",
        "split",
        selected_col,
        "oracle_utility",
        "oracle_quality",
    ]
    keep = [col for col in keep if col in base.columns]
    base = base[keep].drop_duplicates("query_id").copy()
    base = base.rename(columns={selected_col: "base_model_id"})
    return base.sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def build_probe_inputs(outputs: pd.DataFrame, base: pd.DataFrame, *, splits: set[str]) -> pd.DataFrame:
    local_outputs = outputs[~outputs["model_id"].astype(str).isin(FRONTIERS)].copy()
    local_outputs = local_outputs[local_outputs["status"].astype(str).eq("success")].copy()
    rows_by_query = {
        str(query_id): group.sort_values("model_id").to_dict("records")
        for query_id, group in local_outputs.groupby("query_id", sort=False)
    }
    rows: list[dict[str, Any]] = []
    for item in base.itertuples(index=False):
        split = str(item.split)
        if split not in splits:
            continue
        candidates = []
        for row in rows_by_query.get(str(item.query_id), []):
            answer = str(row.get("parsed_answer", "") or "")
            if not answer:
                continue
            candidates.append(
                {
                    "model_id": str(row["model_id"]),
                    "answer": answer,
                    "is_base": str(row["model_id"]) == str(item.base_model_id),
                }
            )
        rows.append(
            {
                "query_id": str(item.query_id),
                "query_text": str(item.query_text),
                "split": split,
                "benchmark": str(getattr(item, "benchmark", "")),
                "domain": str(getattr(item, "domain", "")),
                "metric": str(getattr(item, "metric", "")),
                "base_model_id": str(item.base_model_id),
                "base_is_frontier": str(item.base_model_id) in FRONTIERS,
                "local_candidate_count": len(candidates),
                "local_candidates_json": json.dumps(candidates, sort_keys=True),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def collect_probe(args: argparse.Namespace, probe_inputs: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_local_support_verifier" / safe_part(str(args.model_id))
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = list(probe_inputs.to_dict("records"))
    if not tasks:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = {executor.submit(call_or_load_one, args, raw_dir, task): task for task in tasks}
        for idx, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if idx % 20 == 0 or idx == len(futures):
                print(f"local-support verifier rows {idx}/{len(futures)}")
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def call_or_load_one(args: argparse.Namespace, raw_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    raw_path = raw_dir / f"{safe_part(str(task['query_id']))}_{cache_digest(task)}.json"
    cache_hit = raw_path.exists() and not bool(args.force_rerun)
    status = "success"
    error_type = ""
    started = time.time()
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
        parsed = {"verdict": "unknown", "supported_model": "", "confidence": 0.0, "reason": ""}
    return {
        **{
            key: task[key]
            for key in [
                "query_id",
                "query_text",
                "split",
                "benchmark",
                "domain",
                "metric",
                "base_model_id",
                "base_is_frontier",
                "local_candidate_count",
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
        "supported_model": parsed["supported_model"],
        "confidence": float(parsed["confidence"]),
        "reason": parsed.get("reason", ""),
    }


def verifier_prompt(task: dict[str, Any], *, max_query_chars: int, max_answer_chars: int) -> str:
    candidates = json.loads(str(task["local_candidates_json"]))
    candidate_lines = []
    for item in candidates:
        marker = " [base]" if bool(item.get("is_base")) else ""
        answer = truncate(str(item.get("answer", "")), max_answer_chars) or "[empty]"
        candidate_lines.append(f"- {item['model_id']}{marker}: {answer}")
    valid_models = ", ".join([str(item["model_id"]) for item in candidates])
    return (
        "You are a local RouteCode probe-state verifier. Use only the task and the listed cheap local "
        "candidate answers to decide whether a local action is reliable enough, or whether the router "
        "should keep/escalate to its nonlocal fallback. Do not use benchmark-specific rules.\n"
        "Return exactly one compact JSON object and no prose:\n"
        '{"verdict":"local_supported|no_reliable_local|unknown","supported_model":"MODEL_ID_OR_NONE","confidence":0.0,"reason":"brief"}\n'
        "- supported_model must be one listed local model_id, or NONE.\n"
        "- Use local_supported only when a listed local answer is likely correct or clearly best supported.\n"
        "- Use no_reliable_local when the local answers are contradictory, malformed, empty, or likely wrong.\n"
        "- confidence is between 0 and 1.\n"
        "- reason must be at most 8 words.\n\n"
        f"Allowed supported_model values: {valid_models}, NONE\n"
        f"Current base action: {task['base_model_id']}\n\n"
        f"Task:\n{truncate(str(task['query_text']), max_query_chars)}\n\n"
        "Cheap local candidate answers:\n"
        + ("\n".join(candidate_lines) if candidate_lines else "[none]")
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
    verdict = verdict.replace("-", "_")
    if verdict not in {"local_supported", "no_reliable_local", "unknown"}:
        lowered = text.lower()
        if "local_supported" in lowered or "supported" in lowered:
            verdict = "local_supported"
        elif "no_reliable" in lowered or "escalate" in lowered:
            verdict = "no_reliable_local"
        else:
            verdict = "unknown"
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
    outputs: pd.DataFrame,
    base: pd.DataFrame,
    probe: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_map = {
        (str(row.query_id), str(row.model_id)): row
        for row in outputs.itertuples(index=False)
    }
    available = {
        str(query_id): set(group["model_id"].astype(str))
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][
        ["query_id", "model_id", "utility", "quality_score"]
    ].rename(columns={"model_id": "oracle_model_id", "utility": "oracle_utility", "quality_score": "oracle_quality"})
    base_eval = base.drop(columns=["oracle_utility", "oracle_quality"], errors="ignore").merge(
        oracle, on="query_id", how="left"
    )
    probe_map = probe.set_index("query_id").to_dict("index") if not probe.empty else {}
    train_ids = outputs.loc[outputs["split"].astype(str).eq("train"), "query_id"].astype(str).unique().tolist()
    fallback_frontier = best_action_for_ids(outputs, train_ids, candidate_models=sorted(FRONTIERS))
    policies = enumerate_policies(fallback_frontier)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for policy in policies:
        choices = apply_policy(base_eval, probe_map, available, policy)
        selected = score_choices(choices, output_map)
        for split, group in selected.groupby("split", sort=False):
            values = group["selected_utility"].astype(float).to_numpy()
            oracle_u = float(group["oracle_utility"].astype(float).mean())
            mean_u = float(values.mean()) if len(values) else float("nan")
            ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
            rows.append(
                {
                    **policy,
                    "split": split,
                    "n_queries": int(len(group)),
                    "mean_quality": float(group["selected_quality"].astype(float).mean()),
                    "mean_utility": mean_u,
                    "mean_utility_ci_low": ci_low,
                    "mean_utility_ci_high": ci_high,
                    "mean_normalized_cost": float(group["selected_normalized_cost"].astype(float).mean()),
                    "oracle_mean_quality": float(group["oracle_quality"].astype(float).mean()),
                    "oracle_mean_utility": oracle_u,
                    "oracle_utility_ratio": mean_u / oracle_u if abs(oracle_u) > 1e-12 else np.nan,
                    "frontier_call_rate": float(group["selected_is_frontier"].astype(bool).mean()),
                    "verifier_call_rate": float(group["verifier_available"].astype(bool).mean()),
                    "local_supported_rate": float(group["verdict"].astype(str).eq("local_supported").mean()),
                    "switch_rate": float(group["switched"].astype(bool).mean()),
                    "escalate_rate": float(group["escalated"].astype(bool).mean()),
                    "_utility_values": json.dumps([float(v) for v in values]),
                }
            )
            if split == "test":
                details.append(group.assign(policy=str(policy["policy"]), family=str(policy["family"])))
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    return table, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def enumerate_policies(fallback_frontier: str) -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = [
        {
            "policy": "current_base",
            "family": "reference",
            "kind": "base",
            "confidence": np.nan,
            "fallback_frontier": fallback_frontier,
        },
        {
            "policy": f"always_{fallback_frontier}",
            "family": "reference",
            "kind": "always_frontier",
            "confidence": np.nan,
            "fallback_frontier": fallback_frontier,
        },
    ]
    for threshold in THRESHOLDS:
        policies.extend(
            [
                {
                    "policy": f"switch_any_supported_conf{threshold:g}",
                    "family": "local_support_switch",
                    "kind": "switch_any",
                    "confidence": float(threshold),
                    "fallback_frontier": fallback_frontier,
                },
                {
                    "policy": f"downshift_frontier_supported_conf{threshold:g}",
                    "family": "frontier_cost_suppression",
                    "kind": "downshift_frontier",
                    "confidence": float(threshold),
                    "fallback_frontier": fallback_frontier,
                },
                {
                    "policy": f"local_or_frontier_fallback_conf{threshold:g}",
                    "family": "local_support_cascade",
                    "kind": "local_or_frontier",
                    "confidence": float(threshold),
                    "fallback_frontier": fallback_frontier,
                },
                {
                    "policy": f"escalate_unreliable_base_local_conf{threshold:g}",
                    "family": "local_risk_escalation",
                    "kind": "escalate_unreliable_base_local",
                    "confidence": float(threshold),
                    "fallback_frontier": fallback_frontier,
                },
            ]
        )
    return policies


def apply_policy(
    base: pd.DataFrame,
    probe_map: dict[str, dict[str, Any]],
    available: dict[str, set[str]],
    policy: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in base.itertuples(index=False):
        query_id = str(item.query_id)
        base_model = str(item.base_model_id)
        selected = base_model
        probe = probe_map.get(query_id, {})
        verifier_available = bool(probe)
        verdict = str(probe.get("verdict", "unknown"))
        supported = str(probe.get("supported_model", "") or "")
        confidence = float(probe.get("confidence", 0.0) or 0.0)
        threshold = float(policy.get("confidence", 0.0) or 0.0)
        fallback = str(policy.get("fallback_frontier", ""))
        supported_valid = bool(
            verifier_available
            and verdict == "local_supported"
            and confidence >= threshold
            and supported
            and supported in available.get(query_id, set())
        )
        no_reliable = bool(verifier_available and verdict == "no_reliable_local" and confidence >= threshold)
        escalated = False
        if policy["kind"] == "always_frontier":
            selected = fallback if fallback in available.get(query_id, set()) else base_model
        elif policy["kind"] == "switch_any":
            if supported_valid:
                selected = supported
        elif policy["kind"] == "downshift_frontier":
            if base_model in FRONTIERS and supported_valid:
                selected = supported
        elif policy["kind"] == "local_or_frontier":
            if supported_valid:
                selected = supported
            elif fallback in available.get(query_id, set()):
                selected = fallback
                escalated = selected != base_model
        elif policy["kind"] == "escalate_unreliable_base_local":
            if supported_valid:
                selected = supported
            elif no_reliable and base_model not in FRONTIERS and fallback in available.get(query_id, set()):
                selected = fallback
                escalated = True
        rows.append(
            {
                "query_id": query_id,
                "query_text": str(item.query_text),
                "split": str(item.split),
                "benchmark": str(getattr(item, "benchmark", "")),
                "metric": str(getattr(item, "metric", "")),
                "base_model_id": base_model,
                "selected_model_id": selected,
                "oracle_model_id": str(getattr(item, "oracle_model_id", "")),
                "oracle_utility": float(getattr(item, "oracle_utility", np.nan)),
                "oracle_quality": float(getattr(item, "oracle_quality", np.nan)),
                "verifier_available": verifier_available,
                "verdict": verdict,
                "supported_model": supported,
                "confidence": confidence,
                "switched": selected != base_model,
                "escalated": escalated,
            }
        )
    return pd.DataFrame(rows)


def score_choices(choices: pd.DataFrame, output_map: dict[tuple[str, str], Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in choices.itertuples(index=False):
        row = output_map.get((str(item.query_id), str(item.selected_model_id)))
        if row is None:
            selected_quality = 0.0
            selected_utility = -1e9
            selected_cost = 0.0
            selected_frontier = False
            selected_answer = ""
        else:
            selected_quality = safe_float(getattr(row, "quality_score", 0.0), default=0.0)
            selected_utility = safe_float(getattr(row, "utility", -1e9), default=-1e9)
            selected_cost = safe_float(getattr(row, "normalized_remote_cost", 0.0), default=0.0)
            selected_frontier = bool(getattr(row, "is_frontier", False))
            selected_answer = str(getattr(row, "parsed_answer", "") or "")
        rows.append(
            {
                **item._asdict(),
                "selected_quality": selected_quality,
                "selected_utility": selected_utility,
                "selected_normalized_cost": selected_cost,
                "selected_is_frontier": selected_frontier,
                "selected_answer": selected_answer,
            }
        )
    return pd.DataFrame(rows)


def safe_float(value: Any, *, default: float) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def best_action_for_ids(outputs: pd.DataFrame, ids: list[str], *, candidate_models: list[str]) -> str:
    frame = outputs[
        outputs["query_id"].astype(str).isin(set(ids))
        & outputs["model_id"].astype(str).isin(set(candidate_models))
    ].copy()
    if frame.empty:
        return sorted(candidate_models)[0]
    means = frame.groupby("model_id")["utility"].mean().sort_values(ascending=False)
    return str(means.index[0])


def select_policies(table: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    candidates = table[table["split"].eq("val") & ~table["family"].eq("reference")].copy()
    if not candidates.empty:
        best = candidates.sort_values(
            ["mean_utility", "frontier_call_rate", "switch_rate"],
            ascending=[False, True, True],
        ).head(1)
        policy = str(best.iloc[0]["policy"])
        rows.append(best.assign(selection_rule="val_best_mean_utility"))
        rows.append(
            table[table["split"].eq("test") & table["policy"].eq(policy)]
            .copy()
            .assign(selection_rule="val_best_mean_utility_test")
        )
    refs = table[table["family"].eq("reference") & table["split"].eq("test")].copy()
    if not refs.empty:
        rows.append(refs.assign(selection_rule="reference_test"))
    top = (
        table[table["split"].eq("test") & ~table["family"].eq("reference")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(12)
    )
    if not top.empty:
        rows.append(top.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    return add_bootstrap_ci(selected, bootstrap_samples, seed).drop(columns=["_utility_values"], errors="ignore")


def add_bootstrap_ci(table: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows = []
    for row in table.to_dict("records"):
        values = np.asarray(json.loads(str(row.get("_utility_values", "[]"))), dtype=float)
        low, high = bootstrap_ci(values, bootstrap_samples, seed)
        row["mean_utility_ci_low"] = low
        row["mean_utility_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:96]


def cache_digest(task: dict[str, Any]) -> str:
    text = (
        f"{PROMPT_VERSION}|{task['query_id']}|{task['base_model_id']}|"
        f"{task['local_candidates_json']}"
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def truncate(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.70)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["policy"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#597c6b")
    ax.set_xlabel("Held-out test utility")
    ax.set_title("Current-Base Local Support Verifier")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_local_support_verifier_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, probe: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    selected_cols = [
        "selection_rule",
        "policy",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "mean_normalized_cost",
        "frontier_call_rate",
        "verifier_call_rate",
        "local_supported_rate",
        "switch_rate",
        "escalate_rate",
    ]
    probe_diag = (
        probe.groupby(["split", "verdict"], dropna=False)
        .agg(n=("query_id", "size"), mean_confidence=("confidence", "mean"), mean_latency_s=("latency_s", "mean"))
        .reset_index()
        if not probe.empty
        else pd.DataFrame()
    )
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    text = [
        "# Current-Base Local Support Verifier",
        "",
        "This experiment tests a benchmark-agnostic ProbeCode bridge:",
        "",
        "```text",
        "query + cheap local candidate answers -> local support state -> cost-aware action",
        "```",
        "",
        "The local verifier prompt omits benchmark ID and does not use task-specific checkers.",
        "It makes no GPT, Gemini, or Claude calls.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/205_current_base_local_support_verifier.py",
        "PYTHONPATH=src python experiments/205_current_base_local_support_verifier.py",
        "```",
        "",
        "## Run Config",
        "",
        f"- Served local verifier: `{args.served_model_name}` via `{args.base_url}`",
        f"- Probe rows: `{len(probe)}`",
        f"- Cache-hit rate: `{float(probe['cache_hit'].mean()) if len(probe) else 0.0:.4f}`",
        f"- Splits: `{args.splits}`",
        f"- Prompt version: `{PROMPT_VERSION}`",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[selected_cols]) if not selected.empty else "_No selected rows._",
        "",
        "## Probe Verdict Diagnostics",
        "",
        markdown_table(probe_diag) if not probe_diag.empty else "_No probe rows._",
        "",
        "## Top Held-Out Test Rows",
        "",
        markdown_table(
            top_test[
                [
                    "policy",
                    "family",
                    "mean_quality",
                    "mean_utility",
                    "oracle_utility_ratio",
                    "mean_normalized_cost",
                    "frontier_call_rate",
                    "switch_rate",
                ]
            ]
        ),
        "",
        "## Interpretation",
        "",
        "- Main rows are benchmark-agnostic; they use only query text and cheap local candidate answers.",
        "- `local_support_cascade` is the strongest deployable-style form: local answer if supported, otherwise train-best frontier fallback.",
        "- `frontier_cost_suppression` only downshifts current frontier calls when local evidence is confident.",
        "- Validation chooses the reported policy; top test rows are diagnostic and should not be treated as selected methods.",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    display = frame.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(col) for col in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell.replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
