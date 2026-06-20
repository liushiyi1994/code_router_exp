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

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import score_output


DEFAULT_BENCHMARKS = "gpqa,mmlupro,bbh,gsm8k,math500,livemathbench,aime"
DEFAULT_SCOPES = [
    ("gpqa",),
    ("mmlupro",),
    ("bbh",),
    ("gsm8k",),
    ("math500",),
    ("livemathbench",),
    ("aime",),
    ("gpqa", "mmlupro"),
    ("gpqa", "bbh", "gsm8k", "mmlupro"),
    ("gpqa", "mmlupro", "bbh", "gsm8k", "math500", "livemathbench", "aime"),
]
LOCAL_THRESHOLDS = [0.0, 0.5, 0.7, 0.85]
LOCAL_MODES = ["always", "if_selected_qwen32", "if_not_frontier"]
MODEL_PRIORITY = [
    "deterministic_math_tool",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "qwen3-32b-awq-local",
    "qwen3-14b-awq-local",
    "qwen3-8b-local",
    "qwen3-4b-local",
    "gemini-3.5-flash-strong-solve",
    "gpt-5.5",
    "gemini-3.5-flash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local vLLM solve-and-support verifier fused with residual repair.")
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
            "results/controlled/broad100_targeted_residual_repair_policy/"
            "table_targeted_residual_repair_query_choices.csv"
        ),
    )
    parser.add_argument(
        "--base-policy",
        default="scopegpqa+bbh+gsm8k+mmlupro_selected_qwen32_qwen3-14b-awq-local_none",
    )
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--model-id", default="qwen3-14b-awq-solve-support-verifier")
    parser.add_argument("--max-query-chars", type=int, default=1500)
    parser.add_argument("--max-answer-chars", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--reliable-quality-threshold", type=float, default=0.65)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_local_vllm_solve_support_residual_fusion"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fusion = load_module("experiments/191_variable_verifier_residual_fusion.py", "fusion_191_for_193")

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    benchmarks = {item.strip().lower() for item in str(args.benchmarks).split(",") if item.strip()}
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    query_frame = (
        outputs[
            outputs["split"].astype(str).isin(splits)
            & outputs["benchmark"].astype(str).str.lower().isin(benchmarks)
        ]
        .drop_duplicates("query_id")
        .copy()
        .sort_values(["split", "benchmark", "query_id"])
    )
    if args.limit is not None:
        query_frame = query_frame.head(int(args.limit)).copy()
    verifier = collect_verifier_rows(args, query_frame, outputs)
    verifier.to_csv(args.output_dir / "table_local_vllm_solve_support_verifier_outputs.csv", index=False)

    base = pd.read_csv(args.base_query_choices)
    base = base[base["policy"].astype(str).eq(str(args.base_policy))].copy()
    if base.empty:
        raise RuntimeError(f"Base policy {args.base_policy!r} not found in {args.base_query_choices}.")
    action_map = {str(query_id): group.set_index("model_id").to_dict("index") for query_id, group in outputs.groupby("query_id")}
    oracle = outputs.loc[outputs.groupby("query_id")["utility"].idxmax()][["query_id", "utility", "quality_score"]].rename(
        columns={"utility": "oracle_utility", "quality_score": "oracle_quality"}
    )
    base = fusion.drop_prefixed(base, ["oracle_utility", "oracle_quality"]).merge(oracle, on="query_id", how="left")
    verifier_diag = fusion.verifier_diagnostics(verifier)
    rules = enumerate_local_rules()
    fusion_args = argparse.Namespace(
        lambda_cost=float(args.lambda_cost),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
        reliable_quality_threshold=float(args.reliable_quality_threshold),
    )
    policy_table, query_choices = fusion.evaluate_rules(base, verifier, outputs, action_map, rules, fusion_args)
    selected = fusion.selected_rows(policy_table, fusion_args)
    selected = append_reliable_selection(
        selected,
        policy_table,
        verifier_diag,
        threshold=float(args.reliable_quality_threshold),
    )
    selected_policies = set(selected["policy"].astype(str).tolist()) if not selected.empty else set()
    query_choices_to_write = query_choices[query_choices["policy"].astype(str).isin(selected_policies)].copy()

    verifier_diag.to_csv(args.output_dir / "table_local_vllm_solve_support_verifier_diagnostics.csv", index=False)
    pd.DataFrame(rules).to_csv(args.output_dir / "table_local_vllm_solve_support_rule_library.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_local_vllm_solve_support_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_local_vllm_solve_support_policy_selected.csv", index=False)
    query_choices_to_write.to_csv(args.output_dir / "table_local_vllm_solve_support_query_choices.csv", index=False)
    write_memo(args.output_dir / "LOCAL_VLLM_SOLVE_SUPPORT_RESIDUAL_FUSION_MEMO.md", args, verifier, verifier_diag, selected)
    print(f"Wrote local vLLM solve-support residual fusion results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def enumerate_local_rules() -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = [
        {
            "policy": "base_targeted_residual_repair",
            "scope_json": "[]",
            "threshold": np.nan,
            "mode": "base",
            "selector_kind": "base",
        }
    ]
    for scope in DEFAULT_SCOPES:
        for threshold in LOCAL_THRESHOLDS:
            for mode in LOCAL_MODES:
                rules.append(
                    {
                        "policy": f"scope{'+'.join(scope)}_thr{threshold:g}_{mode}",
                        "scope_json": json.dumps(scope),
                        "threshold": float(threshold),
                        "mode": mode,
                        "selector_kind": "local_grid",
                    }
                )
    return rules


def append_reliable_selection(
    selected: pd.DataFrame,
    policy_table: pd.DataFrame,
    verifier_diag: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    reliable = set(
        verifier_diag[
            verifier_diag["split"].astype(str).eq("val")
            & verifier_diag["verifier_quality"].astype(float).ge(float(threshold))
        ]["benchmark"].astype(str)
    )
    val = policy_table[policy_table["split"].astype(str).eq("val")].copy()
    test = policy_table[policy_table["split"].astype(str).eq("test")].copy()
    base_val = val[val["policy"].astype(str).eq("base_targeted_residual_repair")]
    base_test = test[test["policy"].astype(str).eq("base_targeted_residual_repair")]
    if base_val.empty:
        return selected
    base_metric = float(base_val.iloc[0]["mean_utility_with_probe_cost"])
    candidates: list[pd.Series] = []
    for _, row in val.iterrows():
        if str(row["policy"]) == "base_targeted_residual_repair":
            continue
        scope = set(json.loads(str(row.get("scope_json", "[]"))))
        if not scope or not scope.issubset(reliable):
            continue
        if float(row["mean_utility_with_probe_cost"]) > base_metric + 1e-12:
            candidates.append(row)
    if candidates:
        candidate_frame = pd.DataFrame(candidates)
        best = candidate_frame.sort_values(
            ["mean_utility_with_probe_cost", "probe_call_rate", "frontier_call_rate"],
            ascending=[False, True, True],
        ).iloc[0].copy()
        best["selection_rule"] = "val_best_reliable_verifier_quality"
        rows.append(best)
        match = test[test["policy"].astype(str).eq(str(best["policy"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = "val_best_reliable_verifier_quality_test"
            rows.append(test_row)
    else:
        base_row = base_val.iloc[0].copy()
        base_row["selection_rule"] = "val_best_reliable_verifier_quality_fallback_base"
        rows.append(base_row)
        if not base_test.empty:
            test_row = base_test.iloc[0].copy()
            test_row["selection_rule"] = "val_best_reliable_verifier_quality_fallback_base_test"
            rows.append(test_row)
    if not rows:
        return selected
    appended = pd.DataFrame(rows)
    return pd.concat([selected, appended], ignore_index=True).drop_duplicates(
        ["selection_rule", "policy", "split"],
        keep="first",
    )


def collect_verifier_rows(args: argparse.Namespace, queries: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_local_vllm_solve_support" / safe_part(str(args.model_id)) / f"max_{int(args.max_tokens)}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_map = {str(query_id): group.copy() for query_id, group in outputs.groupby("query_id", sort=False)}
    tasks = [
        build_task(row, output_map.get(str(row["query_id"]), pd.DataFrame()), args)
        for _, row in queries.iterrows()
    ]
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = {executor.submit(call_or_load_one, args, raw_dir, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"local solve-support verifier rows {index}/{len(futures)}")
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"])


def build_task(row: pd.Series, candidates: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    actions = candidate_actions(candidates, max_answer_chars=int(args.max_answer_chars))
    return {
        "query_id": str(row["query_id"]),
        "split": str(row["split"]),
        "benchmark": str(row["benchmark"]),
        "domain": str(row["domain"]),
        "metric": str(row["metric"]),
        "query_text": str(row["query_text"]),
        "gold_answer": str(row["gold_answer"]),
        "candidate_actions": actions,
    }


def candidate_actions(candidates: pd.DataFrame, *, max_answer_chars: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for model_id in MODEL_PRIORITY:
        group = candidates[candidates["model_id"].astype(str).eq(model_id)]
        if group.empty:
            continue
        answer = compact(str(group.iloc[0].get("parsed_answer", "") or ""), max_answer_chars)
        if not answer or answer.lower() in {"nan", "none"}:
            continue
        rows.append({"model_id": model_id, "answer": answer})
        seen.add(model_id)
    for item in candidates.itertuples(index=False):
        model_id = str(item.model_id)
        if model_id in seen:
            continue
        answer = compact(str(getattr(item, "parsed_answer", "") or ""), max_answer_chars)
        if answer and answer.lower() not in {"nan", "none"}:
            rows.append({"model_id": model_id, "answer": answer})
    return rows


def call_or_load_one(args: argparse.Namespace, raw_dir: Path, task: dict[str, Any]) -> dict[str, Any]:
    raw_path = raw_dir / f"{safe_part(task['query_id'])}_{cache_digest(task, args)}.json"
    cache_hit = raw_path.exists() and not bool(args.force_rerun)
    started = time.time()
    status = "success"
    error_type = ""
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        status = str(payload.get("_status", "success"))
        error_type = str(payload.get("_error_type", ""))
    else:
        prompt = verifier_prompt(task, max_query_chars=int(args.max_query_chars))
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
    parsed = parse_json(raw_text)
    answer = str(parsed.get("answer", ""))
    parsed_answer, quality = score_output(answer, str(task["gold_answer"]), str(task["metric"]))
    if status != "success":
        quality = np.nan
    supported = choose_supported_model(parsed_answer, parsed, task["candidate_actions"])
    return {
        "query_id": task["query_id"],
        "split": task["split"],
        "benchmark": task["benchmark"],
        "domain": task["domain"],
        "metric": task["metric"],
        "query_text": task["query_text"],
        "gold_answer": task["gold_answer"],
        "provider_model": str(args.served_model_name),
        "status": status,
        "error_type": error_type,
        "raw_text": raw_text,
        "verifier_answer": answer,
        "parsed_answer": parsed_answer,
        "quality_score": float(quality) if not pd.isna(quality) else np.nan,
        "verifier_confidence": float(parsed.get("confidence", 0.0)),
        "supported_model": supported,
        "verifier_reason": str(parsed.get("reason", ""))[:240],
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_total_usd": 0.0,
        "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
        "cache_hit": bool(cache_hit),
        "raw_output_path": str(raw_path),
    }


def verifier_prompt(task: dict[str, Any], *, max_query_chars: int) -> str:
    metric = str(task["metric"])
    benchmark = str(task["benchmark"])
    if metric == "multiple_choice":
        letters = infer_option_letters(str(task["query_text"]), task["candidate_actions"])
        answer_instruction = f"Your answer must be exactly one option letter from: {'|'.join(letters)}."
    elif benchmark in {"aime", "gsm8k", "math500", "livemathbench"}:
        answer_instruction = "Your answer must be the final exact value only."
    else:
        answer_instruction = "Your answer must be the final exact answer only."
    candidates = "\n".join(
        f"{item['model_id']}: {item['answer']}" for item in task["candidate_actions"]
    )
    return (
        "You are a local RouteCode solve-and-support verifier. Solve independently, then identify whether one listed candidate matches your trusted final answer.\n"
        "Return compact JSON only with keys answer, confidence, supported_model, reason.\n"
        f"{answer_instruction}\n"
        "supported_model must be exactly one listed model id if its answer matches your answer; otherwise use NONE.\n"
        "confidence is 0 to 1. reason is at most 8 words.\n\n"
        f"Benchmark: {benchmark}\n"
        f"Metric: {metric}\n"
        f"Task:\n{compact(str(task['query_text']), max_query_chars)}\n\n"
        "Candidate final answers:\n"
        f"{candidates}\n"
        "Return only the JSON object."
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


def parse_json(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", str(text), flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            maybe = json.loads(match.group(0))
            if isinstance(maybe, dict):
                parsed = maybe
        except json.JSONDecodeError:
            parsed = {}
    answer = str(parsed.get("answer", "")).strip()
    if not answer:
        mcq = re.search(r"\b([A-J])\b", clean.upper())
        answer = mcq.group(1) if mcq else clean[:64]
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


def choose_supported_model(parsed_answer: str, parsed: dict[str, Any], candidates: list[dict[str, str]]) -> str:
    declared = str(parsed.get("supported_model", "") or "").strip()
    candidate_ids = {item["model_id"] for item in candidates}
    if declared in candidate_ids:
        return declared
    if declared.upper() == "NONE":
        return ""
    normalized_answer = normalize_answer(parsed_answer)
    if not normalized_answer:
        return ""
    for model_id in MODEL_PRIORITY:
        for item in candidates:
            if item["model_id"] == model_id and normalize_answer(item["answer"]) == normalized_answer:
                return model_id
    for item in candidates:
        if normalize_answer(item["answer"]) == normalized_answer:
            return item["model_id"]
    return ""


def normalize_answer(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().upper()
    text = text.removeprefix("ANSWER:").strip()
    text = text.strip("$")
    return text


def infer_option_letters(query_text: str, candidates: list[dict[str, str]]) -> list[str]:
    letters = sorted(set(re.findall(r"(?m)(?:^|\n)\s*([A-J])(?:[\).:]|\s*[-])\s+", query_text)))
    if not letters:
        letters = sorted({normalize_answer(item["answer"]) for item in candidates if re.fullmatch(r"[A-J]", normalize_answer(item["answer"]))})
    return letters or list("ABCD")


def cache_digest(task: dict[str, Any], args: argparse.Namespace) -> str:
    text = json.dumps(
        {
            "query_id": task["query_id"],
            "model": args.served_model_name,
            "max_tokens": int(args.max_tokens),
            "version": "local_solve_support_v1",
            "candidates": task["candidate_actions"],
        },
        sort_keys=True,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:96]


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.72)].rstrip() + " ... " + text[-int(max_chars * 0.20) :].lstrip()


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, verifier_diag: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "policy",
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
    ]
    start_command = "scripts/start_vllm_qwen3_32b_awq.sh" if "32B" in str(args.served_model_name) else "scripts/start_vllm_qwen3_14b_awq.sh"
    stop_port = "8007" if "8007" in str(args.base_url) else "8006"
    lines = [
        "# Local vLLM Solve-Support Residual Fusion",
        "",
        "This tests a local solve-and-support verifier fused with Experiment 189 residual repair.",
        "It makes no GPT, Gemini, or Claude calls; local vLLM probe cost is remote-dollar zero.",
        "",
        "## Commands",
        "",
        "```bash",
        start_command,
        (
            "PYTHONPATH=src python experiments/193_local_vllm_solve_support_residual_fusion.py"
            if stop_port == "8006"
            else "PYTHONPATH=src python experiments/193_local_vllm_solve_support_residual_fusion.py "
            "--base-url http://127.0.0.1:8007/v1 --served-model-name Qwen/Qwen3-32B-AWQ "
            "--model-id qwen3-32b-awq-solve-support-verifier "
            "--output-dir results/controlled/broad100_local_vllm_solve_support_residual_fusion_qwen32"
        ),
        f"scripts/stop_vllm_port.sh {stop_port}",
        "```",
        "",
        "## Verifier Rows",
        "",
        f"- Served model: `{args.served_model_name}` via `{args.base_url}`",
        f"- Rows: `{len(verifier)}`",
        f"- Success rows: `{int(verifier['status'].astype(str).eq('success').sum()) if not verifier.empty else 0}`",
        f"- Cache-hit rate: `{float(verifier['cache_hit'].mean()) if not verifier.empty else 0.0:.4f}`",
        f"- Mean latency seconds: `{float(verifier['latency_s'].mean()) if not verifier.empty else 0.0:.4f}`",
        "",
        "## Verifier Diagnostics",
        "",
        markdown_table(verifier_diag) if not verifier_diag.empty else "No verifier diagnostics.",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- A deployable improvement must beat Experiment 189's `0.7736` held-out utility.",
        "- This branch treats local verifier calls as zero remote-dollar cost, but reports probe-call rate and latency.",
        "- The verifier is allowed to support only candidate answers already in the action matrix.",
        "",
        "## Artifacts",
        "",
        f"- Verifier outputs: `{path.parent / 'table_local_vllm_solve_support_verifier_outputs.csv'}`",
        f"- Policy table: `{path.parent / 'table_local_vllm_solve_support_policy_all.csv'}`",
        f"- Selected policies: `{path.parent / 'table_local_vllm_solve_support_policy_selected.csv'}`",
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
