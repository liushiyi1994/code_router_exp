from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


OPTION_BENCHMARKS = {"gpqa", "mmlupro"}
LOCAL_MODELS = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
]
STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a local vLLM option-comparison probe for GPQA/MMLUPro.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--augmented-outputs",
        type=Path,
        default=Path("results/controlled/broad100_train_supervised_strong_gain_gate/model_outputs_with_gemini_strong_all_splits.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_vllm_option_compare_probe"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--benchmarks", default="gpqa,mmlupro")
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_option_compare" / safe_name(args.served_model_name)
    raw_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    base_outputs = package.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    outputs = (
        load_precomputed_outputs(args.augmented_outputs, lambda_cost=float(args.lambda_cost))
        if args.augmented_outputs.exists()
        else base_outputs
    )

    probe = collect_probe_rows(base_outputs, args=args, raw_dir=raw_dir, package=package)
    eval_table, selected = evaluate_probe(package, outputs, probe, lambda_cost=float(args.lambda_cost))
    probe.to_csv(args.output_dir / "table_vllm_option_compare_probe.csv", index=False)
    eval_table.to_csv(args.output_dir / "table_vllm_option_compare_eval.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_option_compare_selected.csv", index=False)
    write_memo(args.output_dir, args, probe, eval_table, selected)
    print(f"Wrote vLLM option compare probe results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_precomputed_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce").fillna(0.0)
    for column in ["cost_total_usd", "latency_s", "normalized_remote_cost"]:
        if column not in outputs:
            outputs[column] = 0.0
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    if "tool_available" not in outputs:
        outputs["tool_available"] = False
    return outputs


def collect_probe_rows(base_outputs: pd.DataFrame, *, args: argparse.Namespace, raw_dir: Path, package) -> pd.DataFrame:
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    benchmarks = {item.strip() for item in str(args.benchmarks).split(",") if item.strip()}
    queries = (
        base_outputs[
            base_outputs["split"].astype(str).isin(splits)
            & base_outputs["benchmark"].astype(str).isin(benchmarks)
        ]
        .drop_duplicates("query_id")
        .sort_values(["split", "benchmark", "query_id"])
        .copy()
    )
    if args.max_rows is not None:
        queries = queries.head(int(args.max_rows)).copy()
    by_query = base_outputs.set_index(["query_id", "model_id"])
    rows: list[dict[str, Any]] = []
    total = len(queries)
    for index, (_, row) in enumerate(queries.iterrows(), start=1):
        payload = call_probe(row, by_query, args=args, raw_dir=raw_dir)
        parsed = parse_probe_text(str(payload.get("_text", "")))
        query_id = str(row["query_id"])
        chosen_model = resolve_selected_model(parsed, by_query, query_id)
        rows.append(
            {
                "query_id": query_id,
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "domain": str(row["domain"]),
                "predicted_answer": parsed["predicted_answer"],
                "selected_model": chosen_model,
                "selected_confidence": float(parsed["confidence"]),
                "need_strong": bool(parsed["need_strong"]),
                "reason": parsed["reason"],
                "status_code": int(payload.get("_status_code", 0) or 0),
                "cache_hit": bool(payload.get("_cache_hit", False)),
                "latency_s": float(payload.get("_latency_s", 0.0) or 0.0),
                "input_tokens": int(payload.get("_input_tokens", 0) or 0),
                "output_tokens": int(payload.get("_output_tokens", 0) or 0),
                "raw_path": str(raw_dir / cache_name(query_id, args.served_model_name)),
                "raw_text": str(payload.get("_text", "")),
            }
        )
        if index % 10 == 0 or index == total:
            print(f"option compare rows {index}/{total}")
    return pd.DataFrame(rows)


def call_probe(row: pd.Series, by_query: pd.DataFrame, *, args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    query_id = str(row["query_id"])
    raw_path = raw_dir / cache_name(query_id, args.served_model_name)
    if raw_path.exists() and not args.force:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        payload["_cache_hit"] = True
        return payload
    prompt = build_prompt(row, by_query)
    started = time.time()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        json={
            "model": args.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": int(args.max_output_tokens),
        },
        timeout=float(args.request_timeout_s),
    )
    payload: dict[str, Any] = {
        "_query_id": query_id,
        "_status_code": int(response.status_code),
        "_latency_s": float(time.time() - started),
    }
    if response.status_code == 200:
        body = response.json()
        payload["_response"] = body
        payload["_text"] = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage", {})
        if isinstance(usage, dict):
            payload["_input_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
            payload["_output_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    else:
        payload["_text"] = response.text[:2000]
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["_cache_hit"] = False
    return payload


def build_prompt(row: pd.Series, by_query: pd.DataFrame) -> str:
    query_id = str(row["query_id"])
    candidate_lines = []
    for model_id in LOCAL_MODELS:
        try:
            model_row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        answer = normalize_option(model_row.get("parsed_answer", ""))
        raw_answer = compact(str(model_row.get("parsed_answer", "")), 80)
        status = str(model_row.get("status", ""))
        candidate_lines.append(f"- {model_id}: status={status}; option={answer}; raw={raw_answer or '[empty]'}")
    query = compact(str(row["query_text"]), 2200)
    return (
        "You are a multiple-choice option verifier for model routing.\n"
        "You see the question and cached option letters from local models. You do not see the gold answer.\n"
        "Independently decide the most likely correct option. Then choose the local model whose option should be used.\n"
        "If no local option looks reliable, set need_strong=true.\n"
        "Return JSON only with keys predicted_answer, selected_model, confidence, need_strong, reason.\n"
        "predicted_answer must be A, B, C, D, or UNKNOWN. selected_model must be one of: "
        + ", ".join([*LOCAL_MODELS, "NONE"])
        + ". confidence is 0 to 1.\n\n"
        f"Benchmark: {row['benchmark']}\n"
        f"Question:\n{query}\n\n"
        "Local option answers:\n"
        + "\n".join(candidate_lines)
        + '\n\nExample: {"predicted_answer":"B","selected_model":"qwen3-14b-awq-local","confidence":0.72,"need_strong":false,"reason":"short"}\n/no_think'
    )


def parse_probe_text(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    predicted = normalize_option(parsed.get("predicted_answer", ""))
    selected_model = str(parsed.get("selected_model", "")).strip()
    if selected_model not in LOCAL_MODELS:
        for model_id in LOCAL_MODELS:
            if model_id in clean:
                selected_model = model_id
                break
    if selected_model not in LOCAL_MODELS:
        selected_model = "NONE"
    try:
        confidence = float(parsed.get("confidence", np.nan))
    except (TypeError, ValueError):
        confidence = np.nan
    if np.isnan(confidence):
        confidence = 0.0
    need_raw = parsed.get("need_strong", selected_model == "NONE")
    need_strong = str(need_raw).strip().lower() in {"true", "yes", "1"} if isinstance(need_raw, str) else bool(need_raw)
    return {
        "predicted_answer": predicted,
        "selected_model": selected_model,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "need_strong": bool(need_strong or selected_model == "NONE"),
        "reason": compact(str(parsed.get("reason", "")), 240),
    }


def resolve_selected_model(parsed: dict[str, Any], by_query: pd.DataFrame, query_id: str) -> str:
    selected = str(parsed["selected_model"])
    if selected in LOCAL_MODELS and action_available(by_query, query_id, selected):
        return selected
    predicted = str(parsed["predicted_answer"])
    if predicted in {"A", "B", "C", "D"}:
        for model_id in ["qwen3-14b-awq-local", "qwen3-32b-awq-local", "qwen3-4b-local", "qwen3-8b-local"]:
            try:
                row = by_query.loc[(query_id, model_id)]
            except KeyError:
                continue
            if normalize_option(row.get("parsed_answer", "")) == predicted and action_available(by_query, query_id, model_id):
                return model_id
    return "NONE"


def evaluate_probe(package, outputs: pd.DataFrame, probe: pd.DataFrame, *, lambda_cost: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    base = {split: normalize_selection(package.observable_local_state_selection(outputs, split=split)) for split in ["val", "test"]}
    tool = {split: normalize_selection(package.profile_v4_selection_for_split(outputs, split=split)) for split in ["val", "test"]}
    base_methods = {"observable_local_state_v5": base, "tool_probe_profile_v4": tool}
    for base_name, by_split in base_methods.items():
        for threshold in [0.0, 0.4, 0.55, 0.7, 0.85, 0.95]:
            for fallback_mode in ["base", "strong_if_needed", "strong_if_low_conf"]:
                method = f"{base_name}_vllm_option_compare_t{threshold:g}_{fallback_mode}"
                for split in ["val", "test"]:
                    split_probe = probe[probe["split"].eq(split)].copy()
                    selected = apply_policy(outputs, by_split[split], split_probe, threshold=threshold, fallback_mode=fallback_mode, split=split)
                    row = evaluate_selection(package, outputs, selected, split=split, method=method, lambda_cost=lambda_cost)
                    row["selector_threshold"] = float(threshold)
                    row["fallback_mode"] = fallback_mode
                    rows.append(row)
        for split in ["val", "test"]:
            rows.append(evaluate_selection(package, outputs, by_split[split], split=split, method=base_name, lambda_cost=lambda_cost))
    eval_table = pd.DataFrame(rows)
    return eval_table, validation_selected(eval_table)


def apply_policy(
    outputs: pd.DataFrame,
    base: pd.Series,
    probe: pd.DataFrame,
    *,
    threshold: float,
    fallback_mode: str,
    split: str,
) -> pd.Series:
    selected = normalize_selection(base)
    query_info = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    for _, row in probe.iterrows():
        query_id = str(row["query_id"])
        if query_id not in query_info.index:
            continue
        if str(query_info.loc[query_id, "benchmark"]) not in OPTION_BENCHMARKS:
            continue
        confidence = float(row["selected_confidence"])
        model_id = str(row["selected_model"])
        use_strong = False
        if fallback_mode == "strong_if_needed":
            use_strong = bool(row["need_strong"]) or confidence < float(threshold) or model_id == "NONE"
        elif fallback_mode == "strong_if_low_conf":
            use_strong = confidence < float(threshold) or model_id == "NONE"
        elif confidence < float(threshold) or model_id == "NONE":
            model_id = str(base.loc[query_id])
        if use_strong and STRONG_MODEL_ID in set(outputs["model_id"].astype(str)):
            model_id = STRONG_MODEL_ID
        selected.loc[query_id] = model_id
    return selected.astype(str)


def evaluate_selection(package, outputs: pd.DataFrame, selected: pd.Series, *, split: str, method: str, lambda_cost: float) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean()) if STRONG_MODEL_ID in set(outputs["model_id"].astype(str)) else 0.0
    return row


def validation_selected(eval_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = eval_table[eval_table["split"].eq("val")].copy()
    if not val.empty:
        best = val.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1)
        if not best.empty:
            method = str(best.iloc[0]["method"])
            rows.append(best.assign(selection_rule="val_best_utility"))
            test = eval_table[eval_table["split"].eq("test") & eval_table["method"].eq(method)]
            if not test.empty:
                rows.append(test.assign(selection_rule="val_best_utility_test"))
    top_test = eval_table[eval_table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(8)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def action_available(by_query: pd.DataFrame, query_id: str, model_id: str) -> bool:
    try:
        row = by_query.loc[(query_id, model_id)]
    except KeyError:
        return False
    return str(row.get("status", "success")) == "success"


def normalize_option(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    if len(text) >= 1 and text[0] in {"A", "B", "C", "D"}:
        return text[0]
    return "UNKNOWN"


def cache_name(query_id: str, model_name: str) -> str:
    digest = hashlib.sha1(f"{query_id}:{model_name}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:90]
    return f"{safe}_{digest}.json"


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:80]


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.75)].rstrip() + " ... " + text[-int(max_chars * 0.18) :].lstrip()


def write_memo(out_dir: Path, args: argparse.Namespace, probe: pd.DataFrame, eval_table: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# vLLM Option-Compare Probe",
        "",
        f"Served model: `{args.served_model_name}`",
        f"Base URL: `{args.base_url}`",
        "Scope: GPQA/MMLUPro val/test option comparison.",
        "No GPT, Gemini, or Claude calls are made by this script.",
        "",
        "## Probe Summary",
        "",
        compact_csv(
            probe.groupby(["split", "benchmark"], as_index=False).agg(
                n=("query_id", "nunique"),
                mean_confidence=("selected_confidence", "mean"),
                need_strong_rate=("need_strong", "mean"),
                mean_latency_s=("latency_s", "mean"),
                cache_hit_rate=("cache_hit", "mean"),
            )
        ),
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected),
        "```",
        "",
        "## Top Held-Out Test Rows",
        "",
        "```csv",
        compact_csv(eval_table[eval_table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)),
        "```",
    ]
    (out_dir / "VLLM_OPTION_COMPARE_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def compact_csv(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
