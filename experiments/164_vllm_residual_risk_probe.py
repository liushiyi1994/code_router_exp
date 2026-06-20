from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
ACTIONS = ["base", "self", "strong"]
DEFAULT_RULE_PARAMS = ("all", "base", 0.50, 3, "math_mmlupro", "self", 1.00, 1, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a residual-targeted local vLLM risk probe.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=Path("results/controlled/broad100_embedding_self_action_gate/embedding_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_vllm_residual_risk_probe_qwen14b"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--model-id", default="qwen3-14b-awq-residual-risk-probe")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--benchmarks", default="gpqa,mmlupro,math500,livemathbench")
    parser.add_argument("--max-query-chars", type=int, default=1800)
    parser.add_argument("--max-answer-chars", type=int, default=160)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    residual = load_module("experiments/163_residual_confidence_rule_policy.py", "residual_rules")
    action_probe = load_module("experiments/155_vllm_action_compare_probe.py", "action_probe")
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_gate")
    calibrated = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_gate")
    pairwise = load_module("experiments/162_pairwise_action_ranker.py", "pairwise_action_ranker")

    outputs = self_gate.load_outputs(args.outputs)
    probe_features = self_gate.load_probe(args.probe_table)
    context = residual.build_context(
        package,
        self_gate,
        calibrated,
        pairwise,
        outputs,
        probe_features,
        embedding_cache_dir=args.embedding_cache_dir,
        self_model_id=str(args.self_model_id),
        max_features=int(args.max_features),
    )
    current_actions = {
        split: residual.apply_rule(context, split, context["base_actions"][split], DEFAULT_RULE_PARAMS)
        for split in ["val", "test"]
    }
    probe_inputs = build_probe_inputs(
        outputs,
        probe_features,
        context,
        current_actions,
        splits=parse_csv(args.splits),
        benchmarks=parse_csv(args.benchmarks),
        self_model_id=str(args.self_model_id),
    )
    if args.limit is not None:
        probe_inputs = probe_inputs.head(int(args.limit)).copy()
    probe = collect_probe(args, action_probe, probe_inputs)
    table = evaluate_probe(
        residual,
        context,
        current_actions,
        probe,
        lambda_cost=float(args.lambda_cost),
        self_model_id=str(args.self_model_id),
    )
    selected = validation_selected(table)

    probe.to_csv(args.output_dir / "table_vllm_residual_risk_probe.csv", index=False)
    table.to_csv(args.output_dir / "table_vllm_residual_risk_eval.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_residual_risk_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "VLLM_RESIDUAL_RISK_MEMO.md", args, probe, table, selected)
    print(f"Wrote vLLM residual-risk probe results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def build_probe_inputs(
    outputs: pd.DataFrame,
    probe_features: pd.DataFrame,
    context: dict[str, Any],
    current_actions: dict[str, np.ndarray],
    *,
    splits: set[str],
    benchmarks: set[str],
    self_model_id: str,
) -> pd.DataFrame:
    by_query = outputs.set_index(["query_id", "model_id"])
    probe_by_query = probe_features.set_index("query_id")
    rows: list[dict[str, Any]] = []
    local_models = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]
    for split in ["val", "test"]:
        if split not in splits:
            continue
        frame = context["frames"][split]
        for index, row in frame.reset_index(drop=True).iterrows():
            query_id = str(row["query_id"])
            benchmark = str(row["benchmark"])
            if benchmarks and benchmark not in benchmarks:
                continue
            action_idx = int(current_actions[split][index])
            action = ACTIONS[action_idx]
            local_answers = []
            for model_id in local_models:
                key = (query_id, model_id)
                if key in by_query.index:
                    local_answers.append({"model_id": model_id, "answer": str(by_query.loc[key].get("parsed_answer", ""))})
            probe_row = probe_by_query.loc[query_id] if query_id in probe_by_query.index else pd.Series(dtype=object)
            rows.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "benchmark": benchmark,
                    "domain": str(row["domain"]),
                    "metric": str(row["metric"]),
                    "query_text": str(row["query_text"]),
                    "current_action": action,
                    "base_model_id": str(row["model_base"]),
                    "base_answer": str(row["base_answer_norm"]),
                    "self_answer": str(row["majority_answer_norm"]),
                    "self_vote_frac": float(row["vote_frac"]),
                    "self_vote_margin": float(row["vote_margin"]),
                    "self_vote_entropy": float(row["vote_entropy"]),
                    "local_agree": float(row["local_agree_with_majority_count"]),
                    "local_answers_json": json.dumps(local_answers, sort_keys=True),
                    "sample_texts_json": str(probe_row.get("sample_texts_json", "[]")),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"])


def collect_probe(args: argparse.Namespace, action_probe, probe_inputs: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_residual_risk" / safe_part(str(args.model_id))
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = len(probe_inputs)
    for index, task in enumerate(probe_inputs.itertuples(index=False), start=1):
        raw_path = raw_dir / f"{safe_part(str(task.query_id))}_{safe_part(str(task.current_action))}.json"
        cache_hit = raw_path.exists() and not args.force_rerun
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status = str(payload.get("_status", "success"))
            error_type = str(payload.get("_error_type", ""))
        else:
            prompt = risk_prompt(
                task,
                max_query_chars=int(args.max_query_chars),
                max_answer_chars=int(args.max_answer_chars),
            )
            try:
                payload = action_probe.call_vllm_chat(
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
        raw_text = action_probe.extract_text(payload)
        parsed = parse_risk_output(raw_text)
        if status != "success":
            parsed = default_parse()
        rows.append(
            {
                "query_id": task.query_id,
                "split": task.split,
                "benchmark": task.benchmark,
                "domain": task.domain,
                "metric": task.metric,
                "current_action": task.current_action,
                "model_id": args.model_id,
                "status": status,
                "error_type": error_type,
                "cache_hit": bool(cache_hit),
                "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
                "raw_output_path": str(raw_path),
                "raw_text": raw_text,
                **parsed,
            }
        )
        if index % 20 == 0 or index == total:
            print(f"residual risk probe rows {index}/{total}", flush=True)
    return pd.DataFrame(rows)


def risk_prompt(task, *, max_query_chars: int, max_answer_chars: int) -> str:
    local_answers = json.loads(str(task.local_answers_json))
    local_lines = [
        f"- {item['model_id']}: {truncate(str(item.get('answer', '')), max_answer_chars) or '[empty]'}"
        for item in local_answers
    ]
    return (
        "You are a local residual-risk judge for an LLM router. You do not know the gold answer.\n"
        "The router already has a current action. Your job is only to catch obvious residual mistakes.\n"
        "Set frontier_needed=true only when local evidence looks likely wrong enough to pay for a stronger frontier solver.\n"
        "Set local_safe=true when a planned strong/frontier call looks unnecessary because local/self evidence is coherent.\n"
        "For ties or weak evidence, keep both booleans false. Choose local_action as base or self if local_safe is true.\n"
        "Return JSON only with keys frontier_needed, frontier_confidence, local_safe, local_action, local_confidence, reason.\n\n"
        f"Benchmark: {task.benchmark}\n"
        f"Metric: {task.metric}\n"
        f"Current action: {task.current_action}\n"
        f"Question:\n{truncate(str(task.query_text), max_query_chars)}\n\n"
        f"Base model: {task.base_model_id}\n"
        f"Base answer: {truncate(str(task.base_answer), max_answer_chars) or '[empty]'}\n"
        f"Self-consistency answer: {truncate(str(task.self_answer), max_answer_chars) or '[empty]'}\n"
        f"Self vote_frac={float(task.self_vote_frac):.3f}, margin={float(task.self_vote_margin):.3f}, entropy={float(task.self_vote_entropy):.3f}, local_agree={float(task.local_agree):.1f}\n"
        "Local model answers:\n"
        + "\n".join(local_lines)
        + '\n\nJSON example: {"frontier_needed":false,"frontier_confidence":0.22,"local_safe":true,"local_action":"self","local_confidence":0.81,"reason":"short"}\n/no_think'
    )


def parse_risk_output(raw_text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    frontier_needed = bool(parsed.get("frontier_needed", False))
    local_safe = bool(parsed.get("local_safe", False))
    local_action = str(parsed.get("local_action", "self")).strip().lower()
    if local_action not in {"base", "self"}:
        local_action = "self"
    return {
        "frontier_needed": frontier_needed,
        "frontier_confidence": clip_float(parsed.get("frontier_confidence", 0.0)),
        "local_safe": local_safe,
        "local_action": local_action,
        "local_confidence": clip_float(parsed.get("local_confidence", 0.0)),
        "reason": truncate(str(parsed.get("reason", "")), 240),
    }


def default_parse() -> dict[str, Any]:
    return {
        "frontier_needed": False,
        "frontier_confidence": 0.0,
        "local_safe": False,
        "local_action": "self",
        "local_confidence": 0.0,
        "reason": "",
    }


def evaluate_probe(
    residual,
    context: dict[str, Any],
    current_actions: dict[str, np.ndarray],
    probe: pd.DataFrame,
    *,
    lambda_cost: float,
    self_model_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            residual.fast_eval(
                context,
                split,
                current_actions[split],
                method="residual_rule_baseline",
                family="baseline",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )
        oracle_actions = np.argmax(context["metrics"][split]["utility"], axis=1)
        rows.append(
            residual.fast_eval(
                context,
                split,
                oracle_actions,
                method="diagnostic_oracle_between_base_self_strong",
                family="diagnostic_oracle",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )
        split_probe = probe[probe["split"].eq(split)].copy()
        for mode in ["escalate", "suppress", "both"]:
            for frontier_threshold in [0.40, 0.55, 0.70, 0.85]:
                for local_threshold in [0.40, 0.55, 0.70, 0.85]:
                    actions = apply_probe_actions(
                        context,
                        split,
                        current_actions[split],
                        split_probe,
                        mode=mode,
                        frontier_threshold=frontier_threshold,
                        local_threshold=local_threshold,
                    )
                    row = residual.fast_eval(
                        context,
                        split,
                        actions,
                        method=f"vllm_residual_risk_{mode}_ft{frontier_threshold:g}_lt{local_threshold:g}",
                        family="vllm_residual_risk",
                        lambda_cost=lambda_cost,
                        self_model_id=self_model_id,
                    )
                    row.update(
                        {
                            "mode": mode,
                            "frontier_threshold": float(frontier_threshold),
                            "local_threshold": float(local_threshold),
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def apply_probe_actions(
    context: dict[str, Any],
    split: str,
    base_actions: np.ndarray,
    probe: pd.DataFrame,
    *,
    mode: str,
    frontier_threshold: float,
    local_threshold: float,
) -> np.ndarray:
    actions = np.asarray(base_actions, dtype=int).copy()
    query_to_index = {str(query_id): idx for idx, query_id in enumerate(context["arrays"][split]["query_id"])}
    for row in probe.itertuples(index=False):
        idx = query_to_index.get(str(row.query_id))
        if idx is None:
            continue
        if mode in {"suppress", "both"} and actions[idx] == 2 and bool(row.local_safe) and float(row.local_confidence) >= local_threshold:
            actions[idx] = 0 if str(row.local_action) == "base" else 1
        if mode in {"escalate", "both"} and actions[idx] != 2 and bool(row.frontier_needed) and float(row.frontier_confidence) >= frontier_threshold:
            actions[idx] = 2
    return actions


def validation_selected(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[table["split"].eq("val") & table["family"].ne("diagnostic_oracle")].sort_values(
        ["mean_utility", "mean_quality"],
        ascending=False,
    )
    if not val.empty:
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = table[table["split"].eq("test") & table["method"].eq(method)]
        if not test.empty:
            rows.append(test.head(1).assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def clip_float(value: Any) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."


def safe_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:90]


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#61758d")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("vLLM Residual-Risk Probe")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_vllm_residual_risk_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, probe: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
        "mode",
        "frontier_threshold",
        "local_threshold",
        "selection_rule",
    ]
    probe_summary = (
        probe.groupby(["split", "benchmark"], as_index=False)
        .agg(
            n=("query_id", "nunique"),
            success_rate=("status", lambda values: float((values == "success").mean())),
            frontier_needed_rate=("frontier_needed", "mean"),
            local_safe_rate=("local_safe", "mean"),
            mean_frontier_confidence=("frontier_confidence", "mean"),
            mean_local_confidence=("local_confidence", "mean"),
        )
        .sort_values(["split", "benchmark"])
    )
    lines = [
        "# vLLM Residual-Risk Probe",
        "",
        f"Served model: `{args.served_model_name}` via `{args.base_url}`.",
        "This run uses local vLLM only; no GPT, Gemini, or Claude API calls are made.",
        "",
        "## Probe Summary",
        "",
        "```csv",
        compact_csv(probe_summary, max_rows=80),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=32),
        "```",
        "",
        "## All Evaluation Rows",
        "",
        "```csv",
        compact_csv(table[[column for column in cols if column in table.columns]], max_rows=60),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
