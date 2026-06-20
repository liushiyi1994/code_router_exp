from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
LOCAL_MODELS = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local", DEFAULT_SELF_MODEL_ID]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a local vLLM binary frontier-need probe.")
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
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_vllm_frontier_need_probe_qwen14b"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument("--model-id", default="qwen3-14b-awq-frontier-need-probe")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--benchmarks", default="aime,bbh,gpqa,gsm8k,livemathbench,math500,mmlupro")
    parser.add_argument("--max-query-chars", type=int, default=1200)
    parser.add_argument("--max-answer-chars", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=72)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    action_probe = load_module("experiments/155_vllm_action_compare_probe.py", "action_probe")
    frontier = load_module("experiments/157_frontier_need_predictor.py", "frontier_need")

    outputs = frontier.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    probe_features = frontier.load_probe(args.probe_table)
    frontier_ids = frontier.frontier_model_ids(outputs)
    local_outputs = outputs[~outputs["model_id"].isin(frontier_ids)].copy()
    base = {
        split: frontier.normalize_selection(package.observable_local_state_selection(local_outputs, split=split))
        for split in ["train", "val", "test"]
    }
    train_frame = frontier.build_feature_frame(outputs, probe_features, base["train"], frontier_ids, split="train")
    frontier_lookup = frontier.frontier_train_lookup(train_frame, frontier_ids)
    probe_inputs = build_probe_inputs(
        outputs,
        probe_features,
        base,
        splits=parse_csv_set(args.splits),
        benchmarks=parse_csv_set(args.benchmarks),
    )
    if args.limit is not None:
        probe_inputs = probe_inputs.head(int(args.limit)).copy()
    probe = collect_probe(args, action_probe, probe_inputs, frontier_ids)
    eval_table, selected = evaluate_probe(
        package,
        frontier,
        outputs,
        base,
        probe,
        frontier_ids,
        frontier_lookup,
        lambda_cost=float(args.lambda_cost),
    )
    probe.to_csv(args.output_dir / "table_vllm_frontier_need_probe.csv", index=False)
    eval_table.to_csv(args.output_dir / "table_vllm_frontier_need_eval.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_frontier_need_selected.csv", index=False)
    write_figure(args.output_dir, eval_table)
    write_memo(args.output_dir / "VLLM_FRONTIER_NEED_MEMO.md", args, probe, eval_table, selected)
    print(f"Wrote vLLM frontier-need probe results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def build_probe_inputs(
    outputs: pd.DataFrame,
    probe_features: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    splits: set[str],
    benchmarks: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_query = outputs.set_index(["query_id", "model_id"])
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    probe_by_query = probe_features.set_index("query_id") if not probe_features.empty else pd.DataFrame()
    for split, selected in base.items():
        if split not in splits:
            continue
        for query_id, base_model in selected.items():
            query_id = str(query_id)
            if query_id not in query_info.index:
                continue
            info = query_info.loc[query_id]
            benchmark = str(info.get("benchmark", ""))
            if benchmarks and benchmark not in benchmarks:
                continue
            base_key = (query_id, str(base_model))
            if base_key not in by_query.index:
                continue
            probe_row = probe_by_query.loc[query_id] if not probe_by_query.empty and query_id in probe_by_query.index else pd.Series(dtype=object)
            local_answers = []
            for model_id in LOCAL_MODELS:
                key = (query_id, model_id)
                if key not in by_query.index:
                    continue
                row = by_query.loc[key]
                local_answers.append(
                    {
                        "model_id": model_id,
                        "status": str(row.get("status", "")),
                        "answer": str(row.get("parsed_answer", "")),
                    }
                )
            rows.append(
                {
                    "query_id": query_id,
                    "split": split,
                    "benchmark": benchmark,
                    "domain": str(info.get("domain", "")),
                    "metric": str(info.get("metric", "")),
                    "query_text": str(info.get("query_text", "")),
                    "local_model_id": str(base_model),
                    "local_answer": str(by_query.loc[base_key].get("parsed_answer", "")),
                    "self_answer": str(probe_row.get("majority_answer_norm", "")),
                    "self_vote_frac": float(probe_row.get("vote_frac", 0.0) or 0.0),
                    "self_vote_margin": float(probe_row.get("vote_margin", 0.0) or 0.0),
                    "self_vote_entropy": float(probe_row.get("vote_entropy", 0.0) or 0.0),
                    "local_answers_json": json.dumps(local_answers, sort_keys=True),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"])


def collect_probe(args: argparse.Namespace, action_probe, probe_inputs: pd.DataFrame, frontier_ids: list[str]) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_frontier_need" / safe_part(str(args.model_id))
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = len(probe_inputs)
    for index, task in enumerate(probe_inputs.itertuples(index=False), start=1):
        raw_path = raw_dir / f"{safe_part(str(task.query_id))}_{cache_digest(str(task.query_id), str(task.local_model_id))}.json"
        cache_hit = raw_path.exists() and not args.force_rerun
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status = str(payload.get("_status", "success"))
            error_type = str(payload.get("_error_type", ""))
        else:
            prompt = frontier_prompt(
                task,
                frontier_ids,
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
        parsed = parse_frontier_output(raw_text, frontier_ids)
        if status != "success":
            parsed = {"frontier_needed": False, "frontier_model": "", "confidence": 0.0, "reason": ""}
        rows.append(
            {
                "query_id": task.query_id,
                "split": task.split,
                "benchmark": task.benchmark,
                "domain": task.domain,
                "metric": task.metric,
                "local_model_id": task.local_model_id,
                "model_id": args.model_id,
                "status": status,
                "error_type": error_type,
                "cache_hit": bool(cache_hit),
                "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
                "raw_output_path": str(raw_path),
                "raw_text": raw_text,
                "frontier_needed": bool(parsed["frontier_needed"]),
                "frontier_model": parsed["frontier_model"],
                "confidence": float(parsed["confidence"]),
                "reason": parsed.get("reason", ""),
            }
        )
        if index % 20 == 0 or index == total:
            print(f"frontier probe rows {index}/{total}", flush=True)
    return pd.DataFrame(rows)


def frontier_prompt(task, frontier_ids: list[str], *, max_query_chars: int, max_answer_chars: int) -> str:
    local_answers = json.loads(str(task.local_answers_json))
    local_lines = []
    for item in local_answers:
        answer = truncate(str(item["answer"]), max_answer_chars) or "[empty]"
        local_lines.append(f"- {item['model_id']}: {answer}")
    choices = ", ".join(frontier_ids)
    return (
        "You are a conservative cost-aware router. You do not know the gold answer.\n"
        "Decide whether the current local route is likely wrong enough to justify paying for a frontier model.\n"
        "Use frontier only for clear risk: malformed local answer, contradictory local answers, low self-consistency, hard expert knowledge, or arithmetic/math where local answers look unreliable.\n"
        "Do not use frontier just because a task is long. Prefer local for normal/tie cases.\n"
        f"If frontier_needed is true, choose one frontier_model from: {choices}.\n"
        "Return JSON only with keys frontier_needed, frontier_model, confidence, reason.\n\n"
        f"Benchmark: {task.benchmark}\n"
        f"Metric: {task.metric}\n"
        f"Question:\n{truncate(str(task.query_text), max_query_chars)}\n\n"
        f"Current local route: {task.local_model_id}\n"
        f"Current local answer: {truncate(str(task.local_answer), max_answer_chars) or '[empty]'}\n\n"
        "Other local/self answers:\n"
        + "\n".join(local_lines)
        + "\n\n"
        f"Self-consistency majority answer: {truncate(str(task.self_answer), max_answer_chars) or '[empty]'}\n"
        f"Self-consistency vote_frac={float(task.self_vote_frac):.3f}, vote_margin={float(task.self_vote_margin):.3f}, entropy={float(task.self_vote_entropy):.3f}\n\n"
        'JSON example: {"frontier_needed":false,"frontier_model":"","confidence":0.72,"reason":"local answers agree"}\n/no_think'
    )


def parse_frontier_output(raw_text: str, frontier_ids: list[str]) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    needed_value = parsed.get("frontier_needed", False)
    if isinstance(needed_value, str):
        frontier_needed = needed_value.strip().lower() in {"true", "yes", "1", "needed", "frontier"}
    else:
        frontier_needed = bool(needed_value)
    frontier_model = str(parsed.get("frontier_model", "")).strip()
    if frontier_model not in frontier_ids:
        low = text.lower()
        frontier_model = ""
        for model_id in frontier_ids:
            if model_id.lower() in low:
                frontier_model = model_id
                break
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "frontier_needed": frontier_needed,
        "frontier_model": frontier_model,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason": truncate(str(parsed.get("reason", "")), 240),
    }


def evaluate_probe(
    package,
    frontier,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    probe: pd.DataFrame,
    frontier_ids: list[str],
    frontier_lookup: dict[str, str],
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            frontier.evaluate_selection(
                package, outputs, base[split], split=split, method="local_observable_state", family="reference", lambda_cost=lambda_cost
            )
        )
        rows.append(
            frontier.evaluate_selection(
                package,
                outputs,
                frontier.oracle_between_local_and_frontier(outputs, base[split], frontier_ids),
                split=split,
                method="diagnostic_oracle_between_local_and_frontier",
                family="diagnostic_oracle",
                lambda_cost=lambda_cost,
            )
        )
        split_probe = probe[probe["split"].eq(split)].copy()
        for threshold in [0.0, 0.4, 0.55, 0.7, 0.85]:
            for cap in [0.25, 0.35, 0.40, 0.50, 1.00]:
                for mode in ["lookup", "direct_model", "strong_only"]:
                    selected = apply_probe_policy(
                        base[split],
                        split_probe,
                        query_info=frontier.query_info(outputs, split),
                        threshold=threshold,
                        cap=cap,
                        mode=mode,
                        frontier_ids=frontier_ids,
                        frontier_lookup=frontier_lookup,
                    )
                    method = f"vllm_frontier_need_{mode}_t{threshold:g}_cap{cap:g}"
                    rows.append(
                        frontier.evaluate_selection(
                            package,
                            outputs,
                            selected,
                            split=split,
                            method=method,
                            family="vllm_frontier_need",
                            lambda_cost=lambda_cost,
                        )
                    )
    table = pd.DataFrame(rows)
    return table, validation_selected(table)


def apply_probe_policy(
    base: pd.Series,
    probe: pd.DataFrame,
    *,
    query_info: pd.DataFrame,
    threshold: float,
    cap: float,
    mode: str,
    frontier_ids: list[str],
    frontier_lookup: dict[str, str],
) -> pd.Series:
    selected = base.copy()
    eligible = probe[(probe["frontier_needed"].astype(bool)) & (probe["confidence"].astype(float) >= float(threshold))].copy()
    eligible = eligible.sort_values("confidence", ascending=False)
    if cap < 1.0:
        eligible = eligible.head(max(1, int(np.floor(float(cap) * len(selected)))))
    for row in eligible.itertuples(index=False):
        query_id = str(row.query_id)
        if mode == "strong_only":
            selected.loc[query_id] = STRONG_MODEL_ID
        elif mode == "direct_model":
            selected.loc[query_id] = str(row.frontier_model) if str(row.frontier_model) in frontier_ids else frontier_ids[0]
        elif mode == "lookup":
            benchmark = str(query_info.loc[query_id, "benchmark"]) if query_id in query_info.index else ""
            selected.loc[query_id] = str(frontier_lookup.get(benchmark, frontier_ids[0]))
        else:
            raise ValueError(mode)
    return selected.astype(str)


def validation_selected(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[table["split"].eq("val") & ~table["family"].eq("diagnostic_oracle")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
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


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#6f6a4a")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("vLLM Frontier-Need Probe")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_vllm_frontier_need_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, probe: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
    ]
    probe_summary = (
        probe.groupby(["split", "benchmark", "frontier_needed"], as_index=False)
        .agg(n=("query_id", "nunique"), mean_confidence=("confidence", "mean"), success_rate=("status", lambda x: float((x == "success").mean())))
        .sort_values(["split", "benchmark", "frontier_needed"])
    )
    lines = [
        "# vLLM Frontier-Need Probe",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Served model: `{args.served_model_name}` via `{args.base_url}`.",
        "This run uses local vLLM only; no GPT, Gemini, or Claude API calls are made.",
        "",
        "## Probe Summary",
        "",
        "```csv",
        compact_csv(probe_summary, max_rows=100),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[c for c in cols if c in selected.columns] + [c for c in ["selection_rule"] if c in selected.columns]], max_rows=28),
        "```",
        "",
        "## All Evaluation Rows",
        "",
        "```csv",
        compact_csv(table[[c for c in cols if c in table.columns]], max_rows=60),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."


def safe_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:90]


def cache_digest(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
