from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
LOCAL_MODELS = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a local vLLM action-compare probe for base/self/strong routing.")
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
        default=Path("results/controlled/broad100_vllm_action_compare_probe"),
    )
    parser.add_argument("--base-method", default="tool_probe_profile_v4_no_strong")
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--model-id", default="qwen3-32b-awq-action-compare-probe")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--benchmarks", default="gpqa,mmlupro,livemathbench,math500")
    parser.add_argument("--max-query-chars", type=int, default=2200)
    parser.add_argument("--max-answer-chars", type=int, default=220)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--prompt-style", choices=["standard", "conservative"], default="standard")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_feature_gate")
    fast_gate = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_action_gate")
    outputs = self_gate.load_outputs(args.outputs)
    probe_features = self_gate.load_probe(args.probe_table)
    requested_splits = parse_csv_set(args.splits)
    base_splits = requested_splits | {"val", "test"}
    base = {
        split: base_selection(package, fast_gate, outputs, base_method=str(args.base_method), split=split, self_model_id=str(args.self_model_id))
        for split in sorted(base_splits, key=lambda value: {"train": 0, "val": 1, "test": 2}.get(value, 99))
    }
    probe_inputs = build_probe_inputs(
        package,
        outputs,
        probe_features,
        base,
        splits=requested_splits,
        benchmarks=parse_csv_set(args.benchmarks),
        self_model_id=str(args.self_model_id),
    )
    if args.limit is not None:
        probe_inputs = probe_inputs.head(int(args.limit)).copy()
    probe = collect_action_probe(args, probe_inputs)
    eval_table, selected = evaluate_action_policies(
        package,
        outputs,
        base,
        probe,
        lambda_cost=float(args.lambda_cost),
        self_model_id=str(args.self_model_id),
    )
    probe.to_csv(args.output_dir / "table_vllm_action_compare_probe.csv", index=False)
    eval_table.to_csv(args.output_dir / "table_vllm_action_compare_eval.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_action_compare_selected.csv", index=False)
    write_figure(args.output_dir, eval_table)
    write_memo(args.output_dir / "VLLM_ACTION_COMPARE_MEMO.md", args, probe, eval_table, selected)
    print(f"Wrote vLLM action-compare probe results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def base_selection(package, fast_gate, outputs: pd.DataFrame, *, base_method: str, split: str, self_model_id: str) -> pd.Series:
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()
    if base_method == "tool_probe_profile_v4_no_strong":
        return normalize_selection(package.profile_v4_selection_for_split(outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}))
    if base_method == "observable_local_state_v5_no_strong":
        return normalize_selection(fast_gate.fast_observable_local_state_selection(package, outputs_no_strong_self, split=split))
    raise ValueError(f"Unknown base method: {base_method}")


def build_probe_inputs(
    package,
    outputs: pd.DataFrame,
    probe_features: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    splits: set[str],
    benchmarks: set[str],
    self_model_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    by_query = outputs.set_index(["query_id", "model_id"])
    probe_by_query = probe_features.set_index("query_id")
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
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
            if query_id not in probe_by_query.index:
                continue
            base_key = (query_id, str(base_model))
            self_key = (query_id, self_model_id)
            strong_key = (query_id, STRONG_MODEL_ID)
            if base_key not in by_query.index or self_key not in by_query.index or strong_key not in by_query.index:
                continue
            probe_row = probe_by_query.loc[query_id]
            local_lines = []
            for model_id in LOCAL_MODELS:
                key = (query_id, model_id)
                if key in by_query.index:
                    row = by_query.loc[key]
                    local_lines.append(
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
                    "base_model_id": str(base_model),
                    "base_answer": str(by_query.loc[base_key].get("parsed_answer", "")),
                    "self_answer": str(probe_row.get("majority_answer_norm", "") or probe_row.get("majority_answer", "")),
                    "self_vote_frac": float(probe_row.get("vote_frac", 0.0) or 0.0),
                    "self_vote_margin": float(probe_row.get("vote_margin", 0.0) or 0.0),
                    "self_vote_entropy": float(probe_row.get("vote_entropy", 0.0) or 0.0),
                    "local_answers_json": json.dumps(local_lines, sort_keys=True),
                }
            )
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"])


def collect_action_probe(args: argparse.Namespace, probe_inputs: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_action_compare" / safe_part(str(args.model_id)) / safe_part(str(args.base_method))
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    total = len(probe_inputs)
    for index, task in enumerate(probe_inputs.itertuples(index=False), start=1):
        raw_path = raw_dir / f"{safe_part(str(task.query_id))}_{cache_digest(str(task.query_id), str(task.base_model_id))}.json"
        cache_hit = raw_path.exists() and not args.force_rerun
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status = str(payload.get("_status", "success"))
            error_type = str(payload.get("_error_type", ""))
        else:
            prompt = action_prompt(
                task,
                max_query_chars=int(args.max_query_chars),
                max_answer_chars=int(args.max_answer_chars),
                prompt_style=str(args.prompt_style),
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
        parsed = parse_action_output(raw_text)
        if status != "success":
            parsed = {"action": "base", "confidence": 0.0, "reason": ""}
        rows.append(
            {
                "query_id": task.query_id,
                "split": task.split,
                "benchmark": task.benchmark,
                "domain": task.domain,
                "metric": task.metric,
                "base_model_id": task.base_model_id,
                "model_id": args.model_id,
                "status": status,
                "error_type": error_type,
                "cache_hit": bool(cache_hit),
                "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
                "raw_output_path": str(raw_path),
                "raw_text": raw_text,
                "action": parsed["action"],
                "confidence": float(parsed["confidence"]),
                "reason": parsed.get("reason", ""),
            }
        )
        if index % 20 == 0 or index == total:
            print(f"action compare rows {index}/{total}")
    return pd.DataFrame(rows)


def action_prompt(task, *, max_query_chars: int, max_answer_chars: int, prompt_style: str) -> str:
    local_answers = json.loads(str(task.local_answers_json))
    local_lines = []
    for item in local_answers:
        local_lines.append(f"- {item['model_id']}: {truncate(str(item['answer']), max_answer_chars) or '[empty]'}")
    if prompt_style == "conservative":
        instruction = (
            "You are a conservative routing action selector. You do not know the gold answer.\n"
            "Default to base unless there is clear evidence that base is unreliable.\n"
            "Choose exactly one action:\n"
            "- base: keep the currently selected base answer. Use this for normal cases and ties.\n"
            "- self: use the local self-consistency majority only when it clearly fixes an empty, malformed, or contradicted base answer.\n"
            "- strong: call a stronger frontier/strong solver only when local answers are inconsistent, low-confidence, malformed, or the task likely needs stronger reasoning.\n"
            "Do not choose self just because a majority answer exists. Do not choose strong just because the task is hard.\n"
        )
    else:
        instruction = (
            "You are a local routing action selector. You do not know the gold answer.\n"
            "Choose the cheapest reliable action for this task:\n"
            "- base: keep the currently selected base answer.\n"
            "- self: use the local self-consistency majority answer.\n"
            "- strong: call a stronger frontier/strong solver because local answers look unreliable.\n"
            "Use strong only when the local evidence is weak, inconsistent, malformed, or likely wrong.\n"
        )
    return (
        instruction
        + "Return JSON only with keys action, confidence, reason. action must be base, self, or strong.\n"
        "Use short reasons. Never mention the gold answer.\n\n"
        f"Benchmark: {task.benchmark}\n"
        f"Metric: {task.metric}\n"
        f"Question:\n{truncate(str(task.query_text), max_query_chars)}\n\n"
        f"Base model: {task.base_model_id}\n"
        f"Base answer: {truncate(str(task.base_answer), max_answer_chars) or '[empty]'}\n\n"
        "Local model answers:\n"
        + "\n".join(local_lines)
        + "\n\n"
        f"Self-consistency majority answer: {truncate(str(task.self_answer), max_answer_chars) or '[empty]'}\n"
        f"Self-consistency vote_frac={float(task.self_vote_frac):.3f}, vote_margin={float(task.self_vote_margin):.3f}, entropy={float(task.self_vote_entropy):.3f}\n\n"
        'JSON example: {"action":"self","confidence":0.74,"reason":"short"}\n/no_think'
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


def parse_action_output(raw_text: str) -> dict[str, Any]:
    text = re.sub(r"<think>.*?</think>", "", str(raw_text or ""), flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    action = str(parsed.get("action", "")).strip().lower()
    if action not in {"base", "self", "strong"}:
        low = text.lower()
        if "strong" in low:
            action = "strong"
        elif "self" in low:
            action = "self"
        else:
            action = "base"
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "action": action,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "reason": truncate(str(parsed.get("reason", "")), 240),
    }


def evaluate_action_policies(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    probe: pd.DataFrame,
    *,
    lambda_cost: float,
    self_model_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            evaluate_selection(
                package,
                outputs,
                base[split],
                split=split,
                method="base_policy",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )
        rows.append(
            evaluate_selection(
                package,
                outputs,
                oracle_between_base_self_strong(outputs, base[split], self_model_id=self_model_id),
                split=split,
                method="diagnostic_oracle_between_base_self_strong",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )
        split_probe = probe[probe["split"].eq(split)].copy()
        for threshold in [0.0, 0.4, 0.55, 0.7, 0.85]:
            for mode in ["direct", "strong_only", "self_only"]:
                selected = apply_probe_policy(
                    base[split],
                    split_probe,
                    threshold=threshold,
                    mode=mode,
                    self_model_id=self_model_id,
                )
                method = f"vllm_action_compare_{mode}_t{threshold:g}"
                rows.append(
                    evaluate_selection(
                        package,
                        outputs,
                        selected,
                        split=split,
                        method=method,
                        lambda_cost=lambda_cost,
                        self_model_id=self_model_id,
                    )
                )
    table = pd.DataFrame(rows)
    return table, validation_selected(table)


def apply_probe_policy(
    base: pd.Series,
    probe: pd.DataFrame,
    *,
    threshold: float,
    mode: str,
    self_model_id: str,
) -> pd.Series:
    selected = normalize_selection(base)
    for row in probe.itertuples(index=False):
        query_id = str(row.query_id)
        action = str(row.action)
        if float(row.confidence) < float(threshold):
            continue
        if mode == "direct":
            if action == "self":
                selected.loc[query_id] = self_model_id
            elif action == "strong":
                selected.loc[query_id] = STRONG_MODEL_ID
        elif mode == "strong_only":
            if action == "strong":
                selected.loc[query_id] = STRONG_MODEL_ID
        elif mode == "self_only":
            if action == "self":
                selected.loc[query_id] = self_model_id
        else:
            raise ValueError(mode)
    return selected


def oracle_between_base_self_strong(outputs: pd.DataFrame, base: pd.Series, *, self_model_id: str) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    selected = normalize_selection(base)
    for query_id, base_model in base.items():
        best_model = str(base_model)
        best_utility = -float("inf")
        best_quality = -float("inf")
        for model_id in [str(base_model), self_model_id, STRONG_MODEL_ID]:
            key = (str(query_id), str(model_id))
            if key not in by_query.index:
                continue
            row = by_query.loc[key]
            utility = float(row["utility"])
            quality = float(row["quality_score"])
            if utility > best_utility or (abs(utility - best_utility) <= 1e-12 and quality > best_quality):
                best_model = str(model_id)
                best_utility = utility
                best_quality = quality
        selected.loc[str(query_id)] = best_model
    return selected


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    lambda_cost: float,
    self_model_id: str,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["self_action_rate"] = float(selected_rows["model_id"].eq(self_model_id).mean())
    return row


def validation_selected(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[table["split"].eq("val") & ~table["method"].str.startswith("diagnostic_")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    )
    if not val.empty:
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = table[table["split"].eq("test") & table["method"].eq(method)]
        if not test.empty:
            rows.append(test.assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def truncate(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= max_chars else text[: max(0, max_chars - 3)] + "..."


def safe_part(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")[:90]


def cache_digest(*parts: str) -> str:
    return hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:16]


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#58766a")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("vLLM Action-Compare Probe")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_vllm_action_compare_utility.pdf")
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
        probe.groupby(["split", "benchmark", "action"], as_index=False)
        .agg(n=("query_id", "nunique"), mean_confidence=("confidence", "mean"), success_rate=("status", lambda x: float((x == "success").mean())))
        .sort_values(["split", "benchmark", "action"])
    )
    lines = [
        "# vLLM Action-Compare Probe",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
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
        compact_csv(selected[cols + [c for c in ["selection_rule"] if c in selected.columns]], max_rows=20),
        "```",
        "",
        "## All Evaluation Rows",
        "",
        "```csv",
        compact_csv(table[cols], max_rows=40),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
