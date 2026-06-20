from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import normalize_answer, score_output


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect/evaluate local vLLM self-consistency routing probes.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_train_supervised_strong_gain_gate/model_outputs_with_gemini_strong_all_splits.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--model-id", default=SELF_MODEL_ID)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--benchmarks", default="", help="Comma-separated filter. Empty means all benchmarks.")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--max-query-chars", type=int, default=1600)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    outputs = load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    tasks = build_tasks(outputs, splits=parse_csv_set(args.splits), benchmarks=parse_csv_set(args.benchmarks))
    if args.limit is not None:
        tasks = tasks.head(int(args.limit)).copy()
    probe = collect_self_consistency(args, tasks)
    probe.to_csv(args.output_dir / "table_vllm_self_consistency_probe.csv", index=False)
    expanded_outputs = add_self_consistency_action(outputs, probe, model_id=str(args.model_id))
    expanded_outputs.to_parquet(args.output_dir / "model_outputs_with_self_consistency.parquet", index=False)
    eval_table, selected = evaluate_policies(
        package,
        outputs,
        expanded_outputs,
        probe,
        self_model_id=str(args.model_id),
        lambda_cost=float(args.lambda_cost),
    )
    eval_table.to_csv(args.output_dir / "table_vllm_self_consistency_eval.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_self_consistency_selected.csv", index=False)
    diagnostics = build_diagnostics(outputs, probe)
    diagnostics.to_csv(args.output_dir / "table_vllm_self_consistency_diagnostics.csv", index=False)
    write_memo(args.output_dir / "VLLM_SELF_CONSISTENCY_MEMO.md", args, probe, diagnostics, eval_table, selected)
    print(f"Wrote vLLM self-consistency probe results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    for column in ["quality_score", "cost_total_usd", "normalized_remote_cost", "latency_s"]:
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    if "utility" not in outputs:
        outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    return outputs


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def build_tasks(outputs: pd.DataFrame, *, splits: set[str], benchmarks: set[str]) -> pd.DataFrame:
    queries = outputs.drop_duplicates("query_id").copy()
    if splits:
        queries = queries[queries["split"].astype(str).isin(splits)].copy()
    if benchmarks:
        queries = queries[queries["benchmark"].astype(str).isin(benchmarks)].copy()
    return queries.sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def collect_self_consistency(args: argparse.Namespace, tasks: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_self_consistency" / safe_part(str(args.model_id))
    raw_dir.mkdir(parents=True, exist_ok=True)

    def one(task: pd.Series) -> dict[str, Any]:
        return collect_one(args, task, raw_dir=raw_dir)

    rows: list[dict[str, Any]] = []
    total = len(tasks)
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in tasks.iterrows()]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == total:
                print(f"self-consistency rows {index}/{total}")
    return pd.DataFrame(rows).sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)


def collect_one(args: argparse.Namespace, task: pd.Series, *, raw_dir: Path) -> dict[str, Any]:
    query_id = str(task["query_id"])
    raw_path = raw_dir / f"{safe_part(query_id)}_{cache_digest(query_id, str(args.model_id), str(args.samples))}.json"
    cache_hit = raw_path.exists() and not args.force_rerun
    started = time.time()
    status = "success"
    error_type = ""
    if cache_hit:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        status = str(payload.get("_status", "success"))
        error_type = str(payload.get("_error_type", ""))
    else:
        prompt = build_prompt(task, max_query_chars=int(args.max_query_chars))
        try:
            payload = call_vllm_chat(
                base_url=str(args.base_url),
                served_model_name=str(args.served_model_name),
                prompt=prompt,
                samples=int(args.samples),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_tokens=int(args.max_output_tokens),
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

    sample_texts = extract_texts(payload) if status == "success" else []
    summary = summarize_samples(sample_texts, gold=str(task["gold_answer"]), metric=str(task["metric"]))
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return {
        "query_id": query_id,
        "split": str(task["split"]),
        "benchmark": str(task["benchmark"]),
        "domain": str(task.get("domain", "")),
        "metric": str(task["metric"]),
        "gold_answer": str(task["gold_answer"]),
        "model_id": str(args.model_id),
        "served_model_name": str(args.served_model_name),
        "status": status,
        "error_type": error_type,
        "cache_hit": bool(cache_hit),
        "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "raw_output_path": str(raw_path),
        "sample_texts_json": json.dumps(sample_texts, ensure_ascii=True),
        **summary,
    }


def build_prompt(task: pd.Series, *, max_query_chars: int) -> str:
    metric = str(task.get("metric", "exact_final_answer"))
    query_text = compact(str(task["query_text"]), max_query_chars)
    if metric == "pass_at_1":
        instruction = (
            "Write a complete Python solution for the task. Output only Python code, with no markdown fences and no explanation."
        )
    elif metric in {"multiple_choice", "exact_or_multiple_choice"} or normalize_answer(str(task.get("gold_answer", "")))[:1].upper() in {
        "A",
        "B",
        "C",
        "D",
    }:
        instruction = "Answer the multiple-choice question with only one letter: A, B, C, or D."
    else:
        instruction = "Solve the task and output only the final answer. Do not include reasoning, units, or markdown."
    return (
        "You are a local probe for a model router. The router needs a short answer sample, not an explanation.\n"
        f"{instruction}\n\n"
        f"Benchmark: {task['benchmark']}\n"
        f"Task:\n{query_text}\n"
        "/no_think"
    )


def call_vllm_chat(
    *,
    base_url: str,
    served_model_name: str,
    prompt: str,
    samples: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "n": int(samples),
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


def extract_texts(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for choice in payload.get("choices", []) or []:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message", {}) if isinstance(choice.get("message", {}), dict) else {}
        text = str(message.get("content", "") or "")
        texts.append(strip_thinking(text).strip())
    return texts


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.DOTALL | re.IGNORECASE)


def summarize_samples(sample_texts: list[str], *, gold: str, metric: str) -> dict[str, Any]:
    parsed_answers: list[str] = []
    answer_norms: list[str] = []
    qualities: list[float] = []
    valid_norms: list[str] = []
    for text in sample_texts:
        parsed, quality = score_output(text, gold, metric)
        norm = normalize_answer(parsed)
        parsed_answers.append(parsed)
        answer_norms.append(norm)
        qualities.append(float(quality))
        if norm and norm not in {"nan", "none", "unknown"}:
            valid_norms.append(norm)
    counts = Counter(valid_norms)
    if counts:
        majority_norm, top_count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    else:
        majority_norm, top_count = "", 0
    majority_indices = [idx for idx, norm in enumerate(answer_norms) if norm == majority_norm]
    majority_answer = parsed_answers[majority_indices[0]] if majority_indices else ""
    majority_quality = max([qualities[idx] for idx in majority_indices], default=0.0)
    valid_count = len(valid_norms)
    total_samples = max(len(sample_texts), 1)
    vote_frac = float(top_count / total_samples)
    second_count = sorted(counts.values(), reverse=True)[1] if len(counts) > 1 else 0
    vote_margin = float((top_count - second_count) / total_samples)
    entropy = vote_entropy(counts, total_samples)
    return {
        "n_samples": int(len(sample_texts)),
        "valid_count": int(valid_count),
        "majority_answer": majority_answer,
        "majority_answer_norm": majority_norm,
        "majority_quality": float(majority_quality),
        "top_vote_count": int(top_count),
        "vote_frac": float(vote_frac),
        "vote_margin": float(vote_margin),
        "vote_entropy": float(entropy),
        "all_samples_agree": bool(valid_count > 0 and top_count == valid_count),
        "any_sample_correct": bool(any(quality > 0 for quality in qualities)),
        "mean_sample_quality": float(np.mean(qualities)) if qualities else 0.0,
        "parsed_answers_json": json.dumps(parsed_answers, ensure_ascii=True),
        "answer_norms_json": json.dumps(answer_norms, ensure_ascii=True),
        "sample_qualities_json": json.dumps(qualities),
    }


def vote_entropy(counts: Counter[str], total_samples: int) -> float:
    if not counts or total_samples <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = float(count) / float(total_samples)
        if p > 0:
            entropy -= p * math.log2(p)
    return float(entropy)


def add_self_consistency_action(outputs: pd.DataFrame, probe: pd.DataFrame, *, model_id: str) -> pd.DataFrame:
    query_rows = outputs.drop_duplicates("query_id").set_index("query_id")
    rows: list[dict[str, Any]] = []
    for _, probe_row in probe.iterrows():
        query_id = str(probe_row["query_id"])
        if query_id not in query_rows.index:
            continue
        template = query_rows.loc[query_id].to_dict()
        quality = float(probe_row.get("majority_quality", 0.0) or 0.0)
        template.update(
            {
                "query_id": query_id,
                "model_id": model_id,
                "provider": "local",
                "is_local": True,
                "is_frontier": False,
                "is_probe": True,
                "status": str(probe_row.get("status", "")),
                "error_type": str(probe_row.get("error_type", "")),
                "raw_output_path": str(probe_row.get("raw_output_path", "")),
                "parsed_answer": str(probe_row.get("majority_answer", "")),
                "quality_score": quality,
                "cost_input_usd": 0.0,
                "cost_output_usd": 0.0,
                "cost_total_usd": 0.0,
                "normalized_remote_cost": 0.0,
                "utility": quality,
                "latency_s": float(probe_row.get("latency_s", 0.0) or 0.0),
                "input_tokens": int(probe_row.get("input_tokens", 0) or 0),
                "output_tokens": int(probe_row.get("output_tokens", 0) or 0),
            }
        )
        rows.append(template)
    if not rows:
        return outputs.copy()
    return pd.concat([outputs, pd.DataFrame(rows)], ignore_index=True)


def evaluate_policies(
    package,
    base_outputs: pd.DataFrame,
    expanded_outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    probe_by_split = {
        split: probe[probe["split"].eq(split)].set_index("query_id")
        for split in ["val", "test"]
    }
    base_by_split = {
        split: normalize_selection(package.observable_local_state_selection(base_outputs, split=split))
        for split in ["val", "test"]
    }
    tool_by_split = {
        split: normalize_selection(package.profile_v4_selection_for_split(base_outputs, split=split))
        for split in ["val", "test"]
    }
    bases = {
        "observable_local_state_v5": base_by_split,
        "tool_probe_profile_v4": tool_by_split,
    }
    for base_name, selections in bases.items():
        for split in ["val", "test"]:
            rows.append(
                evaluate_selection(
                    package,
                    expanded_outputs,
                    selections[split],
                    probe_by_split[split],
                    split=split,
                    method=base_name,
                    family="base",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
            rows.append(
                evaluate_selection(
                    package,
                    expanded_outputs,
                    oracle_between_actions(expanded_outputs, selections[split], [self_model_id, STRONG_MODEL_ID]),
                    probe_by_split[split],
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
        for min_votes in [2, 3]:
            for threshold in [0.34, 0.50, 0.67, 0.84, 1.00]:
                method = f"{base_name}_self_if_votes{min_votes}_frac{threshold:g}"
                for split in ["val", "test"]:
                    selected = apply_self_policy(
                        selections[split],
                        probe_by_split[split],
                        self_model_id=self_model_id,
                        min_votes=min_votes,
                        threshold=threshold,
                    )
                    rows.append(
                        evaluate_selection(
                            package,
                            expanded_outputs,
                            selected,
                            probe_by_split[split],
                            split=split,
                            method=method,
                            family="self_direct",
                            lambda_cost=lambda_cost,
                            self_model_id=self_model_id,
                            threshold=threshold,
                            min_votes=min_votes,
                        )
                    )
        for threshold in [0.34, 0.50, 0.67, 0.84, 1.00]:
            method = f"{base_name}_strong_if_self_frac_lt{threshold:g}"
            for split in ["val", "test"]:
                selected = apply_strong_low_conf_policy(
                    selections[split],
                    probe_by_split[split],
                    threshold=threshold,
                )
                rows.append(
                    evaluate_selection(
                        package,
                        expanded_outputs,
                        selected,
                        probe_by_split[split],
                        split=split,
                        method=method,
                        family="self_conf_strong_gate",
                        lambda_cost=lambda_cost,
                        self_model_id=self_model_id,
                        threshold=threshold,
                    )
                )
        for low_threshold in [0.34, 0.50, 0.67]:
            for high_threshold in [0.67, 0.84, 1.00]:
                if low_threshold >= high_threshold:
                    continue
                method = f"{base_name}_self_high{high_threshold:g}_strong_low{low_threshold:g}"
                for split in ["val", "test"]:
                    selected = apply_self_or_strong_policy(
                        selections[split],
                        probe_by_split[split],
                        self_model_id=self_model_id,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                    )
                    rows.append(
                        evaluate_selection(
                            package,
                            expanded_outputs,
                            selected,
                            probe_by_split[split],
                            split=split,
                            method=method,
                            family="self_and_strong_gate",
                            lambda_cost=lambda_cost,
                            self_model_id=self_model_id,
                            low_threshold=low_threshold,
                            high_threshold=high_threshold,
                        )
                    )
    table = pd.DataFrame(rows)
    return table, validation_selected_rows(table)


def apply_self_policy(
    base: pd.Series,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    min_votes: int,
    threshold: float,
) -> pd.Series:
    selected = normalize_selection(base)
    for query_id, row in probe.iterrows():
        if int(row.get("top_vote_count", 0) or 0) >= int(min_votes) and float(row.get("vote_frac", 0.0) or 0.0) >= float(
            threshold
        ):
            selected.loc[str(query_id)] = self_model_id
    return selected


def apply_strong_low_conf_policy(base: pd.Series, probe: pd.DataFrame, *, threshold: float) -> pd.Series:
    selected = normalize_selection(base)
    for query_id, row in probe.iterrows():
        if float(row.get("vote_frac", 0.0) or 0.0) < float(threshold):
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected


def apply_self_or_strong_policy(
    base: pd.Series,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    selected = normalize_selection(base)
    for query_id, row in probe.iterrows():
        vote_frac = float(row.get("vote_frac", 0.0) or 0.0)
        top_count = int(row.get("top_vote_count", 0) or 0)
        if top_count >= 2 and vote_frac >= float(high_threshold):
            selected.loc[str(query_id)] = self_model_id
        elif vote_frac < float(low_threshold):
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected


def oracle_between_actions(outputs: pd.DataFrame, base: pd.Series, extra_actions: list[str]) -> pd.Series:
    by_query = outputs.drop_duplicates(["query_id", "model_id"], keep="last").set_index(["query_id", "model_id"])
    selected = normalize_selection(base)
    for query_id, base_model in base.items():
        query_id = str(query_id)
        candidates = [str(base_model), *extra_actions]
        best_model = str(base_model)
        best_utility = -float("inf")
        best_quality = -float("inf")
        for model_id in candidates:
            if (query_id, model_id) not in by_query.index:
                continue
            row = by_query.loc[(query_id, model_id)]
            utility = float(row["utility"])
            quality = float(row["quality_score"])
            if utility > best_utility or (abs(utility - best_utility) <= 1e-12 and quality > best_quality):
                best_model = model_id
                best_utility = utility
                best_quality = quality
        selected.loc[query_id] = best_model
    return selected


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    probe: pd.DataFrame,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
    self_model_id: str,
    **extra: Any,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["self_action_rate"] = float(selected_rows["model_id"].eq(self_model_id).mean())
    probe_n = int(probe.index.nunique()) if "query_id" not in probe.columns else int(probe["query_id"].nunique())
    row["self_probe_call_rate"] = float(probe_n / max(selected_rows["query_id"].nunique(), 1)) if not probe.empty else 0.0
    row["self_probe_mean_latency_s"] = float(pd.to_numeric(probe.get("latency_s", pd.Series(dtype=float)), errors="coerce").mean()) if not probe.empty else 0.0
    row.update(extra)
    return row


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)]
        if not test.empty:
            rows.append(test.head(1).assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_diagnostics(outputs: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    base = outputs[outputs["split"].isin(["val", "test"])].copy()
    local = base[base["is_local"].astype(bool) & ~base["model_id"].eq("deterministic_math_tool")].copy()
    local_best = local.loc[local.groupby("query_id")["utility"].idxmax(), ["query_id", "model_id", "quality_score", "utility"]]
    strong = base[base["model_id"].eq(STRONG_MODEL_ID)][["query_id", "quality_score", "utility"]].rename(
        columns={"quality_score": "strong_quality", "utility": "strong_utility"}
    )
    frame = probe.merge(local_best, on="query_id", how="left", suffixes=("", "_best_local")).merge(strong, on="query_id", how="left")
    frame["self_beats_best_local"] = frame["majority_quality"].astype(float) > frame["quality_score"].astype(float)
    frame["strong_beats_self"] = frame["strong_utility"].astype(float) > frame["majority_quality"].astype(float)
    frame["high_vote"] = frame["vote_frac"].astype(float) >= 0.67
    return (
        frame.groupby(["split", "benchmark", "high_vote"], dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            self_quality=("majority_quality", "mean"),
            mean_sample_quality=("mean_sample_quality", "mean"),
            any_sample_correct_rate=("any_sample_correct", "mean"),
            vote_frac=("vote_frac", "mean"),
            vote_entropy=("vote_entropy", "mean"),
            best_local_quality=("quality_score", "mean"),
            strong_quality=("strong_quality", "mean"),
            self_beats_best_local_rate=("self_beats_best_local", "mean"),
            strong_beats_self_rate=("strong_beats_self", "mean"),
        )
        .reset_index()
        .sort_values(["split", "benchmark", "high_vote"])
    )


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:90]


def cache_digest(*parts: str) -> str:
    return hashlib.sha1(":".join(parts).encode("utf-8")).hexdigest()[:12]


def compact(text: str, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: int(max_chars * 0.78)].rstrip() + " ... " + text[-int(max_chars * 0.16) :].lstrip()


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_memo(
    path: Path,
    args: argparse.Namespace,
    probe: pd.DataFrame,
    diagnostics: pd.DataFrame,
    eval_table: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    status = probe["status"].value_counts(dropna=False).sort_index().to_dict() if not probe.empty else {}
    summary = (
        probe.groupby(["split", "benchmark"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            self_quality=("majority_quality", "mean"),
            mean_sample_quality=("mean_sample_quality", "mean"),
            any_sample_correct_rate=("any_sample_correct", "mean"),
            high_vote_rate=("vote_frac", lambda s: float((pd.to_numeric(s, errors="coerce") >= 0.67).mean())),
            mean_vote_frac=("vote_frac", "mean"),
            mean_latency_s=("latency_s", "mean"),
            cache_hit_rate=("cache_hit", "mean"),
        )
        .sort_values(["split", "benchmark"])
        if not probe.empty
        else pd.DataFrame()
    )
    lines = [
        "# vLLM Self-Consistency Probe",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Local probe model: `{args.model_id}` served as `{args.served_model_name}` from `{args.base_url}`.",
        f"Samples per query: `{args.samples}`. Temperature: `{args.temperature}`. Top-p: `{args.top_p}`.",
        "This script uses local vLLM only; it makes no GPT, Gemini, or Claude calls.",
        f"Rows: `{len(probe)}`. Status counts: `{status}`.",
        "",
        "## Probe Summary",
        "",
        "```csv",
        compact_csv(summary),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected),
        "```",
        "",
        "## Held-Out Test Rows",
        "",
        "```csv",
        compact_csv(eval_table[eval_table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(25)),
        "```",
        "",
        "## Self-Consistency Diagnostics",
        "",
        "```csv",
        compact_csv(diagnostics.head(80)),
        "```",
        "",
        "## Interpretation",
        "",
        "- This tests local self-consistency as cheap evidence for wrong-local-winner and value-of-compute decisions.",
        "- The self-consistency action is local and zero API-cost in utility accounting, but the memo reports local probe latency separately.",
        "- If a new self-consistency action raises the oracle, the relevant comparison is against the expanded cost-aware oracle.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
