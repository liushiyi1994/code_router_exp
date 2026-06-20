from __future__ import annotations

import argparse
import json
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_openai_compatible_text,
    normalize_answer,
    post_json,
    score_output,
    usage_from_openai_compatible,
)


LOCAL_MODEL = "qwen3-8b-local"
REASONING_MODEL = "qwen3-8b-reasoning-local"
SERVED_MODEL = "Qwen/Qwen3-8B"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect/evaluate Qwen8 thinking-enabled local routing probes.")
    parser.add_argument("--query-table", default="results/controlled/gemini_self_consistency_gate/query_table_with_self_consistency.csv")
    parser.add_argument("--output-dir", default="results/controlled/qwen8_reasoning_gate")
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--splits", default="", help="Comma-separated split filter, e.g. test or val,test.")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def load_gold_rows(query_ids: set[str]) -> pd.DataFrame:
    run_dirs = [
        Path("results/controlled/math500_qwen8_live_pilot_1024"),
        Path("results/controlled/livemathbench_live_pilot_1024"),
        Path("results/controlled/aime_qwen8_live_pilot_1024"),
    ]
    rows: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path, columns=["query_id", "gold_answer", "metric"])
        frame = frame[frame["query_id"].astype(str).isin(query_ids)]
        if not frame.empty:
            rows.append(frame.drop_duplicates("query_id", keep="first"))
    if not rows:
        return pd.DataFrame(columns=["query_id", "gold_answer", "metric"])
    return pd.concat(rows, ignore_index=True).drop_duplicates("query_id", keep="first")


def call_qwen_reasoning(base_url: str, query_text: str, max_output_tokens: int, timeout_s: float = 180.0) -> dict[str, Any]:
    payload = {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an exact-answer solver. Think internally but output only the final answer. "
                    "Do not output steps, explanation, markdown, or units unless required."
                ),
            },
            {"role": "user", "content": query_text},
        ],
        "temperature": 0,
        "max_tokens": int(max_output_tokens),
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Authorization": "Bearer local-routecode", "Content-Type": "application/json"}
    return post_json(f"{base_url.rstrip('/')}/chat/completions", payload, headers, timeout_s)


def collect_reasoning_rows(
    query_table: pd.DataFrame,
    output_dir: Path,
    *,
    base_url: str,
    max_output_tokens: int,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_qwen8_reasoning"
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(row: pd.Series) -> dict[str, object]:
        query_id = str(row["query_id"])
        cache_path = cache_dir / f"{query_id.replace(':', '_')}.json"
        cache_hit = cache_path.exists()
        start = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_qwen_reasoning(base_url, str(row["query_text"]), max_output_tokens)
            except (urllib.error.URLError, TimeoutError, TimeoutError, Exception) as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        text = extract_openai_compatible_text(payload) if payload.get("_status", status) == "success" else ""
        input_tokens, output_tokens = usage_from_openai_compatible(payload, 0, max_output_tokens)
        return {
            "query_id": query_id,
            "reasoning_status": str(payload.get("_status", status)),
            "reasoning_error_type": str(payload.get("_error_type", error_type)),
            "reasoning_text": text,
            "reasoning_answer": normalize_answer(text),
            "reasoning_input_tokens": int(input_tokens),
            "reasoning_output_tokens": int(output_tokens),
            "reasoning_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "reasoning_cache_hit": cache_hit,
            "reasoning_raw_path": str(cache_path),
        }

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in query_table.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def add_reasoning_scores(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    scored = [
        score_output(str(answer), str(gold), str(metric))
        for answer, gold, metric in zip(table["reasoning_text"], table["gold_answer"], table["metric"])
    ]
    table[f"{REASONING_MODEL}_answer"] = [parsed for parsed, _ in scored]
    table[f"{REASONING_MODEL}_quality"] = [quality for _, quality in scored]
    table[f"{REASONING_MODEL}_cost"] = 0.0
    table[f"{REASONING_MODEL}_utility_selected_cost"] = table[f"{REASONING_MODEL}_quality"].astype(float)
    table["reasoning_gemini_agree"] = table[f"{REASONING_MODEL}_answer"].astype(str).map(normalize_answer) == table[
        f"{GEMINI_MODEL}_answer"
    ].astype(str).map(normalize_answer)
    table["reasoning_qwen8_answer_only_agree"] = table[f"{REASONING_MODEL}_answer"].astype(str).map(normalize_answer) == table[
        f"{LOCAL_MODEL}_answer"
    ].astype(str).map(normalize_answer)
    return table


def evaluate_actions(
    table: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    split: str,
    lambda_cost: float,
    cost_norm: float,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    reasoning_final: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        if action == "reasoning":
            quality = float(row[f"{REASONING_MODEL}_quality"])
            cost = 0.0
            gemini = False
            gpt = False
            reasoning = True
        elif action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"])
            gemini = True
            gpt = False
            reasoning = False
        elif action == "gemini_then_gpt_guarded":
            quality = float(row[f"{GPT_MODEL}_quality"]) if bool(row["gpt_answer_available"]) else float(
                row[f"{GEMINI_MODEL}_quality"]
            )
            cost = float(row[f"{GEMINI_MODEL}_cost"]) + float(row[f"{GPT_MODEL}_cost"])
            gemini = True
            gpt = True
            reasoning = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gemini_calls.append(gemini)
        gpt_calls.append(gpt)
        reasoning_final.append(reasoning)
    current_oracle_quality = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    expanded_oracle_quality = table[
        [f"{LOCAL_MODEL}_quality", f"{REASONING_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]
    ].max(axis=1)
    current_oracle_utility = table[
        [f"{LOCAL_MODEL}_utility_selected_cost", f"{GEMINI_MODEL}_utility_selected_cost", "gemini_then_gpt_guarded_utility"]
    ].max(axis=1)
    expanded_oracle_utility = table[
        [
            f"{LOCAL_MODEL}_utility_selected_cost",
            f"{REASONING_MODEL}_utility_selected_cost",
            f"{GEMINI_MODEL}_utility_selected_cost",
            "gemini_then_gpt_guarded_utility",
        ]
    ].max(axis=1)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(np.mean(qualities) - lambda_cost * (np.mean(costs) / cost_norm))
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_current_oracle": float(current_oracle_quality.mean() - mean_quality),
        "utility_ratio_to_current_oracle": float(mean_utility / current_oracle_utility.mean()),
        "quality_gap_to_expanded_oracle": float(expanded_oracle_quality.mean() - mean_quality),
        "utility_ratio_to_expanded_oracle": float(mean_utility / expanded_oracle_utility.mean()),
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum()),
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gemini_calls, gpt_calls)])),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "reasoning_final_rate": float(np.mean(reasoning_final)),
        "remote_cost_total_usd": float(np.sum(costs)),
        "action_counts": json.dumps(actions.value_counts().to_dict(), sort_keys=True),
    }


def policy_actions(table: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "all_qwen8_reasoning":
        return pd.Series("reasoning", index=table.index)
    if policy == "reasoning_gemini_agree_else_gpt":
        actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
        actions.loc[table["reasoning_gemini_agree"].astype(bool)] = "reasoning"
        return actions
    if policy == "reasoning_or_qwen_answer_agree_else_gpt":
        actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
        accept = table["reasoning_gemini_agree"].astype(bool) | table["reasoning_qwen8_answer_only_agree"].astype(bool)
        actions.loc[accept] = "reasoning"
        return actions
    if policy == "reasoning_else_gemini_on_short_else_gpt":
        actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
        actions.loc[table["query_len"].astype(int).le(220)] = "reasoning"
        return actions
    raise ValueError(policy)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_table = pd.read_csv(args.query_table)
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    if splits:
        query_table = query_table[query_table["split"].astype(str).isin(splits)].copy()
    if args.max_rows:
        query_table = query_table.head(int(args.max_rows)).copy()
    if not {"gold_answer", "metric"}.issubset(query_table.columns):
        gold = load_gold_rows(set(query_table["query_id"].astype(str)))
        query_table = query_table.merge(gold, on="query_id", how="left")
    probe = collect_reasoning_rows(
        query_table,
        output_dir,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
    )
    probe_path = output_dir / "table_qwen8_reasoning_outputs.csv"
    probe.to_csv(probe_path, index=False)
    table = query_table.merge(probe, on="query_id", how="left")
    table = add_reasoning_scores(table)
    table_path = output_dir / "query_table_with_qwen8_reasoning.csv"
    table.to_csv(table_path, index=False)

    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    rows: list[dict[str, object]] = []
    policies = [
        "all_qwen8_reasoning",
        "reasoning_gemini_agree_else_gpt",
        "reasoning_or_qwen_answer_agree_else_gpt",
        "reasoning_else_gemini_on_short_else_gpt",
    ]
    for split, frame in table.groupby("split", sort=False):
        for policy in policies:
            rows.append(
                evaluate_actions(
                    frame,
                    policy_actions(frame, policy),
                    method=policy,
                    split=str(split),
                    lambda_cost=args.lambda_cost,
                    cost_norm=cost_norm,
                )
            )
    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    results_path = output_dir / "table_qwen8_reasoning_gate.csv"
    results.to_csv(results_path, index=False)

    memo = [
        "# Qwen8 Reasoning Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Rows with Qwen8 reasoning outputs: `{len(table)}`.",
        "The probe uses local vLLM Qwen3-8B with `enable_thinking=True`; remote dollar cost is zero.",
        "",
        "## Results",
        "",
        markdown_table(
            results[
                [
                    "method",
                    "split",
                    "n_queries",
                    "mean_quality",
                    "mean_utility",
                    "quality_gap_to_current_oracle",
                    "utility_ratio_to_current_oracle",
                    "quality_gap_to_expanded_oracle",
                    "utility_ratio_to_expanded_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "frontier_call_rate",
                    "gpt_call_rate",
                    "reasoning_final_rate",
                    "action_counts",
                ]
            ]
        ),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{probe_path}`",
        f"- `{table_path}`",
    ]
    memo_path = output_dir / "QWEN8_REASONING_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
