from __future__ import annotations

import argparse
import json
import re
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


BASE_LOCAL_MODEL = "qwen3-8b-local"
THINKING_MODEL = "qwen3-14b-awq-thinking-local"
SERVED_MODEL = "Qwen/Qwen3-14B-AWQ"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect/evaluate Qwen14-AWQ thinking-enabled local math probes.")
    parser.add_argument("--query-table", default="results/controlled/gemini_metadata_gate/query_table_with_gemini_metadata.csv")
    parser.add_argument("--output-dir", default="results/controlled/qwen14_thinking_probe")
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=768)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--splits", default="test", help="Comma-separated split filter, e.g. test or val,test.")
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def final_answer_text(text: object) -> str:
    raw = str(text or "")
    without_think = re.sub(r"<think>.*?</think>", "\n", raw, flags=re.IGNORECASE | re.DOTALL)
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", without_think)
    if boxed:
        return boxed[-1].strip()
    marked = re.split(r"(?i)(?:final\s+answer|answer)\s*[:：]", without_think)
    if len(marked) > 1:
        without_think = marked[-1]
    lines = [line.strip() for line in without_think.splitlines() if line.strip()]
    if lines:
        without_think = lines[-1]
    return without_think.strip().strip("*").strip()


def call_qwen14_thinking(base_url: str, query_text: str, max_output_tokens: int, timeout_s: float = 240.0) -> dict[str, Any]:
    payload = {
        "model": SERVED_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Solve the math problem. You may think internally. End with a single line "
                    "`Final answer: <answer>` and no extra text after that line."
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


def collect_probe_rows(
    query_table: pd.DataFrame,
    output_dir: Path,
    *,
    base_url: str,
    max_output_tokens: int,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_qwen14_thinking"
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
                payload = call_qwen14_thinking(base_url, str(row["query_text"]), max_output_tokens)
            except (urllib.error.URLError, TimeoutError, Exception) as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        text = extract_openai_compatible_text(payload) if payload.get("_status", status) == "success" else ""
        final_text = final_answer_text(text)
        parsed, quality = score_output(final_text, str(row["gold_answer"]), str(row["metric"]))
        input_tokens, output_tokens = usage_from_openai_compatible(payload, 0, max_output_tokens)
        return {
            "query_id": query_id,
            "thinking_status": str(payload.get("_status", status)),
            "thinking_error_type": str(payload.get("_error_type", error_type)),
            "thinking_text": text,
            "thinking_final_text": final_text,
            f"{THINKING_MODEL}_answer": parsed,
            f"{THINKING_MODEL}_quality": quality,
            "thinking_input_tokens": int(input_tokens),
            "thinking_output_tokens": int(output_tokens),
            "thinking_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "thinking_cache_hit": cache_hit,
            "thinking_raw_path": str(cache_path),
        }

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in query_table.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def evaluate_quality(table: pd.DataFrame, lambda_cost: float, cost_norm: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        quality_oracle = frame[[f"{BASE_LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
        expanded_quality_oracle = frame[
            [
                f"{BASE_LOCAL_MODEL}_quality",
                f"{THINKING_MODEL}_quality",
                f"{GEMINI_MODEL}_quality",
                f"{GPT_MODEL}_quality",
            ]
        ].max(axis=1)
        cost_oracle = frame[
            [
                f"{BASE_LOCAL_MODEL}_utility_selected_cost",
                f"{GEMINI_MODEL}_utility_selected_cost",
                f"{GPT_MODEL}_utility_selected_cost",
            ]
        ].max(axis=1)
        thinking_quality = frame[f"{THINKING_MODEL}_quality"].astype(float)
        thinking_utility = thinking_quality
        rows.append(
            {
                "method": "all_qwen14_awq_thinking",
                "split": str(split),
                "n_queries": int(len(frame)),
                "mean_quality": float(thinking_quality.mean()),
                "mean_utility": float(thinking_utility.mean()),
                "quality_gap_to_current_oracle": float(quality_oracle.mean() - thinking_quality.mean()),
                "quality_gap_to_expanded_oracle": float(expanded_quality_oracle.mean() - thinking_quality.mean()),
                "utility_ratio_to_current_cost_oracle": float(thinking_utility.mean() / cost_oracle.mean())
                if abs(float(cost_oracle.mean())) > 1e-12
                else np.nan,
                "normalized_remote_cost_vs_all_gpt": 0.0,
                "frontier_call_rate": 0.0,
                "gpt_call_rate": 0.0,
                "local_final_rate": 1.0,
                "p95_latency_s": float(np.quantile(frame["thinking_latency_s"].astype(float), 0.95)),
            }
        )
    return pd.DataFrame(rows)


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
    probe = collect_probe_rows(
        query_table,
        output_dir,
        base_url=args.base_url,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
    )
    probe_path = output_dir / "table_qwen14_thinking_outputs.csv"
    probe.to_csv(probe_path, index=False)
    table = query_table.merge(probe, on="query_id", how="left")
    table[f"{THINKING_MODEL}_utility_selected_cost"] = table[f"{THINKING_MODEL}_quality"].astype(float)
    table_path = output_dir / "query_table_with_qwen14_thinking.csv"
    table.to_csv(table_path, index=False)
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    results = evaluate_quality(table, args.lambda_cost, cost_norm)
    results_path = output_dir / "table_qwen14_thinking_probe.csv"
    results.to_csv(results_path, index=False)
    memo_path = output_dir / "QWEN14_THINKING_PROBE_MEMO.md"
    memo = [
        "# Qwen14-AWQ Thinking Probe Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Rows: `{len(table)}`. This script uses local vLLM only and makes no API calls.",
        f"Max output tokens: `{args.max_output_tokens}`.",
        "",
        "## Results",
        "",
        markdown_table(results),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{probe_path}`",
        f"- `{table_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
