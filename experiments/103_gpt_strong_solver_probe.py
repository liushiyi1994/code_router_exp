from __future__ import annotations

import argparse
import hashlib
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
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    score_output,
    usage_from_openai,
)


GPT = "gpt-5.5"
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a stronger GPT-5.5 exact-math solver configuration.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/gpt_solver_cache_repair/query_table_expanded_local_pool_gpt_repaired.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/gpt_strong_solver_probe")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-api-spend-usd", type=float, default=8.50)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def prompt_for(row: pd.Series) -> str:
    query = str(row["query_text"]).strip()
    return f"""Solve the problem carefully. You may use private scratch work.

Return compact JSON only, with the final exact answer as a string:
{{"answer":"..."}}

Problem:
{query}
"""


def estimate_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def token_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def cache_name(query_id: str, effort: str, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{effort}:{max_output_tokens}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def openai_payloads(prompt: str, max_output_tokens: int, reasoning_effort: str) -> list[dict[str, Any]]:
    base = {
        "model": GPT,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "text": {"verbosity": "low"},
    }
    payloads: list[dict[str, Any]] = []
    if reasoning_effort:
        payloads.append(base | {"reasoning": {"effort": reasoning_effort}})
    payloads.extend([base | {"reasoning": {"effort": "minimal"}}, dict(base)])
    return payloads


def call_openai(prompt: str, api_key: str, max_output_tokens: int, reasoning_effort: str, timeout_s: float = 240.0) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""
    for payload in openai_payloads(prompt, max_output_tokens, reasoning_effort):
        try:
            return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code == 400:
                continue
            raise
    raise RuntimeError(last_error or "OpenAI request failed.")


def parse_answer(text: object) -> str:
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if "answer" in payload:
                return str(payload["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return raw


def collect_rows(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    reasoning_effort: str,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_strong_solver" / GPT / f"effort_{reasoning_effort}_max_{max_output_tokens}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = [prompt_for(row) for _, row in frame.iterrows()]
    missing = [
        prompt
        for prompt, (_, row) in zip(prompts, frame.iterrows())
        if not (cache_dir / cache_name(str(row["query_id"]), reasoning_effort, max_output_tokens)).exists()
    ]
    estimated = estimate_cost(missing, max_output_tokens)
    if estimated > max_api_spend_usd:
        raise RuntimeError(f"Estimated uncached GPT strong-solver spend ${estimated:.4f} exceeds cap ${max_api_spend_usd:.4f}.")

    def one(row: pd.Series, prompt: str) -> dict[str, Any]:
        query_id = str(row["query_id"])
        raw_path = cache_dir / cache_name(query_id, reasoning_effort, max_output_tokens)
        cache_hit = raw_path.exists()
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_openai(prompt, api_key, max_output_tokens, reasoning_effort)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - started
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row_status = str(payload.get("_status", status))
        text = extract_openai_text(payload) if row_status == "success" else ""
        answer = parse_answer(text)
        input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens) if row_status == "success" else (0, 0)
        metric = str(row.get("metric", "exact_final_answer") or "exact_final_answer")
        parsed, quality = score_output(answer, str(row["gold_answer"]), metric)
        if row_status != "success":
            quality = np.nan
        return {
            "query_id": query_id,
            "strong_status": row_status,
            "strong_error_type": str(payload.get("_error_type", error_type)),
            "strong_text": text,
            "strong_answer": parsed,
            "strong_quality": float(quality),
            "strong_input_tokens": int(input_tokens),
            "strong_output_tokens": int(output_tokens),
            "strong_latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
            "strong_cache_hit": cache_hit,
            "strong_raw_path": str(raw_path),
            "strong_cost": token_cost(int(input_tokens), int(output_tokens)),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row, prompt) for (_, row), prompt in zip(frame.iterrows(), prompts)]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
        all_gpt_cost = max(float(frame[f"{GPT}_cost"].sum()), 1e-12)
        strong_cost = float(frame["strong_cost"].sum())
        quality = float(frame["strong_quality"].mean())
        base_quality = float(frame[f"{GPT}_quality"].mean())
        expanded_oracle = float(frame["expanded_quality_oracle"].mean())
        utility = quality - lambda_cost * (strong_cost / all_gpt_cost)
        base_utility = base_quality - lambda_cost
        rows.append(
            {
                "split": split,
                "n_queries": int(len(frame)),
                "base_gpt_quality": base_quality,
                "strong_gpt_quality": quality,
                "expanded_quality_oracle": expanded_oracle,
                "strong_gap_to_expanded_oracle": expanded_oracle - quality,
                "strong_gain_over_base_gpt": quality - base_quality,
                "base_gpt_utility_vs_all_gpt_norm": base_utility,
                "strong_utility_vs_all_gpt_norm": utility,
                "strong_cost_total_usd": strong_cost,
                "strong_cost_vs_base_gpt_cost": strong_cost / all_gpt_cost,
                "mean_latency_s": float(frame["strong_latency_s"].mean()),
                "p95_latency_s": float(frame["strong_latency_s"].quantile(0.95)),
                "n_base_wrong_strong_right": int(((frame[f"{GPT}_quality"] < 1) & (frame["strong_quality"] > 0)).sum()),
                "n_base_right_strong_wrong": int(((frame[f"{GPT}_quality"] > 0) & (frame["strong_quality"] < 1)).sum()),
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
    table = pd.read_csv(args.query_table)
    if "metric" not in table.columns:
        table["metric"] = "exact_final_answer"
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    eval_table = table[table["split"].astype(str).isin(splits)].copy()
    if args.max_rows:
        eval_table = eval_table.head(int(args.max_rows)).copy()
    api_key = resolve_key(load_env_values(args.env_file), ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("OpenAI API key not found.")
    rows = collect_rows(
        eval_table,
        output_dir,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        reasoning_effort=args.reasoning_effort,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    rows_path = output_dir / "table_gpt_strong_solver_outputs.csv"
    rows.to_csv(rows_path, index=False)
    merged = table.merge(rows, on="query_id", how="left")
    merged_path = output_dir / "query_table_with_gpt_strong_solver.csv"
    merged.to_csv(merged_path, index=False)
    summary = summarize(merged[merged["split"].eq("train") | merged["strong_status"].notna()].copy(), args.lambda_cost)
    summary_path = output_dir / "table_gpt_strong_solver_summary.csv"
    summary.to_csv(summary_path, index=False)
    detail_cols = [
        "query_id",
        "dataset",
        "split",
        "gold_answer",
        f"{GPT}_answer",
        f"{GPT}_quality",
        "strong_answer",
        "strong_quality",
        "strong_cost",
        "strong_latency_s",
    ]
    changed = merged[
        merged["strong_status"].notna()
        & (
            ((merged[f"{GPT}_quality"] < 1) & (merged["strong_quality"] > 0))
            | ((merged[f"{GPT}_quality"] > 0) & (merged["strong_quality"] < 1))
        )
    ][detail_cols].copy()
    changed_path = output_dir / "table_gpt_strong_solver_changes.csv"
    changed.to_csv(changed_path, index=False)
    memo_path = output_dir / "GPT_STRONG_SOLVER_PROBE_MEMO.md"
    memo = [
        "# GPT Strong Solver Probe Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Model: `{GPT}`.",
        f"Reasoning effort requested: `{args.reasoning_effort}`.",
        f"Max output tokens: `{args.max_output_tokens}`.",
        "The prompt includes no gold answer. It asks for compact JSON with only the final answer.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Base/Strong Disagreements",
        "",
        markdown_table(changed.head(40)),
        "",
        "## Files",
        "",
        f"- `{rows_path}`",
        f"- `{merged_path}`",
        f"- `{summary_path}`",
        f"- `{changed_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
