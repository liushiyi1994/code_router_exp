from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_gemini_text,
    load_env_values,
    normalize_answer,
    resolve_key,
    score_output,
)


GEMINI_MODEL = "gemini-3.5-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a cached Gemini strong-solver probe on exact-math rows.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv",
    )
    parser.add_argument(
        "--rescue-table",
        default="results/controlled/strong_inclusive_oracle_audit/table_heldout_oracle_rescue_cases.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/gemini_strong_solver_probe")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--category",
        default="strong_oracle",
        choices=["strong_oracle", "strong_only_rescue", "all"],
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--thinking-budget", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-spend-usd", type=float, default=2.0)
    return parser.parse_args()


def prompt_for(row: pd.Series) -> str:
    return (
        "Solve this exact-answer math problem carefully. You may reason internally, but return only the final "
        "answer with no explanation, no units unless required, and no Markdown.\n\n"
        f"Problem:\n{row['query_text']}"
    )


def call_gemini(
    *,
    prompt: str,
    api_key: str,
    max_output_tokens: int,
    thinking_budget: int,
    temperature: float,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    generation_config: dict[str, Any] = {
        "maxOutputTokens": int(max_output_tokens),
        "temperature": float(temperature),
    }
    if thinking_budget >= 0:
        generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def gemini_cost(payload: dict[str, Any]) -> float:
    usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
    prompt_tokens = int(usage.get("promptTokenCount", 0) or 0)
    candidate_tokens = int(usage.get("candidatesTokenCount", 0) or 0)
    thoughts_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)
    # Keep the same conservative Gemini Flash pricing convention used by prior controlled probes.
    return prompt_tokens * (1.50 / 1_000_000) + (candidate_tokens + thoughts_tokens) * (9.00 / 1_000_000)


def select_rows(query_table: pd.DataFrame, rescue_table: pd.DataFrame, split: str, category: str) -> pd.DataFrame:
    selected = query_table[query_table["split"].astype(str).eq(split)].copy()
    if category == "all":
        return selected
    if category == "strong_only_rescue":
        ids = set(
            rescue_table.loc[
                rescue_table["category"].astype(str).eq("strong_only_rescue"),
                "query_id",
            ].astype(str)
        )
        return selected[selected["query_id"].astype(str).isin(ids)].copy()
    ids = set(
        rescue_table.loc[
            rescue_table["oracle_model"].astype(str).eq("strong-gpt-5.5"),
            "query_id",
        ].astype(str)
    )
    return selected[selected["query_id"].astype(str).isin(ids)].copy()


def collect(args: argparse.Namespace, rows: pd.DataFrame, api_key: str) -> pd.DataFrame:
    output_dir = Path(args.output_dir)
    cache_dir = (
        output_dir
        / "raw_gemini_strong_solver"
        / GEMINI_MODEL
        / f"think_{args.thinking_budget}_max_{args.max_output_tokens}"
    )
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
                payload = call_gemini(
                    prompt=prompt_for(row),
                    api_key=api_key,
                    max_output_tokens=args.max_output_tokens,
                    thinking_budget=args.thinking_budget,
                    temperature=args.temperature,
                )
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:1000], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        text = extract_gemini_text(payload) if payload.get("_status", status) == "success" else ""
        parsed = normalize_answer(text)
        scored_answer, quality = score_output(parsed, str(row["gold_answer"]), str(row["metric"]))
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        return {
            "query_id": query_id,
            "split": str(row["split"]),
            "dataset": str(row["dataset"]),
            "status": str(payload.get("_status", status)),
            "error_type": str(payload.get("_error_type", error_type)),
            "answer_text": text,
            "parsed_answer": scored_answer,
            "gold_answer": str(row["gold_answer"]),
            "metric": str(row["metric"]),
            "quality": float(quality),
            "prompt_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "candidate_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "thoughts_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
            "total_tokens": int(usage.get("totalTokenCount", 0) or 0),
            "cost_usd": gemini_cost(payload),
            "latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "cache_hit": bool(cache_hit),
            "raw_path": str(cache_path),
        }

    existing_cost = 0.0
    if cache_dir.exists():
        for path in cache_dir.glob("*.json"):
            try:
                existing_cost += gemini_cost(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                pass
    if existing_cost > args.max_spend_usd:
        raise SystemExit(f"Existing cached cost estimate ${existing_cost:.4f} exceeds max spend.")

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in rows.iterrows()]
        for future in as_completed(futures):
            results.append(future.result())
            spent = sum(float(row.get("cost_usd", 0.0) or 0.0) for row in results if not row.get("cache_hit"))
            if spent > args.max_spend_usd:
                raise SystemExit(f"Stopping: new spend estimate ${spent:.4f} exceeds max spend.")
    return pd.DataFrame(results)


def write_memo(output_dir: Path, outputs: pd.DataFrame, args: argparse.Namespace) -> None:
    successful = outputs[outputs["status"].eq("success")]
    lines = [
        "# Gemini Strong-Solver Probe Memo",
        "",
        f"Model: `{GEMINI_MODEL}`.",
        f"Rows: `{len(outputs)}`. Successful: `{len(successful)}`.",
        f"Split/category: `{args.split}` / `{args.category}`.",
        f"Thinking budget: `{args.thinking_budget}`. Max output tokens: `{args.max_output_tokens}`.",
        f"Total estimated Gemini cost: `${outputs['cost_usd'].sum():.4f}`.",
        f"Mean quality on successful rows: `{successful['quality'].mean() if len(successful) else 0.0:.4f}`.",
        "",
        "Per-row outputs:",
        "",
        "| query_id | dataset | quality | cost_usd | latency_s | parsed_answer | gold_answer |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in outputs.sort_values("query_id").iterrows():
        lines.append(
            f"| {row['query_id']} | {row['dataset']} | {float(row['quality']):.4f} | "
            f"{float(row['cost_usd']):.6f} | {float(row['latency_s']):.2f} | "
            f"{str(row['parsed_answer']).replace('|', '/')} | {str(row['gold_answer']).replace('|', '/')} |"
        )
    lines.append("")
    output_dir.joinpath("GEMINI_STRONG_SOLVER_PROBE_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_table = pd.read_csv(args.query_table)
    rescue_table = pd.read_csv(args.rescue_table)
    rows = select_rows(query_table, rescue_table, args.split, args.category).sort_values("query_id")
    if args.max_rows is not None:
        rows = rows.head(int(args.max_rows))
    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise SystemExit("Missing Gemini API key in env file.")
    outputs = collect(args, rows, api_key)
    outputs = outputs.sort_values("query_id")
    outputs.to_csv(output_dir / "table_gemini_strong_solver_outputs.csv", index=False)
    write_memo(output_dir, outputs, args)
    print(f"Wrote {len(outputs)} rows to {output_dir / 'table_gemini_strong_solver_outputs.csv'}")
    print(f"Estimated Gemini cost: ${outputs['cost_usd'].sum():.4f}")
    print(f"Mean quality: {outputs['quality'].mean() if len(outputs) else 0.0:.4f}")


if __name__ == "__main__":
    main()
