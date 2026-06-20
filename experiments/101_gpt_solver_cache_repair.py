from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_openai_text,
    load_env_values,
    normalize_answer,
    resolve_key,
    score_output,
    usage_from_openai,
)


GPT_MODEL = "gpt-5.5"
GEMINI_MODEL = "gemini-3.5-flash"
LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
INPUT_PRICE_PER_MTOK = 5.00
OUTPUT_PRICE_PER_MTOK = 30.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair GPT-5.5 exact-math solver rows whose Responses payload was incomplete."
    )
    parser.add_argument(
        "--query-table",
        default="results/controlled/expanded_local_pool_qwen14/query_table_expanded_local_pool.csv",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[
            "results/controlled/math500_qwen14_awq_live_pilot_1024",
            "results/controlled/livemathbench_live_pilot_1024",
            "results/controlled/aime_qwen8_live_pilot_1024",
        ],
    )
    parser.add_argument("--output-dir", default="results/controlled/gpt_solver_cache_repair")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-estimated-cost-usd", type=float, default=7.50)
    parser.add_argument("--max-total-cost-usd", type=float, default=8.00)
    parser.add_argument("--scope", choices=["incomplete", "test-incomplete"], default="incomplete")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def repair_prompt(query_text: str) -> str:
    return (
        "Solve this exact-scored math problem privately. Return exactly one JSON object and nothing else:\n"
        '{"answer":"FINAL_ANSWER"}\n'
        "The answer value must be only the final exact answer, with no explanation or markdown.\n\n"
        f"Problem:\n{query_text}"
    )


def openai_payloads(prompt: str, max_output_tokens: int) -> list[dict[str, Any]]:
    base: dict[str, Any] = {
        "model": GPT_MODEL,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
    }
    return [
        base | {"reasoning": {"effort": "none"}, "text": {"verbosity": "low"}},
        base | {"reasoning": {"effort": "minimal"}, "text": {"verbosity": "low"}},
        base | {"text": {"verbosity": "low"}},
        dict(base),
    ]


def call_openai_repair(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 180.0) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""
    for payload in openai_payloads(prompt, max_output_tokens):
        try:
            return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code == 400:
                continue
            raise
    raise RuntimeError(last_error or "OpenAI request failed")


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return input_tokens * (INPUT_PRICE_PER_MTOK / 1_000_000) + output_tokens * (
        OUTPUT_PRICE_PER_MTOK / 1_000_000
    )


def token_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PRICE_PER_MTOK / 1_000_000) + output_tokens * (
        OUTPUT_PRICE_PER_MTOK / 1_000_000
    )


def parse_json_answer(text: str) -> str:
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            answer = payload.get("answer", "")
            return str(answer).strip()
        except json.JSONDecodeError:
            pass
    return raw


def payload_status(raw_path: object) -> tuple[str, str]:
    path = Path(str(raw_path))
    if not path.exists():
        return "missing", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return "read_error", type(exc).__name__
    status = str(payload.get("status") or payload.get("_status") or "")
    details = payload.get("incomplete_details") or {}
    reason = str(details.get("reason") or payload.get("_error_type") or "")
    return status, reason


def load_gpt_solver_rows(run_dirs: list[Path], query_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[
            frame["model_id"].eq(GPT_MODEL)
            & frame["status"].eq("success")
            & frame["query_id"].astype(str).isin(query_ids)
        ].copy()
        if frame.empty:
            continue
        statuses = [payload_status(path) for path in frame["raw_output_path"]]
        frame["gpt_payload_status"] = [status for status, _ in statuses]
        frame["gpt_payload_reason"] = [reason for _, reason in statuses]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined["query_id"] = combined["query_id"].astype(str)
    return combined.drop_duplicates("query_id", keep="last")


def select_repair_rows(query_table: pd.DataFrame, gpt_rows: pd.DataFrame, scope: str) -> pd.DataFrame:
    split_by_query = query_table.set_index("query_id")["split"].astype(str)
    rows = gpt_rows[gpt_rows["gpt_payload_status"].eq("incomplete")].copy()
    rows["split"] = rows["query_id"].map(split_by_query)
    if scope == "test-incomplete":
        rows = rows[rows["split"].eq("test")].copy()
    rows["repair_prompt"] = rows["query_text"].astype(str).map(repair_prompt)
    return rows.sort_values(["benchmark", "query_id"]).reset_index(drop=True)


def collect_repairs(
    repair_rows: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    concurrency: int,
    dry_run: bool,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_gpt_solver_repair" / GPT_MODEL
    cache_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return pd.DataFrame(
            [
                {
                    "query_id": row.query_id,
                    "repair_status": "dry_run",
                    "repair_payload_status": "",
                    "repair_payload_reason": "",
                    "repair_text": "",
                    "repair_answer": "",
                    "repair_parsed_answer": "",
                    "repair_quality": np.nan,
                    "repair_input_tokens": 0,
                    "repair_output_tokens": 0,
                    "repair_cost": 0.0,
                    "repair_latency_s": 0.0,
                    "repair_cache_hit": False,
                    "repair_raw_path": str(cache_dir / f"{str(row.query_id).replace(':', '_')}.json"),
                    "repair_replaceable": False,
                }
                for row in repair_rows.itertuples(index=False)
            ]
        )

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
                payload = call_openai_repair(str(row["repair_prompt"]), api_key, max_output_tokens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row_status = str(payload.get("_status", status))
        raw_status = str(payload.get("status") or "")
        details = payload.get("incomplete_details") or {}
        raw_reason = str(details.get("reason") or payload.get("_error_type") or "")
        text = extract_openai_text(payload) if row_status == "success" else ""
        answer = parse_json_answer(text)
        parsed_answer, quality = score_output(answer or text, str(row["gold_answer"]), str(row["metric"]))
        if row_status != "success":
            quality = np.nan
        input_tokens, output_tokens = usage_from_openai(
            payload,
            max(1, len(str(row["repair_prompt"])) // 4),
            max_output_tokens,
        )
        replaceable = bool(row_status == "success" and raw_status in {"completed", ""} and str(answer or text).strip())
        return {
            "query_id": query_id,
            "repair_status": row_status,
            "repair_payload_status": raw_status,
            "repair_payload_reason": raw_reason,
            "repair_text": text,
            "repair_answer": answer,
            "repair_parsed_answer": parsed_answer,
            "repair_quality": float(quality) if not pd.isna(quality) else np.nan,
            "repair_input_tokens": int(input_tokens),
            "repair_output_tokens": int(output_tokens),
            "repair_cost": token_cost(int(input_tokens), int(output_tokens)),
            "repair_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "repair_cache_hit": cache_hit,
            "repair_raw_path": str(cache_path),
            "repair_replaceable": replaceable,
        }

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in repair_rows.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def recompute_derived_columns(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    table = table.copy()
    cost_norm = max(float(pd.to_numeric(table[f"{GPT_MODEL}_cost"], errors="coerce").fillna(0.0).mean()), 1e-12)
    for model_id in [GEMINI_MODEL, GPT_MODEL]:
        table[f"{model_id}_utility_selected_cost"] = table[f"{model_id}_quality"].astype(float) - lambda_cost * (
            table[f"{model_id}_cost"].astype(float) / cost_norm
        )
    for model_id in LOCAL_MODELS:
        if f"{model_id}_quality" in table.columns:
            table[f"{model_id}_utility_direct"] = table[f"{model_id}_quality"].astype(float)
    gemini_answer = table[f"{GEMINI_MODEL}_answer"].fillna("").map(normalize_answer)
    gpt_answer = table[f"{GPT_MODEL}_answer"].fillna("").map(normalize_answer)
    table["gemini_gpt_agree"] = gemini_answer.eq(gpt_answer) & gemini_answer.ne("")
    table["gpt_answer_available"] = gpt_answer.ne("") & table[f"{GPT_MODEL}_quality"].notna()
    table["gemini_then_gpt_cost"] = table[f"{GEMINI_MODEL}_cost"].astype(float) + table[f"{GPT_MODEL}_cost"].astype(float)
    table["gemini_then_gpt_utility"] = table[f"{GPT_MODEL}_quality"].astype(float) - lambda_cost * (
        table["gemini_then_gpt_cost"].astype(float) / cost_norm
    )
    table["gemini_then_gpt_guarded_quality"] = np.where(
        table["gpt_answer_available"].astype(bool),
        table[f"{GPT_MODEL}_quality"].astype(float),
        table[f"{GEMINI_MODEL}_quality"].astype(float),
    )
    table["gemini_then_gpt_guarded_utility"] = table["gemini_then_gpt_guarded_quality"].astype(float) - lambda_cost * (
        table["gemini_then_gpt_cost"].astype(float) / cost_norm
    )
    local_quality_cols = [f"{model_id}_quality" for model_id in LOCAL_MODELS if f"{model_id}_quality" in table.columns]
    if local_quality_cols:
        table["local_oracle_quality"] = table[local_quality_cols].max(axis=1)
        table["expanded_quality_oracle"] = table[local_quality_cols + [f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(
            axis=1
        )
    base_quality_cols = ["qwen3-8b-local_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]
    present_base_quality = [column for column in base_quality_cols if column in table.columns]
    if present_base_quality:
        table["quality_oracle_model"] = table[present_base_quality].idxmax(axis=1).str.replace("_quality", "", regex=False)
    base_utility_cols = [
        "qwen3-8b-local_utility_selected_cost",
        f"{GEMINI_MODEL}_utility_selected_cost",
        f"{GPT_MODEL}_utility_selected_cost",
    ]
    present_base_utility = [column for column in base_utility_cols if column in table.columns]
    if present_base_utility:
        table["cost_oracle_model_selected_cost"] = (
            table[present_base_utility].idxmax(axis=1).str.replace("_utility_selected_cost", "", regex=False)
        )
    return table


def apply_repairs(query_table: pd.DataFrame, repairs: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    table = query_table.copy()
    if "gpt_repair_used" not in table.columns:
        table["gpt_repair_used"] = pd.Series(False, index=table.index, dtype=bool)
    for column in [
        "gpt_repair_status",
        "gpt_repair_payload_status",
        "gpt_repair_payload_reason",
        "gpt_repair_raw_path",
        "gpt_repair_old_answer",
    ]:
        if column not in table.columns:
            table[column] = pd.Series("", index=table.index, dtype=object)
    for column in ["gpt_repair_old_quality", "gpt_repair_old_cost", "gpt_repair_old_latency"]:
        if column not in table.columns:
            table[column] = pd.Series(np.nan, index=table.index, dtype=float)
    repair_by_query = repairs.set_index("query_id") if not repairs.empty else pd.DataFrame()
    for idx, row in table.iterrows():
        query_id = str(row["query_id"])
        if repairs.empty or query_id not in repair_by_query.index:
            continue
        repair = repair_by_query.loc[query_id]
        table.at[idx, "gpt_repair_status"] = str(repair.get("repair_status", ""))
        table.at[idx, "gpt_repair_payload_status"] = str(repair.get("repair_payload_status", ""))
        table.at[idx, "gpt_repair_payload_reason"] = str(repair.get("repair_payload_reason", ""))
        table.at[idx, "gpt_repair_raw_path"] = str(repair.get("repair_raw_path", ""))
        if not bool(repair.get("repair_replaceable", False)):
            continue
        table.at[idx, "gpt_repair_used"] = True
        table.at[idx, "gpt_repair_old_answer"] = str(row.get(f"{GPT_MODEL}_answer", ""))
        table.at[idx, "gpt_repair_old_quality"] = float(row.get(f"{GPT_MODEL}_quality", np.nan))
        table.at[idx, "gpt_repair_old_cost"] = float(row.get(f"{GPT_MODEL}_cost", np.nan))
        table.at[idx, "gpt_repair_old_latency"] = float(row.get(f"{GPT_MODEL}_latency", np.nan))
        repaired_answer = str(repair.get("repair_parsed_answer", ""))
        table.at[idx, f"{GPT_MODEL}_answer"] = repaired_answer
        table.at[idx, f"{GPT_MODEL}_answer_len"] = len(normalize_answer(repaired_answer))
        table.at[idx, f"{GPT_MODEL}_quality"] = float(repair.get("repair_quality", np.nan))
        table.at[idx, f"{GPT_MODEL}_cost"] = float(repair.get("repair_cost", 0.0))
        table.at[idx, f"{GPT_MODEL}_latency"] = float(repair.get("repair_latency_s", 0.0))
    return recompute_derived_columns(table, lambda_cost)


def split_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        row: dict[str, object] = {"split": split, "n_queries": int(len(frame))}
        for model_id in LOCAL_MODELS:
            if f"{model_id}_quality" in frame.columns:
                row[f"{model_id}_quality"] = float(frame[f"{model_id}_quality"].mean())
        for model_id in [GEMINI_MODEL, GPT_MODEL]:
            row[f"{model_id}_quality"] = float(frame[f"{model_id}_quality"].mean())
        row["local_oracle_quality"] = float(frame["local_oracle_quality"].mean())
        row["expanded_quality_oracle"] = float(frame["expanded_quality_oracle"].mean())
        row["gpt_repair_used"] = int(frame["gpt_repair_used"].astype(bool).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def best_quality_under_frontier(frame: pd.DataFrame, max_frontier: int) -> float:
    dp = {0: 0.0}
    for _, row in frame.iterrows():
        local_quality = float(row["local_oracle_quality"])
        frontier_quality = max(float(row[f"{GEMINI_MODEL}_quality"]), float(row[f"{GPT_MODEL}_quality"]))
        next_dp: dict[int, float] = {}
        for used_frontier, quality in dp.items():
            next_dp[used_frontier] = max(next_dp.get(used_frontier, -1.0), quality + local_quality)
            if used_frontier + 1 <= max_frontier:
                next_dp[used_frontier + 1] = max(
                    next_dp.get(used_frontier + 1, -1.0), quality + frontier_quality
                )
        dp = next_dp
    return max(dp.values()) / len(frame)


def frontier_bounds(table: pd.DataFrame, split: str, target_quality_gap: float = 0.03) -> pd.DataFrame:
    frame = table[table["split"].eq(split)].copy()
    oracle_quality = float(frame["expanded_quality_oracle"].mean())
    target_quality = oracle_quality - target_quality_gap
    rows: list[dict[str, object]] = []
    for rate in [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]:
        cap = min(len(frame), int(np.floor(rate * len(frame))))
        rows.append(
            {
                "split": split,
                "frontier_rate_cap": rate,
                "frontier_call_cap": cap,
                "max_quality": best_quality_under_frontier(frame, cap),
            }
        )
    min_calls = None
    for calls in range(len(frame) + 1):
        if best_quality_under_frontier(frame, calls) >= target_quality - 1e-12:
            min_calls = calls
            break
    rows.append(
        {
            "split": split,
            "frontier_rate_cap": "min_for_target",
            "frontier_call_cap": min_calls,
            "max_quality": best_quality_under_frontier(frame, int(min_calls)) if min_calls is not None else np.nan,
        }
    )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
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
    query_table["query_id"] = query_table["query_id"].astype(str)
    gpt_rows = load_gpt_solver_rows([Path(path) for path in args.run_dirs], set(query_table["query_id"]))
    if gpt_rows.empty:
        raise ValueError("No GPT solver rows found to audit.")
    repair_rows = select_repair_rows(query_table, gpt_rows, args.scope)
    audit_path = output_dir / "table_gpt_incomplete_audit.csv"
    repair_rows.drop(columns=["repair_prompt"], errors="ignore").to_csv(audit_path, index=False)

    prompts_to_call: list[str] = []
    cache_dir = output_dir / "raw_gpt_solver_repair" / GPT_MODEL
    for row in repair_rows.itertuples(index=False):
        cache_path = cache_dir / f"{str(row.query_id).replace(':', '_')}.json"
        if not cache_path.exists():
            prompts_to_call.append(str(row.repair_prompt))
    estimated = estimate_prompt_cost(prompts_to_call, args.max_output_tokens)
    if estimated > float(args.max_estimated_cost_usd):
        raise RuntimeError(
            f"Estimated uncached GPT repair spend ${estimated:.4f} exceeds cap ${args.max_estimated_cost_usd:.4f}."
        )

    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key and not args.dry_run:
        raise RuntimeError("OpenAI API key not found.")

    repairs = collect_repairs(
        repair_rows,
        output_dir,
        api_key=api_key or "",
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
    )
    if float(repairs["repair_cost"].sum()) > float(args.max_total_cost_usd):
        raise RuntimeError(
            f"Actual GPT repair spend ${repairs['repair_cost'].sum():.4f} exceeds cap ${args.max_total_cost_usd:.4f}."
        )

    repairs_path = output_dir / "table_gpt_solver_repairs.csv"
    repairs.to_csv(repairs_path, index=False)
    repaired_table = apply_repairs(query_table, repairs, args.lambda_cost)
    repaired_table_path = output_dir / "query_table_expanded_local_pool_gpt_repaired.csv"
    repaired_table.to_csv(repaired_table_path, index=False)

    summary = split_summary(repaired_table)
    summary_path = output_dir / "table_repaired_expanded_local_pool_summary.csv"
    summary.to_csv(summary_path, index=False)
    bounds = frontier_bounds(repaired_table, "test")
    bounds_path = output_dir / "table_repaired_frontier_bounds.csv"
    bounds.to_csv(bounds_path, index=False)

    test_summary = summary[summary["split"].eq("test")]
    memo = [
        "# GPT Solver Cache Repair Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Scope: `{args.scope}`.",
        f"Audited GPT solver rows: `{len(gpt_rows)}`.",
        f"Incomplete GPT rows selected for repair: `{len(repair_rows)}`.",
        f"Uncached estimated repair spend before calls: `${estimated:.4f}`.",
        f"Actual repair spend in this artifact: `${repairs['repair_cost'].sum():.4f}`.",
        f"Repair calls using cache: `{int(repairs['repair_cache_hit'].sum())}` of `{len(repairs)}`.",
        f"Replaceable repaired rows: `{int(repairs['repair_replaceable'].astype(bool).sum())}`.",
        "",
        "This is a solver-output cache repair. It does not use gold labels in the prompt, and it does not change the routing method. It corrects GPT rows where the original Responses payload was cut off by `max_output_tokens`.",
        "",
        "## Held-Out Summary",
        "",
        markdown_table(test_summary),
        "",
        "## Held-Out Frontier Bound",
        "",
        markdown_table(bounds),
        "",
        "## Files",
        "",
        f"- `{audit_path}`",
        f"- `{repairs_path}`",
        f"- `{repaired_table_path}`",
        f"- `{summary_path}`",
        f"- `{bounds_path}`",
    ]
    memo_path = output_dir / "GPT_SOLVER_CACHE_REPAIR_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")

    print(f"Wrote {repaired_table_path}")
    print(f"Wrote {repairs_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
