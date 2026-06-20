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
    extract_openai_text,
    load_env_values,
    normalize_answer,
    post_json,
    resolve_key,
    score_output,
    usage_from_openai,
)


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
COMPACT_MODEL = "gpt-5.5-compact-rescue"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT compact rescue probes on mixed exact math.")
    parser.add_argument("--query-table", default="results/controlled/gemini_verifier_gate/query_table_with_verifier.csv")
    parser.add_argument("--output-dir", default="results/controlled/gpt_compact_rescue_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--probe-scope",
        choices=["qwen_gemini_disagree", "gemini_verifier_uncertain", "union_uncertain", "all"],
        default="union_uncertain",
    )
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


def compact_prompt(row: pd.Series) -> str:
    return (
        "You are a compact exact-answer rescue model. Solve privately.\n"
        "Given a problem and Gemini's candidate final answer, return JSON only:\n"
        "{\"decision\":\"accept\" or \"replace\", \"answer\":\"final answer\"}\n"
        "Use accept only if the candidate is correct. If it is wrong, use replace and put the corrected final answer.\n"
        "No explanation, no markdown.\n\n"
        f"Problem:\n{row['query_text']}\n\n"
        f"Gemini candidate final answer:\n{row[f'{GEMINI_MODEL}_answer']}"
    )


def call_openai_compact(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 90.0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": GPT_MODEL,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            payload.pop("text", None)
            return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
        raise


def should_probe(row: pd.Series, scope: str) -> bool:
    disagree = not bool(row.get("qwen8_gemini_agree", False))
    verifier_uncertain = str(row.get("verifier_verdict", "")).upper() != "YES"
    if scope == "qwen_gemini_disagree":
        return disagree
    if scope == "gemini_verifier_uncertain":
        return verifier_uncertain
    if scope == "union_uncertain":
        return disagree or verifier_uncertain
    if scope == "all":
        return True
    raise ValueError(scope)


def collect_compact_rows(
    query_table: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    max_output_tokens: int,
    concurrency: int,
    probe_scope: str,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_gpt_compact" / GPT_MODEL
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe_table = query_table[query_table.apply(lambda row: should_probe(row, probe_scope), axis=1)].copy()

    def one(row: pd.Series) -> dict[str, object]:
        query_id = str(row["query_id"])
        cache_path = cache_dir / f"{query_id.replace(':', '_')}.json"
        cache_hit = cache_path.exists()
        start = time.time()
        status = "success"
        error_type = ""
        prompt = compact_prompt(row)
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_openai_compact(prompt, api_key, max_output_tokens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row_status = str(payload.get("_status", status))
        text = extract_openai_text(payload) if row_status == "success" else ""
        if row_status == "success":
            input_tokens, output_tokens = usage_from_openai(payload, 0, max_output_tokens)
        else:
            input_tokens, output_tokens = 0, 0
        return {
            "query_id": query_id,
            "compact_status": str(payload.get("_status", status)),
            "compact_error_type": str(payload.get("_error_type", error_type)),
            "compact_text": text,
            "compact_input_tokens": int(input_tokens),
            "compact_output_tokens": int(output_tokens),
            "compact_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "compact_cache_hit": cache_hit,
            "compact_raw_path": str(cache_path),
        }

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in probe_table.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def parse_compact_answer(text: object) -> tuple[str, str]:
    raw = str(text or "").strip()
    decision = "replace"
    answer = raw
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            decision = str(payload.get("decision", decision)).strip().lower()
            answer = str(payload.get("answer", answer)).strip()
        except json.JSONDecodeError:
            pass
    if decision not in {"accept", "replace"}:
        decision = "replace"
    return decision, normalize_answer(answer)


def add_compact_scores(table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    table = table.copy()
    for column, default in [
        ("compact_text", ""),
        ("compact_input_tokens", 0),
        ("compact_output_tokens", 0),
        ("compact_latency_s", 0.0),
        ("compact_status", "not_called"),
    ]:
        if column not in table.columns:
            table[column] = default
        table[column] = table[column].fillna(default)
    parsed = [parse_compact_answer(text) for text in table["compact_text"]]
    table["compact_decision"] = [decision for decision, _ in parsed]
    table[f"{COMPACT_MODEL}_answer"] = [answer for _, answer in parsed]
    accept_mask = table["compact_decision"].eq("accept")
    table.loc[accept_mask, f"{COMPACT_MODEL}_answer"] = table.loc[accept_mask, f"{GEMINI_MODEL}_answer"].astype(str)
    scored = [
        score_output(str(answer), str(gold), str(metric))
        for answer, gold, metric in zip(table[f"{COMPACT_MODEL}_answer"], table["gold_answer"], table["metric"])
    ]
    table[f"{COMPACT_MODEL}_answer"] = [answer for answer, _ in scored]
    table[f"{COMPACT_MODEL}_quality"] = [quality for _, quality in scored]
    table["compact_cost"] = table["compact_input_tokens"].astype(float) * (5.00 / 1_000_000) + table[
        "compact_output_tokens"
    ].astype(float) * (30.00 / 1_000_000)
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    table[f"{COMPACT_MODEL}_utility_selected_cost"] = table[f"{COMPACT_MODEL}_quality"].astype(float) - 0.35 * (
        table["compact_cost"].astype(float) / cost_norm
    )
    return table, cost_norm


def policy_actions(table: pd.DataFrame, policy: str) -> pd.Series:
    if policy == "qwen_agree_else_compact":
        actions = pd.Series("compact", index=table.index)
        actions.loc[table["qwen8_gemini_agree"].astype(bool)] = "gemini"
        return actions
    if policy == "verifier_yes_else_compact":
        actions = pd.Series("compact", index=table.index)
        actions.loc[table["verifier_verdict"].eq("YES")] = "gemini"
        return actions
    if policy == "qwen_or_verifier_yes_else_compact":
        actions = pd.Series("compact", index=table.index)
        accept = table["qwen8_gemini_agree"].astype(bool) | table["verifier_verdict"].eq("YES")
        actions.loc[accept] = "gemini"
        return actions
    if policy == "compact_everywhere":
        return pd.Series("compact", index=table.index)
    raise ValueError(policy)


def evaluate_policy(
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
    gemini_calls: list[bool] = []
    compact_calls: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        if action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"])
            gemini = True
            compact = False
        elif action == "compact":
            if str(row.get("compact_status", "not_called")) == "not_called":
                quality = float(row[f"{GEMINI_MODEL}_quality"])
                cost = float(row[f"{GEMINI_MODEL}_cost"])
                gemini = True
                compact = False
            else:
                quality = float(row[f"{COMPACT_MODEL}_quality"])
                cost = float(row["compact_cost"])
                gemini = False
                compact = True
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gemini_calls.append(gemini)
        compact_calls.append(compact)
    current_oracle_quality = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    current_oracle_utility = table[
        [f"{LOCAL_MODEL}_utility_selected_cost", f"{GEMINI_MODEL}_utility_selected_cost", "gemini_then_gpt_guarded_utility"]
    ].max(axis=1)
    expanded_oracle_quality = table[
        [f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality", f"{COMPACT_MODEL}_quality"]
    ].max(axis=1)
    expanded_oracle_utility = table[
        [
            f"{LOCAL_MODEL}_utility_selected_cost",
            f"{GEMINI_MODEL}_utility_selected_cost",
            "gemini_then_gpt_guarded_utility",
            f"{COMPACT_MODEL}_utility_selected_cost",
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
        "frontier_call_rate": float(np.mean([g or c for g, c in zip(gemini_calls, compact_calls)])),
        "gemini_call_rate": float(np.mean(gemini_calls)),
        "gpt_compact_call_rate": float(np.mean(compact_calls)),
        "remote_cost_total_usd": float(np.sum(costs)),
        "action_counts": json.dumps(actions.value_counts().to_dict(), sort_keys=True),
    }


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
    if args.max_rows:
        query_table = query_table.head(int(args.max_rows)).copy()
    if not {"gold_answer", "metric"}.issubset(query_table.columns):
        gold = load_gold_rows(set(query_table["query_id"].astype(str)))
        query_table = query_table.merge(gold, on="query_id", how="left")
    env_values = load_env_values(args.env_file)
    api_key = resolve_key(env_values, ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise ValueError("OpenAI API key not found.")
    compact = collect_compact_rows(
        query_table,
        output_dir,
        api_key=api_key,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        probe_scope=args.probe_scope,
    )
    compact_path = output_dir / "table_gpt_compact_outputs.csv"
    compact.to_csv(compact_path, index=False)
    table = query_table.merge(compact, on="query_id", how="left")
    table, cost_norm = add_compact_scores(table)
    table_path = output_dir / "query_table_with_gpt_compact.csv"
    table.to_csv(table_path, index=False)

    rows: list[dict[str, object]] = []
    policies = [
        "qwen_agree_else_compact",
        "verifier_yes_else_compact",
        "qwen_or_verifier_yes_else_compact",
        "compact_everywhere",
    ]
    for split, frame in table.groupby("split", sort=False):
        for policy in policies:
            rows.append(
                evaluate_policy(
                    frame,
                    policy_actions(frame, policy),
                    method=policy,
                    split=str(split),
                    lambda_cost=args.lambda_cost,
                    cost_norm=cost_norm,
                )
            )
    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    results_path = output_dir / "table_gpt_compact_rescue_gate.csv"
    results.to_csv(results_path, index=False)
    test_rows = results[results["split"].eq("test")].copy()
    memo = [
        "# GPT Compact Rescue Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Probe scope: `{args.probe_scope}`.",
        f"Rows with compact GPT calls: `{len(compact)}`.",
        f"Compact GPT total cost: `${table['compact_cost'].sum():.4f}`.",
        "The compact probe asks GPT-5.5 for a correction decision and final answer. It is charged as a model call, not treated as a free router feature.",
        "",
        "## Held-Out Test Results",
        "",
        markdown_table(
            test_rows[
                [
                    "method",
                    "n_queries",
                    "mean_quality",
                    "mean_utility",
                    "quality_gap_to_current_oracle",
                    "utility_ratio_to_current_oracle",
                    "quality_gap_to_expanded_oracle",
                    "utility_ratio_to_expanded_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "frontier_call_rate",
                    "gpt_compact_call_rate",
                    "remote_cost_total_usd",
                    "action_counts",
                ]
            ]
        ),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{compact_path}`",
        f"- `{table_path}`",
    ]
    memo_path = output_dir / "GPT_COMPACT_RESCUE_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
