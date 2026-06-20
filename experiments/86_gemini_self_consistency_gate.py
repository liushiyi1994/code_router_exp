from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_gemini_text,
    load_env_values,
    normalize_answer,
    resolve_key,
    score_output,
)


GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
LOCAL_MODEL = "qwen3-8b-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gemini self-consistency probes on mixed exact math.")
    parser.add_argument("--query-table", default="results/controlled/gemini_verifier_gate/query_table_with_verifier.csv")
    parser.add_argument("--output-dir", default="results/controlled/gemini_self_consistency_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def consistency_prompt(row: pd.Series) -> str:
    return (
        "Solve the problem independently. Return only the final answer with no explanation.\n\n"
        f"Problem:\n{row['query_text']}"
    )


def call_gemini(prompt: str, api_key: str, timeout_s: float = 90.0) -> dict:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": 96,
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


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


def collect_probe_rows(query_table: pd.DataFrame, output_dir: Path, api_key: str, concurrency: int) -> pd.DataFrame:
    cache_dir = output_dir / "raw_self_consistency" / GEMINI_MODEL
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(row: pd.Series) -> dict[str, object]:
        query_id = str(row["query_id"])
        cache_path = cache_dir / f"{query_id.replace(':', '_')}.json"
        prompt = consistency_prompt(row)
        cache_hit = cache_path.exists()
        start = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_gemini(prompt, api_key)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        text = extract_gemini_text(payload) if payload.get("_status", status) == "success" else ""
        parsed = normalize_answer(text)
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        return {
            "query_id": query_id,
            "self_status": str(payload.get("_status", status)),
            "self_error_type": str(payload.get("_error_type", error_type)),
            "self_text": text,
            "self_answer": parsed,
            "self_input_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "self_output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "self_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "self_cache_hit": cache_hit,
            "self_raw_path": str(cache_path),
        }

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, row) for _, row in query_table.iterrows()]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def add_probe_cost_and_quality(table: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    table = table.copy()
    table["self_cost"] = table["self_input_tokens"].astype(float) * (1.50 / 1_000_000) + table[
        "self_output_tokens"
    ].astype(float) * (9.00 / 1_000_000)
    table["self_gemini_agree"] = table["self_answer"].astype(str).map(normalize_answer) == table[
        f"{GEMINI_MODEL}_answer"
    ].astype(str).map(normalize_answer)
    table["self_qwen8_agree"] = table["self_answer"].astype(str).map(normalize_answer) == table[
        f"{LOCAL_MODEL}_answer"
    ].astype(str).map(normalize_answer)
    if {"gold_answer", "metric"}.issubset(table.columns):
        scored = [
            score_output(str(answer), str(gold), str(metric))
            for answer, gold, metric in zip(table["self_answer"], table["gold_answer"], table["metric"])
        ]
        table["self_answer"] = [parsed for parsed, _ in scored]
        table["self_quality"] = [quality for _, quality in scored]
    else:
        table["self_quality"] = np.nan
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    return table, cost_norm


def policy_actions(table: pd.DataFrame, policy: str) -> tuple[pd.Series, pd.Series]:
    actions = pd.Series("gemini", index=table.index)
    use_self_probe = pd.Series(False, index=table.index)
    if policy == "self_agree_else_gpt":
        use_self_probe.loc[:] = True
        actions.loc[~table["self_gemini_agree"].astype(bool)] = "gemini_then_gpt_guarded"
        return actions, use_self_probe
    if policy == "qwen_agree_else_self_agree_else_gpt":
        disagree = ~table["qwen8_gemini_agree"].astype(bool)
        use_self_probe.loc[disagree] = True
        actions.loc[disagree & ~table["self_gemini_agree"].astype(bool)] = "gemini_then_gpt_guarded"
        return actions, use_self_probe
    if policy == "qwen_or_self_agree_else_gpt":
        use_self_probe.loc[:] = True
        accept = table["qwen8_gemini_agree"].astype(bool) | table["self_gemini_agree"].astype(bool)
        actions.loc[~accept] = "gemini_then_gpt_guarded"
        return actions, use_self_probe
    if policy == "self_or_verifier_yes_else_gpt":
        use_self_probe.loc[:] = True
        accept = table["self_gemini_agree"].astype(bool) | table["verifier_verdict"].eq("YES")
        actions.loc[~accept] = "gemini_then_gpt_guarded"
        return actions, use_self_probe
    if policy == "qwen_or_self_or_verifier_yes_else_gpt":
        disagree = ~table["qwen8_gemini_agree"].astype(bool)
        use_self_probe.loc[disagree] = True
        accept = (
            table["qwen8_gemini_agree"].astype(bool)
            | table["self_gemini_agree"].astype(bool)
            | table["verifier_verdict"].eq("YES")
        )
        actions.loc[~accept] = "gemini_then_gpt_guarded"
        return actions, use_self_probe
    raise ValueError(policy)


def evaluate_policy(
    table: pd.DataFrame,
    actions: pd.Series,
    use_self_probe: pd.Series,
    method: str,
    split: str,
    lambda_cost: float,
    cost_norm: float,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        cost = float(row[f"{GEMINI_MODEL}_cost"])
        if bool(use_self_probe.loc[idx]):
            cost += float(row["self_cost"])
        if action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            gemini = True
            gpt = False
        elif action == "gemini_then_gpt_guarded":
            quality = float(row[f"{GPT_MODEL}_quality"]) if bool(row["gpt_answer_available"]) else float(
                row[f"{GEMINI_MODEL}_quality"]
            )
            cost += float(row[f"{GPT_MODEL}_cost"])
            gemini = True
            gpt = True
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gemini_calls.append(gemini)
        gpt_calls.append(gpt)
    oracle_quality = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    oracle_utility = table[
        [f"{LOCAL_MODEL}_utility_selected_cost", f"{GEMINI_MODEL}_utility_selected_cost", "gemini_then_gpt_guarded_utility"]
    ].max(axis=1)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(np.mean(qualities) - lambda_cost * (np.mean(costs) / cost_norm))
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_oracle": float(oracle_quality.mean() - mean_quality),
        "utility_ratio_to_cost_oracle": float(mean_utility / oracle_utility.mean())
        if abs(float(oracle_utility.mean())) > 1e-12
        else np.nan,
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum()),
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gemini_calls, gpt_calls)])),
        "gemini_call_rate": float(np.mean(gemini_calls)),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "self_probe_rate": float(np.mean(use_self_probe)),
        "self_cost_total_usd": float(table.loc[use_self_probe, "self_cost"].sum()),
        "remote_cost_total_usd": float(np.sum(costs)),
        "action_counts": json.dumps(actions.value_counts().to_dict(), sort_keys=True),
    }


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
    if args.max_rows:
        query_table = query_table.head(int(args.max_rows)).copy()
    api_key = resolve_key(load_env_values(args.env_file), ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise ValueError("Gemini API key not found.")

    probe = collect_probe_rows(query_table, output_dir, api_key, args.concurrency)
    probe_path = output_dir / "table_gemini_self_consistency_outputs.csv"
    probe.to_csv(probe_path, index=False)

    gold = load_gold_rows(set(query_table["query_id"].astype(str)))
    table = query_table.merge(probe, on="query_id", how="left").merge(gold, on="query_id", how="left")
    table, cost_norm = add_probe_cost_and_quality(table)
    table_path = output_dir / "query_table_with_self_consistency.csv"
    table.to_csv(table_path, index=False)

    rows: list[dict[str, object]] = []
    policies = [
        "self_agree_else_gpt",
        "qwen_agree_else_self_agree_else_gpt",
        "qwen_or_self_agree_else_gpt",
        "self_or_verifier_yes_else_gpt",
        "qwen_or_self_or_verifier_yes_else_gpt",
    ]
    for split, frame in table.groupby("split", sort=False):
        for policy in policies:
            actions, use_self_probe = policy_actions(frame, policy)
            rows.append(
                evaluate_policy(
                    frame,
                    actions,
                    use_self_probe,
                    policy,
                    str(split),
                    args.lambda_cost,
                    cost_norm,
                )
            )
    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    results_path = output_dir / "table_gemini_self_consistency_gate.csv"
    results.to_csv(results_path, index=False)

    test_rows = results[results["split"].eq("test")].copy()
    memo = [
        "# Gemini Self-Consistency Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Rows with self-consistency outputs: `{len(table)}`.",
        f"Self-consistency probe cost total: `${table['self_cost'].sum():.4f}`.",
        "",
        "The probe asks Gemini 3.5 Flash to solve independently with thinking disabled. Policies charge the original Gemini answer, the self-consistency probe when used, and GPT only when the gate escalates.",
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
                    "quality_gap_to_oracle",
                    "utility_ratio_to_cost_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "frontier_call_rate",
                    "gpt_call_rate",
                    "self_probe_rate",
                    "remote_cost_total_usd",
                    "action_counts",
                ]
            ]
        ),
        "",
        "## Probe Signal",
        "",
        markdown_table(pd.crosstab(table["self_gemini_agree"], table[f"{GEMINI_MODEL}_quality"]).reset_index()),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{probe_path}`",
        f"- `{table_path}`",
    ]
    memo_path = output_dir / "GEMINI_SELF_CONSISTENCY_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
