from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate capped GPT rescue from cached GPT token traces.")
    parser.add_argument("--query-table", default="results/controlled/gemini_verifier_gate/query_table_with_verifier.csv")
    parser.add_argument("--output-dir", default="results/controlled/budgeted_gpt_rescue_sim")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--caps", default="128,192,256,320,384,448,512,640,768,896,1024")
    return parser.parse_args()


def load_gpt_trace(query_ids: set[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in [
        Path("results/controlled/math500_qwen8_live_pilot_1024"),
        Path("results/controlled/livemathbench_live_pilot_1024"),
        Path("results/controlled/aime_qwen8_live_pilot_1024"),
    ]:
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(
            path,
            columns=["query_id", "model_id", "input_tokens", "output_tokens", "parsed_answer", "quality_score"],
        )
        frame = frame[frame["model_id"].eq(GPT_MODEL) & frame["query_id"].astype(str).isin(query_ids)]
        if not frame.empty:
            frames.append(frame.drop_duplicates("query_id", keep="first"))
    if not frames:
        raise ValueError("No cached GPT trace rows found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("query_id", keep="first")


def evaluate_policy(table: pd.DataFrame, cap: int, policy: str, lambda_cost: float, cost_norm: float) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    capped_answer_available: list[bool] = []
    for _, row in table.iterrows():
        if policy == "qwen_agree_else_capped_gpt":
            call_gpt = not bool(row["qwen8_gemini_agree"])
        elif policy == "verifier_yes_else_capped_gpt":
            call_gpt = str(row["verifier_verdict"]) != "YES"
        elif policy == "qwen_or_verifier_yes_else_capped_gpt":
            call_gpt = not (bool(row["qwen8_gemini_agree"]) or str(row["verifier_verdict"]) == "YES")
        elif policy == "disagree_and_verifier_no_else_gemini":
            call_gpt = (not bool(row["qwen8_gemini_agree"])) and str(row["verifier_verdict"]) != "YES"
        else:
            raise ValueError(policy)

        cost = float(row[f"{GEMINI_MODEL}_cost"])
        quality = float(row[f"{GEMINI_MODEL}_quality"])
        has_capped_answer = False
        if call_gpt:
            # Approximation from the cached full GPT trace: if the full response needed more
            # than `cap` output tokens, a capped run is treated as incomplete and falls back
            # to the already-paid Gemini answer.
            cost += float(row["gpt_input_tokens"]) * (5.00 / 1_000_000) + min(float(row["gpt_output_tokens"]), cap) * (
                30.00 / 1_000_000
            )
            has_capped_answer = bool(row["gpt_answer_available"]) and float(row["gpt_output_tokens"]) <= cap
            if has_capped_answer:
                quality = float(row[f"{GPT_MODEL}_quality"])
        qualities.append(quality)
        costs.append(cost)
        gpt_calls.append(call_gpt)
        capped_answer_available.append(has_capped_answer)
    oracle_quality = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    oracle_utility = table[
        [f"{LOCAL_MODEL}_utility_selected_cost", f"{GEMINI_MODEL}_utility_selected_cost", "gemini_then_gpt_guarded_utility"]
    ].max(axis=1)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(np.mean(qualities) - lambda_cost * (np.mean(costs) / cost_norm))
    return {
        "policy": policy,
        "cap_output_tokens": int(cap),
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_oracle": float(oracle_quality.mean() - mean_quality),
        "utility_ratio_to_cost_oracle": float(mean_utility / oracle_utility.mean()),
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum()),
        "frontier_call_rate": 1.0,
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "capped_gpt_answer_rate": float(np.mean(capped_answer_available)),
        "remote_cost_total_usd": float(np.sum(costs)),
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    trace = load_gpt_trace(set(table["query_id"].astype(str))).rename(
        columns={
            "input_tokens": "gpt_input_tokens",
            "output_tokens": "gpt_output_tokens",
            "parsed_answer": "gpt_trace_answer",
            "quality_score": "gpt_trace_quality",
        }
    )
    table = table.merge(trace.drop(columns=["model_id"]), on="query_id", how="left")
    table_path = out_dir / "query_table_with_gpt_trace.csv"
    table.to_csv(table_path, index=False)
    caps = [int(item) for item in args.caps.split(",") if item.strip()]
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    policies = [
        "qwen_agree_else_capped_gpt",
        "verifier_yes_else_capped_gpt",
        "qwen_or_verifier_yes_else_capped_gpt",
        "disagree_and_verifier_no_else_gemini",
    ]
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        for cap in caps:
            for policy in policies:
                row = evaluate_policy(frame, cap, policy, args.lambda_cost, cost_norm)
                row["split"] = str(split)
                rows.append(row)
    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    results_path = out_dir / "table_budgeted_gpt_rescue_sim.csv"
    results.to_csv(results_path, index=False)
    test_rows = results[results["split"].eq("test")].copy()
    feasible = test_rows[test_rows["quality_gap_to_oracle"].le(0.03)]
    best = feasible.sort_values("utility_ratio_to_cost_oracle", ascending=False).head(1)
    memo = [
        "# Budgeted GPT Rescue Simulation Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        "This is a simulation from cached full GPT traces, not a new API run. It approximates a capped GPT call as incomplete when the cached full response needed more output tokens than the cap.",
        "",
        "## Best Held-Out Feasible Row",
        "",
        markdown_table(
            best[
                [
                    "policy",
                    "cap_output_tokens",
                    "n_queries",
                    "mean_quality",
                    "quality_gap_to_oracle",
                    "utility_ratio_to_cost_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "gpt_call_rate",
                    "capped_gpt_answer_rate",
                ]
            ]
        )
        if not best.empty
        else "_No held-out row reached the 3-point quality target._",
        "",
        "## Held-Out Top Rows",
        "",
        markdown_table(
            test_rows.sort_values(["quality_gap_to_oracle", "utility_ratio_to_cost_oracle"], ascending=[True, False])
            .head(12)[
                [
                    "policy",
                    "cap_output_tokens",
                    "mean_quality",
                    "quality_gap_to_oracle",
                    "utility_ratio_to_cost_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "gpt_call_rate",
                    "capped_gpt_answer_rate",
                ]
            ]
        ),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{table_path}`",
    ]
    memo_path = out_dir / "BUDGETED_GPT_RESCUE_SIM_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
