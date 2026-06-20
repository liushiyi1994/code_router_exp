from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import normalize_answer, score_output


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
SMALL_LOCAL_MODELS = ("qwen3-0.6b-probe", "qwen3-4b-local")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mixed exact-math local-consensus admission gates.")
    parser.add_argument("--query-table", default="results/controlled/gemini_self_consistency_gate/query_table_with_self_consistency.csv")
    parser.add_argument("--output-dir", default="results/controlled/mixed_local_consensus_gate")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def load_small_local_outputs(query_ids: set[str]) -> pd.DataFrame:
    run_dirs = [
        Path("results/controlled/exact_math_live_pilot_1024"),
        Path("results/controlled/math500_qwen8_live_pilot_1024"),
        Path("results/controlled/livemathbench_smalllocals_live_pilot_1024"),
    ]
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[
            frame["status"].eq("success")
            & frame["model_id"].isin(SMALL_LOCAL_MODELS)
            & frame["query_id"].astype(str).isin(query_ids)
        ].copy()
        if frame.empty:
            continue
        rescored = [
            score_output(str(parsed), str(gold), str(metric))
            for parsed, gold, metric in zip(frame["parsed_answer"], frame["gold_answer"], frame["metric"])
        ]
        frame["parsed_answer"] = [parsed for parsed, _ in rescored]
        frame["quality_score"] = [quality for _, quality in rescored]
        frames.append(frame[["query_id", "model_id", "parsed_answer", "quality_score", "latency_s"]])
    if not frames:
        return pd.DataFrame(columns=["query_id", "model_id", "parsed_answer", "quality_score", "latency_s"])
    return pd.concat(frames, ignore_index=True).drop_duplicates(["query_id", "model_id"], keep="last")


def add_small_local_columns(table: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for model_id in SMALL_LOCAL_MODELS:
        model_rows = outputs[outputs["model_id"].eq(model_id)].set_index("query_id")
        table[f"{model_id}_answer"] = table["query_id"].map(model_rows["parsed_answer"]).fillna("").map(normalize_answer)
        table[f"{model_id}_quality"] = table["query_id"].map(model_rows["quality_score"]).astype(float)
        table[f"{model_id}_latency"] = table["query_id"].map(model_rows["latency_s"]).astype(float)
    return table


def local_consensus_mask(table: pd.DataFrame, mode: str) -> pd.Series:
    masks: list[bool] = []
    for _, row in table.iterrows():
        answers = [
            normalize_answer(str(row.get(f"{LOCAL_MODEL}_answer", ""))),
            normalize_answer(str(row.get("qwen3-4b-local_answer", ""))),
            normalize_answer(str(row.get("qwen3-0.6b-probe_answer", ""))),
        ]
        answers = [answer for answer in answers if answer and answer != "nan"]
        if mode == "qwen8_4b":
            masks.append(len(answers) > 1 and answers[0] == answers[1])
        elif mode == "any_pair":
            masks.append(any(answers.count(answer) >= 2 for answer in set(answers)))
        elif mode == "all_three":
            masks.append(len(answers) >= 3 and len(set(answers)) == 1)
        else:
            raise ValueError(mode)
    return pd.Series(masks, index=table.index)


def policy_actions(table: pd.DataFrame, mode: str, *, use_self: bool) -> pd.Series:
    actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
    actions.loc[table["qwen8_gemini_agree"].astype(bool)] = "gemini"
    if use_self:
        self_accept = (~table["qwen8_gemini_agree"].astype(bool)) & table["self_gemini_agree"].astype(bool)
        actions.loc[self_accept] = "gemini"
    actions.loc[local_consensus_mask(table, mode)] = "local"
    return actions


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
    local_final: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        if action == "local":
            quality = float(row[f"{LOCAL_MODEL}_quality"])
            cost = 0.0
            gemini = False
            gpt = False
            local = True
        elif action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"])
            gemini = True
            gpt = False
            local = False
        elif action == "gemini_then_gpt_guarded":
            quality = float(row[f"{GPT_MODEL}_quality"]) if bool(row["gpt_answer_available"]) else float(
                row[f"{GEMINI_MODEL}_quality"]
            )
            cost = float(row[f"{GEMINI_MODEL}_cost"]) + float(row[f"{GPT_MODEL}_cost"])
            gemini = True
            gpt = True
            local = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gemini_calls.append(gemini)
        gpt_calls.append(gpt)
        local_final.append(local)
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
        "utility_ratio_to_cost_oracle": float(mean_utility / oracle_utility.mean()),
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum()),
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gemini_calls, gpt_calls)])),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "local_final_rate": float(np.mean(local_final)),
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
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    outputs = load_small_local_outputs(set(table["query_id"].astype(str)))
    table = add_small_local_columns(table, outputs)
    table_path = out_dir / "query_table_with_small_locals.csv"
    table.to_csv(table_path, index=False)
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        for mode in ["qwen8_4b", "any_pair", "all_three"]:
            for use_self in [False, True]:
                method = f"{mode}{'_self' if use_self else ''}"
                actions = policy_actions(frame, mode, use_self=use_self)
                rows.append(
                    evaluate_actions(
                        frame,
                        actions,
                        method=method,
                        split=str(split),
                        lambda_cost=args.lambda_cost,
                        cost_norm=cost_norm,
                    )
                )
    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    results_path = out_dir / "table_mixed_local_consensus_gate.csv"
    results.to_csv(results_path, index=False)
    test_rows = results[results["split"].eq("test")].copy()
    memo = [
        "# Mixed Local Consensus Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        "Uses cached Qwen3-0.6B, Qwen3-4B, and Qwen3-8B outputs; no model calls are made by this script.",
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
                    "local_final_rate",
                    "remote_cost_total_usd",
                    "action_counts",
                ]
            ]
        ),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{table_path}`",
    ]
    memo_path = out_dir / "MIXED_LOCAL_CONSENSUS_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
