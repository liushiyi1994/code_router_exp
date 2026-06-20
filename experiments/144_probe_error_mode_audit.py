from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit broad100 probe-routing error modes against cost-aware oracle.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--augmented-outputs",
        type=Path,
        default=Path("results/controlled/broad100_train_supervised_strong_gain_gate/model_outputs_with_gemini_strong_all_splits.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_probe_error_mode_audit"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    strong_gate = load_module("experiments/141_train_supervised_strong_gain_gate.py", "strong_gate")

    suites: list[tuple[str, pd.DataFrame]] = [
        ("base", package.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))),
    ]
    if args.augmented_outputs.exists():
        suites.append(("augmented_strong", load_precomputed_outputs(args.augmented_outputs, lambda_cost=float(args.lambda_cost))))

    query_rows: list[pd.DataFrame] = []
    eval_rows: list[dict[str, Any]] = []
    for matrix_name, outputs in suites:
        selections = build_selections(package, strong_gate, outputs)
        for split in ["val", "test"]:
            cost_oracle = outputs[outputs["split"].eq(split)].loc[
                outputs[outputs["split"].eq(split)].groupby("query_id")["utility"].idxmax()
            ]
            quality_oracle = outputs[outputs["split"].eq(split)].loc[
                outputs[outputs["split"].eq(split)].groupby("query_id")["quality_score"].idxmax()
            ]
            for method, by_split in selections.items():
                selected = by_split.get(split)
                if selected is None or selected.empty:
                    continue
                selected_rows = package.selected_to_rows(outputs, selected, split=split)
                if selected_rows.empty:
                    continue
                row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=float(args.lambda_cost))
                row["matrix_name"] = matrix_name
                if STRONG_MODEL_ID in set(outputs["model_id"].astype(str)):
                    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
                eval_rows.append(row)
                query_rows.append(audit_query_rows(outputs, selected_rows, cost_oracle, method=method, matrix_name=matrix_name))

    details = pd.concat(query_rows, ignore_index=True) if query_rows else pd.DataFrame()
    eval_table = pd.DataFrame(eval_rows)
    summary = summarize_errors(details)
    recall = model_recall(details)
    benchmark_gap = benchmark_gaps(details)
    top_misses = top_miss_examples(details)

    eval_table.to_csv(args.output_dir / "table_error_mode_policy_eval.csv", index=False)
    details.to_csv(args.output_dir / "table_error_mode_query_details.csv", index=False)
    summary.to_csv(args.output_dir / "table_error_mode_summary.csv", index=False)
    recall.to_csv(args.output_dir / "table_error_mode_oracle_recall.csv", index=False)
    benchmark_gap.to_csv(args.output_dir / "table_error_mode_benchmark_gap.csv", index=False)
    top_misses.to_csv(args.output_dir / "table_error_mode_top_misses.csv", index=False)
    write_memo(args.output_dir, eval_table, summary, recall, benchmark_gap, top_misses)
    print(f"Wrote probe error-mode audit to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_precomputed_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce").fillna(0.0)
    for column in ["cost_total_usd", "latency_s", "normalized_remote_cost"]:
        if column not in outputs:
            outputs[column] = 0.0
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    if "tool_available" not in outputs:
        outputs["tool_available"] = False
    return outputs


def build_selections(package, strong_gate, outputs: pd.DataFrame) -> dict[str, dict[str, pd.Series]]:
    selections: dict[str, dict[str, pd.Series]] = {
        "observable_local_state_v5": {},
        "tool_probe_profile_v4": {},
        "benchmark_train_utility_lookup": {},
    }
    for split in ["val", "test"]:
        selections["observable_local_state_v5"][split] = normalize(package.observable_local_state_selection(outputs, split=split))
        selections["tool_probe_profile_v4"][split] = normalize(package.profile_v4_selection_for_split(outputs, split=split))
        selections["benchmark_train_utility_lookup"][split] = benchmark_train_lookup(outputs, split=split)

    if STRONG_MODEL_ID in set(outputs["model_id"].astype(str)):
        for base_name in ["observable_local_state_v5", "tool_probe_profile_v4"]:
            method = f"{base_name}_oracle_between_base_and_strong"
            selections[method] = {}
            for split in ["val", "test"]:
                selections[method][split] = normalize(strong_gate.oracle_between_base_and_strong(outputs, selections[base_name][split]))
    return selections


def normalize(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def benchmark_train_lookup(outputs: pd.DataFrame, *, split: str) -> pd.Series:
    train = outputs[outputs["split"].eq("train") & outputs["model_id"].ne("deterministic_math_tool")].copy()
    table = (
        train.groupby(["benchmark", "model_id"], as_index=False)
        .agg(mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"), mean_cost=("normalized_remote_cost", "mean"))
        .sort_values(["benchmark", "mean_utility", "mean_quality", "mean_cost"], ascending=[True, False, False, True])
        .drop_duplicates("benchmark")
        .set_index("benchmark")["model_id"]
    )
    target = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    return pd.Series({str(query_id): str(table.get(row["benchmark"], "qwen3-14b-awq-local")) for query_id, row in target.iterrows()})


def audit_query_rows(
    outputs: pd.DataFrame,
    selected_rows: pd.DataFrame,
    cost_oracle: pd.DataFrame,
    *,
    method: str,
    matrix_name: str,
) -> pd.DataFrame:
    oracle_cols = [
        "query_id",
        "model_id",
        "quality_score",
        "utility",
        "normalized_remote_cost",
        "cost_total_usd",
        "is_frontier",
        "is_local",
        "parsed_answer",
    ]
    oracle = cost_oracle[oracle_cols].rename(
        columns={
            "model_id": "oracle_model",
            "quality_score": "oracle_quality",
            "utility": "oracle_utility",
            "normalized_remote_cost": "oracle_norm_cost",
            "cost_total_usd": "oracle_cost_usd",
            "is_frontier": "oracle_is_frontier",
            "is_local": "oracle_is_local",
            "parsed_answer": "oracle_answer",
        }
    )
    selected_cols = [
        "query_id",
        "query_text",
        "benchmark",
        "domain",
        "metric",
        "split",
        "model_id",
        "quality_score",
        "utility",
        "normalized_remote_cost",
        "cost_total_usd",
        "is_frontier",
        "is_local",
        "parsed_answer",
    ]
    details = selected_rows[selected_cols].rename(
        columns={
            "model_id": "selected_model",
            "quality_score": "selected_quality",
            "utility": "selected_utility",
            "normalized_remote_cost": "selected_norm_cost",
            "cost_total_usd": "selected_cost_usd",
            "is_frontier": "selected_is_frontier",
            "is_local": "selected_is_local",
            "parsed_answer": "selected_answer",
        }
    )
    details = details.merge(oracle, on="query_id", how="left")
    details["matrix_name"] = matrix_name
    details["method"] = method
    details["utility_gap"] = details["oracle_utility"] - details["selected_utility"]
    details["quality_gap"] = details["oracle_quality"] - details["selected_quality"]
    details["error_mode"] = [error_mode(row) for _, row in details.iterrows()]
    details["query_snippet"] = details["query_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 220)
    return details


def error_mode(row: pd.Series) -> str:
    selected = str(row["selected_model"])
    oracle = str(row["oracle_model"])
    if selected == oracle:
        return "hit_same_model"
    if oracle == STRONG_MODEL_ID and selected != STRONG_MODEL_ID:
        return "missed_gemini_strong"
    if selected == STRONG_MODEL_ID and oracle != STRONG_MODEL_ID:
        return "unneeded_gemini_strong"
    selected_frontier = bool(row.get("selected_is_frontier", False))
    oracle_frontier = bool(row.get("oracle_is_frontier", False))
    if selected_frontier and not oracle_frontier:
        return "unneeded_frontier_or_api"
    if not selected_frontier and oracle_frontier:
        return "missed_frontier_or_api"
    selected_local = bool(row.get("selected_is_local", False))
    oracle_local = bool(row.get("oracle_is_local", False))
    if selected_local and oracle_local:
        return "wrong_local_winner"
    return "other_model_mismatch"


def summarize_errors(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    return (
        details.groupby(["matrix_name", "method", "split", "error_mode"], as_index=False)
        .agg(
            n=("query_id", "nunique"),
            mean_utility_gap=("utility_gap", "mean"),
            total_utility_gap=("utility_gap", "sum"),
            mean_quality_gap=("quality_gap", "mean"),
            frontier_rate=("selected_is_frontier", "mean"),
        )
        .sort_values(["matrix_name", "split", "method", "total_utility_gap"], ascending=[True, True, True, False])
    )


def model_recall(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    grouped = details.groupby(["matrix_name", "method", "split", "benchmark", "oracle_model"], as_index=False).agg(
        oracle_n=("query_id", "nunique"),
        selected_hits=("selected_model", lambda values: int(0)),
    )
    hit_counts = (
        details[details["selected_model"].astype(str).eq(details["oracle_model"].astype(str))]
        .groupby(["matrix_name", "method", "split", "benchmark", "oracle_model"], as_index=False)
        .agg(selected_hits=("query_id", "nunique"))
    )
    grouped = grouped.drop(columns=["selected_hits"]).merge(
        hit_counts, on=["matrix_name", "method", "split", "benchmark", "oracle_model"], how="left"
    )
    grouped["selected_hits"] = grouped["selected_hits"].fillna(0).astype(int)
    grouped["recall"] = grouped["selected_hits"] / grouped["oracle_n"].clip(lower=1)
    return grouped.sort_values(["matrix_name", "split", "method", "oracle_n", "recall"], ascending=[True, True, True, False, True])


def benchmark_gaps(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    return (
        details.groupby(["matrix_name", "method", "split", "benchmark"], as_index=False)
        .agg(
            n=("query_id", "nunique"),
            mean_selected_quality=("selected_quality", "mean"),
            mean_oracle_quality=("oracle_quality", "mean"),
            mean_quality_gap=("quality_gap", "mean"),
            mean_selected_utility=("selected_utility", "mean"),
            mean_oracle_utility=("oracle_utility", "mean"),
            mean_utility_gap=("utility_gap", "mean"),
            total_utility_gap=("utility_gap", "sum"),
            frontier_rate=("selected_is_frontier", "mean"),
        )
        .sort_values(["matrix_name", "split", "method", "total_utility_gap"], ascending=[True, True, True, False])
    )


def top_miss_examples(details: pd.DataFrame, n: int = 80) -> pd.DataFrame:
    if details.empty:
        return pd.DataFrame()
    cols = [
        "matrix_name",
        "method",
        "split",
        "benchmark",
        "query_id",
        "error_mode",
        "selected_model",
        "oracle_model",
        "selected_quality",
        "oracle_quality",
        "quality_gap",
        "selected_utility",
        "oracle_utility",
        "utility_gap",
        "selected_answer",
        "oracle_answer",
        "query_snippet",
    ]
    misses = details[details["utility_gap"].gt(1e-12)].copy()
    return misses.sort_values(["utility_gap", "quality_gap"], ascending=False)[cols].head(int(n))


def write_memo(
    output_dir: Path,
    eval_table: pd.DataFrame,
    summary: pd.DataFrame,
    recall: pd.DataFrame,
    benchmark_gap: pd.DataFrame,
    top_misses: pd.DataFrame,
) -> None:
    lines = [
        "# Broad100 Probe Error-Mode Audit",
        "",
        "This is a diagnostic audit for the probe-signal experiment queue. It uses cached outputs only and makes no provider calls.",
        "",
        "## Policy Eval",
        "",
        "```csv",
        compact_csv(eval_table, ["matrix_name", "method", "split", "mean_quality", "mean_utility", "cost_oracle_mean_utility", "oracle_utility_ratio", "frontier_call_rate", "strong_call_rate"]),
        "```",
        "",
        "## Largest Error Modes",
        "",
        "```csv",
        compact_csv(summary.sort_values("total_utility_gap", ascending=False).head(30)),
        "```",
        "",
        "## Largest Benchmark Gaps",
        "",
        "```csv",
        compact_csv(benchmark_gap.sort_values("total_utility_gap", ascending=False).head(30)),
        "```",
        "",
        "## Lowest Oracle-Model Recall Rows",
        "",
        "```csv",
        compact_csv(recall[(recall["oracle_n"] >= 3)].sort_values(["recall", "oracle_n"], ascending=[True, False]).head(30)),
        "```",
        "",
        "## Top Miss Examples",
        "",
        "```csv",
        compact_csv(top_misses.head(20)),
        "```",
        "",
        "## Interpretation",
        "",
        "- Use this table to choose the next probe: the largest positive utility-gap rows are the routing misses that matter.",
        "- `wrong_local_winner` means the next probe should improve local-model selection rather than frontier escalation.",
        "- `missed_gemini_strong` means the next probe should estimate value of test-time compute or strong solving.",
        "- `unneeded_frontier_or_api` means cost-aware routing is over-escalating.",
    ]
    (output_dir / "PROBE_ERROR_MODE_AUDIT_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def compact_csv(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    if frame.empty:
        return ""
    if columns is not None:
        frame = frame[[column for column in columns if column in frame.columns]].copy()
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        elif column.endswith("_json"):
            out[column] = out[column].map(lambda value: json.dumps(value) if isinstance(value, dict) else str(value))
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
