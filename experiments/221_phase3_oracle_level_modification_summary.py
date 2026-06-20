from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


RESIDUAL_TEST_IDS = {
    "aime:hybrid:53",
    "aime:hybrid:55",
    "aime:hybrid:56",
    "aime:hybrid:8",
    "livemathbench:test:83",
    "livemathbench:test:9",
    "math500:test:81",
    "math500:test:97",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize which Phase 3 method modification reaches the configured "
            "oracle-level target. This is a no-call packaging script."
        )
    )
    parser.add_argument(
        "--broad-current",
        type=Path,
        default=Path("results/controlled/broad100_current_best_method_package/table_broad100_current_best_main_eval.csv"),
    )
    parser.add_argument(
        "--no-tool-bound",
        type=Path,
        default=Path("results/controlled/broad100_no_tool_feasibility_bound/table_no_tool_feasibility_bound.csv"),
    )
    parser.add_argument(
        "--gpt-residual-bounds",
        type=Path,
        default=Path("results/controlled/broad100_gpt_strong_residual2048/table_gpt_strong_math_action_bounds.csv"),
    )
    parser.add_argument(
        "--gpt-residual-augmented",
        type=Path,
        default=Path("results/controlled/broad100_gpt_strong_residual2048/model_outputs_with_gpt_strong_math_action.parquet"),
    )
    parser.add_argument(
        "--gpt-residual-512",
        type=Path,
        default=Path("results/controlled/broad100_gpt_strong_residual512_loss/table_gpt_strong_math_action_outputs.csv"),
    )
    parser.add_argument(
        "--exact-main",
        type=Path,
        default=Path("results/controlled/table_phase3_exact_math_main_eval.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/phase3_oracle_level_modification"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(broad_current_rows(pd.read_csv(args.broad_current)))
    rows.extend(no_tool_rows(pd.read_csv(args.no_tool_bound)))
    if args.gpt_residual_bounds.exists():
        rows.extend(gpt_residual_bound_rows(pd.read_csv(args.gpt_residual_bounds)))
    if args.gpt_residual_augmented.exists():
        rows.append(forced_gpt_quality_repair_row(pd.read_parquet(args.gpt_residual_augmented), float(args.lambda_cost)))
    if args.gpt_residual_512.exists():
        rows.append(gpt_512_row(pd.read_csv(args.gpt_residual_512)))
    rows.extend(exact_rows(pd.read_csv(args.exact_main)))

    table = pd.DataFrame(rows)
    for column in ["quality_gap_to_full_oracle", "oracle_utility_ratio", "frontier_call_rate"]:
        table[column] = pd.to_numeric(table[column], errors="coerce")
    table["quality_within_3pt"] = table["quality_gap_to_full_oracle"] <= 0.03
    table["utility_at_least_95pct"] = table["oracle_utility_ratio"] >= 0.95
    table["utility_at_least_97pct"] = table["oracle_utility_ratio"] >= 0.97
    table["frontier_rate_le_0p40"] = table["frontier_call_rate"] <= 0.40
    table["configured_oracle_level_gate"] = (
        table["quality_within_3pt"] & table["utility_at_least_95pct"] & table["frontier_rate_le_0p40"]
    )
    table.to_csv(args.output_dir / "table_oracle_level_modification.csv", index=False)
    write_memo(args.output_dir / "ORACLE_LEVEL_METHOD_MODIFICATION_MEMO.md", table)
    print(f"Wrote oracle-level modification summary to {args.output_dir}")


def broad_current_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for role, label, note in [
        (
            "current_best_validation_selected",
            "verifiability_action_pool_current_best",
            "Validation-selected Broad100 method with learned verifiability plus verifiable local/tool actions.",
        ),
        (
            "routecode_state_policy",
            "compact_routecode_state_policy",
            "Compact RouteCode-style state-action policy; slightly lower utility than current best.",
        ),
    ]:
        row = one(frame, package_role=role)
        rows.append(
            {
                "modification": label,
                "scope": "Broad100 held-out test",
                "mean_quality": float(row["mean_quality"]),
                "mean_utility": float(row["mean_utility"]),
                "quality_gap_to_full_oracle": float(row["quality_gap_to_full_oracle"]),
                "oracle_utility_ratio": float(row["oracle_utility_ratio"]),
                "frontier_call_rate": float(row["frontier_call_rate"]),
                "source": "results/controlled/broad100_current_best_method_package/table_broad100_current_best_main_eval.csv",
                "interpretation": note,
            }
        )
    return rows


def no_tool_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    row = one(frame, split="test", bound_role="no_tool_oracle_vs_full")
    return [
        {
            "modification": "clean_no_tool_oracle_upper_bound",
            "scope": "Broad100 held-out test",
            "mean_quality": float(row["mean_quality"]),
            "mean_utility": float(row["mean_utility"]),
            "quality_gap_to_full_oracle": float(row["quality_gap_to_full_oracle"]),
            "oracle_utility_ratio": float(row["oracle_utility_ratio"]),
            "frontier_call_rate": float(row["frontier_call_rate"]),
            "source": "results/controlled/broad100_no_tool_feasibility_bound/table_no_tool_feasibility_bound.csv",
            "interpretation": "Negative feasibility bound: even perfect routing over the no-tool action pool misses the full oracle.",
        }
    ]


def gpt_residual_bound_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for method, label, note in [
        (
            "no_tool_gpt_strong_oracle_vs_full",
            "no_tool_plus_gpt_strong_cost_aware_oracle",
            "Adds GPT-5.5 strong-solve as a model action, but cost-aware oracle rarely selects it.",
        ),
        (
            "extratrees_d3_leaf8_thr0.0397_gpt_strong_math_else_local",
            "validation_threshold_gpt_strong_policy",
            "Validation-selected threshold policy using the GPT-strong action; misses the target.",
        ),
    ]:
        subset = frame[frame["split"].astype(str).eq("test") & frame["method"].astype(str).eq(method)]
        if subset.empty:
            continue
        row = subset.iloc[0]
        rows.append(
            {
                "modification": label,
                "scope": "Broad100 held-out test",
                "mean_quality": float(row["mean_quality"]),
                "mean_utility": float(row["mean_utility"]),
                "quality_gap_to_full_oracle": float(row["quality_gap_to_full_oracle"]),
                "oracle_utility_ratio": float(row["oracle_utility_ratio"]),
                "frontier_call_rate": float(row["frontier_call_rate"]),
                "source": "results/controlled/broad100_gpt_strong_residual2048/table_gpt_strong_math_action_bounds.csv",
                "interpretation": note,
            }
        )
    return rows


def forced_gpt_quality_repair_row(frame: pd.DataFrame, lambda_cost: float) -> dict[str, Any]:
    test = frame[frame["split"].astype(str).eq("test")].copy()
    for column in ["quality_score", "normalized_remote_cost"]:
        test[column] = pd.to_numeric(test[column], errors="coerce").fillna(0.0)
    test["utility"] = test["quality_score"] - lambda_cost * test["normalized_remote_cost"]
    full = test.sort_values("utility").groupby("query_id").tail(1)

    forced_rows = []
    for query_id, group in test.groupby("query_id"):
        if query_id in RESIDUAL_TEST_IDS:
            strong = group[group["model_id"].astype(str).eq("gpt-5.5-strong-solve")]
            if not strong.empty:
                forced_rows.append(strong.iloc[0])
                continue
        fallback = group[~group["model_id"].astype(str).eq("deterministic_math_tool")].sort_values("utility").iloc[-1]
        forced_rows.append(fallback)
    forced = pd.DataFrame(forced_rows)
    frontier_rate = (
        forced["is_frontier"].astype(bool).mean()
        if "is_frontier" in forced.columns
        else forced["model_id"].astype(str).str.contains("gpt|gemini", case=False, regex=True).mean()
    )
    return {
        "modification": "force_gpt_strong_on_8_quality_residuals",
        "scope": "Broad100 held-out test counterfactual",
        "mean_quality": float(forced["quality_score"].mean()),
        "mean_utility": float(forced["utility"].mean()),
        "quality_gap_to_full_oracle": float(full["quality_score"].mean() - forced["quality_score"].mean()),
        "oracle_utility_ratio": float(forced["utility"].mean() / max(full["utility"].mean(), 1e-12)),
        "frontier_call_rate": float(frontier_rate),
        "source": "results/controlled/broad100_gpt_strong_residual2048/model_outputs_with_gpt_strong_math_action.parquet",
        "interpretation": "Quality reaches the full oracle, but utility fails because GPT-strong residual calls are too expensive.",
    }


def gpt_512_row(frame: pd.DataFrame) -> dict[str, Any]:
    successes = pd.to_numeric(frame["quality_score"], errors="coerce").fillna(0.0).sum()
    total = len(frame)
    return {
        "modification": "gpt_strong_512_token_cap",
        "scope": "8 Broad100 residual rows",
        "mean_quality": float(successes / max(total, 1)),
        "mean_utility": "",
        "quality_gap_to_full_oracle": 1.0 - float(successes / max(total, 1)),
        "oracle_utility_ratio": 0.0,
        "frontier_call_rate": 1.0,
        "source": "results/controlled/broad100_gpt_strong_residual512_loss/table_gpt_strong_math_action_outputs.csv",
        "interpretation": "Cheaper max-token cap failed: all eight calls truncated/no parsed final answer.",
    }


def exact_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    oracle = one(frame, method="exact_math_cost_aware_oracle")
    rows = []
    for method, label, note in [
        (
            "exact_math_tool_augmented_min_cost",
            "exact_math_tool_augmented_min_cost",
            "Validation-selected exact-math method meets quality, utility, cost, and frontier-rate gates.",
        ),
        (
            "exact_math_tool_augmented_quality_conservative",
            "exact_math_quality_oracle_matching_policy",
            "Matches exact-math oracle quality but pays more cost, so utility is below 95%.",
        ),
    ]:
        row = one(frame, method=method)
        rows.append(
            {
                "modification": label,
                "scope": "66-row mixed exact-math held-out test",
                "mean_quality": float(row["quality_mean"]),
                "mean_utility": float(row["utility_cost_aware"]),
                "quality_gap_to_full_oracle": float(oracle["quality_mean"]) - float(row["quality_mean"]),
                "oracle_utility_ratio": float(row["utility_cost_aware"]) / max(float(oracle["utility_cost_aware"]), 1e-12),
                "frontier_call_rate": float(row["frontier_call_rate"]),
                "source": "results/controlled/table_phase3_exact_math_main_eval.csv",
                "interpretation": note,
            }
        )
    return rows


def one(frame: pd.DataFrame, **eq: str) -> pd.Series:
    subset = frame.copy()
    for column, value in eq.items():
        subset = subset[subset[column].astype(str).eq(str(value))]
    if subset.empty:
        raise RuntimeError(f"Missing row matching {eq}")
    return subset.iloc[0]


def write_memo(path: Path, table: pd.DataFrame) -> None:
    winners = table[table["configured_oracle_level_gate"]].copy()
    lines = [
        "# Oracle-Level Method Modification Summary",
        "",
        "This memo answers which tested modification can reach the configured Phase 3 oracle-level target.",
        "The target is not exact equality to the post-hoc oracle; it is the configured gate: within 3 quality points, at least 95% oracle utility, and at most 40% frontier calls.",
        "",
        "## Bottom Line",
        "",
        "- The clean no-tool method cannot reach the full-action-pool oracle; its own oracle misses the full target.",
        "- Forcing GPT-5.5 strong-solve on the eight residual quality-loss rows reaches oracle quality but fails utility because those calls are too expensive.",
        "- The method class that reaches the configured target is the learned-verifiability / verifiable-action-pool method.",
        "- The compact RouteCode state policy also reaches the 3-point and 95% utility gates, but the current-best residual flip has higher utility.",
        "",
        "## Passing Modifications",
        "",
        markdown_table(winners),
        "",
        "## All Checked Modifications",
        "",
        markdown_table(table),
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/216_broad100_current_best_method_package.py",
        "PYTHONPATH=src python experiments/217_broad100_no_tool_feasibility_bound.py",
        "PYTHONPATH=src python experiments/218_phase3_final_claim_package.py",
        "PYTHONPATH=src python experiments/221_phase3_oracle_level_modification_summary.py",
        "```",
        "",
        "## Interpretation",
        "",
        "The practical modification is not a better threshold over the same no-tool choices. The action pool must include cheap verifiable local actions, then RouteCode/ProbeCode can learn or observe when those actions are safe enough to avoid frontier calls.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame.copy()
    for column in ["mean_quality", "mean_utility", "quality_gap_to_full_oracle", "oracle_utility_ratio", "frontier_call_rate"]:
        if column in out.columns:
            out[column] = out[column].map(lambda value: "" if value == "" or pd.isna(value) else f"{float(value):.4f}")
    columns = list(out.columns)
    lines = [
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for _, row in out.iterrows():
        lines.append("|" + "|".join(str(row[column]).replace("\n", " ") for column in columns) + "|")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
