from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


BASE_METHOD = "base_learned_verifiability_global"
CURRENT_BEST_SELECTION_RULE = "val_base_tethered_residual_flip_test"
CURRENT_BEST_METHOD = "et_flip_leaf4_thr0.8502_capNone"
TEST_DIAGNOSTIC_SELECTION_RULE = "top_test_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the current cached Broad100 best method. This is a no-call "
            "artifact pass that combines the learned-verifiability package with "
            "the residual oracle-gap repair selected on validation."
        )
    )
    parser.add_argument(
        "--base-main-eval",
        type=Path,
        default=Path("results/controlled/broad100_target_method_package/table_broad100_target_method_main_eval.csv"),
    )
    parser.add_argument(
        "--base-ablation",
        type=Path,
        default=Path("results/controlled/broad100_target_method_package/table_broad100_target_method_ablation.csv"),
    )
    parser.add_argument(
        "--residual-selected",
        type=Path,
        default=Path(
            "results/controlled/broad100_residual_oracle_gap_repair/"
            "table_residual_oracle_gap_repair_selected.csv"
        ),
    )
    parser.add_argument(
        "--residual-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_residual_oracle_gap_repair/"
            "table_residual_oracle_gap_repair_query_choices.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_current_best_method_package"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(args.base_main_eval)
    ablation = pd.read_csv(args.base_ablation) if args.base_ablation.exists() else pd.DataFrame()
    residual = pd.read_csv(args.residual_selected)
    choices = pd.read_csv(args.residual_query_choices) if args.residual_query_choices.exists() else pd.DataFrame()

    main_eval = build_main_eval(base, residual)
    summary = build_summary(main_eval)
    action_mix = build_action_mix(main_eval)

    main_eval.to_csv(args.output_dir / "table_broad100_current_best_main_eval.csv", index=False)
    summary.to_csv(args.output_dir / "table_broad100_current_best_summary.csv", index=False)
    action_mix.to_csv(args.output_dir / "table_broad100_current_best_action_mix.csv", index=False)
    if not choices.empty:
        current_choices = choices[
            choices["method"].astype(str).eq(CURRENT_BEST_METHOD)
            & choices["split"].astype(str).eq("test")
            & choices["selection_rule"].astype(str).eq(CURRENT_BEST_SELECTION_RULE)
        ].copy()
        current_choices.to_csv(args.output_dir / "table_broad100_current_best_query_choices.csv", index=False)

    write_figure(args.output_dir, main_eval)
    write_memo(args.output_dir / "BROAD100_CURRENT_BEST_METHOD_PACKAGE.md", args, main_eval, summary, ablation)
    print(f"Wrote current Broad100 best-method package to {args.output_dir}")


def build_main_eval(base: pd.DataFrame, residual: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []

    for method, role in [
        ("oracle_local_vs_large_gate", "oracle_upper_bound"),
        ("always_best_local_action", "local_reference"),
        ("always_best_large_action", "large_reference"),
        ("extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75", "previous_base_package_method"),
        ("gb_depth2_thr0.9844_state_k8", "routecode_state_policy"),
    ]:
        subset = base[
            base["split"].astype(str).eq("test")
            & base["method"].astype(str).eq(method)
            & base["action_pool_variant"].astype(str).eq("full_action_pool")
        ].copy()
        if not subset.empty:
            row = subset.iloc[0].copy()
            row["package_role"] = role
            row["selection_basis"] = "source_package_validation_selection"
            row["claim_status"] = "valid_reference" if "reference" in role or role == "oracle_upper_bound" else "valid_selected"
            row["source_artifact"] = "broad100_target_method_package"
            rows.append(row)

    for rule, role, status in [
        ("base_reference_test", "residual_base_reference", "valid_reference"),
        (CURRENT_BEST_SELECTION_RULE, "current_best_validation_selected", "valid_selected_current_best"),
        ("val_primary_gate_best_utility_test", "aggressive_validation_selected", "valid_selected_but_weaker_on_test"),
        (TEST_DIAGNOSTIC_SELECTION_RULE, "test_only_diagnostic_headroom", "diagnostic_test_selected_not_deployable"),
    ]:
        subset = residual[
            residual["split"].astype(str).eq("test")
            & residual["selection_rule"].astype(str).eq(rule)
        ].copy()
        if subset.empty:
            continue
        row = subset.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0].copy()
        row["package_role"] = role
        row["selection_basis"] = rule
        row["claim_status"] = status
        row["source_artifact"] = "broad100_residual_oracle_gap_repair"
        rows.append(row)

    table = pd.DataFrame(rows).reset_index(drop=True)
    table["quality_within_3pt_oracle"] = table["quality_gap_to_full_oracle"].astype(float) <= 0.03
    table["utility_at_least_95pct_oracle"] = table["oracle_utility_ratio"].astype(float) >= 0.95
    table["utility_at_least_97pct_oracle"] = table["oracle_utility_ratio"].astype(float) >= 0.97
    table["frontier_rate_le_0p40"] = table["frontier_call_rate"].astype(float) <= 0.40
    table["primary_numeric_target"] = (
        table["quality_within_3pt_oracle"]
        & table["utility_at_least_95pct_oracle"]
        & table["frontier_rate_le_0p40"]
    )
    return table


def build_summary(main_eval: pd.DataFrame) -> pd.DataFrame:
    current = pick_role(main_eval, "current_best_validation_selected")
    base = pick_role(main_eval, "residual_base_reference")
    oracle = pick_role(main_eval, "oracle_upper_bound")
    diagnostic = pick_role(main_eval, "test_only_diagnostic_headroom", required=False)
    rows: list[dict[str, Any]] = [
        {
            "item": "current_best_method",
            "value": str(current["method"]),
            "evidence": str(current["selection_basis"]),
        },
        {
            "item": "test_quality",
            "value": f"{float(current['mean_quality']):.6f}",
            "evidence": "held-out Broad100 test split",
        },
        {
            "item": "oracle_quality",
            "value": f"{float(oracle['mean_quality']):.6f}",
            "evidence": "post-hoc local-vs-large oracle",
        },
        {
            "item": "quality_gap_to_oracle",
            "value": f"{float(current['quality_gap_to_full_oracle']):.6f}",
            "evidence": "oracle quality - current best quality",
        },
        {
            "item": "test_utility",
            "value": f"{float(current['mean_utility']):.6f}",
            "evidence": "quality - lambda * normalized cost",
        },
        {
            "item": "oracle_utility",
            "value": f"{float(oracle['mean_utility']):.6f}",
            "evidence": "post-hoc local-vs-large oracle",
        },
        {
            "item": "oracle_utility_ratio",
            "value": f"{float(current['oracle_utility_ratio']):.6f}",
            "evidence": "current best utility / oracle utility",
        },
        {
            "item": "frontier_call_rate",
            "value": f"{float(current['frontier_call_rate']):.6f}",
            "evidence": "held-out Broad100 test split",
        },
        {
            "item": "base_utility_delta",
            "value": f"{float(current['mean_utility']) - float(base['mean_utility']):.6f}",
            "evidence": "current best utility - base learned-verifiability utility",
        },
        {
            "item": "base_frontier_rate_delta",
            "value": f"{float(current['frontier_call_rate']) - float(base['frontier_call_rate']):.6f}",
            "evidence": "current best frontier rate - base frontier rate",
        },
        {
            "item": "valid_oracle_level_target",
            "value": str(bool(current["primary_numeric_target"] and current["utility_at_least_97pct_oracle"])),
            "evidence": "within 3 quality points and >=97% oracle utility with <=40% frontier calls",
        },
    ]
    if diagnostic is not None:
        rows.append(
            {
                "item": "test_only_diagnostic_utility_ratio",
                "value": f"{float(diagnostic['oracle_utility_ratio']):.6f}",
                "evidence": "selected on test only; not deployable evidence",
            }
        )
    return pd.DataFrame(rows)


def build_action_mix(main_eval: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in main_eval.to_dict("records"):
        raw = str(record.get("selected_actions_json", "{}"))
        try:
            actions = json.loads(raw)
        except json.JSONDecodeError:
            actions = {}
        for action, count in actions.items():
            rows.append(
                {
                    "method": record["method"],
                    "package_role": record["package_role"],
                    "selection_basis": record["selection_basis"],
                    "selected_action": action,
                    "n_queries": int(count),
                    "share": int(count) / max(int(record["n_queries"]), 1),
                }
            )
    return pd.DataFrame(rows).sort_values(["package_role", "n_queries"], ascending=[True, False])


def write_figure(out_dir: Path, main_eval: pd.DataFrame) -> None:
    plot = main_eval[
        main_eval["package_role"].isin(
            [
                "oracle_upper_bound",
                "residual_base_reference",
                "current_best_validation_selected",
                "routecode_state_policy",
                "test_only_diagnostic_headroom",
            ]
        )
    ].copy()
    plot["label"] = plot["package_role"] + "\n" + plot["method"]
    plot = plot.sort_values("mean_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    colors = []
    for role in plot["package_role"]:
        if role == "oracle_upper_bound":
            colors.append("#2f4b7c")
        elif role == "current_best_validation_selected":
            colors.append("#3f7f5f")
        elif role == "test_only_diagnostic_headroom":
            colors.append("#9d6b53")
        else:
            colors.append("#6f7f8f")
    ax.barh(plot["label"], plot["mean_utility"], color=colors)
    ax.set_xlabel("Held-out Broad100 test mean utility")
    ax.set_title("Current Broad100 Best Method vs Oracle and Base")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_broad100_current_best_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    main_eval: pd.DataFrame,
    summary: pd.DataFrame,
    ablation: pd.DataFrame,
) -> None:
    current = pick_role(main_eval, "current_best_validation_selected")
    base = pick_role(main_eval, "residual_base_reference")
    oracle = pick_role(main_eval, "oracle_upper_bound")
    diagnostic = pick_role(main_eval, "test_only_diagnostic_headroom", required=False)
    no_tool = pd.DataFrame()
    if not ablation.empty:
        no_tool = ablation[
            ablation["split"].astype(str).eq("test")
            & ablation["method"].astype(str).eq("extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75")
            & ablation["action_pool_variant"].astype(str).eq("no_tool_local_pool_ablation")
        ].copy()

    lines = [
        "# Broad100 Current Best Method Package",
        "",
        "This package names the current cached Broad100 best deployable method and separates it from oracle and test-selected diagnostics.",
        "It performs no provider calls, no vLLM calls, and no local generation calls. It only combines existing cached evaluation tables.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/216_broad100_current_best_method_package.py",
        "PYTHONPATH=src python experiments/122_phase3_goal_completion_audit.py",
        "```",
        "",
        "## Current Best",
        "",
        f"- Method: `{current['method']}`.",
        f"- Selection rule: `{current['selection_basis']}`.",
        f"- Held-out test quality: `{float(current['mean_quality']):.4f}` vs oracle `{float(oracle['mean_quality']):.4f}`.",
        f"- Quality gap to oracle: `{float(current['quality_gap_to_full_oracle']):.4f}`.",
        f"- Held-out test utility: `{float(current['mean_utility']):.4f}` vs oracle `{float(oracle['mean_utility']):.4f}`.",
        f"- Oracle utility ratio: `{float(current['oracle_utility_ratio']):.4f}`.",
        f"- Frontier-call rate: `{float(current['frontier_call_rate']):.4f}`.",
        f"- Primary numeric target met: `{bool(current['primary_numeric_target'])}`.",
        f"- >=97% oracle utility target met: `{bool(current['utility_at_least_97pct_oracle'])}`.",
        "",
        "## Delta From Previous Base",
        "",
        f"- Base learned-verifiability utility: `{float(base['mean_utility']):.4f}`.",
        f"- Current best utility delta: `{float(current['mean_utility']) - float(base['mean_utility']):.4f}`.",
        f"- Base frontier-call rate: `{float(base['frontier_call_rate']):.4f}`.",
        f"- Current best frontier-call-rate delta: `{float(current['frontier_call_rate']) - float(base['frontier_call_rate']):.4f}`.",
        "",
        "## Diagnostic Headroom",
        "",
    ]
    if diagnostic is not None:
        lines.extend(
            [
                f"- Best test-only diagnostic: `{diagnostic['method']}` with utility ratio `{float(diagnostic['oracle_utility_ratio']):.4f}` and frontier rate `{float(diagnostic['frontier_call_rate']):.4f}`.",
                "- This row is not valid deployable evidence because its threshold is selected on test.",
            ]
        )
    else:
        lines.append("- No test-only diagnostic row was found.")
    lines.extend(
        [
            "",
            "## No-Tool Caveat",
            "",
        ]
    )
    if not no_tool.empty:
        row = no_tool.iloc[0]
        lines.extend(
            [
                f"- Removing deterministic-tool local actions drops held-out quality to `{float(row['mean_quality']):.4f}` and oracle utility ratio to `{float(row['oracle_utility_ratio']):.4f}`.",
                "- Therefore the current Broad100 success should be described as a verifiability/action-pool bridge, not a clean no-tool benchmark-agnostic router.",
            ]
        )
    else:
        lines.append("- No no-tool ablation row was found in the source package.")
    lines.extend(
        [
            "",
            "## Summary Table",
            "",
            "```csv",
            compact_csv(summary),
            "```",
            "",
            "## Main Eval Rows",
            "",
            "```csv",
            compact_csv(
                main_eval[
                    [
                        "package_role",
                        "method",
                        "selection_basis",
                        "claim_status",
                        "mean_quality",
                        "mean_utility",
                        "quality_gap_to_full_oracle",
                        "oracle_utility_ratio",
                        "frontier_call_rate",
                        "large_call_rate",
                        "primary_numeric_target",
                    ]
                ].sort_values("mean_utility", ascending=False)
            ),
            "```",
            "",
            "## Artifacts",
            "",
            "- `table_broad100_current_best_main_eval.csv`",
            "- `table_broad100_current_best_summary.csv`",
            "- `table_broad100_current_best_action_mix.csv`",
            "- `table_broad100_current_best_query_choices.csv`",
            "- `fig_broad100_current_best_utility.pdf`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_role(frame: pd.DataFrame, role: str, *, required: bool = True) -> pd.Series | None:
    subset = frame[frame["package_role"].astype(str).eq(role)]
    if subset.empty:
        if required:
            raise RuntimeError(f"Missing package role {role}")
        return None
    return subset.iloc[0]


def compact_csv(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return frame.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
