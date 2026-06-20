from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SELECTED_REPAIR_RULE = "val_frontier_cap_best_utility_test"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the cached Broad100 no-tool action-pool feasibility bound. "
            "This makes no provider, vLLM, or local generation calls."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--repair-selected",
        type=Path,
        default=Path(
            "results/controlled/broad100_no_tool_verifiability_repair/"
            "table_no_tool_verifiability_repair_selected.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_no_tool_feasibility_bound"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp213 = load_module("experiments/213_broad100_target_method_package.py", "exp213_for_217")
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    target_table = pd.read_csv(args.target_table)
    full_target = exp213.rebuild_target_pool(
        target_table,
        outputs,
        exp213.FULL_LOCAL_ACTIONS,
        exp213.LARGE_ACTIONS,
        float(args.lambda_cost),
    )
    no_tool_target = exp213.rebuild_target_pool(
        target_table,
        outputs,
        exp213.NO_TOOL_LOCAL_ACTIONS,
        exp213.LARGE_ACTIONS,
        float(args.lambda_cost),
    )

    bound_rows, bound_details = build_bound_rows(exp213, full_target, no_tool_target, args)
    normalized = build_repair_normalized_table(bound_rows, args.repair_selected)

    bound_rows.to_csv(args.output_dir / "table_no_tool_feasibility_bound.csv", index=False)
    bound_details.to_csv(args.output_dir / "table_no_tool_feasibility_bound_query_choices.csv", index=False)
    normalized.to_csv(args.output_dir / "table_no_tool_repair_oracle_normalized.csv", index=False)
    write_figure(args.output_dir, bound_rows, normalized)
    write_memo(args.output_dir / "NO_TOOL_FEASIBILITY_BOUND_MEMO.md", args, bound_rows, normalized)
    print(f"Wrote no-tool feasibility bound to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_bound_rows(
    exp213: Any,
    full_target: pd.DataFrame,
    no_tool_target: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        full_frame = full_target[full_target["split"].eq(split)].copy()
        no_tool_frame = no_tool_target[no_tool_target["split"].eq(split)].copy()

        full_choose = full_frame["large_utility"].to_numpy(dtype=float) >= full_frame["local_utility"].to_numpy(dtype=float)
        row, detail = exp213.evaluate_policy(
            full_frame,
            full_choose,
            oracle_reference=full_frame,
            split=split,
            method="full_action_pool_oracle",
            family="feasibility_bound",
            action_pool_variant="full_action_pool",
            lambda_cost=float(args.lambda_cost),
        )
        row.update({"bound_role": "full_oracle", "reference_oracle": "full_action_pool_oracle"})
        rows.append(row)
        details.append(detail)

        no_tool_choose = no_tool_frame["large_utility"].to_numpy(dtype=float) >= no_tool_frame["local_utility"].to_numpy(dtype=float)
        row, detail = exp213.evaluate_policy(
            no_tool_frame,
            no_tool_choose,
            oracle_reference=full_frame,
            split=split,
            method="no_tool_action_pool_oracle_vs_full_oracle",
            family="feasibility_bound",
            action_pool_variant="no_tool_action_pool",
            lambda_cost=float(args.lambda_cost),
        )
        row.update({"bound_role": "no_tool_oracle_vs_full", "reference_oracle": "full_action_pool_oracle"})
        rows.append(row)
        details.append(detail)

        row, _detail = exp213.evaluate_policy(
            no_tool_frame,
            no_tool_choose,
            oracle_reference=no_tool_frame,
            split=split,
            method="no_tool_action_pool_oracle",
            family="feasibility_bound",
            action_pool_variant="no_tool_action_pool",
            lambda_cost=float(args.lambda_cost),
        )
        row.update({"bound_role": "no_tool_oracle_self_reference", "reference_oracle": "no_tool_action_pool_oracle"})
        rows.append(row)

    table = exp213.add_target_gates(pd.DataFrame(rows))
    return table, pd.concat(details, ignore_index=True)


def build_repair_normalized_table(bound_rows: pd.DataFrame, repair_path: Path) -> pd.DataFrame:
    repair = pd.read_csv(repair_path)
    selected = repair[
        repair["split"].astype(str).eq("test")
        & repair["selection_rule"].astype(str).eq(SELECTED_REPAIR_RULE)
    ].copy()
    if selected.empty:
        selected = repair[
            repair["split"].astype(str).eq("test")
            & repair["selection_rule"].astype(str).eq("val_best_utility_test")
        ].copy()
    selected = selected.head(1)
    if selected.empty:
        return pd.DataFrame()

    test_bounds = bound_rows[bound_rows["split"].eq("test")].copy()
    full = test_bounds[test_bounds["bound_role"].eq("full_oracle")].iloc[0]
    no_tool = test_bounds[test_bounds["bound_role"].eq("no_tool_oracle_self_reference")].iloc[0]
    row = selected.iloc[0].to_dict()
    return pd.DataFrame(
        [
            normalized_row(row, full, "full_action_pool_oracle"),
            normalized_row(row, no_tool, "no_tool_action_pool_oracle"),
        ]
    )


def normalized_row(row: dict[str, Any], oracle: pd.Series, oracle_name: str) -> dict[str, Any]:
    mean_quality = float(row["mean_quality"])
    mean_utility = float(row["mean_utility"])
    oracle_quality = float(oracle["mean_quality"])
    oracle_utility = float(oracle["mean_utility"])
    return {
        "method": str(row["method"]),
        "selection_rule": str(row["selection_rule"]),
        "reference_oracle": oracle_name,
        "split": "test",
        "n_queries": int(row["n_queries"]),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "oracle_mean_quality": oracle_quality,
        "oracle_mean_utility": oracle_utility,
        "quality_gap_to_reference_oracle": oracle_quality - mean_quality,
        "utility_gap_to_reference_oracle": oracle_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / max(oracle_utility, 1e-12),
        "frontier_call_rate": float(row["frontier_call_rate"]),
        "large_call_rate": float(row["large_call_rate"]),
        "meets_3pt_quality_to_reference": (oracle_quality - mean_quality) <= 0.03,
        "meets_95pct_utility_to_reference": mean_utility >= 0.95 * oracle_utility,
        "meets_97pct_utility_to_reference": mean_utility >= 0.97 * oracle_utility,
        "meets_frontier_cap_0p40": float(row["frontier_call_rate"]) <= 0.40,
    }


def write_figure(out_dir: Path, bounds: pd.DataFrame, normalized: pd.DataFrame) -> None:
    test_bounds = bounds[bounds["split"].eq("test")].copy()
    plot_rows = []
    for _, row in test_bounds.iterrows():
        if row["bound_role"] in {"full_oracle", "no_tool_oracle_vs_full"}:
            plot_rows.append(
                {
                    "label": str(row["bound_role"]),
                    "utility": float(row["mean_utility"]),
                    "quality": float(row["mean_quality"]),
                    "kind": "oracle_bound",
                }
            )
    for _, row in normalized.iterrows():
        if str(row["reference_oracle"]) == "full_action_pool_oracle":
            plot_rows.append(
                {
                    "label": "selected_no_tool_repair",
                    "utility": float(row["mean_utility"]),
                    "quality": float(row["mean_quality"]),
                    "kind": "selected_method",
                }
            )
    plot = pd.DataFrame(plot_rows).sort_values("utility", ascending=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#2f4b7c" if kind == "oracle_bound" else "#9d6b53" for kind in plot["kind"]]
    ax.barh(plot["label"], plot["utility"], color=colors)
    ax.set_xlabel("Held-out Broad100 test mean utility")
    ax.set_title("No-Tool Action-Pool Feasibility Bound")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_no_tool_feasibility_bound.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, bounds: pd.DataFrame, normalized: pd.DataFrame) -> None:
    test_bounds = bounds[bounds["split"].eq("test")].copy()
    full = test_bounds[test_bounds["bound_role"].eq("full_oracle")].iloc[0]
    no_tool_vs_full = test_bounds[test_bounds["bound_role"].eq("no_tool_oracle_vs_full")].iloc[0]
    no_tool_self = test_bounds[test_bounds["bound_role"].eq("no_tool_oracle_self_reference")].iloc[0]
    repair_full = normalized[normalized["reference_oracle"].eq("full_action_pool_oracle")].iloc[0]
    repair_no_tool = normalized[normalized["reference_oracle"].eq("no_tool_action_pool_oracle")].iloc[0]

    lines = [
        "# Broad100 No-Tool Feasibility Bound",
        "",
        "This cached diagnostic asks whether a clean no-tool Broad100 method can reach the full-action-pool oracle target at all.",
        "It makes no provider calls, no vLLM calls, and no local generation calls.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/217_broad100_no_tool_feasibility_bound.py",
        "```",
        "",
        "## Main Test Result",
        "",
        f"- Full action-pool oracle: quality `{float(full['mean_quality']):.4f}`, utility `{float(full['mean_utility']):.4f}`, frontier rate `{float(full['frontier_call_rate']):.4f}`.",
        f"- No-tool action-pool oracle against the same full oracle: quality `{float(no_tool_vs_full['mean_quality']):.4f}`, utility `{float(no_tool_vs_full['mean_utility']):.4f}`, full-oracle utility ratio `{float(no_tool_vs_full['oracle_utility_ratio']):.4f}`, quality gap `{float(no_tool_vs_full['quality_gap_to_full_oracle']):.4f}`.",
        f"- Therefore the no-tool action pool itself misses the Phase 3 full-oracle target: `meets_primary_numeric_target={bool(no_tool_vs_full['meets_primary_numeric_target'])}`.",
        "",
        "## Current No-Tool Repair In Context",
        "",
        f"- Selected no-tool repair versus full oracle: quality gap `{float(repair_full['quality_gap_to_reference_oracle']):.4f}`, utility ratio `{float(repair_full['oracle_utility_ratio']):.4f}`.",
        f"- The same selected repair versus the no-tool oracle: quality gap `{float(repair_no_tool['quality_gap_to_reference_oracle']):.4f}`, utility ratio `{float(repair_no_tool['oracle_utility_ratio']):.4f}`.",
        f"- No-tool oracle self-reference utility: `{float(no_tool_self['mean_utility']):.4f}`.",
        "",
        "## Interpretation",
        "",
        "- A clean no-tool policy cannot meet the current Broad100 full-oracle target unless the action pool is improved.",
        "- The selected no-tool repair is much closer when normalized to the no-tool oracle, so the remaining full-oracle gap is substantially an action-pool gap, not just a router-observability gap.",
        "- For the current Phase 3 target, verifiable local/tool actions are not an incidental shortcut; they are required to make the full-action-pool oracle-level target feasible on cached Broad100.",
        "",
        "## Bound Rows",
        "",
        "```csv",
        compact_csv(
            test_bounds[
                [
                    "method",
                    "bound_role",
                    "mean_quality",
                    "mean_utility",
                    "quality_gap_to_full_oracle",
                    "oracle_utility_ratio",
                    "frontier_call_rate",
                    "meets_primary_numeric_target",
                ]
            ]
        ),
        "```",
        "",
        "## Repair Normalized Rows",
        "",
        "```csv",
        compact_csv(normalized),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return frame.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
