from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble final Phase 3 Broad100 ablation table.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    table = build_ablation_table(config)
    table.to_csv(out_dir / "table_final_ablation.csv", index=False)
    write_figure(out_dir / "fig_ablation_utility.pdf", table)
    write_memo(out_dir / "ABLATION_MEMO.md", table)
    print(f"Wrote final ablation artifacts to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def build_ablation_table(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    main_eval = pd.read_csv(Path(config["outputs"]["root"]) / "main_eval/table_main_routing_eval.csv")
    target_ablation = pd.read_csv("results/controlled/broad100_target_method_package/table_broad100_target_method_ablation.csv")
    probe_selected = pd.read_csv("results/controlled/broad100_probe_state_routecode/table_probe_state_policy_selected.csv")
    decision_selected = pd.read_csv(
        "results/controlled/broad100_decision_aware_probe_state_routecode/table_decision_aware_probe_state_selected.csv"
    )
    onboarding = pd.read_csv(Path(config["outputs"]["root"]) / "new_model_onboarding/table_new_model_onboarding.csv")

    current = main_eval[main_eval["method_role"].eq("probecode_statecal")].iloc[0]
    rows.append(row_from_main(current, "full_method", "validation selected ProbeCode-StateCal current best", deployable=True))

    rows.append(
        row_from_target(
            target_ablation,
            method="extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75",
            variant="no_tool_local_pool_ablation",
            ablation="no_verifiable_tool_actions",
            notes="Removes deterministic/verifiable tool-style local actions from the selected learned-verifiability package.",
            deployable=True,
        )
    )
    rows.append(
        row_from_target(
            target_ablation,
            method="gb_depth2_thr0.9844_state_k8",
            variant="full_action_pool",
            ablation="compact_routecode_state_policy",
            notes="Uses compact learned RouteCode state policy without the residual flip that creates the final current-best method.",
            deployable=True,
        )
    )
    rows.append(
        row_from_probe_selected(
            probe_selected,
            method="benchmark_lookup_train",
            ablation="no_probe_local_behavior_features",
            notes="Uses train benchmark lookup only; this removes local probe-behavior features and is a diagnostic label baseline.",
            deployable=False,
        )
    )
    rows.append(
        row_from_probe_selected(
            probe_selected,
            method="probe_utility_ridge_alpha100",
            ablation="direct_probe_action_predictor_no_state",
            notes="Direct utility predictor over probe features, without an explicit RouteCode state table.",
            deployable=True,
        )
    )
    rows.append(
        row_from_decision_selected(
            decision_selected,
            method="et_actionprob_direct_depthnone_leaf4",
            ablation="direct_action_probability_no_state",
            notes="Decision-aware direct action predictor without discrete state abstraction.",
            deployable=True,
        )
    )
    rows.append(
        row_from_target(
            target_ablation,
            method="always_best_local_action",
            variant="full_action_pool",
            ablation="local_only_action_pool_diagnostic",
            notes="Per-query local-only upper bound within the cached local/verifiable action pool.",
            deployable=False,
        )
    )
    rows.append(
        row_from_target(
            target_ablation,
            method="always_best_large_action",
            variant="full_action_pool",
            ablation="large_only_action_pool_diagnostic",
            notes="Per-query large/frontier-style upper bound within the cached large action pool.",
            deployable=False,
        )
    )
    rows.extend(calibration_ablation_rows(onboarding))
    table = pd.DataFrame(rows)
    full_utility = float(table[table["ablation"].eq("full_method")]["mean_utility"].iloc[0])
    full_quality = float(table[table["ablation"].eq("full_method")]["mean_quality"].iloc[0])
    table["utility_delta_vs_full"] = table["mean_utility"].astype(float) - full_utility
    table["quality_delta_vs_full"] = table["mean_quality"].astype(float) - full_quality
    return table.sort_values("mean_utility", ascending=False).reset_index(drop=True)


def row_from_main(row: pd.Series, ablation: str, notes: str, *, deployable: bool) -> dict[str, Any]:
    return {
        "ablation": ablation,
        "source_method": row["method"],
        "source_family": row["method_role"],
        "split": "test",
        "n_queries": int(row["n_queries"]),
        "mean_quality": float(row["mean_quality"]),
        "mean_utility": float(row["mean_utility"]),
        "oracle_utility_ratio": float(row["oracle_utility_ratio"]),
        "frontier_call_rate": float(row["frontier_call_rate"]),
        "remote_cost_per_1k_queries": float(row["remote_cost_per_1k_queries"]),
        "deployable": bool(deployable),
        "notes": notes,
    }


def row_from_target(df: pd.DataFrame, *, method: str, variant: str, ablation: str, notes: str, deployable: bool) -> dict[str, Any]:
    subset = df[df["split"].astype(str).eq("test") & df["method"].astype(str).eq(method) & df["action_pool_variant"].astype(str).eq(variant)]
    if subset.empty:
        raise ValueError(f"Missing target ablation row: method={method} variant={variant}")
    row = subset.iloc[0]
    return {
        "ablation": ablation,
        "source_method": method,
        "source_family": row.get("family", ""),
        "split": "test",
        "n_queries": int(row["n_queries"]),
        "mean_quality": float(row["mean_quality"]),
        "mean_utility": float(row["mean_utility"]),
        "oracle_utility_ratio": float(row["oracle_utility_ratio"]),
        "frontier_call_rate": float(row["frontier_call_rate"]),
        "remote_cost_per_1k_queries": float(row["remote_cost_total_usd"]) / max(float(row["n_queries"]), 1.0) * 1000.0,
        "deployable": bool(deployable),
        "notes": notes,
    }


def row_from_probe_selected(df: pd.DataFrame, *, method: str, ablation: str, notes: str, deployable: bool) -> dict[str, Any]:
    subset = df[df["eval_split"].astype(str).eq("test") & df["method"].astype(str).eq(method)]
    if subset.empty:
        raise ValueError(f"Missing probe selected row: {method}")
    row = subset.iloc[0]
    return row_from_selected(row, ablation=ablation, notes=notes, deployable=deployable)


def row_from_decision_selected(df: pd.DataFrame, *, method: str, ablation: str, notes: str, deployable: bool) -> dict[str, Any]:
    subset = df[df["eval_split"].astype(str).eq("test") & df["method"].astype(str).eq(method)]
    if subset.empty:
        raise ValueError(f"Missing decision selected row: {method}")
    row = subset.iloc[0]
    return row_from_selected(row, ablation=ablation, notes=notes, deployable=deployable)


def row_from_selected(row: pd.Series, *, ablation: str, notes: str, deployable: bool) -> dict[str, Any]:
    return {
        "ablation": ablation,
        "source_method": row["method"],
        "source_family": row.get("family", ""),
        "split": "test",
        "n_queries": int(row["n_queries"]),
        "mean_quality": float(row["mean_quality"]),
        "mean_utility": float(row["mean_utility"]),
        "oracle_utility_ratio": float(row["oracle_utility_ratio"]),
        "frontier_call_rate": float(row["frontier_call_rate"]),
        "remote_cost_per_1k_queries": float("nan"),
        "deployable": bool(deployable),
        "notes": notes,
    }


def calibration_ablation_rows(onboarding: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    budget = 160
    source = onboarding[onboarding["budget"].eq(budget)].copy()
    for method, ablation, notes in [
        ("active_route_state_calibration", "active_state_calibration_budget160", "Active state calibration at 160 new-model evaluations."),
        ("uniform_route_state_calibration", "no_active_calibration_uniform_budget160", "Uniform state calibration at the same budget."),
        ("random_query_route_state_calibration", "random_query_calibration_budget160", "Random query calibration at the same budget."),
    ]:
        subset = source[source["method"].astype(str).eq(method)]
        if subset.empty:
            continue
        row = subset.groupby("method", as_index=False).agg(
            n_queries=("n_test_queries", "mean"),
            mean_quality=("mean_quality", "mean"),
            mean_utility=("mean_utility", "mean"),
            oracle_utility_ratio=("mean_utility", "mean"),
            frontier_call_rate=("frontier_call_rate", "mean"),
            remote_cost_per_1k_queries=("remote_cost_per_1k_queries", "mean"),
        ).iloc[0]
        rows.append(
            {
                "ablation": ablation,
                "source_method": method,
                "source_family": "new_model_calibration",
                "split": "test_simulated_onboarding",
                "n_queries": int(row["n_queries"]),
                "mean_quality": float(row["mean_quality"]),
                "mean_utility": float(row["mean_utility"]),
                "oracle_utility_ratio": float("nan"),
                "frontier_call_rate": float(row["frontier_call_rate"]),
                "remote_cost_per_1k_queries": float(row["remote_cost_per_1k_queries"]),
                "deployable": True,
                "notes": notes,
            }
        )
    return rows


def write_figure(path: Path, table: pd.DataFrame) -> None:
    plot = table[~table["split"].astype(str).str.contains("onboarding")].sort_values("mean_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5.8))
    colors = ["#426b69" if flag else "#888888" for flag in plot["deployable"]]
    ax.barh(plot["ablation"], plot["mean_utility"], color=colors)
    ax.set_xlabel("Mean utility")
    ax.set_title("Phase 3 Final Broad100 Ablations")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(path: Path, table: pd.DataFrame) -> None:
    lines = [
        "# Final Phase 3 Ablation",
        "",
        "This is a cache-backed ablation consolidation for the Broad100 final package. It makes no model calls.",
        "",
        "## Rows",
        "",
    ]
    for row in table.to_dict("records"):
        lines.append(
            f"- `{row['ablation']}`: utility `{float(row['mean_utility']):.4f}`, "
            f"quality `{float(row['mean_quality']):.4f}`, delta vs full `{float(row['utility_delta_vs_full']):.4f}`; "
            f"{row['notes']}"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Removing verifiable/tool-style actions causes a large utility drop in the cached Broad100 package.",
            "- Direct no-state predictors remain well below the final current-best method.",
            "- The onboarding rows show active calibration did not beat uniform/random in the cached Broad100 simulation at this budget.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

