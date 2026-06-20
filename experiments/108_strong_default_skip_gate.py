from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


LOCAL_MODELS = [
    "qwen3-0.6b-probe",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate strong-default skip gates using cached exact-math outcomes."
    )
    parser.add_argument(
        "--query-table",
        default="results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/strong_default_skip_gate")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--utility-ratio-target", type=float, default=0.95)
    return parser.parse_args()


def truthy(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def model_quality_cost(row: pd.Series, action: str) -> tuple[float, float, str]:
    if action == STRONG_GPT:
        return float(row.get("strong_quality", 0.0) or 0.0), float(row.get("strong_cost", 0.0) or 0.0), STRONG_GPT
    if action == GEMINI:
        return (
            float(row.get(f"{GEMINI}_quality", 0.0) or 0.0),
            float(row.get(f"{GEMINI}_cost", 0.0) or 0.0),
            GEMINI,
        )
    if action == BASE_GPT:
        return (
            float(row.get(f"{BASE_GPT}_quality", 0.0) or 0.0),
            float(row.get(f"{BASE_GPT}_cost", 0.0) or 0.0),
            BASE_GPT,
        )
    if action == "local_ensemble":
        return float(row.get("local_ensemble_quality", 0.0) or 0.0), 0.0, "local"
    if action in LOCAL_MODELS:
        return float(row.get(f"{action}_quality", 0.0) or 0.0), 0.0, "local"
    raise ValueError(f"Unknown action: {action}")


def observation_cost(row: pd.Series, observed: Iterable[str], paid_model: str) -> float:
    cost = 0.0
    for model_id in observed:
        if model_id == paid_model:
            continue
        if model_id == GEMINI:
            cost += float(row.get(f"{GEMINI}_cost", 0.0) or 0.0)
        elif model_id == BASE_GPT:
            cost += float(row.get(f"{BASE_GPT}_cost", 0.0) or 0.0)
        elif model_id == STRONG_GPT:
            cost += float(row.get("strong_cost", 0.0) or 0.0)
    return cost


def evaluate_policy(
    frame: pd.DataFrame,
    action_fn: Callable[[pd.Series], str],
    *,
    observed: Iterable[str],
    lambda_cost: float,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    actions: list[str] = []
    for _, row in frame.iterrows():
        action = action_fn(row)
        quality, cost, paid_model = model_quality_cost(row, action)
        qualities.append(quality)
        costs.append(cost + observation_cost(row, observed, paid_model))
        actions.append(action)

    quality_array = np.asarray(qualities, dtype=float)
    cost_array = np.asarray(costs, dtype=float)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    utility = quality_array - float(lambda_cost) * cost_array * len(frame) / strong_norm
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    return {
        "n_queries": int(len(frame)),
        "mean_quality": float(quality_array.mean()),
        "quality_gap_to_oracle": float(oracle_quality - quality_array.mean()),
        "normalized_cost_vs_all_strong": float(cost_array.sum() / strong_norm),
        "mean_utility": float(utility.mean()),
        "utility_ratio_to_oracle": float(utility.mean() / oracle_utility),
        "action_counts": json.dumps(dict(Counter(actions)), sort_keys=True),
    }


def agreement_conditions(table: pd.DataFrame) -> list[tuple[str, Callable[[pd.Series], bool]]]:
    conditions: list[tuple[str, Callable[[pd.Series], bool]]] = []
    for column in [
        "all_three_agree",
        "small_pair_agree",
        "qwen8_4b_agree",
        "qwen8_06b_agree",
        "agree__qwen3-8b-local__qwen3-14b-awq-local",
        "agree__qwen3-4b-local__qwen3-14b-awq-local",
        "agree__qwen3-0.6b-probe__qwen3-14b-awq-local",
    ]:
        if column in table.columns:
            conditions.append((column, lambda row, column=column: truthy(row.get(column))))
    for threshold in [2, 3, 4]:
        conditions.append(
            (
                f"local_votes_ge_{threshold}",
                lambda row, threshold=threshold: float(row.get("local_ensemble_votes", 0.0) or 0.0)
                >= threshold,
            )
        )
    return conditions


def gemini_conditions(table: pd.DataFrame) -> list[tuple[str, Callable[[pd.Series], bool]]]:
    conditions: list[tuple[str, Callable[[pd.Series], bool]]] = []
    for column in [
        "qwen3-0.6b-probe_gemini_agree",
        "qwen3-4b-local_gemini_agree",
        "qwen3-8b-local_gemini_agree",
        "qwen3-14b-awq-local_gemini_agree",
        "gemini_gpt_agree",
    ]:
        if column in table.columns:
            conditions.append((column, lambda row, column=column: truthy(row.get(column))))
    return conditions


def dataset_allowed(dataset: str, mode: str) -> bool:
    return (
        mode == "all"
        or (mode == "non_aime" and dataset != "aime")
        or (mode == "math500_only" and dataset == "math500")
        or (mode == "livemath_only" and dataset == "livemathbench")
    )


def build_rows(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    splits = {split: frame.copy() for split, frame in table.groupby("split", sort=False)}
    rows: list[dict[str, object]] = []

    def append_policy(
        *,
        policy: str,
        policy_family: str,
        observed: tuple[str, ...],
        action_fn: Callable[[pd.Series], str],
    ) -> None:
        row: dict[str, object] = {
            "policy": policy,
            "policy_family": policy_family,
            "observed_models": ",".join(observed),
        }
        for split in ["val", "test"]:
            metrics = evaluate_policy(splits[split], action_fn, observed=observed, lambda_cost=lambda_cost)
            row.update({f"{split}_{key}": value for key, value in metrics.items()})
        rows.append(row)

    for default_action in [STRONG_GPT, BASE_GPT]:
        for safe_action in ["local_ensemble", "qwen3-8b-local", "qwen3-14b-awq-local"]:
            for condition_name, condition in agreement_conditions(table):
                for dataset_mode in ["all", "non_aime", "math500_only", "livemath_only"]:

                    def action_fn(
                        row: pd.Series,
                        *,
                        default_action: str = default_action,
                        safe_action: str = safe_action,
                        condition: Callable[[pd.Series], bool] = condition,
                        dataset_mode: str = dataset_mode,
                    ) -> str:
                        if dataset_allowed(str(row["dataset"]), dataset_mode) and condition(row):
                            return safe_action
                        return default_action

                    append_policy(
                        policy=f"{safe_action}_if_{condition_name}_{dataset_mode}_else_{default_action}",
                        policy_family="local_skip",
                        observed=(),
                        action_fn=action_fn,
                    )

    for default_action in [STRONG_GPT, BASE_GPT]:
        for safe_action in [GEMINI, "local_ensemble", "qwen3-8b-local", "qwen3-14b-awq-local"]:
            for condition_name, condition in gemini_conditions(table):

                def action_fn(
                    row: pd.Series,
                    *,
                    default_action: str = default_action,
                    safe_action: str = safe_action,
                    condition: Callable[[pd.Series], bool] = condition,
                ) -> str:
                    if condition(row):
                        return safe_action
                    return default_action

                append_policy(
                    policy=f"{safe_action}_if_{condition_name}_else_{default_action}_with_gemini_obs",
                    policy_family="gemini_observed_skip",
                    observed=(GEMINI,),
                    action_fn=action_fn,
                )

    return pd.DataFrame(rows)


def select_rows(
    table: pd.DataFrame,
    *,
    quality_gap_target: float,
    cost_target: float,
    utility_ratio_target: float,
) -> pd.DataFrame:
    feasible = table[
        (table["val_quality_gap_to_oracle"] <= quality_gap_target)
        & (table["val_normalized_cost_vs_all_strong"] <= cost_target)
        & (table["val_utility_ratio_to_oracle"] >= utility_ratio_target)
    ].copy()
    if len(feasible):
        selected = feasible.sort_values(
            ["val_normalized_cost_vs_all_strong", "val_quality_gap_to_oracle"],
            ascending=[True, True],
        ).head(1)
        selected["selection_rule"] = "validation_feasible_min_cost"
    else:
        under_cost = table[table["val_normalized_cost_vs_all_strong"] <= cost_target].copy()
        if len(under_cost):
            selected = under_cost.sort_values(
                ["val_quality_gap_to_oracle", "val_utility_ratio_to_oracle"],
                ascending=[True, False],
            ).head(1)
            selected["selection_rule"] = "no_validation_feasible_best_gap_under_cost"
        else:
            selected = table.sort_values(
                ["val_utility_ratio_to_oracle", "val_quality_gap_to_oracle"],
                ascending=[False, True],
            ).head(1)
            selected["selection_rule"] = "no_policy_under_cost_best_validation_utility"

    diagnostic = table[table["test_normalized_cost_vs_all_strong"] <= cost_target].copy()
    if len(diagnostic):
        diagnostic = diagnostic.sort_values(
            ["test_quality_gap_to_oracle", "test_utility_ratio_to_oracle"],
            ascending=[True, False],
        ).head(1)
        diagnostic["selection_rule"] = "best_heldout_diagnostic_under_cost"
        selected = pd.concat([selected, diagnostic], ignore_index=True)
    return selected


def write_memo(
    output_dir: Path,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    quality_gap_target: float,
    cost_target: float,
    utility_ratio_target: float,
) -> None:
    feasible_val = table[
        (table["val_quality_gap_to_oracle"] <= quality_gap_target)
        & (table["val_normalized_cost_vs_all_strong"] <= cost_target)
        & (table["val_utility_ratio_to_oracle"] >= utility_ratio_target)
    ]
    feasible_test = table[
        (table["test_quality_gap_to_oracle"] <= quality_gap_target)
        & (table["test_normalized_cost_vs_all_strong"] <= cost_target)
        & (table["test_utility_ratio_to_oracle"] >= utility_ratio_target)
    ]
    cols = [
        "selection_rule",
        "policy",
        "val_mean_quality",
        "val_quality_gap_to_oracle",
        "val_normalized_cost_vs_all_strong",
        "val_utility_ratio_to_oracle",
        "test_mean_quality",
        "test_quality_gap_to_oracle",
        "test_normalized_cost_vs_all_strong",
        "test_utility_ratio_to_oracle",
        "test_action_counts",
    ]

    def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
        rows = []
        rows.append("| " + " | ".join(columns) + " |")
        rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for _, row in frame[columns].iterrows():
            values = []
            for column in columns:
                value = row[column]
                if isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value).replace("|", "\\|"))
            rows.append("| " + " | ".join(values) + " |")
        return "\n".join(rows)

    memo = [
        "# Strong-Default Skip-Gate Memo",
        "",
        "Purpose: test whether the Phase 3 target can be reached by defaulting to strong GPT and skipping it only when cached local or Gemini agreement signals make the row look safe.",
        "",
        f"Policies evaluated: `{len(table)}`.",
        f"Validation-feasible policies under gap <= `{quality_gap_target}`, cost <= `{cost_target}`, utility ratio >= `{utility_ratio_target}`: `{len(feasible_val)}`.",
        f"Held-out diagnostic feasible policies under the same gates: `{len(feasible_test)}`.",
        "",
        "Selected rows:",
        "",
        markdown_table(selected, cols),
        "",
        "Interpretation: this inverse-cascade family also fails. Strong GPT alone has enough quality, but the available agreement signals cannot skip enough strong calls while preserving the strong-inclusive oracle quality and utility targets.",
        "",
    ]
    (output_dir / "STRONG_DEFAULT_SKIP_GATE_MEMO.md").write_text("\n".join(memo), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    policies = build_rows(table, args.lambda_cost)
    selected = select_rows(
        policies,
        quality_gap_target=args.quality_gap_target,
        cost_target=args.cost_target,
        utility_ratio_target=args.utility_ratio_target,
    )
    policies.to_csv(output_dir / "table_strong_default_skip_gate.csv", index=False)
    selected.to_csv(output_dir / "table_strong_default_skip_gate_selected.csv", index=False)
    write_memo(
        output_dir,
        policies,
        selected,
        quality_gap_target=args.quality_gap_target,
        cost_target=args.cost_target,
        utility_ratio_target=args.utility_ratio_target,
    )
    print(f"Wrote {len(policies)} policies to {output_dir / 'table_strong_default_skip_gate.csv'}")
    print(f"Wrote selected rows to {output_dir / 'table_strong_default_skip_gate_selected.csv'}")


if __name__ == "__main__":
    main()
