from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOOL_MODEL_ID = "deterministic_math_tool"
FRONTIER_MODELS = {"gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve"}
FULL_LOCAL_ACTIONS = (
    "deterministic_math_tool",
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
)
NO_TOOL_LOCAL_ACTIONS = (
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
)
LARGE_ACTIONS = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
SELECTED_GLOBAL_METHOD = "extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75"
SELECTED_STATE_METHOD = "gb_depth2_thr0.9844_state_k8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Package the cached Broad100 target-level learned-verifiability method, "
            "including no-tool-pool ablations and exact target-gate evidence."
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
        "--learned-selected",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_policy_selected.csv"
        ),
    )
    parser.add_argument(
        "--learned-scores",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_scores.csv"
        ),
    )
    parser.add_argument(
        "--learned-assignments",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_assignments.csv"
        ),
    )
    parser.add_argument(
        "--target-status",
        type=Path,
        default=Path(
            "results/controlled/broad100_target_level_method_status/"
            "table_broad100_target_gate_comparison.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_target_method_package"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_composed_171_for_213")
    policy_fns = exp171.candidate_policy_functions()
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    base_target = pd.read_csv(args.target_table)
    full_target = rebuild_target_pool(base_target, outputs, FULL_LOCAL_ACTIONS, LARGE_ACTIONS, args.lambda_cost)
    no_tool_target = rebuild_target_pool(base_target, outputs, NO_TOOL_LOCAL_ACTIONS, LARGE_ACTIONS, args.lambda_cost)
    scores = pd.read_csv(args.learned_scores)
    assignments = pd.read_csv(args.learned_assignments)
    selected_source = pd.read_csv(args.learned_selected)

    method_rows: list[dict[str, Any]] = []
    assignment_rows: list[pd.DataFrame] = []

    for split in ["val", "test"]:
        frame = full_target[full_target["split"].eq(split)].copy()
        oracle_choose = frame["large_utility"].to_numpy(dtype=float) >= frame["local_utility"].to_numpy(dtype=float)
        for method, family, choose in [
            ("oracle_local_vs_large_gate", "diagnostic_oracle", oracle_choose),
            ("always_best_local_action", "reference", np.zeros(len(frame), dtype=bool)),
            ("always_best_large_action", "reference", np.ones(len(frame), dtype=bool)),
        ]:
            row, detail = evaluate_policy(
                frame,
                choose,
                oracle_reference=frame,
                split=split,
                method=method,
                family=family,
                action_pool_variant="full_action_pool",
                lambda_cost=args.lambda_cost,
            )
            method_rows.append(row)
            assignment_rows.append(detail)

    global_spec = parse_global_method(SELECTED_GLOBAL_METHOD)
    global_flags = predicted_verifiability_flags(scores, global_spec["classifier"], global_spec["threshold"])
    state_spec = parse_state_method(SELECTED_STATE_METHOD)
    state_info = state_assignments(assignments, SELECTED_STATE_METHOD)
    state_policy = selected_state_policy(selected_source, SELECTED_STATE_METHOD)

    policy_specs = [
        {
            "method": SELECTED_GLOBAL_METHOD,
            "family": "learned_verifiability_global",
            "policy_name": global_spec["policy_name"],
            "choose_fn": lambda frame: policy_fns[global_spec["policy_name"]](merge_route_signal(frame, global_flags)),
            "classifier": global_spec["classifier"],
            "threshold": global_spec["threshold"],
            "state_policy_json": "",
        },
        {
            "method": SELECTED_STATE_METHOD,
            "family": "learned_verifiability_state",
            "policy_name": "state_policy",
            "choose_fn": lambda frame: compose_state_policy(
                merge_route_signal(frame, state_info[["query_id", "pred_tool_available"]]),
                state_info[["query_id", "probe_state"]],
                state_policy,
                policy_fns,
            ),
            "classifier": state_spec["classifier"],
            "threshold": state_spec["threshold"],
            "state_policy_json": json.dumps(state_policy, sort_keys=True),
        },
    ]

    for spec in policy_specs:
        for variant, target in [
            ("full_action_pool", full_target),
            ("no_tool_local_pool_ablation", no_tool_target),
        ]:
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].copy()
                oracle_ref = full_target[full_target["split"].eq(split)].copy()
                choose = np.asarray(spec["choose_fn"](frame), dtype=bool)
                row, detail = evaluate_policy(
                    frame,
                    choose,
                    oracle_reference=oracle_ref,
                    split=split,
                    method=spec["method"],
                    family=spec["family"],
                    action_pool_variant=variant,
                    lambda_cost=args.lambda_cost,
                )
                row.update(
                    {
                        "classifier": spec["classifier"],
                        "threshold": spec["threshold"],
                        "policy_name": spec["policy_name"],
                        "state_policy_json": spec["state_policy_json"],
                    }
                )
                method_rows.append(row)
                assignment_rows.append(detail)

    main_eval = add_target_gates(pd.DataFrame(method_rows))
    details = pd.concat(assignment_rows, ignore_index=True)
    action_mix = summarize_action_mix(details)
    status_rows = load_status_rows(args.target_status)

    main_eval.to_csv(args.output_dir / "table_broad100_target_method_main_eval.csv", index=False)
    main_eval[main_eval["action_pool_variant"].str.contains("ablation|full", regex=True)].to_csv(
        args.output_dir / "table_broad100_target_method_ablation.csv", index=False
    )
    details.to_csv(args.output_dir / "table_broad100_target_method_assignments.csv", index=False)
    action_mix.to_csv(args.output_dir / "table_broad100_target_method_action_mix.csv", index=False)
    status_rows.to_csv(args.output_dir / "table_broad100_target_method_status_source.csv", index=False)
    write_figure(args.output_dir, main_eval)
    write_memo(args.output_dir / "BROAD100_TARGET_METHOD_PACKAGE.md", args, main_eval, action_mix, status_rows)
    print(f"Wrote Broad100 target-method package to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rebuild_target_pool(
    base_target: pd.DataFrame,
    outputs: pd.DataFrame,
    local_actions: tuple[str, ...],
    large_actions: tuple[str, ...],
    lambda_cost: float,
) -> pd.DataFrame:
    by_query_model = outputs.set_index(["query_id", "model_id"], drop=False)
    rows: list[dict[str, Any]] = []
    for source in base_target.to_dict("records"):
        query_id = str(source["query_id"])
        local = best_available_action(by_query_model, query_id, local_actions)
        large = best_available_action(by_query_model, query_id, large_actions)
        if local is None or large is None:
            continue
        row = dict(source)
        row.update(
            {
                "best_local_action": str(local["model_id"]),
                "best_large_action": str(large["model_id"]),
                "local_quality": float(local["quality_score"]),
                "large_quality": float(large["quality_score"]),
                "local_normalized_cost": float(local["normalized_remote_cost"]),
                "large_normalized_cost": float(large["normalized_remote_cost"]),
                "local_cost_usd": float(local["cost_total_usd"]),
                "large_cost_usd": float(large["cost_total_usd"]),
                "local_latency_s": float(local["latency_s"]),
                "large_latency_s": float(large["latency_s"]),
                "local_utility": float(local["quality_score"]) - float(lambda_cost) * float(local["normalized_remote_cost"]),
                "large_utility": float(large["quality_score"]) - float(lambda_cost) * float(large["normalized_remote_cost"]),
                "large_is_frontier": bool(str(large["model_id"]) in FRONTIER_MODELS),
                "best_large_family": large_family(str(large["model_id"])),
            }
        )
        row["delta_large"] = float(row["large_utility"]) - float(row["local_utility"])
        row["need_large"] = bool(row["delta_large"] >= 0.0)
        row["need_large_positive_gain"] = bool(row["delta_large"] > 1e-12)
        rows.append(row)
    return pd.DataFrame(rows)


def best_available_action(
    by_query_model: pd.DataFrame, query_id: str, actions: tuple[str, ...]
) -> pd.Series | None:
    candidates: list[pd.Series] = []
    for action in actions:
        key = (query_id, action)
        if key in by_query_model.index:
            candidates.append(by_query_model.loc[key].copy())
    if not candidates:
        return None
    frame = pd.DataFrame(candidates)
    return frame.sort_values(["utility", "quality_score", "normalized_remote_cost"], ascending=[False, False, True]).iloc[0]


def parse_global_method(method: str) -> dict[str, Any]:
    match = re.match(r"^(?P<classifier>.+)_thr(?P<threshold>[0-9.]+)_(?P<policy_name>.+)$", method)
    if not match:
        raise ValueError(f"Cannot parse global method: {method}")
    return {
        "classifier": match.group("classifier"),
        "threshold": float(match.group("threshold")),
        "policy_name": match.group("policy_name"),
    }


def parse_state_method(method: str) -> dict[str, Any]:
    match = re.match(r"^(?P<classifier>.+)_thr(?P<threshold>[0-9.]+)_state_k(?P<k>[0-9]+)$", method)
    if not match:
        raise ValueError(f"Cannot parse state method: {method}")
    return {
        "classifier": match.group("classifier"),
        "threshold": float(match.group("threshold")),
        "k": int(match.group("k")),
    }


def predicted_verifiability_flags(scores: pd.DataFrame, classifier: str, threshold: float) -> pd.DataFrame:
    subset = scores[scores["classifier"].astype(str).eq(classifier)].copy()
    if subset.empty:
        raise RuntimeError(f"No learned-verifiability score rows for classifier {classifier}")
    threshold_values = subset["threshold"].astype(float).dropna().unique()
    chosen = float(threshold_values[np.argmin(np.abs(threshold_values - float(threshold)))])
    subset = subset[np.isclose(subset["threshold"].astype(float), chosen)].copy()
    return (
        subset[["query_id", "pred_tool_available"]]
        .drop_duplicates("query_id")
        .assign(pred_tool_available=lambda frame: frame["pred_tool_available"].astype(bool))
    )


def state_assignments(assignments: pd.DataFrame, method: str) -> pd.DataFrame:
    subset = assignments[assignments["method"].astype(str).eq(method)].copy()
    if subset.empty:
        raise RuntimeError(f"No learned-verifiability assignments for {method}")
    return subset[["query_id", "probe_state", "pred_tool_available"]].drop_duplicates("query_id")


def selected_state_policy(selected: pd.DataFrame, method: str) -> dict[str, str]:
    subset = selected[
        selected["method"].astype(str).eq(method)
        & selected["split"].astype(str).eq("test")
        & selected["selection_rule"].astype(str).eq("val_best_utility_test")
    ].copy()
    if subset.empty:
        subset = selected[selected["method"].astype(str).eq(method)].copy()
    if subset.empty:
        raise RuntimeError(f"No selected policy row for {method}")
    value = str(subset.iloc[0].get("state_policy_json", "{}"))
    policy = json.loads(value)
    return {str(key): str(val) for key, val in policy.items()}


def merge_route_signal(frame: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=["tool_available"], errors="ignore").merge(flags, on="query_id", how="left")
    out["pred_tool_available"] = out["pred_tool_available"].fillna(False).astype(bool)
    # Candidate policies read tool_available. Here it is the learned route-time
    # verifiability signal, not the true direct tool flag.
    out["tool_available"] = out["pred_tool_available"]
    return out


def compose_state_policy(
    frame: pd.DataFrame,
    states: pd.DataFrame,
    state_policy: dict[str, str],
    policy_fns: dict[str, Callable[[pd.DataFrame], np.ndarray]],
) -> np.ndarray:
    work = frame.merge(states, on="query_id", how="left")
    choose = np.zeros(len(work), dtype=bool)
    for state in sorted(work["probe_state"].dropna().unique()):
        positions = np.where(work["probe_state"].to_numpy() == state)[0]
        policy_name = state_policy.get(str(int(state)), "always_local")
        choose[positions] = policy_fns[policy_name](work.iloc[positions].copy())
    return choose


def evaluate_policy(
    frame: pd.DataFrame,
    choose_large: np.ndarray,
    *,
    oracle_reference: pd.DataFrame,
    split: str,
    method: str,
    family: str,
    action_pool_variant: str,
    lambda_cost: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    choose_large = np.asarray(choose_large, dtype=bool)
    selected_action = np.where(choose_large, frame["best_large_action"], frame["best_local_action"])
    quality = np.where(choose_large, frame["large_quality"], frame["local_quality"]).astype(float)
    utility = np.where(choose_large, frame["large_utility"], frame["local_utility"]).astype(float)
    norm_cost = np.where(choose_large, frame["large_normalized_cost"], frame["local_normalized_cost"]).astype(float)
    usd_cost = np.where(choose_large, frame["large_cost_usd"], frame["local_cost_usd"]).astype(float)
    latency = np.where(choose_large, frame["large_latency_s"], frame["local_latency_s"]).astype(float)
    oracle_utility = np.maximum(oracle_reference["local_utility"], oracle_reference["large_utility"]).astype(float)
    oracle_quality = np.maximum(oracle_reference["local_quality"], oracle_reference["large_quality"]).astype(float)
    positives = oracle_reference["need_large"].astype(bool).to_numpy()
    tp = int(np.sum(choose_large & positives))
    fp = int(np.sum(choose_large & ~positives))
    fn = int(np.sum(~choose_large & positives))
    tn = int(np.sum(~choose_large & ~positives))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    detail = pd.DataFrame(
        {
            "query_id": frame["query_id"].astype(str).to_numpy(),
            "split": split,
            "benchmark": frame["benchmark"].astype(str).to_numpy(),
            "method": method,
            "family": family,
            "action_pool_variant": action_pool_variant,
            "choose_large": choose_large,
            "selected_action": selected_action,
            "selected_is_frontier": [str(action) in FRONTIER_MODELS for action in selected_action],
            "selected_quality": quality,
            "selected_utility": utility,
            "selected_normalized_cost": norm_cost,
            "selected_latency_s": latency,
            "full_oracle_utility": oracle_utility.to_numpy(dtype=float),
            "full_oracle_quality": oracle_quality.to_numpy(dtype=float),
        }
    )
    detail["utility_regret_to_full_oracle"] = detail["full_oracle_utility"] - detail["selected_utility"]
    detail["quality_regret_to_full_oracle"] = detail["full_oracle_quality"] - detail["selected_quality"]
    oracle_mean_utility = float(np.mean(oracle_utility))
    oracle_mean_quality = float(np.mean(oracle_quality))
    mean_utility = float(np.mean(utility))
    mean_quality = float(np.mean(quality))
    row = {
        "method": method,
        "family": family,
        "action_pool_variant": action_pool_variant,
        "split": split,
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "normalized_cost_mean": float(np.mean(norm_cost)),
        "remote_cost_total_usd": float(np.sum(usd_cost)),
        "mean_latency_s": float(np.mean(latency)),
        "p95_latency_s": float(np.quantile(latency, 0.95)),
        "full_oracle_mean_quality": oracle_mean_quality,
        "full_oracle_mean_utility": oracle_mean_utility,
        "quality_gap_to_full_oracle": oracle_mean_quality - mean_quality,
        "utility_gap_to_full_oracle": oracle_mean_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / max(oracle_mean_utility, 1e-12),
        "large_call_rate": float(np.mean(choose_large)),
        "frontier_call_rate": float(np.mean([str(action) in FRONTIER_MODELS for action in selected_action])),
        "need_large_precision": float(precision),
        "need_large_recall": float(recall),
        "need_large_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "selected_actions_json": json.dumps(pd.Series(selected_action).value_counts().sort_index().to_dict(), sort_keys=True),
        "lambda_cost": float(lambda_cost),
    }
    return row, detail


def add_target_gates(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["quality_target"] = out["full_oracle_mean_quality"] - 0.03
    out["utility_95pct_target"] = 0.95 * out["full_oracle_mean_utility"]
    out["utility_97pct_target"] = 0.97 * out["full_oracle_mean_utility"]
    out["meets_3pt_quality"] = out["mean_quality"] >= out["quality_target"]
    out["meets_95pct_utility"] = out["mean_utility"] >= out["utility_95pct_target"]
    out["meets_97pct_utility"] = out["mean_utility"] >= out["utility_97pct_target"]
    out["meets_frontier_cap_0p40"] = out["frontier_call_rate"] <= 0.40
    out["meets_primary_numeric_target"] = (
        out["meets_3pt_quality"] & out["meets_95pct_utility"] & out["meets_frontier_cap_0p40"]
    )
    return out


def summarize_action_mix(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.groupby(["method", "family", "action_pool_variant", "split", "selected_action"], as_index=False)
        .agg(
            n_queries=("query_id", "size"),
            mean_quality=("selected_quality", "mean"),
            mean_utility=("selected_utility", "mean"),
            mean_regret=("utility_regret_to_full_oracle", "mean"),
        )
        .sort_values(["split", "method", "action_pool_variant", "n_queries"], ascending=[True, True, True, False])
    )


def load_status_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    keep = frame[
        frame["method"].astype(str).isin(
            [
                SELECTED_GLOBAL_METHOD,
                SELECTED_STATE_METHOD,
                "current_base",
                "main_no_benchmark_no_tool_k2",
            ]
        )
    ].copy()
    return keep


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].copy()
    plot = plot[
        plot["method"].isin(
            [
                "oracle_local_vs_large_gate",
                SELECTED_GLOBAL_METHOD,
                SELECTED_STATE_METHOD,
                "always_best_local_action",
                "always_best_large_action",
            ]
        )
    ].copy()
    plot["label"] = plot["method"] + " / " + plot["action_pool_variant"]
    plot = plot.sort_values("mean_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6.5))
    colors = ["#426b69" if value else "#9d6b53" for value in plot["meets_primary_numeric_target"]]
    ax.barh(plot["label"], plot["mean_utility"], color=colors)
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Broad100 Target Method and No-Tool Ablations")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_broad100_target_method_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    main_eval: pd.DataFrame,
    action_mix: pd.DataFrame,
    status_rows: pd.DataFrame,
) -> None:
    test = main_eval[main_eval["split"].eq("test")].copy()
    cols = [
        "method",
        "family",
        "action_pool_variant",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "full_oracle_mean_quality",
        "full_oracle_mean_utility",
        "quality_gap_to_full_oracle",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "large_call_rate",
        "meets_primary_numeric_target",
        "meets_97pct_utility",
    ]
    mix_cols = ["method", "action_pool_variant", "split", "selected_action", "n_queries", "mean_utility", "mean_regret"]
    global_full = pick_row(test, SELECTED_GLOBAL_METHOD, "full_action_pool")
    global_no_tool = pick_row(test, SELECTED_GLOBAL_METHOD, "no_tool_local_pool_ablation")
    state_full = pick_row(test, SELECTED_STATE_METHOD, "full_action_pool")
    lines = [
        "# Broad100 Target Method Package",
        "",
        "This package reconstructs the cached learned-verifiability ProbeCode policy and tests whether it reaches the Phase 3 target gates.",
        "It makes no provider, vLLM, or local generation calls; all rows are rebuilt from cached Broad100 outputs.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/213_broad100_target_method_package.py",
        "```",
        "",
        "## Main Result",
        "",
        f"- Selected global learned-verifiability policy: `{SELECTED_GLOBAL_METHOD}`.",
        f"- Held-out test quality/utilty: `{global_full['mean_quality']:.4f}` / `{global_full['mean_utility']:.4f}`.",
        f"- Full cost-aware oracle quality/utility: `{global_full['full_oracle_mean_quality']:.4f}` / `{global_full['full_oracle_mean_utility']:.4f}`.",
        f"- Quality gap: `{global_full['quality_gap_to_full_oracle']:.4f}`; oracle utility ratio: `{global_full['oracle_utility_ratio']:.4f}`.",
        f"- Frontier-call rate: `{global_full['frontier_call_rate']:.4f}`.",
        f"- Primary numeric target met: `{bool(global_full['meets_primary_numeric_target'])}`.",
        "",
        "## What Carries The Result",
        "",
        f"- The no-tool local-pool ablation for the same policy has quality/utilty `{global_no_tool['mean_quality']:.4f}` / `{global_no_tool['mean_utility']:.4f}` and target pass `{bool(global_no_tool['meets_primary_numeric_target'])}`.",
        f"- The selected RouteCode state policy `{SELECTED_STATE_METHOD}` has quality/utilty `{state_full['mean_quality']:.4f}` / `{state_full['mean_utility']:.4f}` and target pass `{bool(state_full['meets_primary_numeric_target'])}`.",
        "- Interpretation: the cached target-level method works numerically, but the deterministic-tool/local-verifiability part is a major contributor and should be described as a verifiability/action-pool bridge, not as a fully clean benchmark-agnostic result.",
        "",
        "## Test Rows",
        "",
        "```csv",
        compact_csv(test[[col for col in cols if col in test.columns]].sort_values("mean_utility", ascending=False)),
        "```",
        "",
        "## Test Action Mix",
        "",
        "```csv",
        compact_csv(
            action_mix[action_mix["split"].eq("test")][[col for col in mix_cols if col in action_mix.columns]],
            max_rows=80,
        ),
        "```",
        "",
        "## Source Status Rows",
        "",
        "```csv",
        compact_csv(status_rows, max_rows=20),
        "```",
        "",
        "## Artifacts",
        "",
        "- `table_broad100_target_method_main_eval.csv`",
        "- `table_broad100_target_method_ablation.csv`",
        "- `table_broad100_target_method_assignments.csv`",
        "- `table_broad100_target_method_action_mix.csv`",
        "- `fig_broad100_target_method_utility.pdf`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick_row(frame: pd.DataFrame, method: str, variant: str) -> pd.Series:
    subset = frame[frame["method"].eq(method) & frame["action_pool_variant"].eq(variant)]
    if subset.empty:
        raise RuntimeError(f"Missing row for {method} / {variant}")
    return subset.iloc[0]


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def large_family(model_id: str) -> str:
    if model_id in FRONTIER_MODELS:
        return "frontier"
    if model_id == "qwen3-32b-awq-selfconsistency-n3-local":
        return "self_consistency"
    return "strong_local"


if __name__ == "__main__":
    main()
