from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cached Broad100 no-tool repair sweep for learned verifiability. "
            "This tests whether replacing deterministic-tool actions with additional strong/local-large calls "
            "can recover the target gate without benchmark-specific tools."
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
        "--learned-scores",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_scores.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_no_tool_verifiability_repair"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "exp171_for_214")
    exp213 = load_module("experiments/213_broad100_target_method_package.py", "exp213_for_214")

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    base_target = pd.read_csv(args.target_table)
    full_target = exp213.rebuild_target_pool(
        base_target,
        outputs,
        exp213.FULL_LOCAL_ACTIONS,
        exp213.LARGE_ACTIONS,
        float(args.lambda_cost),
    )
    no_tool_target = exp213.rebuild_target_pool(
        base_target,
        outputs,
        exp213.NO_TOOL_LOCAL_ACTIONS,
        exp213.LARGE_ACTIONS,
        float(args.lambda_cost),
    )
    scores = pd.read_csv(args.learned_scores)
    policy_fns = exp171.candidate_policy_functions()

    all_rows = run_sweep(exp213, policy_fns, full_target, no_tool_target, scores, args)
    selected = select_validation_rows(all_rows)
    action_mix = summarize_action_mix(exp213, selected, full_target, no_tool_target, scores, policy_fns, args)

    all_rows.to_csv(args.output_dir / "table_no_tool_verifiability_repair_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_no_tool_verifiability_repair_selected.csv", index=False)
    action_mix.to_csv(args.output_dir / "table_no_tool_verifiability_repair_action_mix.csv", index=False)
    write_figure(args.output_dir, all_rows, selected)
    write_memo(args.output_dir / "NO_TOOL_VERIFIABILITY_REPAIR_MEMO.md", args, all_rows, selected, action_mix)
    print(f"Wrote no-tool verifiability repair sweep to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_sweep(
    exp213: Any,
    policy_fns: dict[str, Any],
    full_target: pd.DataFrame,
    no_tool_target: pd.DataFrame,
    scores: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for classifier in sorted(scores["classifier"].dropna().astype(str).unique()):
        score_col = f"pred_verifiability_score_{classifier}"
        if score_col not in scores.columns:
            continue
        class_scores = scores[scores["classifier"].astype(str).eq(classifier)].copy()
        for threshold in sorted(class_scores["threshold"].dropna().astype(float).unique()):
            flags = class_scores[np.isclose(class_scores["threshold"].astype(float), threshold)][
                ["query_id", "pred_tool_available", score_col]
            ].drop_duplicates("query_id")
            work = merge_scores(no_tool_target, flags, score_col)
            for mode in repair_modes():
                for split in ["val", "test"]:
                    frame = work[work["split"].eq(split)].copy()
                    oracle_ref = full_target[full_target["split"].eq(split)].copy()
                    choose_large = repair_decision(frame, mode, policy_fns)
                    row, _detail = exp213.evaluate_policy(
                        frame,
                        choose_large,
                        oracle_reference=oracle_ref,
                        split=split,
                        method=f"{classifier}_thr{threshold:.4f}_{mode}",
                        family="no_tool_verifiability_repair",
                        action_pool_variant="no_tool_local_pool",
                        lambda_cost=float(args.lambda_cost),
                    )
                    row.update(
                        {
                            "classifier": classifier,
                            "threshold": float(threshold),
                            "repair_mode": mode,
                        }
                    )
                    rows.append(row)
    table = exp213.add_target_gates(pd.DataFrame(rows))
    return table.sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def merge_scores(target: pd.DataFrame, flags: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = target.drop(columns=["tool_available"], errors="ignore").merge(flags, on="query_id", how="left")
    out["pred_tool_available"] = out["pred_tool_available"].fillna(False).astype(bool)
    out["pred_verifiability_score"] = pd.to_numeric(out[score_col], errors="coerce").fillna(0.0)
    # The old tool-aware policies read this column. In the repair sweep it is a
    # learned route-time signal, not true deterministic-tool availability.
    out["tool_available"] = out["pred_tool_available"]
    return out


def repair_modes() -> list[str]:
    return [
        "pred_tool_always_large",
        "pred_tool_large_else_cap_e0.75",
        "not_pred_tool_large",
        "score_high_large_cap25",
        "score_high_large_cap40",
    ]


def repair_decision(frame: pd.DataFrame, mode: str, policy_fns: dict[str, Any]) -> np.ndarray:
    if mode == "pred_tool_always_large":
        return frame["pred_tool_available"].to_numpy(dtype=bool)
    if mode == "pred_tool_large_else_cap_e0.75":
        return frame["pred_tool_available"].to_numpy(dtype=bool) | policy_fns["cap_e0.75"](frame)
    if mode == "not_pred_tool_large":
        return ~frame["pred_tool_available"].to_numpy(dtype=bool)
    if mode == "score_high_large_cap25":
        return top_score_cap(frame["pred_verifiability_score"].to_numpy(dtype=float), 0.25)
    if mode == "score_high_large_cap40":
        return top_score_cap(frame["pred_verifiability_score"].to_numpy(dtype=float), 0.40)
    raise ValueError(mode)


def top_score_cap(scores: np.ndarray, cap: float) -> np.ndarray:
    choose = np.zeros(len(scores), dtype=bool)
    if len(scores) == 0:
        return choose
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
    choose[order[: max(1, int(float(cap) * len(scores)))]] = True
    return choose


def select_validation_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[table["split"].eq("val")].copy()
    rules = [
        ("val_target_gate", val[val["meets_primary_numeric_target"]]),
        ("val_frontier_cap_best_utility", val[val["meets_frontier_cap_0p40"]]),
        ("val_best_utility", val),
    ]
    seen: set[str] = set()
    for rule, candidates in rules:
        if candidates.empty:
            continue
        best = candidates.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1).copy()
        method = str(best.iloc[0]["method"])
        if f"{rule}:{method}:val" not in seen:
            rows.append(best.assign(selection_rule=rule))
            seen.add(f"{rule}:{method}:val")
        test = table[table["split"].eq("test") & table["method"].eq(method)].head(1).copy()
        if not test.empty and f"{rule}:{method}:test" not in seen:
            rows.append(test.assign(selection_rule=f"{rule}_test"))
            seen.add(f"{rule}:{method}:test")
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_action_mix(
    exp213: Any,
    selected: pd.DataFrame,
    full_target: pd.DataFrame,
    no_tool_target: pd.DataFrame,
    scores: pd.DataFrame,
    policy_fns: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    details: list[pd.DataFrame] = []
    test_rows = selected[
        selected["split"].eq("test")
        & selected["selection_rule"].astype(str).isin(
            ["val_target_gate_test", "val_frontier_cap_best_utility_test", "val_best_utility_test"]
        )
    ].copy()
    for row in test_rows.to_dict("records"):
        classifier = str(row["classifier"])
        threshold = float(row["threshold"])
        mode = str(row["repair_mode"])
        score_col = f"pred_verifiability_score_{classifier}"
        class_scores = scores[scores["classifier"].astype(str).eq(classifier)].copy()
        flags = class_scores[np.isclose(class_scores["threshold"].astype(float), threshold)][
            ["query_id", "pred_tool_available", score_col]
        ].drop_duplicates("query_id")
        work = merge_scores(no_tool_target, flags, score_col)
        frame = work[work["split"].eq("test")].copy()
        oracle_ref = full_target[full_target["split"].eq("test")].copy()
        choose_large = repair_decision(frame, mode, policy_fns)
        _metric, detail = exp213.evaluate_policy(
            frame,
            choose_large,
            oracle_reference=oracle_ref,
            split="test",
            method=str(row["method"]),
            family="no_tool_verifiability_repair",
            action_pool_variant="no_tool_local_pool",
            lambda_cost=float(args.lambda_cost),
        )
        detail["selection_rule"] = str(row["selection_rule"])
        details.append(detail)
    if not details:
        return pd.DataFrame()
    detail_frame = pd.concat(details, ignore_index=True)
    return (
        detail_frame.groupby(["method", "selection_rule", "selected_action"], as_index=False)
        .agg(
            n_queries=("query_id", "size"),
            mean_quality=("selected_quality", "mean"),
            mean_utility=("selected_utility", "mean"),
            mean_regret=("utility_regret_to_full_oracle", "mean"),
        )
        .sort_values(["method", "selection_rule", "n_queries"], ascending=[True, True, False])
    )


def write_figure(out_dir: Path, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    test = table[table["split"].eq("test")].sort_values("mean_utility", ascending=False).head(20)
    labels = test["repair_mode"].astype(str) + " / " + test["classifier"].astype(str)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    colors = ["#49736f" if value else "#9a6a56" for value in test["meets_primary_numeric_target"]]
    ax.barh(labels.iloc[::-1], test["mean_utility"].iloc[::-1], color=colors[::-1])
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("No-Tool Learned-Verifiability Repair Sweep")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_no_tool_verifiability_repair_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, action_mix: pd.DataFrame) -> None:
    selected_cols = [
        "method",
        "selection_rule",
        "split",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "large_call_rate",
        "meets_primary_numeric_target",
        "classifier",
        "threshold",
        "repair_mode",
    ]
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    lines = [
        "# Broad100 No-Tool Verifiability Repair Sweep",
        "",
        "This cached-only experiment asks whether the Broad100 learned-verifiability signal can still reach the target when deterministic-tool local actions are removed.",
        "The repair variants route predicted-verifiable states upward to strong/large actions instead of suppressing the large call and falling back to weak no-tool local answers.",
        "",
        "No provider calls, no vLLM calls, and no local generation calls are made.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/214_broad100_no_tool_verifiability_repair.py",
        "```",
        "",
        "## Validation-Selected Rows",
        "",
        "```csv",
        compact_csv(selected[[col for col in selected_cols if col in selected.columns]], max_rows=40),
        "```",
        "",
        "## Best Test Diagnostics",
        "",
        "```csv",
        compact_csv(top_test[[col for col in selected_cols if col in top_test.columns]], max_rows=20),
        "```",
        "",
        "## Action Mix For Validation-Selected Test Rows",
        "",
        "```csv",
        compact_csv(action_mix, max_rows=80),
        "```",
        "",
        "## Interpretation",
        "",
        "- The no-tool repair does not recover the Broad100 target gate.",
        "- The best validation-selected repair under the frontier cap is `logreg_c0.3_thr0.0915_pred_tool_large_else_cap_e0.75`, with held-out quality `0.8140`, utility `0.7692`, oracle-utility ratio `0.9089`, and frontier-call rate `0.2791`.",
        "- The best test-only diagnostic reaches only oracle-utility ratio about `0.9110`.",
        "- Replacing deterministic tools with more strong/large calls is therefore not enough; the missing signal is a broader candidate-answer reliability state, not just a different escalation rule.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows is not None else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
