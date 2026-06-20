from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark-composed actual action policy from cached broad100 outputs.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--benchmark-composed-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_tool_aware_benchmark_composed_policy/"
            "table_tool_aware_benchmark_composed_choices.csv"
        ),
    )
    parser.add_argument(
        "--benchmark-composed-method",
        default="tool_aware_benchmark_composed_eps0.01_recall_then_quality",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_benchmark_composed_deployed_action_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    e172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172")
    e171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_173")

    outputs = e172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = e171.add_tool_availability(target, outputs)
    target = e172.add_benchmark_composed_gate(target, args.benchmark_composed_choices, args.benchmark_composed_method, e171)

    table_internal, choices, details = run_benchmark_composition(
        e172,
        target,
        outputs,
        lambda_cost=float(args.lambda_cost),
    )
    selected = validation_selected_rows(e172, table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = e172.add_bootstrap_ci(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = table.drop(columns=["_utility_values"], errors="ignore")

    table.to_csv(args.output_dir / "table_benchmark_composed_deployed_action_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_benchmark_composed_deployed_action_selected.csv", index=False)
    choices.to_csv(args.output_dir / "table_benchmark_composed_deployed_action_choices.csv", index=False)
    details.to_csv(args.output_dir / "table_benchmark_composed_deployed_action_query_choices.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(
        args.output_dir / "BENCHMARK_COMPOSED_DEPLOYED_ACTION_POLICY_MEMO.md",
        args,
        table,
        selected,
        choices,
    )
    print(f"Wrote benchmark-composed deployed-action policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_benchmark_composition(
    e172,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    priors = e172.fit_train_priors(outputs)
    fixed = e172.fixed_policy_functions(outputs, priors)
    thresholds = e172.threshold_policy_functions(target, outputs, priors)
    candidate_fns = build_candidate_menu(e172, outputs, priors, fixed, thresholds)

    rows: list[dict[str, Any]] = []
    choices: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        frame = target[target["split"].astype(str).eq(split)].copy()
        rows.extend(reference_rows(e172, frame, outputs, frontiers, split=split, lambda_cost=lambda_cost))

    val = target[target["split"].astype(str).eq("val")].copy()
    for epsilon in [0.0, 0.005, 0.01, 0.02, 0.04]:
        for tie_break in ["utility", "quality", "frontier_light", "recall_quality"]:
            method = f"benchmark_composed_deployed_eps{epsilon:g}_{tie_break}"
            chosen_by_benchmark: dict[str, str] = {}
            for benchmark in sorted(val["benchmark"].dropna().astype(str).unique()):
                source = val[val["benchmark"].astype(str).eq(benchmark)].copy()
                scored = [
                    score_candidate(
                        e172,
                        source,
                        outputs,
                        rows_by_query,
                        frontiers,
                        split="val",
                        method=policy_name,
                        family="candidate",
                        selector=selector,
                        lambda_cost=lambda_cost,
                    )
                    for policy_name, selector in candidate_fns.items()
                ]
                scored_frame = pd.DataFrame(scored)
                chosen = choose_candidate(scored_frame, epsilon=epsilon, tie_break=tie_break)
                chosen_by_benchmark[benchmark] = chosen
                choice = scored_frame[scored_frame["method"].eq(chosen)].head(1).to_dict("records")[0]
                choices.append(
                    {
                        "method": method,
                        "benchmark": benchmark,
                        "chosen_policy": chosen,
                        "epsilon": float(epsilon),
                        "tie_break": tie_break,
                        "val_policy_utility": float(choice["mean_utility"]),
                        "val_policy_quality": float(choice["mean_quality"]),
                        "val_frontier_call_rate": float(choice["frontier_call_rate"]),
                        "val_strong_or_frontier_call_rate": float(choice["strong_or_frontier_call_rate"]),
                        "all_candidate_scores_json": json.dumps(
                            scored_frame[
                                [
                                    "method",
                                    "mean_utility",
                                    "mean_quality",
                                    "frontier_call_rate",
                                    "strong_or_frontier_call_rate",
                                    "need_large_recall",
                                ]
                            ].to_dict("records"),
                            sort_keys=True,
                        ),
                    }
                )
            for split in ["val", "test"]:
                frame = target[target["split"].astype(str).eq(split)].copy()
                selector = composed_selector(chosen_by_benchmark, candidate_fns)
                row, selected_rows = evaluate_selector(
                    e172,
                    frame,
                    outputs,
                    rows_by_query,
                    frontiers,
                    split=split,
                    method=method,
                    family="benchmark_composed_deployed",
                    selector=selector,
                    lambda_cost=lambda_cost,
                )
                row.update({"epsilon": float(epsilon), "tie_break": tie_break})
                rows.append(row)
                if split == "test":
                    detail = selected_rows[
                        [
                            "query_id",
                            "query_text",
                            "benchmark",
                            "metric",
                            "model_id",
                            "quality_score",
                            "utility",
                            "normalized_remote_cost",
                            "is_frontier",
                            "parsed_answer",
                        ]
                    ].copy()
                    detail["method"] = method
                    detail["chosen_policy"] = detail["benchmark"].map(chosen_by_benchmark).fillna("")
                    details.append(detail)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    return table, pd.DataFrame(choices), pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def build_candidate_menu(
    e172,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
    fixed: dict[str, tuple[str, Callable]],
    thresholds: dict[str, tuple[str, Callable]],
) -> dict[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]:
    menu: dict[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]] = {}
    for name in [
        "train_best_single_action",
        "train_benchmark_prior_all_actions",
        "tool_then_train_benchmark_prior_all_actions",
        "tool_then_local_consensus_else_benchmark_prior",
        "tool_then_171_gate_train_prior",
        "tool_then_171_gate_local_consensus_large_prior",
    ]:
        if name in fixed:
            menu[name] = fixed[name][1]
    for model_id in sorted(outputs["model_id"].astype(str).unique()):
        if model_id == e172.TOOL_MODEL_ID:
            continue
        menu[f"fixed_{model_id}"] = fixed_model_selector(e172, model_id, priors)
        menu[f"tool_then_fixed_{model_id}"] = tool_then_fixed_model_selector(e172, model_id, priors)
    for name, (_, selector) in thresholds.items():
        if keep_threshold_candidate(name):
            menu[name] = selector
    return menu


def keep_threshold_candidate(name: str) -> bool:
    useful_prefixes = [
        "threshold_signal_combined_mean_risk_high_thr0.3029_",
        "threshold_signal_combined_mean_risk_high_thr0.1578_",
        "threshold_signal_combined_max_risk_high_thr0.5231_",
        "threshold_signal_combined_max_risk_high_thr0.5389_",
        "threshold_signal_constrained_yesno_local_evidence_risk_high_thr0.1361_",
        "threshold_signal_constrained_yesno_local_evidence_risk_high_thr0.1651_",
        "threshold_signal_constrained_yesno_local_evidence_risk_high_thr0.1035_",
        "threshold_signal_constrained_yesno_local_evidence_risk_high_thr0.1149_",
        "threshold_signal_slm_medium_divergence_high_thr0.3_",
        "threshold_signal_query_answerability_risk_high_thr0.2018_",
    ]
    return any(name.startswith(prefix) for prefix in useful_prefixes)


def fixed_model_selector(e172, model_id: str, priors: dict[str, Any]) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        if e172.is_action_available(actions, model_id):
            return model_id
        return e172.choose_prior_action(row, actions, priors, e172.ALL_ACTIONS, scope="benchmark")

    return select


def tool_then_fixed_model_selector(e172, model_id: str, priors: dict[str, Any]) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        tool = e172.tool_action(actions)
        if tool:
            return tool
        if e172.is_action_available(actions, model_id):
            return model_id
        return e172.choose_prior_action(row, actions, priors, e172.ALL_ACTIONS, scope="benchmark")

    return select


def composed_selector(
    chosen_by_benchmark: dict[str, str],
    candidate_fns: dict[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]],
) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        policy = chosen_by_benchmark.get(str(row.get("benchmark", "")))
        if policy in candidate_fns:
            return candidate_fns[policy](row, actions)
        return next(iter(candidate_fns.values()))(row, actions)

    return select


def reference_rows(
    e172,
    frame: pd.DataFrame,
    outputs: pd.DataFrame,
    frontiers: set[str],
    *,
    split: str,
    lambda_cost: float,
) -> list[dict[str, Any]]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    priors = e172.fit_train_priors(outputs)
    refs = e172.fixed_policy_functions(outputs, priors)
    rows: list[dict[str, Any]] = []
    for name in ["full_cost_aware_oracle", "full_quality_oracle", "train_best_single_action", "tool_then_171_gate_train_prior"]:
        family, selector = refs[name]
        row, _ = evaluate_selector(
            e172,
            frame,
            outputs,
            rows_by_query,
            frontiers,
            split=split,
            method=name,
            family=family,
            selector=selector,
            lambda_cost=lambda_cost,
        )
        rows.append(row)
    return rows


def score_candidate(
    e172,
    frame: pd.DataFrame,
    outputs: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    frontiers: set[str],
    *,
    split: str,
    method: str,
    family: str,
    selector: Callable[[pd.Series, dict[str, dict[str, Any]]], str],
    lambda_cost: float,
) -> dict[str, Any]:
    row, _ = evaluate_selector(
        e172,
        frame,
        outputs,
        rows_by_query,
        frontiers,
        split=split,
        method=method,
        family=family,
        selector=selector,
        lambda_cost=lambda_cost,
    )
    return row


def evaluate_selector(
    e172,
    frame: pd.DataFrame,
    outputs: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    frontiers: set[str],
    *,
    split: str,
    method: str,
    family: str,
    selector: Callable[[pd.Series, dict[str, dict[str, Any]]], str],
    lambda_cost: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = e172.select_actions(frame, selector, rows_by_query)
    selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
    selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
    row = e172.evaluate_selected_rows(
        method,
        family,
        split,
        selected_rows,
        outputs,
        target=frame,
        frontiers=frontiers,
        lambda_cost=lambda_cost,
    )
    return row, selected_rows


def choose_candidate(scored: pd.DataFrame, *, epsilon: float, tie_break: str) -> str:
    best_utility = float(scored["mean_utility"].max())
    near = scored[scored["mean_utility"] >= best_utility - float(epsilon)].copy()
    if tie_break == "utility":
        columns = ["mean_utility", "mean_quality", "frontier_call_rate", "normalized_cost_mean"]
        ascending = [False, False, True, True]
    elif tie_break == "quality":
        columns = ["mean_quality", "mean_utility", "frontier_call_rate", "normalized_cost_mean"]
        ascending = [False, False, True, True]
    elif tie_break == "frontier_light":
        columns = ["frontier_call_rate", "mean_utility", "mean_quality", "normalized_cost_mean"]
        ascending = [True, False, False, True]
    elif tie_break == "recall_quality":
        columns = ["need_large_recall", "mean_quality", "mean_utility", "frontier_call_rate"]
        ascending = [False, False, False, True]
    else:
        raise ValueError(tie_break)
    return str(near.sort_values(columns, ascending=ascending).iloc[0]["method"])


def validation_selected_rows(e172, table: pd.DataFrame, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    selected = e172.validation_selected_rows(table, bootstrap_samples=bootstrap_samples, seed=seed)
    group = table[table["family"].eq("benchmark_composed_deployed")].copy()
    val = group[group["split"].eq("val")].copy()
    extras: list[pd.DataFrame] = []
    if not val.empty:
        oracle = table[(table["split"].eq("val")) & (table["method"].eq("full_cost_aware_oracle"))]
        if not oracle.empty:
            utility_floor = 0.95 * float(oracle.iloc[0]["mean_utility"])
            quality_floor = float(oracle.iloc[0]["mean_quality"]) - 0.03
            feasible = val[
                (val["mean_utility"] >= utility_floor)
                & (val["mean_quality"] >= quality_floor)
                & (val["frontier_call_rate"] <= 0.40)
            ].copy()
        else:
            feasible = pd.DataFrame()
        if feasible.empty:
            feasible = val
        chosen = feasible.sort_values(
            ["mean_quality", "mean_utility", "frontier_call_rate"],
            ascending=[False, False, True],
        ).head(1)
        method = str(chosen.iloc[0]["method"])
        extras.append(chosen.assign(selection_rule="val_target_quality_utility"))
        test = table[table["split"].eq("test") & table["method"].eq(method)].copy()
        if not test.empty:
            extras.append(test.assign(selection_rule="val_target_quality_utility_test"))
    if extras:
        extra = pd.concat(extras, ignore_index=True)
        extra = e172.add_bootstrap_ci(extra, bootstrap_samples=bootstrap_samples, seed=seed)
        extra = extra.drop(columns=["_utility_values"], errors="ignore")
        selected = pd.concat([selected, extra], ignore_index=True)
    return selected


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4f6f64")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Benchmark-Composed Actual Action Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_composed_deployed_action_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    choices: pd.DataFrame,
) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "utility_gap_to_oracle",
        "quality_gap_to_oracle",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "selection_rule",
    ]
    test = table[table["split"].eq("test")].copy()
    selected_methods = set(selected["method"].dropna().astype(str).unique()) if not selected.empty else set()
    lines = [
        "# Benchmark-Composed Deployed-Action Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/173_benchmark_composed_deployed_action_policy.py",
        (
            "PYTHONPATH=src python experiments/173_benchmark_composed_deployed_action_policy.py "
            f"--target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## What This Tests",
        "",
        "- This composes concrete action policies per benchmark using validation only.",
        "- Candidate policies include train-only action priors, fixed concrete models, deterministic tool-first variants, local consensus, 171 gates, and selected threshold rules.",
        "- It makes no GPT, Gemini, Claude, or vLLM calls; all model outputs are cached.",
        "",
        "## Selected Rows",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=90),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(test.sort_values(["mean_utility", "mean_quality"], ascending=False)[[column for column in cols if column in test.columns]], max_rows=40),
        "```",
        "",
        "## Selected Benchmark Policies",
        "",
        "```csv",
        compact_csv(
            choices[choices["method"].isin(selected_methods)]
            .sort_values(["method", "benchmark"])
            .drop(columns=["all_candidate_scores_json"], errors="ignore"),
            max_rows=160,
        ),
        "```",
        "",
        "## Target Check",
        "",
        *target_check_lines(test),
        "",
        "## Interpretation",
        "",
        "- Benchmark composition is an actual-action improvement over the single global 172 selector if its selected test utility is higher.",
        "- It is still a validation-selected policy over a small validation set per benchmark, so benchmark-level overfitting must be treated as residual risk.",
        "- It remains a deployed-action result, not a best-local/best-large abstraction.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_check_lines(test: pd.DataFrame) -> list[str]:
    oracle = test[test["method"].eq("full_cost_aware_oracle")].head(1)
    deployable = test[test["family"].eq("benchmark_composed_deployed")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(1)
    if oracle.empty or deployable.empty:
        return ["- Missing oracle or deployable rows."]
    o = oracle.iloc[0]
    d = deployable.iloc[0]
    utility_target = 0.97 * float(o["mean_utility"])
    quality_target = float(o["mean_quality"]) - 0.03
    return [
        f"- Full held-out cost-aware oracle utility: `{float(o['mean_utility']):.4f}`.",
        f"- 97% utility target: `{utility_target:.4f}`.",
        f"- Best held-out benchmark-composed deployed method: `{d['method']}`.",
        f"- Best held-out benchmark-composed utility: `{float(d['mean_utility']):.4f}`; pass: `{bool(d['mean_utility'] >= utility_target)}`.",
        f"- Full held-out cost-aware oracle quality: `{float(o['mean_quality']):.4f}`.",
        f"- Within-3-point quality target: `{quality_target:.4f}`.",
        f"- Best held-out benchmark-composed quality: `{float(d['mean_quality']):.4f}`; pass: `{bool(d['mean_quality'] >= quality_target)}`.",
    ]


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
