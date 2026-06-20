from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOOL_MODEL_ID = "deterministic_math_tool"
CODE_BENCHMARKS = {"humaneval", "mbpp"}
PUBLIC_TEST_LOCAL_POOL = (
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
)
STRONG_OR_FRONTIER_ACTIONS = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    "gemini-3.5-flash",
    "gpt-5.5",
    "gemini-3.5-flash-strong-solve",
)
LOCAL_ACTIONS = (TOOL_MODEL_ID, *PUBLIC_TEST_LOCAL_POOL)
ALL_ACTIONS = tuple(dict.fromkeys((*LOCAL_ACTIONS, *STRONG_OR_FRONTIER_ACTIONS)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-test verifier policy for code-task action identity.")
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
    parser.add_argument("--benchmark-composed-method", default="tool_aware_benchmark_composed_eps0.01_recall_then_quality")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_public_test_verifier_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171")
    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(target, args.benchmark_composed_choices, args.benchmark_composed_method, exp171)
    priors = exp172.fit_train_priors(outputs)
    table_internal, details, code_summary = evaluate_policy_library(target, outputs, exp172=exp172, priors=priors, lambda_cost=float(args.lambda_cost))
    selected = exp172.validation_selected_rows(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = exp172.add_bootstrap_ci(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = table.drop(columns=["_utility_values"], errors="ignore")
    table.to_csv(args.output_dir / "table_public_test_verifier_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_public_test_verifier_policy_selected.csv", index=False)
    details.to_csv(args.output_dir / "table_public_test_verifier_policy_query_choices.csv", index=False)
    code_summary.to_csv(args.output_dir / "table_public_test_verifier_code_summary.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "PUBLIC_TEST_VERIFIER_POLICY_MEMO.md", args, table, selected, code_summary, exp172)
    print(f"Wrote public-test verifier policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate_policy_library(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    exp172,
    priors: dict[str, Any],
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    selectors = fixed_selectors(exp172, priors)
    threshold_selectors = threshold_selectors_from_val(target, exp172, priors)
    all_selectors = {**selectors, **threshold_selectors}
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        frame = target[target["split"].astype(str).eq(split)].copy()
        for method, (family, selector) in all_selectors.items():
            selected = select_actions(frame, rows_by_query, selector, exp172)
            selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                method,
                family,
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            row["code_verifier_call_rate"] = float(selected["code_verifier_used"].mean()) if not selected.empty else 0.0
            row["code_public_pass_rate"] = float(selected["code_public_pass"].mean()) if not selected.empty else 0.0
            rows.append(row)
            if split == "test" and method in detail_methods():
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
                detail["family"] = family
                detail = detail.merge(
                    selected[["query_id", "code_verifier_used", "code_public_pass", "code_verified_model"]],
                    on="query_id",
                    how="left",
                )
                detail_frames.append(detail)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    code_summary = summarize_code_verifier(outputs, exp172)
    return table, details, code_summary


def fixed_selectors(exp172, priors: dict[str, Any]) -> dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]]:
    return {
        "full_cost_aware_oracle": ("diagnostic_oracle", exp172.full_oracle_selector("utility")),
        "full_quality_oracle": ("diagnostic_oracle", exp172.full_oracle_selector("quality_score")),
        "train_best_single_action": (
            "reference",
            lambda row, actions: exp172.choose_prior_action(row, actions, priors, ALL_ACTIONS, scope="global"),
        ),
        "code_public_test_else_171_gate_prior": (
            "public_test_verifier_policy",
            lambda row, actions: code_or_gate_prior(row, actions, exp172, priors),
        ),
        "code_public_test_else_171_gate_consensus": (
            "public_test_verifier_policy",
            lambda row, actions: code_or_gate_consensus(row, actions, exp172, priors),
        ),
        "code_public_test_else_train_benchmark_prior": (
            "public_test_verifier_policy",
            lambda row, actions: code_or_train_prior(row, actions, exp172, priors),
        ),
        "diagnostic_public_code_oracle_else_171_gate_prior": (
            "diagnostic_policy",
            lambda row, actions: code_oracle_or_gate_prior(row, actions, exp172, priors),
        ),
    }


def threshold_selectors_from_val(
    target: pd.DataFrame,
    exp172,
    priors: dict[str, Any],
) -> dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]]:
    selectors: dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]] = {}
    val = target[target["split"].astype(str).eq("val")]
    signal_names = [
        "signal_constrained_plus_cached_mean_risk",
        "signal_constrained_plus_cached_max_risk",
        "signal_combined_mean_risk",
        "signal_constrained_yesno_local_evidence_risk",
    ]
    for signal in [name for name in signal_names if name in val.columns]:
        for threshold in candidate_thresholds(val[signal].to_numpy(dtype=float))[::4]:
            threshold = float(threshold)
            method = f"code_public_test_{signal}_thr{threshold:.4g}"
            selectors[method] = (
                "public_test_threshold_policy",
                lambda row, actions, signal=signal, threshold=threshold: code_or_signal_prior(row, actions, exp172, priors, signal, threshold),
            )
    return selectors


def code_or_gate_prior(row: pd.Series, actions: dict[str, dict[str, Any]], exp172, priors: dict[str, Any]) -> str:
    code = public_test_choice(row, actions, exp172, priors)
    if code:
        return code
    tool = exp172.tool_action(actions)
    if tool:
        return tool
    pool = STRONG_OR_FRONTIER_ACTIONS if bool(row.get("benchmark_composed_need_large", False)) else tuple(model for model in LOCAL_ACTIONS if model != TOOL_MODEL_ID)
    return exp172.choose_prior_action(row, actions, priors, pool, scope="benchmark")


def code_or_gate_consensus(row: pd.Series, actions: dict[str, dict[str, Any]], exp172, priors: dict[str, Any]) -> str:
    code = public_test_choice(row, actions, exp172, priors)
    if code:
        return code
    tool = exp172.tool_action(actions)
    if tool:
        return tool
    if bool(row.get("benchmark_composed_need_large", False)):
        return exp172.choose_prior_action(row, actions, priors, STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
    return exp172.choose_answer_agreement(
        row,
        actions,
        priors,
        pool=tuple(model for model in LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        evidence_pool=tuple(model for model in LOCAL_ACTIONS if model != TOOL_MODEL_ID),
        alpha=0.50,
        beta=0.25,
        fallback_pool=tuple(model for model in LOCAL_ACTIONS if model != TOOL_MODEL_ID),
    )


def code_or_train_prior(row: pd.Series, actions: dict[str, dict[str, Any]], exp172, priors: dict[str, Any]) -> str:
    code = public_test_choice(row, actions, exp172, priors)
    if code:
        return code
    tool = exp172.tool_action(actions)
    if tool:
        return tool
    return exp172.choose_prior_action(row, actions, priors, ALL_ACTIONS, scope="benchmark")


def code_or_signal_prior(
    row: pd.Series,
    actions: dict[str, dict[str, Any]],
    exp172,
    priors: dict[str, Any],
    signal: str,
    threshold: float,
) -> str:
    code = public_test_choice(row, actions, exp172, priors)
    if code:
        return code
    tool = exp172.tool_action(actions)
    if tool:
        return tool
    value = as_float(row.get(signal, -np.inf), default=-np.inf)
    pool = STRONG_OR_FRONTIER_ACTIONS if value >= float(threshold) else tuple(model for model in LOCAL_ACTIONS if model != TOOL_MODEL_ID)
    return exp172.choose_prior_action(row, actions, priors, pool, scope="benchmark")


def code_oracle_or_gate_prior(row: pd.Series, actions: dict[str, dict[str, Any]], exp172, priors: dict[str, Any]) -> str:
    if str(row.get("benchmark", "")) in CODE_BENCHMARKS:
        passing = [model for model in PUBLIC_TEST_LOCAL_POOL if is_public_test_pass(actions, model, exp172)]
        if passing:
            return best_action_by_metric(actions, passing, "utility", exp172)
    return code_or_gate_prior(row, actions, exp172, priors)


def public_test_choice(row: pd.Series, actions: dict[str, dict[str, Any]], exp172, priors: dict[str, Any]) -> str | None:
    if str(row.get("benchmark", "")) not in CODE_BENCHMARKS:
        return None
    passing = [model for model in PUBLIC_TEST_LOCAL_POOL if is_public_test_pass(actions, model, exp172)]
    if not passing:
        return None
    ranking = exp172.prior_rank_index(row, priors, tuple(PUBLIC_TEST_LOCAL_POOL))
    candidates: list[tuple[float, float, float, str]] = []
    for model_id in passing:
        action = actions[model_id]
        prior = float(ranking.get(model_id, 0.0))
        cost = as_float(action.get("normalized_remote_cost", 0.0))
        quality = as_float(action.get("quality_score", 0.0))
        # All passing public tests have the same route-time verifier status; use
        # train prior first, then quality/cost as deterministic tie breakers.
        candidates.append((prior, quality, -cost, model_id))
    return sorted(candidates, reverse=True)[0][3]


def is_public_test_pass(actions: dict[str, dict[str, Any]], model_id: str, exp172) -> bool:
    if not exp172.is_action_available(actions, model_id):
        return False
    parsed = str(actions[model_id].get("parsed_answer", "")).strip().lower()
    return parsed == "passed"


def best_action_by_metric(actions: dict[str, dict[str, Any]], pool: list[str], metric: str, exp172) -> str:
    candidates = []
    for model_id in pool:
        if not exp172.is_action_available(actions, model_id):
            continue
        action = actions[model_id]
        candidates.append(
            (
                as_float(action.get(metric, 0.0)),
                as_float(action.get("quality_score", 0.0)),
                -as_float(action.get("normalized_remote_cost", 0.0)),
                model_id,
            )
        )
    return sorted(candidates, reverse=True)[0][3] if candidates else ""


def select_actions(
    frame: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    selector: Callable[[pd.Series, dict[str, dict[str, Any]]], str],
    exp172,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        query_id = str(row["query_id"])
        actions = rows_by_query[query_id]
        code_model = first_public_test_pass(row, actions, exp172)
        model_id = selector(row, actions)
        if not exp172.is_action_available(actions, model_id):
            model_id = exp172.first_available(actions, ALL_ACTIONS)
        rows.append(
            {
                "query_id": query_id,
                "model_id": str(model_id),
                "code_verifier_used": str(row.get("benchmark", "")) in CODE_BENCHMARKS,
                "code_public_pass": bool(code_model),
                "code_verified_model": str(code_model or ""),
            }
        )
    return pd.DataFrame(rows)


def first_public_test_pass(row: pd.Series, actions: dict[str, dict[str, Any]], exp172) -> str | None:
    if str(row.get("benchmark", "")) not in CODE_BENCHMARKS:
        return None
    for model_id in PUBLIC_TEST_LOCAL_POOL:
        if is_public_test_pass(actions, model_id, exp172):
            return model_id
    return None


def summarize_code_verifier(outputs: pd.DataFrame, exp172) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, benchmark), group in outputs[outputs["benchmark"].isin(CODE_BENCHMARKS)].groupby(["split", "benchmark"]):
        for query_id, qgroup in group.groupby("query_id"):
            actions = qgroup.set_index("model_id").to_dict("index")
            passing = [model for model in PUBLIC_TEST_LOCAL_POOL if is_public_test_pass(actions, model, exp172)]
            rows.append(
                {
                    "split": split,
                    "benchmark": benchmark,
                    "query_id": str(query_id),
                    "n_passing_public_test_local": len(passing),
                    "passing_models_json": json.dumps(passing),
                    "any_public_test_pass": bool(passing),
                }
            )
    detail = pd.DataFrame(rows)
    return (
        detail.groupby(["split", "benchmark"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            any_public_test_pass_rate=("any_public_test_pass", "mean"),
            mean_passing_local_count=("n_passing_public_test_local", "mean"),
        )
        .sort_values(["split", "benchmark"])
    )


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([0.0])
    return np.unique(np.quantile(values, np.linspace(0.0, 1.0, 21)))


def as_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def detail_methods() -> set[str]:
    return {
        "full_cost_aware_oracle",
        "code_public_test_else_171_gate_prior",
        "code_public_test_else_171_gate_consensus",
        "code_public_test_else_train_benchmark_prior",
        "diagnostic_public_code_oracle_else_171_gate_prior",
    }


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#526f6d")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Public-Test Verifier Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_public_test_verifier_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    code_summary: pd.DataFrame,
    exp172,
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
        "code_verifier_call_rate",
        "code_public_pass_rate",
        "need_large_precision",
        "need_large_recall",
        "selection_rule",
    ]
    selected_cols = [column for column in cols if column in selected.columns]
    test = table[table["split"].eq("test")].copy()
    full_oracle = test[test["method"].eq("full_cost_aware_oracle")].head(1)
    deployable_test = test[test["family"].isin(["public_test_verifier_policy", "public_test_threshold_policy", "reference"])]
    best_deployable = deployable_test.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1)
    lines = [
        "# Public-Test Verifier Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/175_public_test_verifier_policy.py",
        (
            "PYTHONPATH=src python experiments/175_public_test_verifier_policy.py "
            f"--target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## What This Tests",
        "",
        "- Cached-only deployed-action selector; no GPT, Gemini, Claude, vLLM, or other model calls.",
        "- For HumanEval and MBPP, use public prompt tests/pass status as a route-time verifier for local code actions.",
        "- Outside code tasks, fall back to existing tool-aware 171 gate or threshold policies.",
        "- This targets concrete action identity, especially choosing the correct local code model and avoiding unnecessary frontier calls on code tasks.",
        "",
        "## Selected Rows",
        "",
        "```csv",
        exp172.compact_csv(selected[selected_cols], max_rows=80),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        exp172.compact_csv(test.sort_values(["mean_utility", "mean_quality"], ascending=False)[[c for c in cols if c in test.columns]], max_rows=40),
        "```",
        "",
        "## Target Check",
        "",
        *exp172.target_check_lines(full_oracle, best_deployable),
        "",
        "## Code Verifier Coverage",
        "",
        "```csv",
        exp172.compact_csv(code_summary, max_rows=None),
        "```",
        "",
        "## Interpretation",
        "",
        "- Passing public tests are treated as route-time evidence because the tests are in the code prompt/metadata.",
        "- Diagnostic rows remain diagnostic only; deployable rows must be selected on validation.",
        "",
        "## Artifacts",
        "",
        f"- All policy table: `{args.output_dir / 'table_public_test_verifier_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_public_test_verifier_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_public_test_verifier_policy_query_choices.csv'}`",
        f"- Code verifier summary: `{args.output_dir / 'table_public_test_verifier_code_summary.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_public_test_verifier_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
