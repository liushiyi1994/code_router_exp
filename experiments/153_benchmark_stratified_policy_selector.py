from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-selected benchmark-stratified policy composition over cached self-consistency gates."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_benchmark_stratified_policy_selector"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--frontier-cap", type=float, default=0.40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_feature_gate")
    fast_gate = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_action_gate")
    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    library = build_policy_library(
        package,
        self_gate,
        fast_gate,
        outputs,
        probe,
        self_model_id=str(args.self_model_id),
        max_features=int(args.max_features),
    )
    candidate_table = evaluate_library(
        package,
        outputs,
        library,
        lambda_cost=float(args.lambda_cost),
        self_model_id=str(args.self_model_id),
    )
    choice_table, composed_table, composed_details = run_composed_selectors(
        package,
        outputs,
        library,
        candidate_table,
        lambda_cost=float(args.lambda_cost),
        self_model_id=str(args.self_model_id),
        frontier_cap=float(args.frontier_cap),
    )
    candidate_table.to_csv(args.output_dir / "table_policy_library_eval.csv", index=False)
    choice_table.to_csv(args.output_dir / "table_benchmark_policy_choices.csv", index=False)
    composed_table.to_csv(args.output_dir / "table_benchmark_stratified_eval.csv", index=False)
    composed_details.to_csv(args.output_dir / "table_benchmark_stratified_details.csv", index=False)
    write_figure(args.output_dir, candidate_table, composed_table)
    write_memo(
        args.output_dir / "BENCHMARK_STRATIFIED_POLICY_SELECTOR_MEMO.md",
        args,
        candidate_table,
        choice_table,
        composed_table,
    )
    print(f"Wrote benchmark-stratified policy selector results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_policy_library(
    package,
    self_gate,
    fast_gate,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    max_features: int,
) -> dict[str, dict[str, pd.Series]]:
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()

    base_specs = {
        "observable_local_state_v5": lambda split: fast_gate.fast_observable_local_state_selection(
            package, outputs_no_self, split=split
        ),
        "observable_local_state_v5_no_strong": lambda split: fast_gate.fast_observable_local_state_selection(
            package, outputs_no_strong_self, split=split
        ),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}
        ),
    }
    library: dict[str, dict[str, pd.Series]] = {}
    for base_name, builder in base_specs.items():
        print(f"building policy candidates for {base_name}")
        base = {split: normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        library[base_name] = {"val": base["val"], "test": base["test"]}

        train = self_gate.build_feature_frame(outputs, probe, base["train"], split="train", self_model_id=self_model_id)
        val = self_gate.build_feature_frame(outputs, probe, base["val"], split="val", self_model_id=self_model_id)
        test = self_gate.build_feature_frame(outputs, probe, base["test"], split="test", self_model_id=self_model_id)
        if train.empty or val.empty or test.empty:
            continue
        for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
            x_train, x_val, x_test = self_gate.featurize(
                train, val, test, feature_view=feature_view, max_features=max_features
            )
            predictions: dict[tuple[str, float], dict[str, pd.DataFrame]] = {}
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                action_predictions: dict[str, pd.DataFrame] = {}
                for action_col in ["utility_base", "utility_self", "utility_strong"]:
                    model = Ridge(alpha=float(alpha), solver="lsqr")
                    model.fit(x_train, train[action_col].to_numpy(dtype=float))
                    action_predictions[action_col] = pd.DataFrame(
                        {
                            "val": np.asarray(model.predict(x_val), dtype=float),
                            "test": np.asarray(model.predict(x_test), dtype=float),
                        }
                    )
                predictions[(feature_view, float(alpha))] = action_predictions
            for (feature_view, alpha), pred in predictions.items():
                val_scores = pd.DataFrame(
                    {
                        "base": pred["utility_base"]["val"].to_numpy(dtype=float),
                        "self": pred["utility_self"]["val"].to_numpy(dtype=float),
                        "strong": pred["utility_strong"]["val"].to_numpy(dtype=float),
                    },
                    index=val["query_id"].astype(str),
                )
                test_scores = pd.DataFrame(
                    {
                        "base": pred["utility_base"]["test"].to_numpy(dtype=float),
                        "self": pred["utility_self"]["test"].to_numpy(dtype=float),
                        "strong": pred["utility_strong"]["test"].to_numpy(dtype=float),
                    },
                    index=test["query_id"].astype(str),
                )
                method = f"{base_name}_self_feature_ridge_{feature_view}_alpha{alpha:g}"
                library[method] = {
                    "val": self_gate.scores_to_selection(base["val"], val_scores, self_model_id=self_model_id),
                    "test": self_gate.scores_to_selection(base["test"], test_scores, self_model_id=self_model_id),
                }
    return library


def evaluate_library(
    package,
    outputs: pd.DataFrame,
    library: dict[str, dict[str, pd.Series]],
    *,
    lambda_cost: float,
    self_model_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, by_split in library.items():
        for split in ["val", "test"]:
            row = evaluate_selection(
                package,
                outputs,
                by_split[split],
                split=split,
                method=method,
                family=method_family(method),
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def run_composed_selectors(
    package,
    outputs: pd.DataFrame,
    library: dict[str, dict[str, pd.Series]],
    candidate_table: pd.DataFrame,
    *,
    lambda_cost: float,
    self_model_id: str,
    frontier_cap: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    choices: list[dict[str, Any]] = []
    composed_rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []

    global_method = select_global_method(candidate_table)
    modes: list[tuple[str, float | None, float | None]] = [
        ("benchmark_val_best", None, None),
        (f"benchmark_val_best_frontier_cap{frontier_cap:.2f}", frontier_cap, None),
    ]
    for delta in [0.02, 0.05, 0.08, 0.12]:
        modes.append((f"benchmark_val_best_shrink_global_delta{delta:.2f}", None, delta))

    for mode, cap, shrink_delta in modes:
        chosen_by_benchmark: dict[str, str] = {}
        for benchmark in sorted(query_info["benchmark"].dropna().astype(str).unique()):
            method = select_benchmark_method(
                package,
                outputs,
                library,
                benchmark=benchmark,
                global_method=global_method,
                frontier_cap=cap,
                shrink_delta=shrink_delta,
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
            chosen_by_benchmark[benchmark] = method
            choices.append(
                benchmark_choice_row(
                    package,
                    outputs,
                    library[method]["val"],
                    benchmark=benchmark,
                    method=method,
                    mode=mode,
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )

        for split in ["val", "test"]:
            selected = compose_by_benchmark(library, chosen_by_benchmark, query_info, split=split)
            method_name = mode
            row = evaluate_selection(
                package,
                outputs,
                selected,
                split=split,
                method=method_name,
                family="benchmark_stratified_selector",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
            row["chosen_methods_json"] = json.dumps(chosen_by_benchmark, sort_keys=True)
            composed_rows.append(row)
            details.append(selection_details(package, outputs, selected, split=split, method=method_name))

    for split in ["val", "test"]:
        selected = library[global_method][split]
        row = evaluate_selection(
            package,
            outputs,
            selected,
            split=split,
            method="global_val_best_policy",
            family="global_selector",
            lambda_cost=lambda_cost,
            self_model_id=self_model_id,
        )
        row["chosen_methods_json"] = json.dumps({"global": global_method}, sort_keys=True)
        composed_rows.append(row)
        details.append(selection_details(package, outputs, selected, split=split, method="global_val_best_policy"))

    # Diagnostic upper bound: choose the best candidate per benchmark using test.
    diagnostic_choices = {
        benchmark: select_benchmark_method(
            package,
            outputs,
            library,
            benchmark=benchmark,
            global_method=global_method,
            frontier_cap=None,
            shrink_delta=None,
            lambda_cost=lambda_cost,
            self_model_id=self_model_id,
            selection_split="test",
        )
        for benchmark in sorted(query_info["benchmark"].dropna().astype(str).unique())
    }
    selected = compose_by_benchmark(library, diagnostic_choices, query_info, split="test")
    row = evaluate_selection(
        package,
        outputs,
        selected,
        split="test",
        method="diagnostic_test_best_per_benchmark_policy",
        family="diagnostic_policy_selector",
        lambda_cost=lambda_cost,
        self_model_id=self_model_id,
    )
    row["chosen_methods_json"] = json.dumps(diagnostic_choices, sort_keys=True)
    composed_rows.append(row)
    details.append(selection_details(package, outputs, selected, split="test", method=row["method"]))

    return (
        pd.DataFrame(choices).sort_values(["mode", "benchmark"]),
        pd.DataFrame(composed_rows).sort_values(["split", "mean_utility"], ascending=[True, False]),
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
    )


def select_global_method(candidate_table: pd.DataFrame) -> str:
    val = candidate_table[candidate_table["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
    return str(val.iloc[0]["method"])


def select_benchmark_method(
    package,
    outputs: pd.DataFrame,
    library: dict[str, dict[str, pd.Series]],
    *,
    benchmark: str,
    global_method: str,
    frontier_cap: float | None,
    shrink_delta: float | None,
    lambda_cost: float,
    self_model_id: str,
    selection_split: str = "val",
) -> str:
    rows = []
    for method, by_split in library.items():
        row = evaluate_selection(
            package,
            outputs,
            by_split[selection_split],
            split=selection_split,
            method=method,
            family=method_family(method),
            lambda_cost=lambda_cost,
            self_model_id=self_model_id,
            benchmark=benchmark,
        )
        if row["n_queries"] <= 0:
            continue
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        return sorted(library)[0]
    feasible = table
    if frontier_cap is not None:
        capped = feasible[feasible["frontier_call_rate"].le(float(frontier_cap))]
        if not capped.empty:
            feasible = capped
    best = feasible.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]
    if shrink_delta is not None:
        global_rows = table[table["method"].eq(global_method)]
        if not global_rows.empty:
            global_utility = float(global_rows.iloc[0]["mean_utility"])
            if float(best["mean_utility"]) - global_utility < float(shrink_delta):
                return str(global_method)
    return str(best["method"])


def benchmark_choice_row(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    benchmark: str,
    method: str,
    mode: str,
    lambda_cost: float,
    self_model_id: str,
) -> dict[str, Any]:
    row = evaluate_selection(
        package,
        outputs,
        selected,
        split="val",
        method=method,
        family=method_family(method),
        lambda_cost=lambda_cost,
        self_model_id=self_model_id,
        benchmark=benchmark,
    )
    row["benchmark"] = benchmark
    row["mode"] = mode
    return row


def compose_by_benchmark(
    library: dict[str, dict[str, pd.Series]],
    chosen_by_benchmark: dict[str, str],
    query_info: pd.DataFrame,
    *,
    split: str,
) -> pd.Series:
    split_queries = query_info[query_info["split"].eq(split)]
    selected = {}
    for query_id, row in split_queries.iterrows():
        benchmark = str(row["benchmark"])
        method = chosen_by_benchmark[benchmark]
        selected[str(query_id)] = str(library[method][split].loc[str(query_id)])
    return pd.Series(selected)


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
    self_model_id: str,
    benchmark: str | None = None,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)].copy()
    if benchmark is not None:
        target = target[target["benchmark"].astype(str).eq(str(benchmark))].copy()
    if target.empty:
        return empty_row(method, split, family, lambda_cost)
    query_ids = set(target["query_id"].astype(str).unique())
    selected = normalize_selection(selected)
    selected = selected[selected.index.astype(str).isin(query_ids)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(target, selected, split=split)
    if selected_rows.empty:
        return empty_row(method, split, family, lambda_cost)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["self_action_rate"] = float(selected_rows["model_id"].eq(self_model_id).mean())
    if benchmark is not None:
        row["benchmark"] = benchmark
    return row


def empty_row(method: str, split: str, family: str, lambda_cost: float) -> dict[str, Any]:
    return {
        "method": method,
        "split": split,
        "n_queries": 0,
        "mean_quality": np.nan,
        "mean_utility": np.nan,
        "quality_oracle_mean_quality": np.nan,
        "cost_oracle_mean_utility": np.nan,
        "quality_gap_to_oracle": np.nan,
        "utility_gap_to_oracle": np.nan,
        "oracle_utility_ratio": np.nan,
        "remote_cost_total_usd": np.nan,
        "normalized_remote_cost_mean": np.nan,
        "frontier_call_rate": np.nan,
        "local_call_rate": np.nan,
        "mean_latency_s": np.nan,
        "p95_latency_s": np.nan,
        "lambda_cost": lambda_cost,
        "selected_models_json": "{}",
        "family": family,
        "strong_call_rate": np.nan,
        "self_action_rate": np.nan,
    }


def selection_details(package, outputs: pd.DataFrame, selected: pd.Series, *, split: str, method: str) -> pd.DataFrame:
    rows = package.selected_to_rows(outputs, selected, split=split).copy()
    if rows.empty:
        return rows
    rows["method"] = method
    return rows[
        [
            "method",
            "split",
            "query_id",
            "benchmark",
            "domain",
            "model_id",
            "quality_score",
            "utility",
            "normalized_remote_cost",
            "is_frontier",
            "latency_s",
        ]
    ]


def method_family(method: str) -> str:
    if method.endswith("_v5") or method.endswith("_v4") or method.endswith("_no_strong"):
        return "base"
    if "self_feature_ridge" in method:
        return "self_feature_ridge"
    return "policy"


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(out_dir: Path, candidate_table: pd.DataFrame, composed_table: pd.DataFrame) -> None:
    test_candidates = candidate_table[candidate_table["split"].eq("test")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    ).head(8)
    test_composed = composed_table[composed_table["split"].eq("test")]
    plot = pd.concat([test_composed, test_candidates], ignore_index=True)
    plot = plot.drop_duplicates("method").sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#566b8f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Benchmark-Stratified Policy Selector")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_stratified_policy_selector.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, candidate_table: pd.DataFrame, choice_table: pd.DataFrame, composed_table: pd.DataFrame) -> None:
    selected_cols = [
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
        "family",
    ]
    choice_cols = [
        "mode",
        "benchmark",
        "method",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
    ]
    lines = [
        "# Benchmark-Stratified Policy Selector",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls. It reuses cached self-consistency rows.",
        "Candidate policies are trained on train rows. Benchmark-level policy choices are selected on validation only; the test-best-per-benchmark row is diagnostic.",
        "",
        "## Composed Policies",
        "",
        "```csv",
        compact_csv(composed_table[selected_cols + [c for c in ["chosen_methods_json"] if c in composed_table.columns]], max_rows=20),
        "```",
        "",
        "## Validation Benchmark Choices",
        "",
        "```csv",
        compact_csv(choice_table[choice_cols], max_rows=40),
        "```",
        "",
        "## Best Candidate Policies On Test",
        "",
        "```csv",
        compact_csv(
            candidate_table[candidate_table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)[selected_cols]
            .head(12)
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- This tests whether the broad100 problem is partly benchmark-composition rather than a single global gate.",
        "- A positive deployable result must beat the global validation-selected policy on held-out test without using the diagnostic test-best row.",
        "- The method still uses benchmark identity, so it is a benchmark-aware router rather than a pure query-only deployable router.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
