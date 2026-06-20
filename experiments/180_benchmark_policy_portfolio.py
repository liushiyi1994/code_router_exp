from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-selected benchmark policy portfolio for broad100.")
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
        default=Path("results/controlled/broad100_benchmark_policy_portfolio"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_180")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_180")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_180")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_180")
    exp178 = load_module("experiments/178_answer_group_verifier_policy.py", "answer_group_178_for_180")
    exp179 = load_module("experiments/179_cached_adjudicator_blend_policy.py", "adjudicator_blend_179_for_180")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(target, args.benchmark_composed_choices, args.benchmark_composed_method, exp171)
    gpt_cost = mean_gpt_solver_cost(outputs)

    library = build_policy_library(
        target,
        outputs,
        exp171=exp171,
        exp172=exp172,
        exp175=exp175,
        exp177=exp177,
        exp178=exp178,
        exp179=exp179,
    )
    library_eval = evaluate_library_by_benchmark(library, outputs, gpt_cost=gpt_cost, lambda_cost=float(args.lambda_cost))
    portfolio_internal, portfolio_choices, portfolio_maps = evaluate_portfolios(
        library,
        library_eval,
        target,
        outputs,
        exp172,
        gpt_cost=gpt_cost,
        lambda_cost=float(args.lambda_cost),
    )
    selected = selected_rows(portfolio_internal, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    portfolio_table = exp172.add_bootstrap_ci(portfolio_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    portfolio_table = portfolio_table.drop(columns=["_utility_values"], errors="ignore")
    selected = selected.drop(columns=["_utility_values"], errors="ignore")

    library_eval.to_csv(args.output_dir / "table_benchmark_policy_library_eval.csv", index=False)
    portfolio_table.to_csv(args.output_dir / "table_benchmark_policy_portfolio_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_benchmark_policy_portfolio_selected.csv", index=False)
    portfolio_choices.to_csv(args.output_dir / "table_benchmark_policy_portfolio_query_choices.csv", index=False)
    portfolio_maps.to_csv(args.output_dir / "table_benchmark_policy_portfolio_maps.csv", index=False)
    write_figure(args.output_dir, portfolio_table)
    write_memo(args.output_dir / "BENCHMARK_POLICY_PORTFOLIO_MEMO.md", args, library, portfolio_table, selected, portfolio_maps)
    print(f"Wrote benchmark policy portfolio results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_policy_library(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    exp171,
    exp172,
    exp175,
    exp177,
    exp178,
    exp179,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    priors = exp172.fit_train_priors(outputs)
    pieces.append(selector_library_choices("172", exp172.fixed_policy_functions(outputs, priors), target, rows_by_query, exp172))
    pieces.append(selector_library_choices("172", exp172.threshold_policy_functions(target, outputs, priors), target, rows_by_query, exp172))
    pieces.append(public_test_choices(target, rows_by_query, exp175, exp172, priors))
    pieces.append(candidate_ranker_choices(target, outputs, priors, exp171, exp172, exp175, exp177, exp179))
    pieces.append(answer_group_choices(target, outputs, exp178))
    pieces.append(adjudicator_blend_choices(target, outputs, priors, exp171, exp172, exp175, exp177, exp179))
    library = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    library["query_id"] = library["query_id"].astype(str)
    library["split"] = library["split"].astype(str)
    library["method"] = library["method"].astype(str)
    library["family"] = library["family"].astype(str)
    library["route_cost_usd"] = pd.to_numeric(library.get("route_cost_usd", 0.0), errors="coerce").fillna(0.0)
    library = library.drop_duplicates(["method", "split", "query_id"], keep="first")
    return library


def selector_library_choices(
    prefix: str,
    selectors: dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]],
    target: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    exp172,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for method, (family, selector) in selectors.items():
        if family == "diagnostic_oracle":
            continue
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            selected = exp172.select_actions(frame, selector, rows_by_query)
            selected["split"] = split
            selected["method"] = f"{prefix}_{method}"
            selected["family"] = family
            selected["route_cost_usd"] = 0.0
            frames.append(selected[["method", "family", "split", "query_id", "model_id", "route_cost_usd"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def public_test_choices(
    target: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    exp175,
    exp172,
    priors: dict[str, Any],
) -> pd.DataFrame:
    selectors = {**exp175.fixed_selectors(exp172, priors), **exp175.threshold_selectors_from_val(target, exp172, priors)}
    frames: list[pd.DataFrame] = []
    for method, (family, selector) in selectors.items():
        if family == "diagnostic_oracle" or method.startswith("diagnostic_"):
            continue
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            selected = exp175.select_actions(frame, rows_by_query, selector, exp172)
            selected["split"] = split
            selected["method"] = f"175_{method}"
            selected["family"] = family
            selected["route_cost_usd"] = 0.0
            frames.append(selected[["method", "family", "split", "query_id", "model_id", "route_cost_usd"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def candidate_ranker_choices(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
    exp171,
    exp172,
    exp175,
    exp177,
    exp179,
) -> pd.DataFrame:
    feature_frame, cat_cols, num_cols = exp177.build_feature_frame(outputs, target)
    base_choices = exp179.fit_base_choices(exp177, exp172, exp175, feature_frame, target, outputs, priors, cat_cols, num_cols)
    frames: list[pd.DataFrame] = []
    for method, frame in base_choices.items():
        out = frame.rename(columns={"base_model_id": "model_id"}).copy()
        out["method"] = f"177_{method}"
        out["family"] = "candidate_ranker_base"
        out["route_cost_usd"] = 0.0
        frames.append(out[["method", "family", "split", "query_id", "model_id", "route_cost_usd"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def answer_group_choices(target: pd.DataFrame, outputs: pd.DataFrame, exp178) -> pd.DataFrame:
    context = exp178.build_context(outputs, target)
    wanted = {
        "answer_group_bench_support_strong_w0_fboverall_thr0.6_localnone_frontiernone",
        "answer_group_bench_support_strong_w0_fboverall_thr0.65_localnone_frontiernone",
        "answer_group_bench_support_strong_w0_fboverall_thr0.75_localnone_frontiernone",
        "answer_group_bench_support_strong_w0_fbfrontier_thr0.65_localnone_frontiergpqa-mmlupro",
        "answer_group_bench_support_w0_fboverall_thr0.65_localnone_frontiernone",
        "answer_group_support_w0_fboverall_thr0.65_localnone_frontiernone",
    }
    frames: list[pd.DataFrame] = []
    for config in exp178.candidate_configs():
        if config.method not in wanted:
            continue
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            selected = exp178.select_actions(frame, config, context)
            selected["split"] = split
            selected["method"] = f"178_{config.method}"
            selected["family"] = "answer_group_verifier_policy"
            selected["route_cost_usd"] = 0.0
            frames.append(selected[["method", "family", "split", "query_id", "model_id", "route_cost_usd"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def adjudicator_blend_choices(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
    exp171,
    exp172,
    exp175,
    exp177,
    exp179,
) -> pd.DataFrame:
    feature_frame, cat_cols, num_cols = exp177.build_feature_frame(outputs, target)
    base_choices = exp179.fit_base_choices(exp177, exp172, exp175, feature_frame, target, outputs, priors, cat_cols, num_cols)
    adjudicators = exp179.load_adjudicators()
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frames: list[pd.DataFrame] = []
    for base_method, base in base_choices.items():
        for source, adjudicator in adjudicators.items():
            for threshold in exp179.THRESHOLDS:
                for benchmarks in exp179.BENCHMARK_SETS:
                    method = f"179_{exp179.blend_method(base_method, source, float(threshold), benchmarks)}"
                    for split in ["val", "test"]:
                        frame = target[target["split"].astype(str).eq(split)].copy()
                        split_base = base[base["split"].eq(split)].copy()
                        choice = exp179.apply_override(split_base, adjudicator, frame, rows_by_query, exp172, float(threshold), benchmarks)
                        choice["method"] = method
                        choice["family"] = "cached_adjudicator_blend"
                        frames.append(choice[["method", "family", "split", "query_id", "model_id", "route_cost_usd"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_library_by_benchmark(
    library: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    gpt_cost: float,
    lambda_cost: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    merged = library.merge(outputs, on=["query_id", "model_id"], how="left", suffixes=("", "_out"))
    merged = merged[merged["split"].astype(str).eq(merged["split_out"].astype(str))]
    merged["route_cost_norm"] = pd.to_numeric(merged["route_cost_usd"], errors="coerce").fillna(0.0) / max(gpt_cost, 1e-12)
    merged["utility_with_route_cost"] = merged["quality_score"].astype(float) - float(lambda_cost) * (
        merged["normalized_remote_cost"].astype(float) + merged["route_cost_norm"].astype(float)
    )
    for (method, family, split, benchmark), group in merged.groupby(["method", "family", "split", "benchmark"], sort=False):
        rows.append(
            {
                "method": method,
                "family": family,
                "split": split,
                "benchmark": benchmark,
                "n_queries": int(group["query_id"].nunique()),
                "mean_quality": float(group["quality_score"].mean()),
                "mean_utility": float(group["utility"].mean()),
                "mean_utility_with_route_cost": float(group["utility_with_route_cost"].mean()),
                "normalized_cost_mean": float(group["normalized_remote_cost"].mean()),
                "frontier_call_rate": float(group["is_frontier"].astype(bool).mean()),
                "strong_or_frontier_call_rate": float(group["model_id"].astype(str).isin(set(["qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local", "gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve"])).mean()),
                "route_cost_norm_mean": float(group["route_cost_norm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_portfolios(
    library: pd.DataFrame,
    library_eval: pd.DataFrame,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    exp172,
    *,
    gpt_cost: float,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    choices: list[pd.DataFrame] = []
    maps: list[pd.DataFrame] = []
    deployable_eval = library_eval[~library_eval["family"].astype(str).str.startswith("diagnostic")].copy()
    val_eval = deployable_eval[deployable_eval["split"].eq("val")].copy()
    test_eval = library_eval[library_eval["split"].eq("test")].copy()
    for objective in ["mean_utility", "mean_utility_with_route_cost"]:
        for cap in [0.25, 0.35, 0.40, 0.45, 1.00]:
            mapping = select_benchmark_mapping(val_eval, objective=objective, frontier_cap=cap)
            method = f"benchmark_portfolio_val_{objective}_frontiercap{cap:g}"
            add_portfolio_rows(method, "validation_benchmark_portfolio", mapping, library, target, outputs, exp172, gpt_cost, lambda_cost, rows, choices)
            maps.append(mapping.assign(portfolio_method=method, selection_split="val", objective=objective, frontier_cap=cap))
    for objective in ["mean_utility", "mean_utility_with_route_cost"]:
        mapping = select_benchmark_mapping(test_eval, objective=objective, frontier_cap=1.00)
        method = f"benchmark_portfolio_test_oracle_{objective}"
        add_portfolio_rows(method, "diagnostic_test_picked_portfolio", mapping, library, target, outputs, exp172, gpt_cost, lambda_cost, rows, choices)
        maps.append(mapping.assign(portfolio_method=method, selection_split="test", objective=objective, frontier_cap=1.00))
    return (
        pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False]),
        pd.concat(choices, ignore_index=True) if choices else pd.DataFrame(),
        pd.concat(maps, ignore_index=True) if maps else pd.DataFrame(),
    )


def select_benchmark_mapping(eval_table: pd.DataFrame, *, objective: str, frontier_cap: float) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for benchmark, group in eval_table.groupby("benchmark", sort=False):
        candidates = group[group["frontier_call_rate"] <= float(frontier_cap)].copy()
        if candidates.empty:
            candidates = group.copy()
        best = candidates.sort_values([objective, "mean_quality", "normalized_cost_mean"], ascending=[False, False, True]).head(1)
        rows.append(best.iloc[0])
    return pd.DataFrame(rows)[
        [
            "benchmark",
            "method",
            "family",
            "n_queries",
            "mean_quality",
            "mean_utility",
            "mean_utility_with_route_cost",
            "frontier_call_rate",
            "route_cost_norm_mean",
        ]
    ].reset_index(drop=True)


def add_portfolio_rows(
    method: str,
    family: str,
    mapping: pd.DataFrame,
    library: pd.DataFrame,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    exp172,
    gpt_cost: float,
    lambda_cost: float,
    rows: list[dict[str, Any]],
    choices: list[pd.DataFrame],
) -> None:
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    for split in ["val", "test"]:
        pieces: list[pd.DataFrame] = []
        for row in mapping.itertuples(index=False):
            query_ids = set(
                target[(target["split"].astype(str).eq(split)) & (target["benchmark"].astype(str).eq(str(row.benchmark)))]["query_id"].astype(str)
            )
            selected = library[
                library["split"].astype(str).eq(split)
                & library["method"].astype(str).eq(str(row.method))
                & library["query_id"].astype(str).isin(query_ids)
            ].copy()
            pieces.append(selected)
        choice = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        choice["portfolio_method"] = method
        selected_rows = choice[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
        selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
        frame = target[target["split"].astype(str).eq(split)].copy()
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
        route_cost_norm = pd.to_numeric(choice.get("route_cost_usd", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float) / max(gpt_cost, 1e-12)
        selected_quality = selected_rows["quality_score"].to_numpy(dtype=float)
        selected_cost = selected_rows["normalized_remote_cost"].to_numpy(dtype=float)
        route_utilities = selected_quality - float(lambda_cost) * (selected_cost + route_cost_norm)
        row["route_cost_norm_mean"] = float(np.mean(route_cost_norm)) if len(route_cost_norm) else 0.0
        row["mean_utility_with_route_cost"] = float(np.mean(route_utilities)) if len(route_utilities) else np.nan
        row["oracle_utility_ratio_with_route_cost"] = float(row["mean_utility_with_route_cost"] / max(float(row["cost_oracle_mean_utility"]), 1e-12))
        row["portfolio_map_json"] = json.dumps(dict(zip(mapping["benchmark"].astype(str), mapping["method"].astype(str))), sort_keys=True)
        row["_utility_values_routecost"] = route_utilities.tolist()
        rows.append(row)
        choices.append(choice)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility", "mean_utility_with_route_cost"]:
        val = table[(table["split"].eq("val")) & table["family"].eq("validation_benchmark_portfolio")].copy()
        if val.empty:
            continue
        best = val.sort_values([objective, "frontier_call_rate"], ascending=[False, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule=f"val_best_{objective}"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule=f"val_best_{objective}_test"))
    diagnostic = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(8)
    if not diagnostic.empty:
        rows.append(diagnostic.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(
        table[["method", "split", "_utility_values"]],
        on=["method", "split"],
        how="left",
    )
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)


def mean_gpt_solver_cost(outputs: pd.DataFrame) -> float:
    return max(
        float(outputs[outputs["model_id"].astype(str).eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5f7a68")
    ax.set_xlabel("Held-out test selected-solver utility")
    ax.set_title("Benchmark Policy Portfolio")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_benchmark_policy_portfolio_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, library: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame, maps: pd.DataFrame) -> None:
    cols = [
        "method",
        "split",
        "selection_rule",
        "family",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_route_cost",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_route_cost",
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "route_cost_norm_mean",
    ]
    lines = [
        "# Benchmark Policy Portfolio",
        "",
        "This cached experiment recomputes validation/test query-level choices for the strongest existing policy families, then selects one policy per benchmark on validation. It makes no GPT, Gemini, Claude, vLLM, or local model calls.",
        "",
        "The portfolio is a method-search diagnostic for benchmark-specific action identity. Test-picked portfolio rows are explicitly diagnostic only.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/180_benchmark_policy_portfolio.py",
        (
            "PYTHONPATH=src python experiments/180_benchmark_policy_portfolio.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Library Size",
        "",
        f"- Candidate policy methods: `{library['method'].nunique()}`",
        f"- Query-choice rows: `{len(library)}`",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]),
        "",
        "## Best Held-Out Rows",
        "",
        markdown_table(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)[[column for column in cols if column in table.columns]]),
        "",
        "## Validation Portfolio Maps",
        "",
        markdown_table(maps[maps["selection_split"].eq("val")].head(80)),
        "",
        "## Interpretation",
        "",
        "- If the validation-selected portfolio improves over individual policies, the bottleneck is partly benchmark-specific policy composition.",
        "- If it still misses target, the remaining issue is not just global threshold selection; the policy library lacks a strong enough task-specific checker.",
        "- Test-picked portfolio rows estimate the ceiling of this library and are not deployable claims.",
        "",
        "## Artifacts",
        "",
        f"- Library benchmark eval: `{args.output_dir / 'table_benchmark_policy_library_eval.csv'}`",
        f"- All portfolio table: `{args.output_dir / 'table_benchmark_policy_portfolio_all.csv'}`",
        f"- Selected portfolio table: `{args.output_dir / 'table_benchmark_policy_portfolio_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_benchmark_policy_portfolio_query_choices.csv'}`",
        f"- Portfolio maps: `{args.output_dir / 'table_benchmark_policy_portfolio_maps.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_benchmark_policy_portfolio_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
