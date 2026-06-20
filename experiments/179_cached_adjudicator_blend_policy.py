from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ADJUDICATOR_SOURCES = {
    "gpt_frontier": Path("results/controlled/broad100_answer_adjudicator/table_broad_answer_adjudications.csv"),
    "gemini_frontier": Path("results/controlled/broad100_answer_adjudicator_gemini/table_broad_answer_adjudications.csv"),
    "gpt_local": Path("results/controlled/broad100_answer_adjudicator_gpt_local_only/table_broad_answer_adjudications.csv"),
    "medium_frontier": Path("results/controlled/broad100_answer_adjudicator_medium/table_broad_answer_adjudications.csv"),
}
THRESHOLDS = (0.0, 0.5, 0.75, 0.9, 0.95, 0.98)
BENCHMARK_SETS = (
    ("gpqa",),
    ("mmlupro",),
    ("gpqa", "mmlupro"),
    ("gpqa", "mmlupro", "bbh"),
    ("gpqa", "mmlupro", "math500"),
    ("gpqa", "mmlupro", "livemathbench"),
    ("gpqa", "mmlupro", "math500", "gsm8k"),
)


@dataclass(frozen=True)
class BasePolicy:
    model_name: str
    action_mode: str
    cost_penalty: float

    @property
    def method(self) -> str:
        return f"{self.model_name}_{self.action_mode}_pen{self.cost_penalty:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached answer-adjudicator override over practical broad100 ranker policies.")
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
        default=Path("results/controlled/broad100_cached_adjudicator_blend_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_179")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_179")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_179")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_179")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    priors = exp172.fit_train_priors(outputs)
    feature_frame, cat_cols, num_cols = exp177.build_feature_frame(outputs, target)
    base_choices = fit_base_choices(exp177, exp172, exp175, feature_frame, target, outputs, priors, cat_cols, num_cols)
    adjudicators = load_adjudicators()

    policy_internal, query_choices = evaluate_blends(
        base_choices,
        adjudicators,
        target,
        outputs,
        exp172,
        lambda_cost=float(args.lambda_cost),
    )
    selected = selected_rows(policy_internal, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    policy_table = exp172.add_bootstrap_ci(policy_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    policy_table = policy_table.drop(columns=["_utility_values"], errors="ignore")
    selected = selected.drop(columns=["_utility_values"], errors="ignore")

    policy_table.to_csv(args.output_dir / "table_cached_adjudicator_blend_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_cached_adjudicator_blend_selected.csv", index=False)
    query_choice_rows(selected, query_choices, outputs).to_csv(
        args.output_dir / "table_cached_adjudicator_blend_query_choices.csv",
        index=False,
    )
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "CACHED_ADJUDICATOR_BLEND_POLICY_MEMO.md", args, policy_table, selected)
    print(f"Wrote cached adjudicator blend policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def base_policies() -> list[BasePolicy]:
    return [
        BasePolicy("hgb_l1", "gate_rank_localplus", 0.25),
        BasePolicy("hgb_l1", "gate_rank_localplus", 0.35),
        BasePolicy("hgb_l2", "gate_rank_localplus", 0.25),
    ]


def fit_base_choices(
    exp177,
    exp172,
    exp175,
    feature_frame: pd.DataFrame,
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    priors: dict[str, Any],
    cat_cols: list[str],
    num_cols: list[str],
) -> dict[str, pd.DataFrame]:
    rows_by_query = exp177.rows_by_query_map(outputs)
    train = feature_frame[feature_frame["split"].astype(str).eq("train")]
    choices: dict[str, pd.DataFrame] = {}
    for base in base_policies():
        config = exp177.RankerConfig(
            model_name=base.model_name,
            action_mode=base.action_mode,
            cost_penalty=base.cost_penalty,
        )
        pipe = exp177.make_pipeline(config.model_name, cat_cols, num_cols)
        pipe.fit(train[cat_cols + num_cols], train["quality_score"].astype(float))
        split_frames: list[pd.DataFrame] = []
        for split in ["val", "test"]:
            candidates = feature_frame[feature_frame["split"].astype(str).eq(split)].copy()
            candidates = exp177.add_predictions(candidates, pipe, config.cost_penalty, cat_cols, num_cols)
            frame = target[target["split"].astype(str).eq(split)].copy()
            selected = exp177.select_actions(frame, candidates, config, rows_by_query, priors, exp172=exp172, exp175=exp175)
            selected["split"] = split
            selected["base_method"] = config.method
            selected = selected.rename(columns={"model_id": "base_model_id"})
            split_frames.append(selected[["query_id", "split", "base_method", "base_model_id"]])
        choices[config.method] = pd.concat(split_frames, ignore_index=True)
    return choices


def load_adjudicators() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for source, path in ADJUDICATOR_SOURCES.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["query_id"] = frame["query_id"].astype(str)
        frame["split"] = frame["split"].astype(str)
        frame["benchmark"] = frame["benchmark"].astype(str).str.lower()
        frame["selected_model"] = frame["selected_model"].astype(str)
        frame["selected_confidence"] = pd.to_numeric(frame["selected_confidence"], errors="coerce").fillna(0.0)
        frame["adjudicator_cost"] = pd.to_numeric(frame["adjudicator_cost"], errors="coerce").fillna(0.0)
        tables[source] = frame
    return tables


def evaluate_blends(
    base_choices: dict[str, pd.DataFrame],
    adjudicators: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    exp172,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    gpt_cost = max(
        float(outputs[outputs["model_id"].astype(str).eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )
    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for base_method, base in base_choices.items():
        base = base.copy()
        for split in ["val", "test"]:
            split_base = base[base["split"].eq(split)].copy()
            split_base["model_id"] = split_base["base_model_id"]
            split_base["route_cost_usd"] = 0.0
            split_base["overrode_base"] = False
            frame = target[target["split"].astype(str).eq(split)].copy()
            rows.append(
                evaluate_choice_frame(
                    split_base,
                    outputs,
                    frame,
                    exp172,
                    frontiers,
                    method=base_method,
                    family="candidate_ranker_base",
                    split=split,
                    lambda_cost=lambda_cost,
                    gpt_cost=gpt_cost,
                    route_cost_norm=0.0,
                    override_rate=0.0,
                    adjudicator_source="none",
                    threshold=np.nan,
                    benchmarks=(),
                )
            )
        for source, adjudicator in adjudicators.items():
            for threshold in THRESHOLDS:
                for benchmarks in BENCHMARK_SETS:
                    method = blend_method(base_method, source, threshold, benchmarks)
                    for split in ["val", "test"]:
                        frame = target[target["split"].astype(str).eq(split)].copy()
                        split_base = base[base["split"].eq(split)].copy()
                        choice = apply_override(split_base, adjudicator, frame, rows_by_query, exp172, threshold, benchmarks)
                        route_cost_norm = float(choice["route_cost_usd"].mean() / gpt_cost)
                        row = evaluate_choice_frame(
                            choice,
                            outputs,
                            frame,
                            exp172,
                            frontiers,
                            method=method,
                            family="cached_adjudicator_blend",
                            split=split,
                            lambda_cost=lambda_cost,
                            gpt_cost=gpt_cost,
                            route_cost_norm=route_cost_norm,
                            override_rate=float(choice["overrode_base"].mean()),
                            adjudicator_source=source,
                            threshold=float(threshold),
                            benchmarks=benchmarks,
                        )
                        rows.append(row)
                        if split == "test":
                            detail_frames.append(choice.assign(method=method, family="cached_adjudicator_blend"))
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return table, details


def apply_override(
    base: pd.DataFrame,
    adjudicator: pd.DataFrame,
    target: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    exp172,
    threshold: float,
    benchmarks: tuple[str, ...],
) -> pd.DataFrame:
    meta = target[["query_id", "benchmark"]].copy()
    meta["query_id"] = meta["query_id"].astype(str)
    meta["benchmark"] = meta["benchmark"].astype(str).str.lower()
    out = base.merge(meta, on="query_id", how="left")
    adj = adjudicator[adjudicator["benchmark"].isin(set(benchmarks))].copy()
    adj = adj[adj["selected_confidence"] >= float(threshold)].copy()
    adj = adj[["query_id", "split", "selected_model", "selected_confidence", "adjudicator_cost"]]
    out = out.merge(adj, on=["query_id", "split"], how="left")
    selected: list[str] = []
    overridden: list[bool] = []
    route_costs: list[float] = []
    for row in out.itertuples(index=False):
        query_id = str(row.query_id)
        actions = rows_by_query[query_id]
        base_model = str(row.base_model_id)
        adj_model = str(getattr(row, "selected_model"))
        can_override = (
            str(row.benchmark) in benchmarks
            and adj_model
            and adj_model != "nan"
            and adj_model in actions
            and exp172.is_action_available(actions, adj_model)
        )
        if can_override:
            selected.append(adj_model)
            overridden.append(adj_model != base_model)
            route_costs.append(float(getattr(row, "adjudicator_cost", 0.0) or 0.0))
        else:
            selected.append(base_model)
            overridden.append(False)
            route_costs.append(0.0)
    out["model_id"] = selected
    out["overrode_base"] = overridden
    out["route_cost_usd"] = route_costs
    return out[
        [
            "query_id",
            "split",
            "benchmark",
            "base_method",
            "base_model_id",
            "model_id",
            "selected_confidence",
            "route_cost_usd",
            "overrode_base",
        ]
    ]


def evaluate_choice_frame(
    choice: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    frontiers: set[str],
    *,
    method: str,
    family: str,
    split: str,
    lambda_cost: float,
    gpt_cost: float,
    route_cost_norm: float,
    override_rate: float,
    adjudicator_source: str,
    threshold: float,
    benchmarks: tuple[str, ...],
) -> dict[str, Any]:
    selected = choice[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
    selected = selected[selected["split"].astype(str).eq(split)].copy()
    row = exp172.evaluate_selected_rows(
        method,
        family,
        split,
        selected,
        outputs,
        target=target,
        frontiers=frontiers,
        lambda_cost=lambda_cost,
    )
    row.update(
        {
            "base_method": choice["base_method"].iloc[0] if "base_method" in choice and not choice.empty else "",
            "adjudicator_source": adjudicator_source,
            "selector_confidence_threshold": threshold,
            "override_benchmarks": ",".join(benchmarks),
            "override_rate": float(override_rate),
            "route_cost_norm_mean": float(route_cost_norm),
            "route_cost_usd_total": float(choice.get("route_cost_usd", pd.Series(dtype=float)).sum()),
        }
    )
    row["mean_utility_with_route_cost"] = float(row["mean_quality"] - float(lambda_cost) * (row["normalized_cost_mean"] + route_cost_norm))
    row["oracle_utility_ratio_with_route_cost"] = float(
        row["mean_utility_with_route_cost"] / max(float(row["cost_oracle_mean_utility"]), 1e-12)
    )
    return row


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cap in [0.40, 0.45]:
        val = table[(table["split"].eq("val")) & (table["frontier_call_rate"] <= cap)].copy()
        if not val.empty:
            best = val.sort_values(["mean_utility", "normalized_cost_mean"], ascending=[False, True]).head(1)
            method = str(best.iloc[0]["method"])
            rows.append(best.assign(selection_rule=f"val_best_solver_utility_frontier_le_{cap:g}"))
            rows.append(
                table[table["split"].eq("test") & table["method"].eq(method)]
                .copy()
                .assign(selection_rule=f"val_best_solver_utility_frontier_le_{cap:g}_test")
            )
            best_route = val.sort_values(["mean_utility_with_route_cost", "normalized_cost_mean"], ascending=[False, True]).head(1)
            route_method = str(best_route.iloc[0]["method"])
            rows.append(best_route.assign(selection_rule=f"val_best_routecost_utility_frontier_le_{cap:g}"))
            rows.append(
                table[table["split"].eq("test") & table["method"].eq(route_method)]
                .copy()
                .assign(selection_rule=f"val_best_routecost_utility_frontier_le_{cap:g}_test")
            )
    val_all = table[table["split"].eq("val")].sort_values(["mean_utility", "normalized_cost_mean"], ascending=[False, True])
    if not val_all.empty:
        method = str(val_all.iloc[0]["method"])
        rows.append(val_all.head(1).copy().assign(selection_rule="val_best_solver_utility_unconstrained"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule="val_best_solver_utility_unconstrained_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(
        table[["method", "split", "_utility_values"]],
        on=["method", "split"],
        how="left",
    )
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)


def query_choice_rows(selected: pd.DataFrame, choices: pd.DataFrame, outputs: pd.DataFrame) -> pd.DataFrame:
    if selected.empty or choices.empty:
        return pd.DataFrame()
    methods = selected[selected["split"].eq("test")]["method"].astype(str).unique().tolist()
    if not methods:
        return pd.DataFrame()
    rows = choices[choices["method"].astype(str).isin(methods)].copy()
    if rows.empty:
        return rows
    details = rows.merge(outputs, on=["query_id", "model_id"], how="left", suffixes=("", "_selected"))
    columns = [
        "method",
        "family",
        "query_id",
        "query_text",
        "benchmark",
        "metric",
        "base_model_id",
        "model_id",
        "quality_score",
        "utility",
        "normalized_remote_cost",
        "is_frontier",
        "parsed_answer",
        "selected_confidence",
        "route_cost_usd",
        "overrode_base",
    ]
    return details[[column for column in columns if column in details.columns]].sort_values(["method", "benchmark", "query_id"])


def blend_method(base_method: str, source: str, threshold: float, benchmarks: tuple[str, ...]) -> str:
    return f"{base_method}_override_{source}_thr{threshold:g}_bench{'-'.join(benchmarks)}"


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#675f8c")
    ax.set_xlabel("Held-out test selected-solver utility")
    ax.set_title("Cached Adjudicator Blend Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cached_adjudicator_blend_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "split",
        "selection_rule",
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
        "override_rate",
        "route_cost_norm_mean",
        "adjudicator_source",
        "selector_confidence_threshold",
        "override_benchmarks",
    ]
    test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)
    lines = [
        "# Cached Adjudicator Blend Policy",
        "",
        "This experiment uses cached answer-adjudicator tables as a cheap-verifier proxy on top of the practical candidate-ranker policies. It makes no GPT, Gemini, Claude, vLLM, or local model calls.",
        "",
        "Route cost is reported separately as adjudicator cost normalized by the mean GPT solver cost and charged in `mean_utility_with_route_cost`.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/179_cached_adjudicator_blend_policy.py",
        (
            "PYTHONPATH=src python experiments/179_cached_adjudicator_blend_policy.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(test.head(16)[[column for column in cols if column in test.columns]]),
        "",
        "## Interpretation",
        "",
        "- Cached adjudicator overrides are only a partial diagnostic signal.",
        "- Validation-selected rows do not reach the 97% oracle-utility target or the within-3-point quality target.",
        "- The best held-out diagnostic improvements are small and test-picked, so they should guide the next verifier design rather than support a deployed-method claim.",
        "- Generic answer adjudication is still weaker than a task-specific checker for GPQA, MMLUPro, AIME, and exact math.",
        "",
        "## Artifacts",
        "",
        f"- All policy table: `{args.output_dir / 'table_cached_adjudicator_blend_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_cached_adjudicator_blend_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_cached_adjudicator_blend_query_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_cached_adjudicator_blend_utility.pdf'}`",
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
