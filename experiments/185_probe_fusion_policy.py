from __future__ import annotations

import argparse
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRICT_THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]
BENCHMARK_SETS: list[tuple[str, ...]] = [("gpqa", "mmlupro"), ("gpqa",), ("mmlupro",)]
FUSION_MODES = ["strict_priority", "strict_repair_base", "strict_veto_local"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="No-call fusion of local-safe and strict-verifier support policies.")
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
        "--local-policy-table",
        type=Path,
        default=Path("results/controlled/broad100_local_safe_gain_gate/table_local_safe_gain_policy_all.csv"),
    )
    parser.add_argument(
        "--strict-verifier",
        type=Path,
        default=Path("results/controlled/broad100_strict_mcq_verifier_policy/table_strict_mcq_verifier_outputs.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_probe_fusion_policy"))
    parser.add_argument("--max-local-specs", type=int, default=24)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_185")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_185")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_185")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_185")
    exp183 = load_module("experiments/183_local_safe_gain_gate.py", "local_safe_183_for_185")
    exp184 = load_module("experiments/184_strict_mcq_verifier_policy.py", "strict_mcq_184_for_185")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    rows_by_query = exp177.rows_by_query_map(outputs)
    base_choices = exp183.build_base_choices(exp177, exp172, exp175, outputs, target, rows_by_query)
    feature_frame = exp183.build_local_safe_features(base_choices, target, outputs, rows_by_query)
    scored = score_local_specs(feature_frame, exp183)
    local_specs = select_local_specs(args.local_policy_table, max_specs=int(args.max_local_specs))
    verifier = pd.read_csv(args.strict_verifier)
    policy_table, query_choices = evaluate_fusions(
        scored,
        local_specs,
        verifier,
        outputs,
        target,
        exp172,
        exp183,
        exp184,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = selected_rows(policy_table, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    query_choices = filter_query_choices(query_choices, policy_table, selected)

    scored.to_csv(args.output_dir / "table_probe_fusion_features.csv", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_probe_fusion_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_probe_fusion_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_probe_fusion_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "PROBE_FUSION_POLICY_MEMO.md", args, local_specs, verifier, policy_table, selected)
    print(f"Wrote probe fusion results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def score_local_specs(feature_frame: pd.DataFrame, exp183) -> pd.DataFrame:
    cat_cols = exp183.categorical_feature_columns()
    num_cols = exp183.numeric_feature_columns()
    train = feature_frame[feature_frame["split"].astype(str).eq("train")].copy()
    predictors = exp183.fit_predictors(train, cat_cols, num_cols)
    scored = feature_frame.copy()
    for name, pipe in predictors.items():
        scored[f"pred_{name}"] = pipe.predict(scored[cat_cols + num_cols])
    return scored


def select_local_specs(path: Path, *, max_specs: int) -> list[tuple[str, str, float]]:
    table = pd.read_csv(path)
    val = table[
        table["split"].astype(str).eq("val") & table["family"].astype(str).eq("local_safe_gain_gate")
    ].copy()
    val = val.sort_values(["mean_utility", "frontier_call_rate", "override_rate"], ascending=[False, True, True])
    capped = val[val["frontier_call_rate"].astype(float) <= 0.30].copy()
    picked = pd.concat([val.head(max_specs), capped.head(max(4, max_specs // 3))], ignore_index=True)
    specs: list[tuple[str, str, float]] = []
    seen: set[str] = set()
    for method in picked["method"].astype(str).tolist():
        parsed = parse_local_method(method)
        if parsed is None or method in seen:
            continue
        seen.add(method)
        specs.append(parsed)
    return specs


def parse_local_method(method: str) -> tuple[str, str, float] | None:
    match = re.fullmatch(r"(pred_[A-Za-z0-9]+)_thr(-?[0-9.]+)", method)
    if not match:
        return None
    return method, match.group(1), float(match.group(2))


def evaluate_fusions(
    scored: pd.DataFrame,
    local_specs: list[tuple[str, str, float]],
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    exp183,
    exp184,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    available = {str(query_id): set(group["model_id"].astype(str)) for query_id, group in outputs.groupby("query_id", sort=False)}
    verifier_map = verifier.set_index("query_id").to_dict("index")
    gpt_cost = exp184.mean_gpt_cost(outputs)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []

    methods: list[dict[str, Any]] = [{"method": "base_candidate_ranker", "family": "reference", "kind": "base"}]
    for local_method, pred_col, threshold in local_specs:
        methods.append(
            {
                "method": f"local_{local_method}",
                "family": "local_safe_reference",
                "kind": "local",
                "local_method": local_method,
                "pred_col": pred_col,
                "local_threshold": threshold,
            }
        )
        for benchmarks in BENCHMARK_SETS:
            bench_name = "-".join(benchmarks)
            for strict_threshold in STRICT_THRESHOLDS:
                for mode in FUSION_MODES:
                    methods.append(
                        {
                            "method": f"fusion_{mode}_{local_method}_strict{strict_threshold:g}_bench{bench_name}",
                            "family": "probe_fusion",
                            "kind": "fusion",
                            "mode": mode,
                            "local_method": local_method,
                            "pred_col": pred_col,
                            "local_threshold": threshold,
                            "strict_threshold": strict_threshold,
                            "benchmarks": set(benchmarks),
                        }
                    )
    for benchmarks in BENCHMARK_SETS:
        bench_name = "-".join(benchmarks)
        for strict_threshold in STRICT_THRESHOLDS:
            methods.append(
                {
                    "method": f"strict_support_only_thr{strict_threshold:g}_bench{bench_name}",
                    "family": "strict_support_reference",
                    "kind": "strict",
                    "strict_threshold": strict_threshold,
                    "benchmarks": set(benchmarks),
                }
            )

    for spec in methods:
        for split in ["val", "test"]:
            split_frame = scored[scored["split"].astype(str).eq(split)].copy()
            target_split = target[target["split"].astype(str).eq(split)].copy()
            choices = choose_fusion_actions(split_frame, spec, exp183, verifier_map, available)
            selected_rows = choices[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                str(spec["method"]),
                str(spec["family"]),
                split,
                selected_rows,
                outputs,
                target=target_split,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            probe_cost = strict_probe_norm_cost(choices, verifier_map, gpt_cost)
            route_util = selected_rows["quality_score"].to_numpy(dtype=float) - float(lambda_cost) * (
                selected_rows["normalized_remote_cost"].to_numpy(dtype=float) + probe_cost
            )
            row.update(
                {
                    "probe_call_rate": float(choices["verifier_probed"].mean()) if not choices.empty else 0.0,
                    "local_override_rate": float(choices["local_overrode_base"].mean()) if not choices.empty else 0.0,
                    "strict_override_rate": float(choices["strict_overrode_local"].mean()) if not choices.empty else 0.0,
                    "override_rate": float(choices["overrode_base"].mean()) if not choices.empty else 0.0,
                    "extra_probe_norm_cost_mean": float(probe_cost.mean()) if len(probe_cost) else 0.0,
                    "mean_utility_with_probe_cost": float(route_util.mean()) if len(route_util) else np.nan,
                    "oracle_utility_ratio_with_probe_cost": float(
                        route_util.mean() / max(float(row["cost_oracle_mean_utility"]), 1e-12)
                    )
                    if len(route_util)
                    else np.nan,
                }
            )
            rows.append(row)
            if split == "test":
                details.append(
                    selected_rows[
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
                    ]
                    .merge(choices, on="query_id", how="left")
                    .assign(method=str(spec["method"]), family=str(spec["family"]))
                )
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    details_frame = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return table, details_frame


def choose_fusion_actions(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    exp183,
    verifier_map: dict[str, dict[str, Any]],
    available: dict[str, set[str]],
) -> pd.DataFrame:
    if spec["kind"] in {"local", "fusion"}:
        local_choices = exp183.choose_actions(
            frame,
            pred_col=str(spec["pred_col"]),
            threshold=float(spec["local_threshold"]),
            family="local_safe_gain_gate",
        )
    else:
        local_choices = exp183.choose_actions(frame, pred_col="", threshold=math.nan, family="reference")

    rows: list[dict[str, Any]] = []
    for item in local_choices.itertuples(index=False):
        query_id = str(item.query_id)
        local_model = str(item.model_id)
        base_model = str(item.base_model_id)
        selected = local_model
        probed = False
        supported = ""
        confidence = 0.0
        strict_overrode = False
        if spec["kind"] in {"strict", "fusion"}:
            item_map = verifier_map.get(query_id)
            if item_map is not None and str(item_map.get("status", "")) == "success":
                benchmark = str(item_map.get("benchmark", "")).lower()
                benchmarks = set(spec.get("benchmarks", set()))
                probed = not benchmarks or benchmark in benchmarks
                confidence = float(item_map.get("verifier_confidence", 0.0) or 0.0)
                supported = str(item_map.get("supported_model", "") or "")
                can_use_support = (
                    probed
                    and confidence >= float(spec.get("strict_threshold", math.inf))
                    and supported in available.get(query_id, set())
                    and supported.upper() != "NONE"
                )
                if can_use_support:
                    mode = str(spec.get("mode", "strict_priority"))
                    if spec["kind"] == "strict" or mode == "strict_priority":
                        selected = supported
                    elif mode == "strict_repair_base" and local_model == base_model:
                        selected = supported
                    elif mode == "strict_veto_local" and local_model != base_model:
                        selected = supported
                    strict_overrode = selected != local_model
        rows.append(
            {
                "query_id": query_id,
                "model_id": selected,
                "base_model_id": base_model,
                "local_model_id": local_model,
                "consensus_model": getattr(item, "consensus_model", ""),
                "local_overrode_base": local_model != base_model,
                "strict_overrode_local": strict_overrode,
                "overrode_base": selected != base_model,
                "verifier_probed": probed,
                "verifier_confidence": confidence,
                "supported_model": supported,
            }
        )
    return pd.DataFrame(rows)


def strict_probe_norm_cost(
    choices: pd.DataFrame,
    verifier_map: dict[str, dict[str, Any]],
    gpt_cost: float,
) -> np.ndarray:
    costs: list[float] = []
    for row in choices.itertuples(index=False):
        if not bool(row.verifier_probed):
            costs.append(0.0)
            continue
        item = verifier_map.get(str(row.query_id))
        if item is None:
            costs.append(0.0)
            continue
        costs.append(float(item.get("cost_total_usd", 0.0) or 0.0) / max(gpt_cost, 1e-12))
    return np.asarray(costs, dtype=float)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility", "mean_utility_with_probe_cost"]:
        val = table[table["split"].eq("val") & table["family"].isin(["probe_fusion", "strict_support_reference"])].copy()
        if val.empty:
            continue
        best = val.sort_values([objective, "frontier_call_rate", "override_rate"], ascending=[False, True, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule=f"val_best_{objective}"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule=f"val_best_{objective}_test"))
    reference = table[table["split"].eq("test") & table["family"].eq("reference")]
    if not reference.empty:
        rows.append(reference.assign(selection_rule="reference_test"))
    local_ref = table[table["split"].eq("test") & table["family"].eq("local_safe_reference")]
    if not local_ref.empty:
        rows.append(local_ref.sort_values(["mean_utility", "mean_quality"], ascending=False).head(5).assign(selection_rule="local_reference_test"))
    top_test = (
        table[table["split"].eq("test") & table["family"].ne("reference")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(16)
    )
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    with_values = table[["method", "split", "_utility_values"]]
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(
        with_values,
        on=["method", "split"],
        how="left",
    )
    selected = exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)
    return selected.drop(columns=["_utility_values"], errors="ignore")


def filter_query_choices(query_choices: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if query_choices.empty:
        return query_choices
    selected_methods = set(selected.get("method", pd.Series(dtype=str)).astype(str).tolist())
    top_methods = set(
        table[table["split"].eq("test")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(24)["method"]
        .astype(str)
        .tolist()
    )
    keep = selected_methods | top_methods
    return query_choices[query_choices["method"].astype(str).isin(keep)].copy()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5a7f78")
    ax.set_xlabel("Held-out test selected-action utility")
    ax.set_title("Probe Fusion Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_probe_fusion_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    local_specs: list[tuple[str, str, float]],
    verifier: pd.DataFrame,
    table: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    selected_cols = [
        "selection_rule",
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_probe_cost",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "frontier_call_rate",
        "probe_call_rate",
        "local_override_rate",
        "strict_override_rate",
        "override_rate",
    ]
    top_cols = [column for column in selected_cols if column in table.columns]
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    text = [
        "# Probe Fusion Policy",
        "",
        "This no-call branch fuses the local-safe gain gate with strict MCQ verifier support.",
        "It does not call GPT, Gemini, Claude, vLLM, or local models; it reuses cached artifacts.",
        "The oracle/action set is the original broad100 action matrix, so verifier answers are probes only.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/185_probe_fusion_policy.py",
        "PYTHONPATH=src python experiments/185_probe_fusion_policy.py",
        "```",
        "",
        "## Inputs",
        "",
        f"- Local specs selected from validation rows: `{args.local_policy_table}`",
        f"- Strict verifier outputs: `{args.strict_verifier}`",
        f"- Number of local specs considered: `{len(local_specs)}`",
        f"- Cached strict verifier rows: `{len(verifier)}`",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in selected_cols if column in selected.columns]])
        if not selected.empty
        else "_No selected rows._",
        "",
        "## Best Held-Out Rows",
        "",
        markdown_table(top_test[top_cols]) if not top_test.empty else "_No held-out rows._",
        "",
        "## Interpretation",
        "",
        "- `mean_utility` charges only the selected final action.",
        "- `mean_utility_with_probe_cost` also charges the cached strict GPT verifier probe when the policy would use it.",
        "- This is a complementarity test: success requires a validation-selected fusion to beat the local-safe gate, not only a test-picked row.",
        "",
        "## Artifacts",
        "",
        f"- All policy rows: `{path.parent / 'table_probe_fusion_policy_all.csv'}`",
        f"- Selected policy rows: `{path.parent / 'table_probe_fusion_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_probe_fusion_query_choices.csv'}`",
        f"- Figure: `{path.parent / 'fig_probe_fusion_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
        else:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else str(value))
    headers = [str(column) for column in formatted.columns]
    rows = formatted.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
