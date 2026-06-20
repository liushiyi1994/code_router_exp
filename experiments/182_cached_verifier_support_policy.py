from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
BENCHMARK_SETS: list[tuple[str, ...]] = [
    ("gpqa", "mmlupro"),
    ("mmlupro",),
    ("gpqa",),
]
SUPPORT_MODES = [
    "support_any",
    "support_answer_match",
    "support_local_only",
    "support_nonfrontier_only",
    "support_strong_frontier_only",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached verifier-supported-action policy for broad100.")
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
        "--verifier-table",
        type=Path,
        default=Path(
            "results/controlled/broad100_task_specific_verifier_action_gpt_mcq_512/"
            "table_task_specific_verifier_outputs.csv"
        ),
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
        default=Path("results/controlled/broad100_cached_verifier_support_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_182")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_182")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_182")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_182")
    exp179 = load_module("experiments/179_cached_adjudicator_blend_policy.py", "adjudicator_blend_179_for_182")
    exp181 = load_module("experiments/181_task_specific_verifier_action.py", "task_verifier_181_for_182")

    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp172.add_benchmark_composed_gate(
        target,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    verifier = load_verifier(args.verifier_table)
    base_choices = exp181.practical_base_choices(target, outputs, exp172, exp175, exp177, exp179)
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    gpt_cost = exp181.mean_gpt_cost(outputs)

    policy_table, query_choices, support_diagnostics = evaluate_support_policies(
        base_choices=base_choices,
        verifier=verifier,
        outputs=outputs,
        target=target,
        rows_by_query=rows_by_query,
        frontiers=frontiers,
        exp172=exp172,
        gpt_cost=gpt_cost,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = selected_rows(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed), exp172=exp172)

    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_cached_verifier_support_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_cached_verifier_support_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_cached_verifier_support_query_choices.csv", index=False)
    support_diagnostics.to_csv(args.output_dir / "table_cached_verifier_support_diagnostics.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(
        args.output_dir / "CACHED_VERIFIER_SUPPORT_POLICY_MEMO.md",
        args,
        verifier,
        support_diagnostics,
        policy_table,
        selected,
    )
    print(f"Wrote cached verifier support policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_verifier(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["query_id"] = frame["query_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["benchmark"] = frame["benchmark"].astype(str).str.lower()
    frame["status"] = frame["status"].astype(str)
    frame["supported_model"] = frame["supported_model"].fillna("").astype(str)
    frame["parsed_answer"] = frame["parsed_answer"].fillna("").astype(str)
    frame["verifier_confidence"] = pd.to_numeric(frame["verifier_confidence"], errors="coerce").fillna(0.0)
    frame["cost_total_usd"] = pd.to_numeric(frame["cost_total_usd"], errors="coerce").fillna(0.0)
    frame["quality_score"] = pd.to_numeric(frame["quality_score"], errors="coerce").fillna(0.0)
    return frame


def evaluate_support_policies(
    *,
    base_choices: pd.DataFrame,
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    frontiers: set[str],
    exp172,
    gpt_cost: float,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    detail: list[pd.DataFrame] = []
    verifier_map = verifier.set_index("query_id").to_dict("index")
    methods = [("base_candidate_ranker", "reference", np.nan, "base", tuple())]
    for benchmarks in BENCHMARK_SETS:
        bench_name = "-".join(benchmarks)
        for support_mode in SUPPORT_MODES:
            for threshold in THRESHOLDS:
                method = f"verifier_supported_{support_mode}_thr{threshold:g}_bench{bench_name}"
                methods.append((method, "cached_verifier_support", threshold, support_mode, benchmarks))
        for threshold in THRESHOLDS:
            method = f"verifier_supported_oracle_between_base_and_support_thr{threshold:g}_bench{bench_name}"
            methods.append((method, "diagnostic_oracle", threshold, "oracle_between_base_and_support", benchmarks))

    for method, family, threshold, support_mode, benchmarks in methods:
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            split_base = base_choices[base_choices["split"].astype(str).eq(split)].copy()
            choice = choose_supported_actions(
                split_base,
                verifier_map,
                rows_by_query,
                support_mode=support_mode,
                threshold=float(threshold) if not pd.isna(threshold) else np.nan,
                benchmarks=set(benchmarks),
            )
            selected_rows = choice[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
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
            route_cost_norm = probe_norm_cost(choice, verifier_map, gpt_cost)
            route_utilities = selected_rows["quality_score"].to_numpy(dtype=float) - float(lambda_cost) * (
                selected_rows["normalized_remote_cost"].to_numpy(dtype=float) + route_cost_norm
            )
            row.update(
                {
                    "threshold": threshold,
                    "support_mode": support_mode,
                    "benchmarks": ",".join(benchmarks) if benchmarks else "",
                    "probe_call_rate": float(choice["verifier_probed"].mean()) if not choice.empty else 0.0,
                    "support_override_rate": float(choice["overrode_base"].mean()) if not choice.empty else 0.0,
                    "supported_model_available_rate": float(choice["supported_model_available"].mean()) if not choice.empty else 0.0,
                    "extra_probe_norm_cost_mean": float(np.mean(route_cost_norm)) if len(route_cost_norm) else 0.0,
                    "mean_utility_with_probe_cost": float(np.mean(route_utilities)) if len(route_utilities) else np.nan,
                    "oracle_utility_ratio_with_probe_cost": float(
                        np.mean(route_utilities) / max(float(row["cost_oracle_mean_utility"]), 1e-12)
                    )
                    if len(route_utilities)
                    else np.nan,
                    "_utility_values_with_probe_cost": route_utilities.tolist(),
                }
            )
            rows.append(row)
            if split == "test":
                detail.append(
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
                    .merge(
                        choice[
                            [
                                "query_id",
                                "base_model_id",
                                "verifier_probed",
                                "overrode_base",
                                "supported_model",
                                "verifier_confidence",
                                "support_mode_reason",
                            ]
                        ],
                        on="query_id",
                        how="left",
                    )
                    .assign(method=method, family=family)
                )
    support_diagnostics = diagnose_supported_models(verifier, outputs, rows_by_query, frontiers)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False])
    details = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()
    return table, details, support_diagnostics


def choose_supported_actions(
    base: pd.DataFrame,
    verifier_map: dict[str, dict[str, Any]],
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    *,
    support_mode: str,
    threshold: float,
    benchmarks: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        query_id = str(row.query_id)
        base_model = str(row.model_id)
        item = verifier_map.get(query_id)
        selected = base_model
        probed = False
        overrode = False
        reason = "base"
        supported = ""
        confidence = 0.0
        available = False
        if support_mode != "base" and item is not None and str(item.get("status", "")) == "success":
            benchmark = str(item.get("benchmark", "")).lower()
            probed = not benchmarks or benchmark in benchmarks
            supported = str(item.get("supported_model", "") or "")
            confidence = float(item.get("verifier_confidence", 0.0) or 0.0)
            available = supported in rows_by_query.get(query_id, {})
            if probed and available and confidence >= threshold and supported.upper() != "NONE":
                supported_row = rows_by_query[query_id][supported]
                base_row = rows_by_query[query_id].get(base_model, {})
                if support_mode == "oracle_between_base_and_support":
                    selected = oracle_choice(base_model, supported, base_row, supported_row)
                    reason = "diagnostic_oracle_support" if selected == supported else "diagnostic_oracle_base"
                elif support_allowed(support_mode, item, supported_row):
                    selected = supported
                    reason = support_mode
                else:
                    reason = f"{support_mode}_blocked"
        if selected != base_model:
            overrode = True
        rows.append(
            {
                "query_id": query_id,
                "split": str(row.split),
                "model_id": selected,
                "base_model_id": base_model,
                "verifier_probed": probed,
                "overrode_base": overrode,
                "supported_model": supported,
                "supported_model_available": available,
                "verifier_confidence": confidence,
                "support_mode_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def support_allowed(support_mode: str, item: dict[str, Any], supported_row: dict[str, Any]) -> bool:
    if support_mode == "support_any":
        return True
    if support_mode == "support_answer_match":
        verifier_answer = normalize_answer(item.get("parsed_answer", ""))
        supported_answer = normalize_answer(supported_row.get("parsed_answer", ""))
        return bool(verifier_answer) and verifier_answer == supported_answer
    is_frontier = bool(supported_row.get("is_frontier", False))
    is_local = bool(supported_row.get("is_local", False))
    if support_mode == "support_local_only":
        return is_local
    if support_mode == "support_nonfrontier_only":
        return not is_frontier
    if support_mode == "support_strong_frontier_only":
        return str(supported_row.get("model_id", "")) in {
            "gemini-3.5-flash",
            "gemini-3.5-flash-strong-solve",
            "gpt-5.5",
            "qwen3-32b-awq-local",
            "qwen3-32b-awq-selfconsistency-n3-local",
        }
    return False


def oracle_choice(base_model: str, supported_model: str, base_row: dict[str, Any], supported_row: dict[str, Any]) -> str:
    base_utility = float(base_row.get("utility", -1e9) or -1e9)
    supported_utility = float(supported_row.get("utility", -1e9) or -1e9)
    if supported_utility > base_utility:
        return supported_model
    return base_model


def probe_norm_cost(choice: pd.DataFrame, verifier_map: dict[str, dict[str, Any]], gpt_cost: float) -> np.ndarray:
    costs: list[float] = []
    for row in choice.itertuples(index=False):
        item = verifier_map.get(str(row.query_id))
        if item is None or not bool(row.verifier_probed):
            costs.append(0.0)
        else:
            costs.append(float(item.get("cost_total_usd", 0.0) or 0.0) / max(gpt_cost, 1e-12))
    return np.asarray(costs, dtype=float)


def diagnose_supported_models(
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    frontiers: set[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in verifier.itertuples(index=False):
        supported = str(item.supported_model or "")
        action = rows_by_query.get(str(item.query_id), {}).get(supported)
        if not supported or supported.upper() == "NONE" or action is None:
            rows.append(
                {
                    "query_id": str(item.query_id),
                    "split": str(item.split),
                    "benchmark": str(item.benchmark),
                    "supported_model": supported,
                    "verifier_confidence": float(item.verifier_confidence),
                    "verifier_quality": float(item.quality_score),
                    "supported_model_available": bool(action is not None),
                    "supported_quality": np.nan,
                    "supported_utility": np.nan,
                    "supported_is_frontier": bool(supported in frontiers),
                    "verifier_matches_supported_answer": False,
                }
            )
            continue
        rows.append(
            {
                "query_id": str(item.query_id),
                "split": str(item.split),
                "benchmark": str(item.benchmark),
                "supported_model": supported,
                "verifier_confidence": float(item.verifier_confidence),
                "verifier_quality": float(item.quality_score),
                "supported_model_available": True,
                "supported_quality": float(action.get("quality_score", 0.0) or 0.0),
                "supported_utility": float(action.get("utility", 0.0) or 0.0),
                "supported_is_frontier": bool(supported in frontiers),
                "verifier_matches_supported_answer": normalize_answer(item.parsed_answer)
                == normalize_answer(action.get("parsed_answer", "")),
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    summary = (
        detail.groupby(["split", "benchmark"], as_index=False)
        .agg(
            n_rows=("query_id", "nunique"),
            valid_support_rate=("supported_model_available", "mean"),
            mean_verifier_quality=("verifier_quality", "mean"),
            mean_supported_quality=("supported_quality", "mean"),
            mean_supported_utility=("supported_utility", "mean"),
            mean_confidence=("verifier_confidence", "mean"),
            answer_match_rate=("verifier_matches_supported_answer", "mean"),
            supported_frontier_rate=("supported_is_frontier", "mean"),
        )
        .sort_values(["split", "benchmark"])
    )
    return summary


def selected_rows(table: pd.DataFrame, *, bootstrap_samples: int, seed: int, exp172) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility", "mean_utility_with_probe_cost"]:
        val = table[table["split"].eq("val") & table["family"].ne("diagnostic_oracle") & table["family"].ne("reference")]
        if val.empty:
            continue
        best = val.sort_values([objective, "frontier_call_rate", "support_override_rate"], ascending=[False, True, True]).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule=f"val_best_{objective}"))
        rows.append(table[table["split"].eq("test") & table["method"].eq(method)].copy().assign(selection_rule=f"val_best_{objective}_test"))
    reference = table[table["split"].eq("test") & table["family"].eq("reference")]
    if not reference.empty:
        rows.append(reference.assign(selection_rule="reference_test"))
    diagnostic = table[table["split"].eq("test") & table["family"].eq("diagnostic_oracle")]
    if not diagnostic.empty:
        rows.append(
            diagnostic.sort_values(["mean_utility", "mean_quality"], ascending=False)
            .head(6)
            .assign(selection_rule="diagnostic_oracle_test")
        )
    top_test = table[table["split"].eq("test") & table["family"].ne("diagnostic_oracle")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    ).head(12)
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
    return exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed).drop(
        columns=["_utility_values"], errors="ignore"
    )


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    if text in {"nan", "none", ""}:
        return ""
    return " ".join(text.replace("$", "").replace("\\boxed", "").replace("{", "").replace("}", "").split())


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5f7f74")
    ax.set_xlabel("Held-out test selected-action utility")
    ax.set_title("Cached Verifier Supported-Action Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cached_verifier_support_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    verifier: pd.DataFrame,
    support_diagnostics: pd.DataFrame,
    table: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    cols = [
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
        "strong_or_frontier_call_rate",
        "probe_call_rate",
        "support_override_rate",
        "supported_model_available_rate",
        "extra_probe_norm_cost_mean",
    ]
    lines = [
        "# Cached Verifier Supported-Action Policy",
        "",
        "This cached follow-up asks whether the GPT task verifier is more useful as evidence for selecting an existing candidate action than as a final answer action.",
        "No new provider, vLLM, or local model calls are made by this script.",
        "",
        "## Inputs",
        "",
        f"- Target table: `{args.target_table}`",
        f"- Cached model outputs: `{args.outputs}`",
        f"- Cached verifier rows: `{args.verifier_table}`",
        f"- Verifier rows: `{len(verifier)}`",
        f"- Cached verifier provider models: `{', '.join(sorted(verifier['provider_model'].dropna().astype(str).unique()))}`",
        f"- Cached verifier total spend represented in table: `${float(verifier['cost_total_usd'].sum()):.4f}`",
        "Claude is not used.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/182_cached_verifier_support_policy.py",
        (
            "PYTHONPATH=src python experiments/182_cached_verifier_support_policy.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--verifier-table {args.verifier_table} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Supported-Model Diagnostics",
        "",
        markdown_table(support_diagnostics),
        "",
        "## Validation-Selected And Reference Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]),
        "",
        "## Best Held-Out Non-Oracle Rows",
        "",
        markdown_table(
            table[table["split"].eq("test") & table["family"].ne("diagnostic_oracle")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)
            .head(12)[[column for column in cols if column in table.columns]]
        ),
        "",
        "## Interpretation",
        "",
        "- The verifier support signal is benchmark-asymmetric: cached GPT verifier rows are much stronger on MMLUPro than GPQA.",
        "- `mean_utility` reports selected action utility only. `mean_utility_with_probe_cost` charges the GPT verifier probe whenever the policy would need to call it.",
        "- Diagnostic oracle rows use gold outcomes to choose between the base action and verifier-supported action. They are upper bounds, not deployable policies.",
        "",
        "## Artifacts",
        "",
        f"- All policy rows: `{args.output_dir / 'table_cached_verifier_support_policy_all.csv'}`",
        f"- Selected policy rows: `{args.output_dir / 'table_cached_verifier_support_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_cached_verifier_support_query_choices.csv'}`",
        f"- Support diagnostics: `{args.output_dir / 'table_cached_verifier_support_diagnostics.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_cached_verifier_support_policy_utility.pdf'}`",
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
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
