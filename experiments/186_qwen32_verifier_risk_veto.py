from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CONF_THRESHOLDS = [0.0, 0.5, 0.7, 0.85, 0.95]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use cached Qwen32 vLLM answer verifier as a local-risk veto.")
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
        "--qwen32-verifier",
        type=Path,
        default=Path("results/controlled/broad100_qwen32_answer_verifier_strong_gate/table_vllm_answer_verifier_probe.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_qwen32_verifier_risk_veto"))
    parser.add_argument("--max-local-specs", type=int, default=32)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_aware_171_for_186")
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172_for_186")
    exp175 = load_module("experiments/175_public_test_verifier_policy.py", "public_test_175_for_186")
    exp177 = load_module("experiments/177_candidate_correctness_ranker_policy.py", "candidate_ranker_177_for_186")
    exp183 = load_module("experiments/183_local_safe_gain_gate.py", "local_safe_183_for_186")
    exp185 = load_module("experiments/185_probe_fusion_policy.py", "probe_fusion_185_for_186")

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
    scored = exp185.score_local_specs(feature_frame, exp183)
    local_specs = exp185.select_local_specs(args.local_policy_table, max_specs=int(args.max_local_specs))
    verifier = load_verifier(args.qwen32_verifier)
    policy_table, query_choices = evaluate_policies(
        scored,
        local_specs,
        verifier,
        outputs,
        target,
        exp172,
        exp183,
        lambda_cost=float(args.lambda_cost),
    )
    policy_table = exp172.add_bootstrap_ci(policy_table, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    selected = selected_rows(policy_table, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    query_choices = filter_query_choices(query_choices, policy_table, selected)

    scored.to_csv(args.output_dir / "table_qwen32_verifier_risk_features.csv", index=False)
    verifier.to_csv(args.output_dir / "table_qwen32_verifier_risk_probe.csv", index=False)
    policy_table.drop(columns=["_utility_values"], errors="ignore").to_csv(
        args.output_dir / "table_qwen32_verifier_risk_policy_all.csv", index=False
    )
    selected.to_csv(args.output_dir / "table_qwen32_verifier_risk_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_qwen32_verifier_risk_query_choices.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "QWEN32_VERIFIER_RISK_VETO_MEMO.md", args, local_specs, verifier, policy_table, selected)
    print(f"Wrote Qwen32 verifier-risk veto results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_verifier(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["split"] = frame["split"].astype(str)
    frame["verdict"] = frame["verdict"].fillna("unknown").astype(str).str.lower()
    frame["confidence"] = pd.to_numeric(frame["confidence"], errors="coerce")
    frame["is_accept"] = frame["verdict"].eq("accept")
    frame["is_escalate"] = frame["verdict"].eq("escalate")
    frame["accept_confidence"] = np.where(frame["is_accept"], frame["confidence"].fillna(0.0), 0.0)
    frame["escalate_flag"] = frame["is_escalate"].astype(float)
    frame["valid_verifier"] = frame["status"].astype(str).eq("success")
    return frame


def evaluate_policies(
    scored: pd.DataFrame,
    local_specs: list[tuple[str, str, float]],
    verifier: pd.DataFrame,
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    exp183,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    verifier_map = verifier.set_index("query_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    methods: list[dict[str, Any]] = [{"method": "base_candidate_ranker", "family": "reference", "kind": "base"}]
    for local_method, pred_col, threshold in local_specs:
        methods.append(
            {
                "method": f"local_{local_method}",
                "family": "local_safe_reference",
                "kind": "local",
                "pred_col": pred_col,
                "local_threshold": threshold,
            }
        )
        methods.append(
            {
                "method": f"qwen32_veto_escalate_{local_method}",
                "family": "qwen32_verifier_veto",
                "kind": "veto_escalate",
                "pred_col": pred_col,
                "local_threshold": threshold,
            }
        )
        for conf in CONF_THRESHOLDS:
            methods.append(
                {
                    "method": f"qwen32_accept_required_conf{conf:g}_{local_method}",
                    "family": "qwen32_verifier_accept_required",
                    "kind": "accept_required",
                    "pred_col": pred_col,
                    "local_threshold": threshold,
                    "confidence_threshold": conf,
                }
            )
            methods.append(
                {
                    "method": f"qwen32_accept_to_local_conf{conf:g}_{local_method}",
                    "family": "qwen32_verifier_accept_to_local",
                    "kind": "accept_to_local",
                    "pred_col": pred_col,
                    "local_threshold": threshold,
                    "confidence_threshold": conf,
                }
            )

    for spec in methods:
        for split in ["val", "test"]:
            split_frame = scored[scored["split"].astype(str).eq(split)].copy()
            target_split = target[target["split"].astype(str).eq(split)].copy()
            choices = choose_actions(split_frame, spec, exp183, verifier_map)
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
            row.update(
                {
                    "verifier_call_rate": float(choices["verifier_available"].mean()) if not choices.empty else 0.0,
                    "local_override_rate": float(choices["local_overrode_base"].mean()) if not choices.empty else 0.0,
                    "veto_rate": float(choices["vetoed_local"].mean()) if not choices.empty else 0.0,
                    "accept_to_local_rate": float(choices["accepted_to_local"].mean()) if not choices.empty else 0.0,
                    "override_rate": float(choices["overrode_base"].mean()) if not choices.empty else 0.0,
                    "mean_utility_with_probe_cost": row["mean_utility"],
                    "oracle_utility_ratio_with_probe_cost": row["oracle_utility_ratio"],
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
    return table, pd.concat(details, ignore_index=True) if details else pd.DataFrame()


def choose_actions(
    frame: pd.DataFrame,
    spec: dict[str, Any],
    exp183,
    verifier_map: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    if spec["kind"] in {"local", "veto_escalate", "accept_required", "accept_to_local"}:
        local_choices = exp183.choose_actions(
            frame,
            pred_col=str(spec.get("pred_col", "")),
            threshold=float(spec.get("local_threshold", math.nan)),
            family="local_safe_gain_gate",
        )
    else:
        local_choices = exp183.choose_actions(frame, pred_col="", threshold=math.nan, family="reference")

    frame_by_query = frame.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for item in local_choices.itertuples(index=False):
        query_id = str(item.query_id)
        base_model = str(item.base_model_id)
        local_model = str(item.model_id)
        selected = local_model
        verifier = verifier_map.get(query_id, {})
        available = bool(verifier.get("valid_verifier", False))
        verdict = str(verifier.get("verdict", "unknown"))
        accept_conf = float(verifier.get("accept_confidence", 0.0) or 0.0)
        is_escalate = bool(verifier.get("is_escalate", False))
        kind = str(spec["kind"])
        vetoed = False
        accepted_to_local = False
        if kind == "veto_escalate" and local_model != base_model and is_escalate:
            selected = base_model
            vetoed = True
        elif kind == "accept_required" and local_model != base_model:
            if not available or verdict != "accept" or accept_conf < float(spec.get("confidence_threshold", 0.0)):
                selected = base_model
                vetoed = True
        elif kind == "accept_to_local":
            row = frame_by_query.loc[query_id]
            consensus_model = str(row.get("consensus_model", "") or "")
            if consensus_model and available and verdict == "accept" and accept_conf >= float(spec.get("confidence_threshold", 0.0)):
                selected = consensus_model
                accepted_to_local = selected != local_model
        rows.append(
            {
                "query_id": query_id,
                "model_id": selected,
                "base_model_id": base_model,
                "local_model_id": local_model,
                "consensus_model": getattr(item, "consensus_model", ""),
                "local_overrode_base": local_model != base_model,
                "vetoed_local": vetoed,
                "accepted_to_local": accepted_to_local,
                "overrode_base": selected != base_model,
                "verifier_available": available,
                "verifier_verdict": verdict,
                "verifier_accept_confidence": accept_conf,
            }
        )
    return pd.DataFrame(rows)


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for objective in ["mean_utility"]:
        val = table[
            table["split"].eq("val")
            & table["family"].isin(
                ["qwen32_verifier_veto", "qwen32_verifier_accept_required", "qwen32_verifier_accept_to_local"]
            )
        ].copy()
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
    return query_choices[query_choices["method"].astype(str).isin(selected_methods | top_methods)].copy()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#6f6f8f")
    ax.set_xlabel("Held-out test selected-action utility")
    ax.set_title("Qwen32 Verifier-Risk Veto Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_qwen32_verifier_risk_policy_utility.pdf")
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
        "oracle_utility_ratio",
        "frontier_call_rate",
        "verifier_call_rate",
        "local_override_rate",
        "veto_rate",
        "accept_to_local_rate",
        "override_rate",
    ]
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    diagnostics = (
        verifier.groupby(["split", "verdict"], dropna=False)
        .agg(n=("query_id", "size"), mean_confidence=("confidence", "mean"), mean_base_quality=("base_quality", "mean"))
        .reset_index()
    )
    text = [
        "# Qwen32 Verifier-Risk Veto",
        "",
        "This no-call branch reuses cached Qwen3-32B-AWQ vLLM answer-verifier outputs as a query-level local-risk signal.",
        "The verifier was collected by Experiment 142, so no GPT, Gemini, Claude, vLLM, or local model calls are made here.",
        "Because the verifier judged an older base answer, this experiment treats its verdict as a risk/veto signal rather than as a direct judgment of the current selected answer.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/186_qwen32_verifier_risk_veto.py",
        "PYTHONPATH=src python experiments/186_qwen32_verifier_risk_veto.py",
        "```",
        "",
        "## Inputs",
        "",
        f"- Local specs selected from validation rows: `{args.local_policy_table}`",
        f"- Cached Qwen32 verifier probe: `{args.qwen32_verifier}`",
        f"- Number of local specs considered: `{len(local_specs)}`",
        f"- Cached verifier rows: `{len(verifier)}`",
        "",
        "## Verifier Diagnostics",
        "",
        markdown_table(diagnostics),
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in selected_cols if column in selected.columns]])
        if not selected.empty
        else "_No selected rows._",
        "",
        "## Best Held-Out Rows",
        "",
        markdown_table(top_test[[column for column in selected_cols if column in top_test.columns]]) if not top_test.empty else "_No held-out rows._",
        "",
        "## Interpretation",
        "",
        "- Local vLLM probe cost is treated as zero remote API cost, but this branch still reports verifier-call rate.",
        "- A useful result would beat the no-probe local-safe reference after validation selection.",
        "- If only test-picked rows improve, treat it as diagnostic evidence rather than a deployable policy.",
        "",
        "## Artifacts",
        "",
        f"- All policy rows: `{path.parent / 'table_qwen32_verifier_risk_policy_all.csv'}`",
        f"- Selected policy rows: `{path.parent / 'table_qwen32_verifier_risk_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_qwen32_verifier_risk_query_choices.csv'}`",
        f"- Figure: `{path.parent / 'fig_qwen32_verifier_risk_policy_utility.pdf'}`",
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
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
