from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVATION_SIGNALS = (
    "signal_act_last_global_anomaly",
    "signal_act_mean_global_anomaly",
    "signal_act_last_benchmark_anomaly",
    "signal_act_mean_benchmark_anomaly",
    "signal_act_last_knn_need_large_k3",
    "signal_act_last_knn_need_large_k5",
    "signal_act_last_knn_need_large_k10",
    "signal_act_mean_knn_need_large_k3",
    "signal_act_mean_knn_need_large_k5",
    "signal_act_mean_knn_need_large_k10",
    "signal_act_last_knn_delta_large_k5",
    "signal_act_mean_knn_delta_large_k5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threshold-only activation anomaly pilot for RouteCode.")
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
        "--activation-cache",
        type=Path,
        default=Path("results/controlled/broad100_qwen4_prefill_activation_router/qwen3_4b_prefill_activations.parquet"),
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
        default=Path("results/controlled/broad100_activation_anomaly_threshold_policy"),
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

    outputs_all = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target_all = pd.read_csv(args.target_table)
    target_all = exp171.add_tool_availability(target_all, outputs_all)
    target_all = exp172.add_benchmark_composed_gate(
        target_all,
        args.benchmark_composed_choices,
        args.benchmark_composed_method,
        exp171,
    )
    activations = load_activations(args.activation_cache)
    target = target_all[target_all["query_id"].astype(str).isin(set(activations.index.astype(str)))].copy()
    target = add_activation_signals(target, activations)
    eval_ids = set(target["query_id"].astype(str))
    outputs_eval = outputs_all[outputs_all["query_id"].astype(str).isin(eval_ids)].copy()
    priors = exp172.fit_train_priors(outputs_all)

    table_internal, details = evaluate_policy_library(
        target,
        outputs_eval,
        exp172=exp172,
        priors=priors,
        lambda_cost=float(args.lambda_cost),
    )
    selected = exp172.validation_selected_rows(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = exp172.add_bootstrap_ci(table_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    table = table.drop(columns=["_utility_values"], errors="ignore")

    target.to_csv(args.output_dir / "table_activation_anomaly_features.csv", index=False)
    table.to_csv(args.output_dir / "table_activation_anomaly_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_activation_anomaly_policy_selected.csv", index=False)
    details.to_csv(args.output_dir / "table_activation_anomaly_policy_query_choices.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "ACTIVATION_ANOMALY_POLICY_MEMO.md", args, target, table, selected)
    print(f"Wrote activation anomaly threshold policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_activations(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["query_id"] = frame["query_id"].astype(str)
    return frame.set_index("query_id").sort_index()


def add_activation_signals(target: pd.DataFrame, activations: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    out["query_id"] = out["query_id"].astype(str)
    out = out[out["query_id"].isin(set(activations.index.astype(str)))].copy()
    out = out.sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)
    train = out[out["split"].astype(str).eq("train")].copy()
    if train.empty:
        raise ValueError("Need train rows to fit activation anomaly statistics.")

    for view in ["last", "mean"]:
        matrix, ids = activation_matrix(activations, out["query_id"].astype(str).tolist(), view)
        train_positions = out.index[out["split"].astype(str).eq("train")].to_numpy()
        train_matrix = matrix[train_positions]
        mean = train_matrix.mean(axis=0)
        scale = train_matrix.std(axis=0)
        scale[scale < 1e-6] = 1.0
        z = (matrix - mean) / scale
        train_z = z[train_positions]
        out[f"signal_act_{view}_global_anomaly"] = row_norm(z - train_z.mean(axis=0))

        bench_signal = np.zeros(len(out), dtype=float)
        for benchmark, group in out.groupby("benchmark", sort=False):
            group_positions = group.index.to_numpy()
            train_group_positions = group.index[group["split"].astype(str).eq("train")].to_numpy()
            if train_group_positions.size == 0:
                centroid = train_z.mean(axis=0)
            else:
                centroid = z[train_group_positions].mean(axis=0)
            bench_signal[group_positions] = row_norm(z[group_positions] - centroid)
        out[f"signal_act_{view}_benchmark_anomaly"] = bench_signal

        for k in [3, 5, 10]:
            out[f"signal_act_{view}_knn_need_large_k{k}"] = train_neighbor_mean(
                z,
                out,
                train_positions,
                train["need_large"].astype(float).to_numpy(),
                k=k,
            )
        out[f"signal_act_{view}_knn_delta_large_k5"] = train_neighbor_mean(
            z,
            out,
            train_positions,
            train["delta_large"].astype(float).clip(lower=0.0).to_numpy(),
            k=5,
        )
    return out


def activation_matrix(activations: pd.DataFrame, query_ids: list[str], view: str) -> tuple[np.ndarray, list[str]]:
    prefix = f"{view}_"
    columns = [column for column in activations.columns if str(column).startswith(prefix)]
    if not columns:
        raise ValueError(f"No activation columns found for view {view}.")
    matrix = activations.loc[query_ids, columns].to_numpy(dtype=np.float32)
    return matrix, query_ids


def row_norm(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    return np.sqrt(np.mean(np.square(values), axis=1))


def train_neighbor_mean(
    z: np.ndarray,
    target: pd.DataFrame,
    train_positions: np.ndarray,
    train_values: np.ndarray,
    *,
    k: int,
) -> np.ndarray:
    train_z = z[train_positions]
    out = np.zeros(z.shape[0], dtype=float)
    for row_idx in range(z.shape[0]):
        distances = row_norm(train_z - z[row_idx])
        if target.iloc[row_idx]["split"] == "train":
            same_train = np.where(train_positions == row_idx)[0]
            if same_train.size:
                distances[same_train[0]] = np.inf
        k_eff = min(int(k), max(len(distances) - int(np.isinf(distances).sum()), 1))
        nearest = np.argsort(distances)[:k_eff]
        out[row_idx] = float(np.mean(train_values[nearest])) if nearest.size else 0.0
    return out


def evaluate_policy_library(
    target: pd.DataFrame,
    outputs: pd.DataFrame,
    *,
    exp172,
    priors: dict[str, Any],
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    selectors = fixed_selectors(exp172, priors)
    selectors.update(threshold_selectors_from_val(target, exp172, priors))

    rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        frame = target[target["split"].astype(str).eq(split)].copy()
        for method, (family, selector) in selectors.items():
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
            row["activation_subset"] = True
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
                detail_frames.append(detail)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    details = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return table, details


def fixed_selectors(exp172, priors: dict[str, Any]) -> dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]]:
    return {
        "full_cost_aware_oracle": ("diagnostic_oracle", exp172.full_oracle_selector("utility")),
        "full_quality_oracle": ("diagnostic_oracle", exp172.full_oracle_selector("quality_score")),
        "target_best_local_action": ("target_reference", lambda row, actions: str(row["best_local_action"])),
        "target_best_large_action": ("target_reference", lambda row, actions: str(row["best_large_action"])),
        "target_local_vs_large_oracle": (
            "diagnostic_oracle",
            lambda row, actions: str(row["best_large_action"]) if bool(row.get("need_large", False)) else str(row["best_local_action"]),
        ),
        "train_best_single_action": (
            "reference",
            lambda row, actions: exp172.choose_prior_action(row, actions, priors, exp172.ALL_ACTIONS, scope="global"),
        ),
        "tool_then_local_consensus_else_benchmark_prior": (
            "deployed_policy",
            lambda row, actions: exp172.choose_tool_or_local_consensus(row, actions, priors, fallback_pool=exp172.ALL_ACTIONS),
        ),
        "tool_then_171_gate_local_consensus_large_prior": (
            "deployed_policy",
            lambda row, actions: exp172.choose_tool_or_gate_consensus(row, actions, priors),
        ),
    }


def threshold_selectors_from_val(
    target: pd.DataFrame,
    exp172,
    priors: dict[str, Any],
) -> dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]]:
    selectors: dict[str, tuple[str, Callable[[pd.Series, dict[str, dict[str, Any]]], str]]] = {}
    val = target[target["split"].astype(str).eq("val")]
    for signal in [name for name in ACTIVATION_SIGNALS if name in target.columns]:
        thresholds = exp172.candidate_thresholds(val[signal].to_numpy(dtype=float))
        for threshold in thresholds:
            threshold = float(threshold)
            selectors[f"activation_lvlarge_{signal}_thr{threshold:.4g}"] = (
                "activation_local_vs_large_threshold",
                local_vs_large_threshold_selector(signal, threshold),
            )
            selectors[f"activation_deployed_{signal}_thr{threshold:.4g}"] = (
                "activation_deployed_threshold",
                deployed_threshold_selector(signal, threshold, exp172, priors),
            )
    return selectors


def local_vs_large_threshold_selector(signal: str, threshold: float) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        value = as_float(row.get(signal, np.nan), default=-np.inf)
        return str(row["best_large_action"]) if value >= threshold else str(row["best_local_action"])

    return select


def deployed_threshold_selector(
    signal: str,
    threshold: float,
    exp172,
    priors: dict[str, Any],
) -> Callable[[pd.Series, dict[str, dict[str, Any]]], str]:
    def select(row: pd.Series, actions: dict[str, dict[str, Any]]) -> str:
        tool = exp172.tool_action(actions)
        if tool:
            return tool
        value = as_float(row.get(signal, np.nan), default=-np.inf)
        if value >= threshold:
            return exp172.choose_prior_action(row, actions, priors, exp172.STRONG_OR_FRONTIER_ACTIONS, scope="benchmark")
        return exp172.choose_answer_agreement(
            row,
            actions,
            priors,
            pool=tuple(model for model in exp172.OBSERVABLE_LOCAL_ACTIONS if model != exp172.TOOL_MODEL_ID),
            evidence_pool=tuple(model for model in exp172.OBSERVABLE_LOCAL_ACTIONS if model != exp172.TOOL_MODEL_ID),
            alpha=0.50,
            beta=0.25,
            fallback_pool=tuple(model for model in exp172.OBSERVABLE_LOCAL_ACTIONS if model != exp172.TOOL_MODEL_ID),
        )

    return select


def select_actions(
    frame: pd.DataFrame,
    rows_by_query: dict[str, dict[str, dict[str, Any]]],
    selector: Callable[[pd.Series, dict[str, dict[str, Any]]], str],
    exp172,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        query_id = str(row["query_id"])
        actions = rows_by_query[query_id]
        model_id = selector(row, actions)
        if not exp172.is_action_available(actions, model_id):
            model_id = exp172.first_available(actions, exp172.ALL_ACTIONS)
        rows.append({"query_id": query_id, "model_id": model_id})
    return pd.DataFrame(rows)


def detail_methods() -> set[str]:
    return {
        "full_cost_aware_oracle",
        "target_local_vs_large_oracle",
        "target_best_local_action",
        "target_best_large_action",
        "tool_then_171_gate_local_consensus_large_prior",
    }


def as_float(value: object, *, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#6b7f4e")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Activation-Anomaly Threshold Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_activation_anomaly_policy_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    target: pd.DataFrame,
    table: pd.DataFrame,
    selected: pd.DataFrame,
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
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "selection_rule",
    ]
    available_cols = [column for column in cols if column in selected.columns]
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    scope = target.groupby(["split", "benchmark"]).size().rename("n_queries").reset_index()
    target_summary = (
        target.groupby("split", as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            local_utility=("local_utility", "mean"),
            large_utility=("large_utility", "mean"),
            local_vs_large_oracle_utility=("delta_large", lambda s: float(np.nan)),
            need_large_rate=("need_large", "mean"),
        )
    )
    target_summary["local_vs_large_oracle_utility"] = [
        float(target[target["split"].astype(str).eq(split)][["local_utility", "large_utility"]].max(axis=1).mean())
        for split in target_summary["split"].astype(str)
    ]
    lines = [
        "# Activation-Anomaly Threshold Policy",
        "",
        "This is a cached, threshold-only probe pilot over the Qwen3-4B prefill activation cache. It makes no GPT, Gemini, Claude, vLLM, or local model calls.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/176_activation_anomaly_threshold_policy.py",
        (
            "PYTHONPATH=src python experiments/176_activation_anomaly_threshold_policy.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--activation-cache {args.activation_cache} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Scope",
        "",
        markdown_table(scope),
        "",
        "## Oracle And References On This Subset",
        "",
        markdown_table(target_summary),
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected[available_cols] if available_cols else selected),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(best_test[[column for column in cols if column in best_test.columns]]),
        "",
        "## Interpretation",
        "",
        "- The signal is fitted only through train activation distribution statistics and train nearest-neighbor summaries; no router is trained.",
        "- Treat local-vs-large threshold rows as diagnostic because they choose the target table's best local or best large action.",
        "- Treat deployed threshold rows as the applied result because they choose concrete cached actions from train priors and local agreement.",
        "- A positive result would improve validation-selected held-out utility without increasing strong/frontier calls materially. A negative result supports the predictability-gap story for cheap activation summaries.",
        "",
        "## Artifacts",
        "",
        f"- Feature table: `{args.output_dir / 'table_activation_anomaly_features.csv'}`",
        f"- All policy table: `{args.output_dir / 'table_activation_anomaly_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_activation_anomaly_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_activation_anomaly_policy_query_choices.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_activation_anomaly_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
