from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
ACTIONS = ["base", "self", "strong"]
BENCHMARK_SETS = {
    "all": ("aime", "bbh", "gpqa", "gsm8k", "humaneval", "livemathbench", "math500", "mbpp", "mmlupro"),
    "stress": ("gpqa", "mmlupro", "math500", "livemathbench"),
    "gpqa": ("gpqa",),
    "gpqa_mmlupro": ("gpqa", "mmlupro"),
    "gpqa_mmlupro_math": ("gpqa", "mmlupro", "math500"),
    "math_mmlupro": ("math500", "mmlupro"),
    "mmlupro": ("mmlupro",),
    "none": tuple(),
}
SUPPRESS_TARGETS = {"self": 1, "base": 0, "base_if_equal_else_self": 3}
RESCUE_FROM = {"self": (1,), "base": (0,), "base_self": (0, 1)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual confidence rules on top of the cached pairwise action ranker.")
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
        "--embedding-cache-dir",
        type=Path,
        default=Path("results/controlled/broad100_embedding_self_action_gate/embedding_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_residual_confidence_rule_policy"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--val-tie-eps", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_gate")
    calibrated = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_gate")
    pairwise = load_module("experiments/162_pairwise_action_ranker.py", "pairwise_action_ranker")

    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    context = build_context(
        package,
        self_gate,
        calibrated,
        pairwise,
        outputs,
        probe,
        embedding_cache_dir=args.embedding_cache_dir,
        self_model_id=str(args.self_model_id),
        max_features=int(args.max_features),
    )
    table, residuals = run_rule_grid(context, lambda_cost=float(args.lambda_cost), self_model_id=str(args.self_model_id))
    selected = validation_selected_rows(table, eps=float(args.val_tie_eps))

    table.to_csv(args.output_dir / "table_residual_confidence_rule_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_residual_confidence_rule_policy_selected.csv", index=False)
    residuals.to_csv(args.output_dir / "table_residual_confidence_rule_policy_residuals.csv", index=False)
    write_figure(args.output_dir, table, selected)
    write_memo(args.output_dir / "RESIDUAL_CONFIDENCE_RULE_POLICY_MEMO.md", args, table, selected)
    print(f"Wrote residual confidence-rule policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_context(
    package,
    self_gate,
    calibrated,
    pairwise,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    embedding_cache_dir: Path,
    self_model_id: str,
    max_features: int,
) -> dict[str, Any]:
    local_agree = calibrated.local_agreement_counts(outputs, self_model_id=self_model_id)
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    base = {
        split: self_gate.normalize_selection(package.profile_v4_selection_for_split(outputs_no_self, split=split))
        for split in ["train", "val", "test"]
    }
    frames = {
        split: calibrated.build_feature_frame_fast(
            outputs,
            probe,
            base[split],
            split=split,
            self_model_id=self_model_id,
            local_agree=local_agree,
        )
        for split in ["train", "val", "test"]
    }
    metrics = {
        split: pairwise.build_action_metrics(outputs, frames[split], split=split, self_model_id=self_model_id)
        for split in ["val", "test"]
    }
    oracle_stats = pairwise.split_oracle_stats(outputs)
    feature_sets = pairwise.build_feature_sets(
        calibrated,
        frames,
        "tool_probe_profile_v4",
        embedding_cache_dir,
        max_features=max_features,
    )
    scores = pairwise.fit_pairwise_scores(
        frames,
        feature_sets["metadata_numeric_text"],
        learner="logistic",
        hp_value=10.0,
    )
    base_actions = {
        split: pairwise.scores_to_action_indices(scores[split], self_bias=-0.05, strong_bias=-0.20, strong_cap=None)
        for split in ["val", "test"]
    }
    arrays = {split: frame_arrays(frames[split], scores[split]) for split in ["val", "test"]}
    return {
        "frames": frames,
        "metrics": metrics,
        "oracle_stats": oracle_stats,
        "scores": scores,
        "base_actions": base_actions,
        "arrays": arrays,
    }


def frame_arrays(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "query_id": frame["query_id"].astype(str).to_numpy(),
        "query_text": frame["query_text"].astype(str).to_numpy(),
        "benchmark": frame["benchmark"].astype(str).to_numpy(),
        "domain": frame["domain"].astype(str).to_numpy(),
        "metric": frame["metric"].astype(str).to_numpy(),
        "vote_frac": frame["vote_frac"].to_numpy(dtype=float),
        "local_agree": frame["local_agree_with_majority_count"].to_numpy(dtype=float),
        "base_equals_self": frame["base_equals_self_majority"].to_numpy(dtype=bool),
        "scores": scores,
    }


def run_rule_grid(context: dict[str, Any], *, lambda_cost: float, self_model_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            fast_eval(
                context,
                split,
                context["base_actions"][split],
                method="pairwise_logistic_baseline",
                family="baseline",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )
        oracle_actions = np.argmax(context["metrics"][split]["utility"], axis=1)
        rows.append(
            fast_eval(
                context,
                split,
                oracle_actions,
                method="diagnostic_oracle_between_base_self_strong",
                family="diagnostic_oracle",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )

    for params in rule_grid():
        method = rule_method_name(params)
        for split in ["val", "test"]:
            actions = apply_rule(context, split, context["base_actions"][split], params)
            row = fast_eval(
                context,
                split,
                actions,
                method=method,
                family="residual_confidence_rule",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
            row.update(params_to_columns(params))
            rows.append(row)

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    selected_method = select_cost_tiebreak_method(table, eps=0.001)
    selected_row = table[table["split"].eq("val") & table["method"].eq(selected_method)].iloc[0]
    residuals = residual_table(context, params_from_row(selected_row))
    return table, residuals


def rule_grid() -> list[tuple[Any, ...]]:
    suppress_sets = ["all", "stress", "gpqa", "gpqa_mmlupro", "gpqa_mmlupro_math"]
    rescue_sets = ["none", "stress", "mmlupro", "math_mmlupro", "gpqa_mmlupro_math"]
    params: list[tuple[Any, ...]] = []
    for suppress_set in suppress_sets:
        for suppress_to in ["base", "self", "base_if_equal_else_self"]:
            for suppress_vote_frac in [0.50, 0.67, 0.90, 1.00]:
                for suppress_local_agree in [1, 2, 3]:
                    for rescue_set in rescue_sets:
                        for rescue_from in ["self", "base_self"]:
                            for rescue_vote_frac in [0.67, 1.00]:
                                for rescue_local_agree in [1, 2]:
                                    for rescue_cap in [None, 0.15, 0.25]:
                                        params.append(
                                            (
                                                suppress_set,
                                                suppress_to,
                                                float(suppress_vote_frac),
                                                int(suppress_local_agree),
                                                rescue_set,
                                                rescue_from,
                                                float(rescue_vote_frac),
                                                int(rescue_local_agree),
                                                rescue_cap,
                                            )
                                        )
    return params


def apply_rule(context: dict[str, Any], split: str, actions: np.ndarray, params: tuple[Any, ...]) -> np.ndarray:
    suppress_set, suppress_to, suppress_vote_frac, suppress_local_agree, rescue_set, rescue_from, rescue_vote_frac, rescue_local_agree, rescue_cap = params
    out = np.asarray(actions, dtype=int).copy()
    arrays = context["arrays"][split]

    suppress_mask = (
        (out == 2)
        & (arrays["vote_frac"] >= float(suppress_vote_frac))
        & (arrays["local_agree"] >= int(suppress_local_agree))
        & np.isin(arrays["benchmark"], BENCHMARK_SETS[str(suppress_set)])
    )
    target = SUPPRESS_TARGETS[str(suppress_to)]
    if target in {0, 1}:
        out[suppress_mask] = target
    else:
        out[suppress_mask] = np.where(arrays["base_equals_self"][suppress_mask], 0, 1)

    rescue_benchmarks = BENCHMARK_SETS[str(rescue_set)]
    if rescue_benchmarks:
        rescue_mask = (
            np.isin(out, RESCUE_FROM[str(rescue_from)])
            & (arrays["vote_frac"] <= float(rescue_vote_frac))
            & (arrays["local_agree"] <= int(rescue_local_agree))
            & np.isin(arrays["benchmark"], rescue_benchmarks)
        )
        rescue_indices = np.where(rescue_mask)[0]
        if rescue_cap is not None and len(rescue_indices) > int(np.floor(float(rescue_cap) * len(out))):
            margins = arrays["scores"][rescue_indices, 2] - np.maximum(
                arrays["scores"][rescue_indices, 0],
                arrays["scores"][rescue_indices, 1],
            )
            keep = set(rescue_indices[np.argsort(margins)[::-1][: int(np.floor(float(rescue_cap) * len(out)))]])
            rescue_mask = np.asarray([idx in keep for idx in range(len(out))], dtype=bool)
        out[rescue_mask] = 2
    return out


def fast_eval(
    context: dict[str, Any],
    split: str,
    actions: np.ndarray,
    *,
    method: str,
    family: str,
    lambda_cost: float,
    self_model_id: str,
) -> dict[str, Any]:
    metrics = context["metrics"][split]
    idx = np.arange(len(actions))
    quality = metrics["quality"][idx, actions]
    utility = metrics["utility"][idx, actions]
    norm_cost = metrics["norm_cost"][idx, actions]
    usd_cost = metrics["usd_cost"][idx, actions]
    latency = metrics["latency"][idx, actions]
    frontier = metrics["frontier"][idx, actions]
    local = metrics["local"][idx, actions]
    selected_models = metrics["selected_models"][idx, actions]
    oracle_utility = context["oracle_stats"][split]["utility"]
    oracle_quality = context["oracle_stats"][split]["quality"]
    unique_models, counts = np.unique(selected_models.astype(str), return_counts=True)
    return {
        "method": method,
        "family": family,
        "split": split,
        "n_queries": int(len(actions)),
        "mean_quality": float(np.mean(quality)),
        "mean_utility": float(np.mean(utility)),
        "quality_oracle_mean_quality": float(np.mean(oracle_quality)),
        "cost_oracle_mean_utility": float(np.mean(oracle_utility)),
        "quality_gap_to_oracle": float(np.mean(oracle_quality) - np.mean(quality)),
        "utility_gap_to_oracle": float(np.mean(oracle_utility) - np.mean(utility)),
        "oracle_utility_ratio": float(np.mean(utility) / np.mean(oracle_utility)),
        "remote_cost_total_usd": float(np.sum(usd_cost)),
        "normalized_remote_cost_mean": float(np.mean(norm_cost)),
        "frontier_call_rate": float(np.mean(frontier)),
        "local_call_rate": float(np.mean(local)),
        "mean_latency_s": float(np.mean(latency)),
        "p95_latency_s": float(np.quantile(latency, 0.95)),
        "lambda_cost": float(lambda_cost),
        "selected_models_json": json.dumps({str(model): int(count) for model, count in zip(unique_models, counts)}, sort_keys=True),
        "strong_call_rate": float(np.mean(selected_models == STRONG_MODEL_ID)),
        "self_action_rate": float(np.mean(selected_models == self_model_id)),
    }


def validation_selected_rows(table: pd.DataFrame, *, eps: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for rule_name, method in [
        ("strict_val_best_utility", select_strict_method(table)),
        (f"val_best_utility_cost_tiebreak_eps{eps:g}", select_cost_tiebreak_method(table, eps=eps)),
    ]:
        for split, suffix in [("val", ""), ("test", "_test")]:
            match = table[table["split"].eq(split) & table["method"].eq(method)]
            if not match.empty:
                rows.append(match.head(1).assign(selection_rule=rule_name + suffix))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def select_strict_method(table: pd.DataFrame) -> str:
    val = table[table["split"].eq("val") & table["family"].ne("diagnostic_oracle")].sort_values(
        ["mean_utility", "mean_quality"],
        ascending=False,
    )
    return str(val.iloc[0]["method"])


def select_cost_tiebreak_method(table: pd.DataFrame, *, eps: float) -> str:
    val = table[table["split"].eq("val") & table["family"].ne("diagnostic_oracle")].copy()
    best = float(val["mean_utility"].max())
    near = val[val["mean_utility"] >= best - float(eps)].copy()
    near["rescue_benchmark_count"] = near["rescue_set"].map(lambda value: len(BENCHMARK_SETS.get(str(value), tuple())))
    near = near.sort_values(
        ["frontier_call_rate", "strong_call_rate", "rescue_benchmark_count", "mean_utility", "mean_quality"],
        ascending=[True, True, False, False, False],
    )
    return str(near.iloc[0]["method"])


def residual_table(context: dict[str, Any], params: tuple[Any, ...]) -> pd.DataFrame:
    actions = apply_rule(context, "test", context["base_actions"]["test"], params)
    oracle_actions = np.argmax(context["metrics"]["test"]["utility"], axis=1)
    metrics = context["metrics"]["test"]
    arrays = context["arrays"]["test"]
    idx = np.arange(len(actions))
    selected_models = metrics["selected_models"][idx, actions]
    oracle_models = metrics["selected_models"][idx, oracle_actions]
    selected_utility = metrics["utility"][idx, actions]
    oracle_utility = metrics["utility"][idx, oracle_actions]
    return pd.DataFrame(
        {
            "query_id": arrays["query_id"],
            "benchmark": arrays["benchmark"],
            "domain": arrays["domain"],
            "metric": arrays["metric"],
            "selected_action": np.asarray(ACTIONS, dtype=object)[actions],
            "oracle_action": np.asarray(ACTIONS, dtype=object)[oracle_actions],
            "selected_model": selected_models,
            "oracle_model": oracle_models,
            "selected_utility": selected_utility,
            "oracle_action_utility": oracle_utility,
            "action_regret": oracle_utility - selected_utility,
            "selected_quality": metrics["quality"][idx, actions],
            "oracle_quality": metrics["quality"][idx, oracle_actions],
            "vote_frac": arrays["vote_frac"],
            "local_agree_with_majority_count": arrays["local_agree"],
            "query_text": arrays["query_text"],
        }
    ).sort_values("action_regret", ascending=False)


def params_to_columns(params: tuple[Any, ...]) -> dict[str, Any]:
    suppress_set, suppress_to, suppress_vote_frac, suppress_local_agree, rescue_set, rescue_from, rescue_vote_frac, rescue_local_agree, rescue_cap = params
    return {
        "suppress_set": suppress_set,
        "suppress_to": suppress_to,
        "suppress_vote_frac": float(suppress_vote_frac),
        "suppress_local_agree": int(suppress_local_agree),
        "rescue_set": rescue_set,
        "rescue_from": rescue_from,
        "rescue_vote_frac": float(rescue_vote_frac),
        "rescue_local_agree": int(rescue_local_agree),
        "rescue_cap": np.nan if rescue_cap is None else float(rescue_cap),
    }


def rule_method_name(params: tuple[Any, ...]) -> str:
    columns = params_to_columns(params)
    cap = "none" if pd.isna(columns["rescue_cap"]) else f"{columns['rescue_cap']:.2f}"
    return (
        "pairwise_logistic_residual_rule"
        f"_sup{columns['suppress_set']}-{columns['suppress_to']}"
        f"_vf{columns['suppress_vote_frac']:.2f}_la{columns['suppress_local_agree']}"
        f"_res{columns['rescue_set']}-{columns['rescue_from']}"
        f"_vf{columns['rescue_vote_frac']:.2f}_la{columns['rescue_local_agree']}_cap{cap}"
    )


def params_from_row(row: pd.Series) -> tuple[Any, ...]:
    rescue_cap = None if pd.isna(row.get("rescue_cap", np.nan)) else float(row["rescue_cap"])
    return (
        str(row["suppress_set"]),
        str(row["suppress_to"]),
        float(row["suppress_vote_frac"]),
        int(row["suppress_local_agree"]),
        str(row["rescue_set"]),
        str(row["rescue_from"]),
        float(row["rescue_vote_frac"]),
        int(row["rescue_local_agree"]),
        rescue_cap,
    )


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(out_dir: Path, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    plot = pd.concat(
        [
            selected[selected["split"].eq("test")],
            table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10),
        ],
        ignore_index=True,
    ).drop_duplicates("method")
    labels = plot["selection_rule"].fillna(plot["family"]).astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#4f728f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Residual Confidence Rule Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_residual_confidence_rule_policy.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
        "suppress_set",
        "suppress_to",
        "suppress_vote_frac",
        "suppress_local_agree",
        "rescue_set",
        "rescue_from",
        "rescue_vote_frac",
        "rescue_local_agree",
        "rescue_cap",
        "selection_rule",
    ]
    lines = [
        "# Residual Confidence Rule Policy",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "",
        "This evaluator makes no provider API or vLLM calls. It composes deterministic confidence rules on top of the validation-selected cached pairwise logistic action ranker.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=40),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(
            table[table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)[[column for column in cols if column in table.columns]],
            max_rows=24,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- The residual rule family suppresses some confident-but-wrong strong calls and rescues selected self/base actions to strong under low-confidence stress conditions.",
        "- The validation-selected lift is small; this is an incremental cached-policy improvement, not a solution to the oracle gap.",
        "- Top held-out rows remain diagnostic when selected by test utility.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
