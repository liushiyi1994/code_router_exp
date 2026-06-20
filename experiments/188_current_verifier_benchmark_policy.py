from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark-composed policy over cached current-action verifier policies."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_action_verifier_qwen14b/"
            "table_current_action_verifier_features.csv"
        ),
    )
    parser.add_argument(
        "--probe",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_action_verifier_qwen14b/"
            "table_current_action_verifier_probe.csv"
        ),
    )
    parser.add_argument(
        "--oracle-targets",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_action_verifier_qwen14b/"
            "table_current_action_oracle_targets.csv"
        ),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_current_verifier_benchmark_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--methods", default="pred_rf_thr-0.0288")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.oracle_targets)
    features = pd.read_csv(args.features)
    probe = pd.read_csv(args.probe)
    outputs = pd.read_parquet(args.outputs)
    exp187 = load_module("experiments/187_current_action_verifier_vllm.py", "current_verifier_187_for_188")
    exp183 = load_module("experiments/183_local_safe_gain_gate.py", "local_safe_183_for_188")
    outputs = outputs.copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    oracle = (
        outputs.groupby("query_id", as_index=False)
        .agg(
            cost_oracle_mean_utility=("utility", "max"),
            quality_oracle_mean_quality=("quality_score", "max"),
        )
    )
    choices = reconstruct_choices(features, probe, outputs, targets, exp187, exp183, args.methods)
    choices = choices.merge(oracle, on="query_id", how="left")
    choices["regret"] = choices["cost_oracle_mean_utility"] - choices["utility"].astype(float)

    library = summarize_library(choices)
    policy_maps = build_policy_maps(choices, library)
    composed_choices = compose_choices(choices, policy_maps)
    policy_table = summarize_policies(composed_choices, args.bootstrap_samples, args.seed)
    selected = selected_rows(policy_table)

    library.to_csv(args.output_dir / "table_current_verifier_benchmark_library.csv", index=False)
    pd.DataFrame(policy_maps).to_csv(args.output_dir / "table_current_verifier_benchmark_policy_map.csv", index=False)
    composed_choices.to_csv(args.output_dir / "table_current_verifier_benchmark_query_choices.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_current_verifier_benchmark_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_current_verifier_benchmark_policy_selected.csv", index=False)
    write_memo(args.output_dir / "CURRENT_VERIFIER_BENCHMARK_POLICY_MEMO.md", args, policy_maps, selected)
    print(f"Wrote benchmark current-verifier policies to {args.output_dir}")


def summarize_library(choices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (split, benchmark, method), group in choices.groupby(["split", "benchmark", "method"], dropna=False):
        rows.append(
            {
                "split": split,
                "benchmark": benchmark,
                "method": method,
                "n_queries": len(group),
                "mean_quality": group["quality_score"].astype(float).mean(),
                "mean_utility": group["utility"].astype(float).mean(),
                "frontier_call_rate": group["is_frontier"].astype(bool).mean(),
                "strong_or_frontier_call_rate": group["selected_model_id"].astype(str).map(is_strong_or_frontier).mean(),
                "regret": group["regret"].astype(float).mean(),
            }
        )
    return pd.DataFrame(rows)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reconstruct_choices(
    features: pd.DataFrame,
    probe: pd.DataFrame,
    outputs: pd.DataFrame,
    targets: pd.DataFrame,
    exp187,
    exp183,
    methods: str,
) -> pd.DataFrame:
    available = {
        str(query_id): set(group["model_id"].astype(str))
        for query_id, group in outputs.groupby("query_id", sort=False)
    }
    probe_map = probe.set_index(["query_id", "method"]).to_dict("index")
    specs: list[dict[str, Any]] = [{"method": "base_candidate_ranker", "family": "reference", "kind": "base"}]
    for local_method in parse_csv(methods):
        parsed = parse_method(local_method)
        if parsed is None:
            continue
        method, pred_col, threshold = parsed
        specs.append(
            {
                "method": f"local_{method}",
                "family": "local_safe_reference",
                "kind": "local",
                "local_method": method,
                "pred_col": pred_col,
                "local_threshold": threshold,
            }
        )
        for confidence in [0.0, 0.5, 0.7, 0.85, 0.95]:
            for kind, family in [
                ("switch", "current_action_verifier_switch"),
                ("reject_to_base", "current_action_verifier_reject_to_base"),
                ("accept_required", "current_action_verifier_accept_required"),
            ]:
                specs.append(
                    {
                        "method": f"current_verifier_{kind}_conf{confidence:g}_{method}",
                        "family": family,
                        "kind": kind,
                        "local_method": method,
                        "pred_col": pred_col,
                        "local_threshold": threshold,
                        "confidence": confidence,
                    }
                )
    frames: list[pd.DataFrame] = []
    target_cols = targets[["query_id", "split", "need_large"]].drop_duplicates("query_id")
    for spec in specs:
        for split in ["val", "test"]:
            split_features = features[features["split"].astype(str).eq(split)].copy()
            choices = exp187.choose_policy_actions(split_features, spec, exp183, probe_map, available)
            if choices.empty:
                continue
            selected = choices[["query_id", "model_id"]].merge(outputs, on=["query_id", "model_id"], how="left")
            selected = selected[selected["split"].astype(str).eq(split)].copy()
            selected = selected.merge(choices.drop(columns=["model_id"]), on="query_id", how="left")
            selected = selected.merge(target_cols, on=["query_id", "split"], how="left")
            selected = selected.rename(columns={"model_id": "selected_model_id"})
            selected["method"] = str(spec["method"])
            selected["family"] = str(spec["family"])
            frames.append(selected)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_method(method: str) -> tuple[str, str, float] | None:
    match = re.fullmatch(r"(pred_[A-Za-z0-9]+)_thr(-?[0-9.]+)", method)
    if not match:
        return None
    return method, match.group(1), float(match.group(2))


def build_policy_maps(choices: pd.DataFrame, library: pd.DataFrame) -> list[dict[str, object]]:
    val = library[library["split"].eq("val")].copy()
    global_scores = (
        choices[choices["split"].eq("val")]
        .groupby("method", as_index=False)
        .agg(mean_utility=("utility", "mean"), frontier_call_rate=("is_frontier", "mean"))
        .sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True])
    )
    global_best = str(global_scores.iloc[0]["method"])
    local_reference = "local_pred_rf_thr-0.0288"
    maps: list[dict[str, object]] = []
    benchmarks = sorted(val["benchmark"].dropna().astype(str).unique())

    for fallback in [global_best, local_reference]:
        maps.append(make_global_map(f"global_{fallback}", benchmarks, fallback))

    for eps in [0.0, 0.005, 0.01, 0.02]:
        maps.extend(
            [
                make_benchmark_map(
                    val,
                    benchmarks,
                    fallback=global_best,
                    name=f"benchmark_best_eps{eps:g}_fallback_global",
                    eps=eps,
                    frontier_cap=None,
                ),
                make_benchmark_map(
                    val,
                    benchmarks,
                    fallback=local_reference,
                    name=f"benchmark_best_eps{eps:g}_fallback_local",
                    eps=eps,
                    frontier_cap=None,
                ),
                make_benchmark_map(
                    val,
                    benchmarks,
                    fallback=global_best,
                    name=f"benchmark_frontiercap0.40_eps{eps:g}_fallback_global",
                    eps=eps,
                    frontier_cap=0.40,
                ),
                make_benchmark_map(
                    val,
                    benchmarks,
                    fallback=local_reference,
                    name=f"benchmark_frontiercap0.40_eps{eps:g}_fallback_local",
                    eps=eps,
                    frontier_cap=0.40,
                ),
            ]
        )
    return maps


def make_global_map(name: str, benchmarks: list[str], method: str) -> dict[str, object]:
    mapping = {benchmark: method for benchmark in benchmarks}
    return {"policy": name, "fallback": method, "eps": 0.0, "frontier_cap": np.nan, "mapping_json": json.dumps(mapping, sort_keys=True)}


def make_benchmark_map(
    val: pd.DataFrame,
    benchmarks: list[str],
    *,
    fallback: str,
    name: str,
    eps: float,
    frontier_cap: float | None,
) -> dict[str, object]:
    mapping: dict[str, str] = {}
    for benchmark in benchmarks:
        subset = val[val["benchmark"].astype(str).eq(benchmark)].copy()
        if frontier_cap is not None:
            capped = subset[subset["frontier_call_rate"].astype(float).le(frontier_cap)].copy()
            if not capped.empty:
                subset = capped
        fallback_score = subset.loc[subset["method"].astype(str).eq(fallback), "mean_utility"]
        baseline = float(fallback_score.max()) if not fallback_score.empty else -np.inf
        subset = subset.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True])
        if subset.empty:
            mapping[benchmark] = fallback
            continue
        best = subset.iloc[0]
        if float(best["mean_utility"]) >= baseline + float(eps):
            mapping[benchmark] = str(best["method"])
        else:
            mapping[benchmark] = fallback
    return {
        "policy": name,
        "fallback": fallback,
        "eps": eps,
        "frontier_cap": np.nan if frontier_cap is None else frontier_cap,
        "mapping_json": json.dumps(mapping, sort_keys=True),
    }


def compose_choices(choices: pd.DataFrame, policy_maps: list[dict[str, object]]) -> pd.DataFrame:
    key = choices.set_index(["split", "benchmark", "query_id", "method"])
    rows: list[pd.DataFrame] = []
    for policy_map in policy_maps:
        mapping = json.loads(str(policy_map["mapping_json"]))
        frames: list[pd.DataFrame] = []
        for split in ["val", "test"]:
            for benchmark, method in mapping.items():
                try:
                    part = key.loc[(split, benchmark, slice(None), method), :].reset_index()
                except KeyError:
                    continue
                part = part.copy()
                part["policy"] = str(policy_map["policy"])
                part["mapped_method"] = method
                frames.append(part)
        if frames:
            rows.append(pd.concat(frames, ignore_index=True))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_policies(choices: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (policy, split), group in choices.groupby(["policy", "split"], dropna=False):
        utility = group["utility"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(utility, bootstrap_samples, seed)
        selected_counts = group["selected_model_id"].astype(str).value_counts().sort_index().to_dict()
        tp = int(((group["need_large"].astype(bool)) & (group["selected_model_id"].astype(str).map(is_large_action))).sum())
        fp = int(((~group["need_large"].astype(bool)) & (group["selected_model_id"].astype(str).map(is_large_action))).sum())
        fn = int(((group["need_large"].astype(bool)) & (~group["selected_model_id"].astype(str).map(is_large_action))).sum())
        rows.append(
            {
                "policy": policy,
                "split": split,
                "n_queries": len(group),
                "mean_quality": group["quality_score"].astype(float).mean(),
                "mean_utility": float(utility.mean()),
                "mean_utility_ci_low": ci_low,
                "mean_utility_ci_high": ci_high,
                "normalized_cost_mean": group["normalized_remote_cost"].astype(float).mean(),
                "cost_oracle_mean_utility": group["cost_oracle_mean_utility"].astype(float).mean(),
                "quality_oracle_mean_quality": group["quality_oracle_mean_quality"].astype(float).mean(),
                "utility_gap_to_oracle": group["regret"].astype(float).mean(),
                "quality_gap_to_oracle": (
                    group["quality_oracle_mean_quality"].astype(float) - group["quality_score"].astype(float)
                ).mean(),
                "oracle_utility_ratio": float(utility.mean() / group["cost_oracle_mean_utility"].astype(float).mean()),
                "within_3pct_oracle_utility": bool(
                    utility.mean() >= 0.97 * group["cost_oracle_mean_utility"].astype(float).mean()
                ),
                "within_3pt_oracle_quality": bool(
                    group["mean_quality"].mean() if "mean_quality" in group else False
                ),
                "frontier_call_rate": group["is_frontier"].astype(bool).mean(),
                "strong_or_frontier_call_rate": group["selected_model_id"].astype(str).map(is_strong_or_frontier).mean(),
                "local_call_rate": (~group["selected_model_id"].astype(str).map(is_large_action)).mean(),
                "need_large_precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "need_large_recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "selected_models_json": json.dumps(selected_counts, sort_keys=True),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table["within_3pt_oracle_quality"] = table["quality_gap_to_oracle"].le(0.03)
    return table.sort_values(["split", "mean_utility"], ascending=[True, False])


def selected_rows(policy_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    val = policy_table[policy_table["split"].eq("val")].copy()
    test = policy_table[policy_table["split"].eq("test")].copy()
    if val.empty or test.empty:
        return pd.DataFrame()
    selectors = {
        "val_best_mean_utility": val.sort_values(
            ["mean_utility", "frontier_call_rate", "strong_or_frontier_call_rate"],
            ascending=[False, True, True],
        ).iloc[0],
        "val_target_quality_then_utility": val.sort_values(
            ["within_3pt_oracle_quality", "mean_utility", "frontier_call_rate"],
            ascending=[False, False, True],
        ).iloc[0],
    }
    capped = val[val["frontier_call_rate"].le(0.40)].copy()
    if not capped.empty:
        selectors["val_best_frontier_le_0.40"] = capped.sort_values(
            ["mean_utility", "frontier_call_rate"],
            ascending=[False, True],
        ).iloc[0]
    for rule, row in selectors.items():
        val_row = row.copy()
        val_row["selection_rule"] = rule
        rows.append(val_row)
        test_match = test[test["policy"].astype(str).eq(str(row["policy"]))]
        if not test_match.empty:
            test_row = test_match.iloc[0].copy()
            test_row["selection_rule"] = f"{rule}_test"
            rows.append(test_row)
    best_test = test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(5).copy()
    for _, row in best_test.iterrows():
        row = row.copy()
        row["selection_rule"] = "top_test_diagnostic"
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(int(samples)):
        idx = rng.integers(0, len(values), len(values))
        means.append(float(values[idx].mean()))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def is_large_action(model_id: str) -> bool:
    return is_strong_or_frontier(model_id)


def is_strong_or_frontier(model_id: str) -> bool:
    value = str(model_id)
    return (
        value in {"qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local"}
        or value.startswith("gemini-")
        or value.startswith("gpt-")
    )


def write_memo(path: Path, args: argparse.Namespace, policy_maps: list[dict[str, object]], selected: pd.DataFrame) -> None:
    lines = [
        "# Current-Verifier Benchmark Policy",
        "",
        "This is a no-new-call composition over cached Experiment 187 current-action verifier policies.",
        "It selects one cached verifier policy per benchmark using validation only.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/188_current_verifier_benchmark_policy.py",
        "```",
        "",
        "## Inputs",
        "",
        f"- Features: `{args.features}`",
        f"- Probe rows: `{args.probe}`",
        f"- Oracle targets: `{args.oracle_targets}`",
        f"- Outputs: `{args.outputs}`",
        f"- Candidate policy maps: `{len(policy_maps)}`",
        "",
        "## Selected Rows",
        "",
    ]
    if selected.empty:
        lines.append("No selected rows were produced.")
    else:
        cols = [
            "selection_rule",
            "policy",
            "split",
            "n_queries",
            "mean_quality",
            "mean_utility",
            "oracle_utility_ratio",
            "frontier_call_rate",
            "strong_or_frontier_call_rate",
        ]
        lines.append(markdown_table(selected[cols]))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A deployable win must beat Experiment 187's validation-selected held-out utility `0.7678`.",
            "- This experiment does not make new model calls; it only tests whether global current-verifier thresholds were too blunt.",
            "- Benchmark composition should be treated cautiously because prior benchmark-level composition overfit validation.",
            "",
            "## Artifacts",
            "",
            f"- Library: `{path.parent / 'table_current_verifier_benchmark_library.csv'}`",
            f"- Policy map: `{path.parent / 'table_current_verifier_benchmark_policy_map.csv'}`",
            f"- All policies: `{path.parent / 'table_current_verifier_benchmark_policy_all.csv'}`",
            f"- Selected policies: `{path.parent / 'table_current_verifier_benchmark_policy_selected.csv'}`",
            f"- Query choices: `{path.parent / 'table_current_verifier_benchmark_query_choices.csv'}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
