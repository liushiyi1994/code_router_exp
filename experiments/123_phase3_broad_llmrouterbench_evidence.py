from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUESTED_BENCHMARKS = {
    "gsm8k": "GSM8K",
    "math500": "MATH500",
    "aime": "AIME",
    "humaneval": "HumanEval",
    "mbpp": "MBPP",
    "livecodebench": "LiveCodeBench",
    "gpqa": "GPQA",
    "mmlupro": "MMLU-Pro",
    "bbh": "BBH",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize broad real LLMRouterBench evidence for Phase 3.")
    parser.add_argument("--outcomes", type=Path, default=Path("data/processed/llmrouterbench_broad20/outcomes.csv"))
    parser.add_argument("--result-dir", type=Path, default=Path("results/llmrouterbench_broad20"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_csv(args.outcomes)
    coverage = build_coverage_table(outcomes)
    methods = build_method_table(args.result_dir)
    claims = build_claim_table(args.result_dir)

    coverage_path = args.output_dir / "table_phase3_broad_llmrouterbench_coverage.csv"
    method_path = args.output_dir / "table_phase3_broad_llmrouterbench_method_summary.csv"
    claim_path = args.output_dir / "table_phase3_broad_llmrouterbench_claim_status.csv"
    memo_path = args.output_dir / "PHASE3_BROAD_LLMROUTERBENCH_EVIDENCE.md"

    coverage.to_csv(coverage_path, index=False)
    methods.to_csv(method_path, index=False)
    claims.to_csv(claim_path, index=False)
    write_memo(memo_path, args.outcomes, args.result_dir, coverage, methods, claims)

    print(f"Wrote broad LLMRouterBench Phase 3 evidence to {memo_path}")
    print(summary_line(coverage))


def build_coverage_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    required = {"query_id", "dataset", "model_id", "quality", "cost_total"}
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"outcomes missing required columns: {missing}")

    rows: list[dict[str, object]] = []
    outcomes = outcomes.copy()
    outcomes["dataset_key"] = outcomes["dataset"].astype(str).str.lower()
    for dataset_key, display_name in REQUESTED_BENCHMARKS.items():
        subset = outcomes[outcomes["dataset_key"].eq(dataset_key)]
        if subset.empty:
            rows.append(
                {
                    "benchmark": display_name,
                    "dataset_key": dataset_key,
                    "present": False,
                    "query_count": 0,
                    "model_count": 0,
                    "row_count": 0,
                    "mean_quality": pd.NA,
                    "best_single_model": "",
                    "best_single_quality": pd.NA,
                    "query_oracle_quality": pd.NA,
                    "oracle_gap": pd.NA,
                    "mean_cost_total": pd.NA,
                    "source_splits": "",
                }
            )
            continue
        model_quality = subset.groupby("model_id")["quality"].mean().sort_values(ascending=False)
        best_model = str(model_quality.index[0])
        best_quality = float(model_quality.iloc[0])
        query_oracle = float(subset.groupby("query_id")["quality"].max().mean())
        source_splits = ",".join(sorted(subset.get("source_split", pd.Series(dtype=str)).astype(str).unique()))
        rows.append(
            {
                "benchmark": display_name,
                "dataset_key": dataset_key,
                "present": True,
                "query_count": int(subset["query_id"].nunique()),
                "model_count": int(subset["model_id"].nunique()),
                "row_count": int(len(subset)),
                "mean_quality": float(subset["quality"].mean()),
                "best_single_model": best_model,
                "best_single_quality": best_quality,
                "query_oracle_quality": query_oracle,
                "oracle_gap": query_oracle - best_quality,
                "mean_cost_total": float(subset["cost_total"].mean()),
                "source_splits": source_splits,
            }
        )
    all_row = {
        "benchmark": "ALL_LLMROUTERBENCH_BROAD20",
        "dataset_key": "all",
        "present": True,
        "query_count": int(outcomes["query_id"].nunique()),
        "model_count": int(outcomes["model_id"].nunique()),
        "row_count": int(len(outcomes)),
        "mean_quality": float(outcomes["quality"].mean()),
        "best_single_model": str(outcomes.groupby("model_id")["quality"].mean().idxmax()),
        "best_single_quality": float(outcomes.groupby("model_id")["quality"].mean().max()),
        "query_oracle_quality": float(outcomes.groupby("query_id")["quality"].max().mean()),
        "oracle_gap": float(outcomes.groupby("query_id")["quality"].max().mean() - outcomes.groupby("model_id")["quality"].mean().max()),
        "mean_cost_total": float(outcomes["cost_total"].mean()),
        "source_splits": ",".join(sorted(outcomes.get("source_split", pd.Series(dtype=str)).astype(str).unique())),
    }
    return pd.concat([pd.DataFrame(rows), pd.DataFrame([all_row])], ignore_index=True)


def build_method_table(result_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    routability = _read_csv(result_dir / "table_routability.csv")
    rate = _read_csv(result_dir / "table_rate_distortion.csv")
    d2 = _read_csv(result_dir / "table_predictability_constrained.csv")
    calibration = _read_csv(result_dir / "table_new_model_integration.csv")

    for method in ["best_single", "dataset_label_lookup", "query_oracle"]:
        selected = routability[routability["method"].eq(method)] if not routability.empty else pd.DataFrame()
        if not selected.empty:
            rows.append(_method_row("broad20_routability", selected.iloc[0]))

    if not rate.empty:
        regret = rate[(rate["method"].eq("regret_routecode_oracle_labels")) & (rate["K"].astype(float) <= 16)]
        if not regret.empty:
            rows.append(_method_row("broad20_low_rate_oracle_best_k_le_16", regret.sort_values("mean_utility", ascending=False).iloc[0]))
        predicted = rate[
            rate["method"].isin(
                [
                    "routecode_predicted_labels",
                    "regret_routecode_predicted_labels",
                    "routecode_mlp_predicted_labels",
                ]
            )
        ]
        if not predicted.empty:
            rows.append(_method_row("broad20_predicted_routecode_best", predicted.sort_values("mean_utility", ascending=False).iloc[0]))

    if not d2.empty:
        centroid = d2[d2["method"].eq("d2_embedding_centroid")]
        if not centroid.empty:
            rows.append(_method_row("broad20_d2_embedding_centroid_best", centroid.sort_values("mean_utility", ascending=False).iloc[0]))
        logistic = d2[d2["method"].eq("d2_logistic_label_predictor")]
        if not logistic.empty:
            rows.append(_method_row("broad20_d2_logistic_label_predictor_best", logistic.sort_values("mean_utility", ascending=False).iloc[0]))

    if not calibration.empty:
        paired = calibration[calibration["method"].isin(["routecode_label_calibration", "direct_retraining_budgeted_knn"])]
        if not paired.empty:
            rows.append(
                {
                    "evidence_family": "broad20_new_model_calibration",
                    "method": "routecode_label_calibration_vs_direct_knn",
                    "K": 16,
                    "alpha": pd.NA,
                    "mean_utility": float(
                        paired[paired["method"].eq("routecode_label_calibration")]["mean_utility"].mean()
                    ),
                    "mean_quality": float(
                        paired[paired["method"].eq("routecode_label_calibration")]["mean_quality"].mean()
                    ),
                    "normalized_cost": float(
                        paired[paired["method"].eq("routecode_label_calibration")]["normalized_cost"].mean()
                    ),
                    "recovered_gap_vs_oracle": float(
                        paired[paired["method"].eq("routecode_label_calibration")]["recovered_gap_vs_oracle"].mean()
                    ),
                    "utility_ci_low": pd.NA,
                    "utility_ci_high": pd.NA,
                    "label_accuracy": pd.NA,
                    "notes": "Mean over broad20 held-out-model calibration sweeps; compare with claim-status memo for matched differences.",
                }
            )
    return pd.DataFrame(rows)


def build_claim_table(result_dir: Path) -> pd.DataFrame:
    claims = _read_csv(result_dir / "table_claim_status.csv")
    if claims.empty:
        return pd.DataFrame(
            columns=["claim_id", "claim", "status", "primary_metric", "primary_value", "threshold", "evidence", "interpretation"]
        )
    return claims


def _method_row(evidence_family: str, row: pd.Series) -> dict[str, object]:
    return {
        "evidence_family": evidence_family,
        "method": row.get("method", ""),
        "K": row.get("K", pd.NA),
        "alpha": row.get("alpha", pd.NA),
        "mean_utility": row.get("mean_utility", pd.NA),
        "mean_quality": row.get("mean_quality", pd.NA),
        "normalized_cost": row.get("normalized_cost", pd.NA),
        "recovered_gap_vs_oracle": row.get("recovered_gap_vs_oracle", pd.NA),
        "utility_ci_low": row.get("utility_ci_low", pd.NA),
        "utility_ci_high": row.get("utility_ci_high", pd.NA),
        "label_accuracy": row.get("label_accuracy", pd.NA),
        "notes": "",
    }


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def write_memo(
    path: Path,
    outcomes_path: Path,
    result_dir: Path,
    coverage: pd.DataFrame,
    methods: pd.DataFrame,
    claims: pd.DataFrame,
) -> None:
    present = coverage[coverage["benchmark"].ne("ALL_LLMROUTERBENCH_BROAD20") & coverage["present"].astype(bool)]
    missing = coverage[coverage["benchmark"].ne("ALL_LLMROUTERBENCH_BROAD20") & ~coverage["present"].astype(bool)]
    all_row = coverage[coverage["benchmark"].eq("ALL_LLMROUTERBENCH_BROAD20")].iloc[0]
    lines = [
        "# Phase 3 Broad LLMRouterBench Evidence",
        "",
        "This memo bridges the existing `results/llmrouterbench_broad20` real outcome-matrix run into the controlled Phase 3 evidence ledger.",
        "It makes no API calls and does not replace the controlled exact-math GPT-5.5/Gemini-3.5 run.",
        "",
        "## Scope",
        "",
        f"- Outcome source: `{outcomes_path}`.",
        f"- Existing result source: `{result_dir}`.",
        f"- Broad20 matrix rows: `{int(all_row['row_count'])}` query-model rows.",
        f"- Broad20 queries: `{int(all_row['query_count'])}`.",
        f"- Broad20 models: `{int(all_row['model_count'])}`.",
        f"- Requested Phase 3 benchmarks present: `{len(present)}/{len(REQUESTED_BENCHMARKS)}`.",
        f"- Missing requested benchmark: `{', '.join(missing['benchmark'].astype(str)) if len(missing) else 'none'}`.",
        "",
        "Important limitation: this is a released LLMRouterBench outcome matrix with its own model pool. It is broad real benchmark evidence, but it is not the same model pool as the controlled exact-math API run (`gpt-5.5`, `gemini-3.5-flash`, local vLLM models).",
        "",
        "## Benchmark Coverage",
        "",
        markdown_table(coverage),
        "",
        "## Method Evidence",
        "",
        markdown_table(methods),
        "",
        "## Claim Status From Broad20",
        "",
        markdown_table(claims),
        "",
        "## Interpretation",
        "",
        "- Broad20 supports the diagnostic claim that low-rate oracle route codes exist: the existing claim audit marks `low_rate_oracle_codes` as `diagnostic_supported`.",
        "- Broad20 does not support the deployable claim that small inferred labels recover most routing performance: the existing claim audit marks `small_inferred_labels` as `not_supported`.",
        "- Broad20 keeps model-pool transfer and new-model calibration alive as diagnostics, but not as final paper-level support.",
        "- The current controlled Phase 3 story is therefore split: exact-math GPT/Gemini/local results support the configured target gates; broad20 supplies broad real benchmark diagnostics but not the same controlled model-pool result.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_line(coverage: pd.DataFrame) -> str:
    present = coverage[coverage["benchmark"].ne("ALL_LLMROUTERBENCH_BROAD20") & coverage["present"].astype(bool)]
    all_row = coverage[coverage["benchmark"].eq("ALL_LLMROUTERBENCH_BROAD20")].iloc[0]
    return (
        f"present_requested={len(present)}/{len(REQUESTED_BENCHMARKS)}; "
        f"queries={int(all_row['query_count'])}; models={int(all_row['model_count'])}; rows={int(all_row['row_count'])}"
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
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
