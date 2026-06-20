from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/controlled/broad100_target_level_method_status")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize which cached Broad100 ProbeCode-style policies reach the Phase 3 target gates, "
            "and separate target-level rows from diagnostic/tool-dependent rows."
        )
    )
    parser.add_argument(
        "--learned-verifiability",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_policy_selected.csv"
        ),
    )
    parser.add_argument(
        "--probe-state",
        type=Path,
        default=Path(
            "results/controlled/broad100_probe_state_composed_yesno_policy/"
            "table_probe_state_composed_policy_selected.csv"
        ),
    )
    parser.add_argument(
        "--conformal",
        type=Path,
        default=Path(
            "results/controlled/broad100_conformal_answer_set_probe_policy/"
            "table_conformal_answer_set_policy_selected.csv"
        ),
    )
    parser.add_argument(
        "--current-base",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_policy_variable_verifier_fusion/"
            "table_current_policy_variable_verifier_selected.csv"
        ),
    )
    parser.add_argument(
        "--learned-verifiability-heldout",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_benchmark_heldout/"
            "table_learned_verifiability_benchmark_heldout_selected.csv"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(load_current_base(args.current_base))
    rows.extend(load_learned_verifiability(args.learned_verifiability))
    rows.extend(load_probe_state(args.probe_state))
    rows.extend(load_conformal(args.conformal))
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No comparison rows were loaded.")
    table = add_target_gates(table)
    heldout = load_benchmark_heldout(args.learned_verifiability_heldout)

    table.to_csv(args.output_dir / "table_broad100_target_gate_comparison.csv", index=False)
    heldout.to_csv(args.output_dir / "table_broad100_target_gate_benchmark_heldout.csv", index=False)
    write_memo(args.output_dir / "BROAD100_TARGET_LEVEL_METHOD_STATUS.md", args, table, heldout)
    print(f"Wrote Broad100 target-level method status to {args.output_dir}")


def load_current_base(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    selected = frame[
        frame["split"].astype(str).eq("test")
        & frame["policy"].astype(str).eq("base_current_policy")
        & frame["selection_rule"].astype(str).eq("base_reference_test")
    ].copy()
    for row in selected.to_dict("records"):
        rows.append(
            normalize_row(
                source="current_policy_variable_verifier_fusion",
                method="current_base",
                family="current_base",
                row=row,
                quality_key="mean_quality",
                utility_key="mean_utility",
                oracle_quality_key="quality_oracle_mean_quality",
                oracle_utility_key="cost_oracle_mean_utility",
                frontier_key="frontier_call_rate",
                selection_rule="base_reference_test",
                status_note="current concrete base policy",
                uses_tool_specific_signal=False,
                uses_benchmark_specific_policy=False,
                diagnostic=False,
            )
        )
    return rows


def load_learned_verifiability(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    keep_methods = {
        "extratrees_d3_leaf8_thr0.5997_tool_cap_e0.75": "target-level learned verifiability global policy",
        "gb_depth2_thr0.9844_state_k8": "target-level learned verifiability state policy",
        "gb_depth2_thr0.9844_tool_always_large": "higher-utility diagnostic policy, more tool-dependent",
        "gb_depth2_thr0.9844_state_k2": "higher-utility diagnostic state policy, more tool-dependent",
    }
    rows: list[dict[str, Any]] = []
    selected = frame[frame["split"].astype(str).eq("test")].copy()
    selected = selected[selected["method"].astype(str).isin(keep_methods)]
    selected = selected[
        selected["selection_rule"].astype(str).isin(
            ["val_best_utility_test", "top_test_diagnostic"]
        )
    ].copy()
    for row in selected.to_dict("records"):
        method = str(row["method"])
        note = keep_methods[method]
        rows.append(
            normalize_row(
                source="learned_verifiability_probe_state",
                method=method,
                family=str(row.get("family", "")),
                row=row,
                quality_key="mean_quality",
                utility_key="mean_utility",
                oracle_quality_key="local_large_oracle_mean_quality",
                oracle_utility_key="local_large_oracle_mean_utility",
                frontier_key="frontier_call_rate",
                selection_rule=str(row.get("selection_rule", "")),
                status_note=note,
                uses_tool_specific_signal="tool_" in method or "tool" in str(row.get("state_policy_json", "")),
                uses_benchmark_specific_policy=False,
                diagnostic=bool(row.get("diagnostic", False)) or str(row.get("selection_rule", "")).startswith("top_test"),
                extra={
                    "large_call_rate": safe_float(row.get("large_call_rate")),
                    "need_large_precision": safe_float(row.get("need_large_precision")),
                    "need_large_recall": safe_float(row.get("need_large_recall")),
                    "state_policy_json": str(row.get("state_policy_json", "")),
                },
            )
        )
    return dedupe(rows)


def load_probe_state(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    selected = frame[
        frame["split"].astype(str).eq("test")
        & frame["family"].astype(str).isin(["probe_state_composed", "probe_state_tool_aware_diagnostic"])
        & frame["selection_rule"].astype(str).isin(["val_best_utility_test", "val_target_gate_test"])
    ].copy()
    for row in selected.to_dict("records"):
        family = str(row.get("family", ""))
        rows.append(
            normalize_row(
                source="probe_state_composed_yesno_policy",
                method=str(row["method"]),
                family=family,
                row=row,
                quality_key="mean_quality",
                utility_key="mean_utility",
                oracle_quality_key="local_large_oracle_mean_quality",
                oracle_utility_key="local_large_oracle_mean_utility",
                frontier_key="frontier_call_rate",
                selection_rule=str(row.get("selection_rule", "")),
                status_note=(
                    "no-tool/no-benchmark probe-state baseline"
                    if family == "probe_state_composed"
                    else "tool-aware positive-control probe-state diagnostic"
                ),
                uses_tool_specific_signal=family == "probe_state_tool_aware_diagnostic",
                uses_benchmark_specific_policy=False,
                diagnostic=bool(row.get("diagnostic", False)),
                extra={
                    "large_call_rate": safe_float(row.get("large_call_rate")),
                    "state_policy_json": str(row.get("state_policy_json", "")),
                },
            )
        )
    return dedupe(rows)


def load_conformal(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    selected = frame[
        frame["eval_split"].astype(str).eq("test")
        & frame["selection_rule"].astype(str).eq("val_best_mean_utility_test")
    ].copy()
    keep = {
        "current_base",
        "conformal_answer_set",
        "cisc_conformal_answer_set",
        "cisc_confidence_threshold",
        "self_consistency_majority_threshold",
    }
    selected = selected[selected["family"].astype(str).isin(keep)]
    for row in selected.to_dict("records"):
        rows.append(
            normalize_row(
                source="conformal_answer_set_probe_policy",
                method=str(row["method"]),
                family=str(row["family"]),
                row=row,
                quality_key="mean_quality",
                utility_key="mean_utility",
                oracle_quality_key="oracle_mean_quality",
                oracle_utility_key="oracle_mean_utility",
                frontier_key="frontier_call_rate",
                selection_rule=str(row.get("selection_rule", "")),
                status_note="answer-set / confidence-informed self-consistency probe",
                uses_tool_specific_signal=False,
                uses_benchmark_specific_policy=False,
                diagnostic=False,
            )
        )
    return dedupe(rows)


def load_benchmark_heldout(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    selected = frame[
        frame["split"].astype(str).eq("test")
        & frame["selection_rule"].astype(str).eq("val_best_utility_test")
    ].copy()
    if selected.empty:
        return pd.DataFrame()
    summary = (
        selected.groupby(["family", "method", "diagnostic"], dropna=False, as_index=False)
        .agg(
            n_heldout_benchmarks=("heldout_benchmark", "nunique"),
            mean_quality=("mean_quality", "mean"),
            mean_utility=("mean_utility", "mean"),
            mean_oracle_utility_ratio=("oracle_utility_ratio", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
            mean_large_call_rate=("large_call_rate", "mean"),
        )
        .sort_values("mean_utility", ascending=False)
    )
    return summary


def normalize_row(
    *,
    source: str,
    method: str,
    family: str,
    row: dict[str, Any],
    quality_key: str,
    utility_key: str,
    oracle_quality_key: str,
    oracle_utility_key: str,
    frontier_key: str,
    selection_rule: str,
    status_note: str,
    uses_tool_specific_signal: bool,
    uses_benchmark_specific_policy: bool,
    diagnostic: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    oracle_quality = safe_float(row.get(oracle_quality_key))
    oracle_utility = safe_float(row.get(oracle_utility_key))
    quality = safe_float(row.get(quality_key))
    utility = safe_float(row.get(utility_key))
    out = {
        "source": source,
        "family": family,
        "method": method,
        "selection_rule": selection_rule,
        "n_queries": int(row.get("n_queries", 0) or 0),
        "mean_quality": quality,
        "mean_utility": utility,
        "oracle_mean_quality": oracle_quality,
        "oracle_mean_utility": oracle_utility,
        "quality_gap_to_oracle": oracle_quality - quality,
        "utility_gap_to_oracle": oracle_utility - utility,
        "oracle_utility_ratio": utility / oracle_utility if oracle_utility else float("nan"),
        "frontier_call_rate": safe_float(row.get(frontier_key)),
        "diagnostic": bool(diagnostic),
        "uses_tool_specific_signal": bool(uses_tool_specific_signal),
        "uses_benchmark_specific_policy": bool(uses_benchmark_specific_policy),
        "status_note": status_note,
    }
    if extra:
        out.update(extra)
    return out


def add_target_gates(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["quality_target"] = out["oracle_mean_quality"] - 0.03
    out["utility_95pct_target"] = out["oracle_mean_utility"] * 0.95
    out["utility_97pct_target"] = out["oracle_mean_utility"] * 0.97
    out["meets_3pt_quality"] = out["mean_quality"] >= out["quality_target"]
    out["meets_95pct_utility"] = out["mean_utility"] >= out["utility_95pct_target"]
    out["meets_97pct_utility"] = out["mean_utility"] >= out["utility_97pct_target"]
    out["meets_frontier_cap_0p40"] = out["frontier_call_rate"] <= 0.40
    out["meets_primary_numeric_target"] = (
        out["meets_3pt_quality"] & out["meets_95pct_utility"] & out["meets_frontier_cap_0p40"]
    )
    out["headline_suitability"] = "not_suitable"
    clean = (
        out["meets_primary_numeric_target"]
        & ~out["diagnostic"]
        & ~out["uses_benchmark_specific_policy"]
        & ~out["uses_tool_specific_signal"]
    )
    out.loc[clean, "headline_suitability"] = "clean_numeric_pass"
    tool_dependent = out["meets_primary_numeric_target"] & out["uses_tool_specific_signal"]
    out.loc[tool_dependent, "headline_suitability"] = "numeric_pass_tool_dependent"
    diagnostic = out["meets_primary_numeric_target"] & out["diagnostic"]
    out.loc[diagnostic, "headline_suitability"] = "numeric_pass_diagnostic"
    return out.sort_values(
        ["meets_primary_numeric_target", "diagnostic", "uses_tool_specific_signal", "mean_utility"],
        ascending=[False, True, True, False],
    )


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, heldout: pd.DataFrame) -> None:
    display_cols = [
        "source",
        "family",
        "method",
        "selection_rule",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "meets_primary_numeric_target",
        "headline_suitability",
        "status_note",
    ]
    clean_pass = table[table["headline_suitability"].eq("clean_numeric_pass")]
    target_pass = table[table["meets_primary_numeric_target"]].copy()
    learned_state = table[table["method"].eq("gb_depth2_thr0.9844_state_k8")].head(1)
    no_tool = table[
        table["source"].eq("probe_state_composed_yesno_policy")
        & table["family"].eq("probe_state_composed")
    ].sort_values("mean_utility", ascending=False).head(1)
    lines = [
        "# Broad100 Target-Level Method Status",
        "",
        "This memo answers whether a small modification can make the current ProbeCode/RouteCode-style method reach the Phase 3 target gates.",
        "It uses only cached Broad100 artifacts and makes no provider, vLLM, or local generation calls.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/212_broad100_target_level_method_status.py",
        "```",
        "",
        "## Short Answer",
        "",
        "- Yes, the cached learned-verifiability modification reaches the numerical target on the standard Broad100 split.",
        "- No, the no-tool/no-benchmark probe-state method does not reach the target.",
        "- Exact oracle-level performance is not realistic without outcome leakage, benchmark-specific verifiers, or a stronger new probe signal.",
        "",
        "## Target-Gate Comparison",
        "",
        markdown_table(table[display_cols]),
        "",
        "## Main Evidence",
        "",
    ]
    if not learned_state.empty:
        row = learned_state.iloc[0]
        lines.extend(
            [
                (
                    f"- Validation-selected learned-verifiability state `{row['method']}` reaches "
                    f"quality `{row['mean_quality']:.4f}` and utility `{row['mean_utility']:.4f}` "
                    f"against oracle quality `{row['oracle_mean_quality']:.4f}` and utility "
                    f"`{row['oracle_mean_utility']:.4f}`."
                ),
                (
                    f"- That is a quality gap of `{row['quality_gap_to_oracle']:.4f}` and an oracle-utility ratio "
                    f"of `{row['oracle_utility_ratio']:.4f}` with frontier-call rate `{row['frontier_call_rate']:.4f}`."
                ),
            ]
        )
    if not no_tool.empty:
        row = no_tool.iloc[0]
        lines.extend(
            [
                (
                    f"- The best selected no-tool/no-benchmark probe-state row in this comparison reaches only "
                    f"quality `{row['mean_quality']:.4f}`, utility `{row['mean_utility']:.4f}`, and "
                    f"oracle-utility ratio `{row['oracle_utility_ratio']:.4f}`."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The modification that works is not another answer-agreement threshold. It is a learned verifiability state that routes verifiable/easy states away from expensive frontier calls and routes uncertain states to large actions.",
            "",
            "However, several target-level rows still choose policies with `tool_` behavior in their state/action table. That makes them valid evidence that verifiability is observable, but not yet a clean benchmark-agnostic main claim. The clean no-tool probe-state result is below target, so the next real method step is to replace tool-specific verifiability with broader candidate-answer reliability.",
            "",
            "## Benchmark-Heldout Check",
            "",
        ]
    )
    if heldout.empty:
        lines.append("No benchmark-heldout summary was available.")
    else:
        heldout_cols = [
            "family",
            "method",
            "diagnostic",
            "n_heldout_benchmarks",
            "mean_quality",
            "mean_utility",
            "mean_oracle_utility_ratio",
            "mean_frontier_call_rate",
        ]
        lines.append(markdown_table(heldout[heldout_cols].head(12)))
        lines.extend(
            [
                "",
                "Benchmark-heldout rows are weaker evidence than the standard split because each held-out benchmark has only its test slice. They are included to show whether the learned verifiability behavior transfers rather than merely fitting one benchmark family.",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Comparison table: `{path.parent / 'table_broad100_target_gate_comparison.csv'}`",
            f"- Benchmark-heldout summary: `{path.parent / 'table_broad100_target_gate_benchmark_heldout.csv'}`",
            f"- Source learned-verifiability table: `{args.learned_verifiability}`",
            f"- Source no-tool probe-state table: `{args.probe_state}`",
            f"- Source conformal answer-set table: `{args.conformal}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    rows = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("source")), str(row.get("method")), str(row.get("selection_rule")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    main()
