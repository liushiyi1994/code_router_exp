from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-heldout transfer for learned verifiability ProbeCode. "
            "Each run trains and selects on other benchmarks, then evaluates on the held-out benchmark test split."
        )
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
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
        "--probe-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_learned_verifiability_benchmark_heldout"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166_for_208")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_composed_171_for_208")
    exp206 = load_module("experiments/206_probe_state_composed_yesno_policy.py", "probe_state_composed_206_for_208")
    exp207 = load_module("experiments/207_learned_verifiability_probe_state.py", "learned_verifiability_207_for_208")

    outputs = pd.read_parquet(args.outputs)
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = exp207.merge_probe_features(target, pd.read_csv(args.probe_features))
    target = exp207.add_generic_text_features(target)
    feature_columns = exp207.generic_verifiability_features(target)

    all_rows: list[pd.DataFrame] = []
    selected_rows: list[pd.DataFrame] = []
    classifier_rows: list[pd.DataFrame] = []
    benchmarks = sorted(target["benchmark"].dropna().astype(str).unique())
    for heldout in benchmarks:
        frame = target[
            ((target["split"].eq("train") | target["split"].eq("val")) & ~target["benchmark"].astype(str).eq(heldout))
            | (target["split"].eq("test") & target["benchmark"].astype(str).eq(heldout))
        ].copy()
        scored, classifier_summary = exp207.fit_verifiability_models(frame, feature_columns, seed=int(args.seed))
        table, _score_table, _assignments, _cards = exp207.evaluate_learned_verifiability(
            scored,
            rc166,
            exp171,
            exp206,
            lambda_cost=float(args.lambda_cost),
            seed=int(args.seed),
        )
        selected = exp206.selected_rows(table, rc166, int(args.bootstrap_samples), int(args.seed))
        all_rows.append(table.assign(heldout_benchmark=heldout))
        selected_rows.append(selected.assign(heldout_benchmark=heldout))
        classifier_rows.append(classifier_summary.assign(heldout_benchmark=heldout))

    all_table = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    selected_table = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    classifier_table = pd.concat(classifier_rows, ignore_index=True) if classifier_rows else pd.DataFrame()
    summary = summarize(selected_table)

    all_table.to_csv(args.output_dir / "table_learned_verifiability_benchmark_heldout_all.csv", index=False)
    selected_table.to_csv(args.output_dir / "table_learned_verifiability_benchmark_heldout_selected.csv", index=False)
    classifier_table.to_csv(args.output_dir / "table_learned_verifiability_benchmark_heldout_classifier_summary.csv", index=False)
    summary.to_csv(args.output_dir / "table_learned_verifiability_benchmark_heldout_summary.csv", index=False)
    write_memo(args.output_dir / "LEARNED_VERIFIABILITY_BENCHMARK_HELDOUT_MEMO.md", args, summary, selected_table, classifier_table)
    print(f"Wrote learned verifiability benchmark-heldout results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summarize(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    test = selected[
        selected["split"].eq("test")
        & selected["selection_rule"].astype(str).isin(["val_best_utility_test", "val_target_gate_test"])
    ].copy()
    if test.empty:
        return pd.DataFrame()
    return (
        test.groupby(["family", "selection_rule"], as_index=False)
        .agg(
            n_heldout=("heldout_benchmark", "nunique"),
            mean_heldout_quality=("mean_quality", "mean"),
            mean_heldout_utility=("mean_utility", "mean"),
            mean_oracle_ratio=("oracle_utility_ratio", "mean"),
            mean_large_call_rate=("large_call_rate", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
        )
        .sort_values(["mean_heldout_utility", "mean_heldout_quality"], ascending=False)
    )


def write_memo(path: Path, args: argparse.Namespace, summary: pd.DataFrame, selected: pd.DataFrame, classifier: pd.DataFrame) -> None:
    top_selected = selected[
        selected["split"].eq("test")
        & selected["selection_rule"].astype(str).isin(["val_best_utility_test", "val_target_gate_test"])
    ].copy()
    cols = [
        "heldout_benchmark",
        "method",
        "family",
        "selection_rule",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "large_call_rate",
        "frontier_call_rate",
        "diagnostic",
    ]
    classifier_test = classifier[classifier["split"].eq("test")].copy()
    lines = [
        "# Learned Verifiability Benchmark-Heldout Transfer",
        "",
        "This cached experiment trains learned verifiability on all non-held-out benchmarks, selects policy settings on non-held-out validation rows, and evaluates on the held-out benchmark test split.",
        "",
        "No GPT, Gemini, Claude, local generation, or vLLM calls are made.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/208_learned_verifiability_benchmark_heldout.py",
        f"PYTHONPATH=src python experiments/208_learned_verifiability_benchmark_heldout.py --target-table {args.target_table} --outputs {args.outputs} --probe-features {args.probe_features} --output-dir {args.output_dir}",
        "```",
        "",
        "## Mean Held-Out Summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip() if not summary.empty else "",
        "```",
        "",
        "## Selected Held-Out Rows",
        "",
        "```csv",
        top_selected[[col for col in cols if col in top_selected.columns]].to_csv(index=False).strip() if not top_selected.empty else "",
        "```",
        "",
        "## Classifier Test Summary",
        "",
        "```csv",
        classifier_test.to_csv(index=False).strip() if not classifier_test.empty else "",
        "```",
        "",
        "## Interpretation",
        "",
        "- Learned verifiability is benchmark-heldout useful only if learned rows stay close to the direct tool-flag positive control.",
        "- If standard-split target results degrade here, the method is learning benchmark or answer-form regularities that do not transfer reliably enough.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
