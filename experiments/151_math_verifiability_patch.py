from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
TOOL_MODEL_ID = "deterministic_math_tool"
MATH_BENCHMARKS = {"math500", "livemathbench"}
LOCAL_MODELS = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached LiveMathBench/MATH500 verifiability fallback patches.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_math_verifiability_patch"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_feature_gate")
    outputs = self_gate.load_outputs(args.outputs)
    table, selected, math_details = run_patches(
        package,
        outputs,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
    )
    table.to_csv(args.output_dir / "table_math_verifiability_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_math_verifiability_selected.csv", index=False)
    math_details.to_csv(args.output_dir / "table_math_verifiability_details.csv", index=False)
    write_memo(args.output_dir / "MATH_VERIFIABILITY_PATCH_MEMO.md", args, table, selected, math_details)
    print(f"Wrote math verifiability patch results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_patches(
    package,
    outputs: pd.DataFrame,
    *,
    self_model_id: str,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([self_model_id, STRONG_MODEL_ID])].copy()
    base_specs = {
        "observable_local_state_v5": lambda split: package.observable_local_state_selection(outputs_no_self, split=split),
        "observable_local_state_v5_no_strong": lambda split: package.observable_local_state_selection(
            outputs_no_strong_self, split=split
        ),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
    }
    train_best = train_best_local_by_benchmark(outputs)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    by_query = outputs.set_index(["query_id", "model_id"])
    for base_name, builder in base_specs.items():
        base = {split: normalize_selection(builder(split)) for split in ["val", "test"]}
        for split in ["val", "test"]:
            candidates: dict[str, tuple[str, pd.Series]] = {base_name: ("base", base[split])}
            for mode in [
                "local_majority_if_frontier",
                "qwen4_if_frontier",
                "train_best_local_if_frontier",
                "local_majority_else_qwen4_if_frontier",
                "local_majority_else_train_best_if_frontier",
            ]:
                candidates[f"{base_name}_math_{mode}"] = (
                    "math_verifiability_patch",
                    patch_math_frontier_calls(package, outputs, by_query, base[split], split=split, mode=mode, train_best=train_best),
                )
            candidates[f"{base_name}_math_local_oracle_if_frontier"] = (
                "diagnostic_oracle",
                patch_math_frontier_with_oracle(outputs, by_query, base[split], split=split),
            )
            for method, (family, selected) in candidates.items():
                for scope in ["all", "math_only"]:
                    row = evaluate_selection(
                        package,
                        outputs,
                        selected,
                        split=split,
                        method=method,
                        family=family,
                        scope=scope,
                        lambda_cost=lambda_cost,
                    )
                    if row:
                        rows.append(row)
                details.append(math_detail_rows(package, outputs, selected, split=split, method=method, family=family))
    table = pd.DataFrame(rows).sort_values(["scope", "split", "mean_utility", "mean_quality"], ascending=[True, True, False, False])
    selected = validation_selected_rows(table)
    detail_table = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    return table, selected, detail_table


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def train_best_local_by_benchmark(outputs: pd.DataFrame) -> dict[str, str]:
    train = outputs[outputs["split"].eq("train") & outputs["benchmark"].isin(MATH_BENCHMARKS) & outputs["model_id"].isin(LOCAL_MODELS)]
    if train.empty:
        return {}
    table = (
        train.groupby(["benchmark", "model_id"], as_index=False)
        .agg(mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"))
        .sort_values(["benchmark", "mean_utility", "mean_quality"], ascending=[True, False, False])
        .drop_duplicates("benchmark")
        .set_index("benchmark")["model_id"]
    )
    return {str(benchmark): str(model_id) for benchmark, model_id in table.items()}


def patch_math_frontier_calls(
    package,
    outputs: pd.DataFrame,
    by_query: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    mode: str,
    train_best: dict[str, str],
) -> pd.Series:
    patched = normalize_selection(selected)
    queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    for query_id, row in queries.iterrows():
        query_id = str(query_id)
        benchmark = str(row.get("benchmark", ""))
        if benchmark not in MATH_BENCHMARKS:
            continue
        current = str(patched.get(query_id, ""))
        if not is_frontier_or_strong(by_query, query_id, current):
            continue
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            patched.loc[query_id] = TOOL_MODEL_ID
            continue
        replacement = replacement_model(by_query, query_id, benchmark=benchmark, mode=mode, train_best=train_best)
        if replacement:
            patched.loc[query_id] = replacement
    return patched


def replacement_model(
    by_query: pd.DataFrame,
    query_id: str,
    *,
    benchmark: str,
    mode: str,
    train_best: dict[str, str],
) -> str | None:
    majority = local_majority_model(by_query, query_id)
    if mode == "local_majority_if_frontier":
        return majority
    if mode == "qwen4_if_frontier":
        return "qwen3-4b-local"
    if mode == "train_best_local_if_frontier":
        return train_best.get(benchmark, "qwen3-4b-local")
    if mode == "local_majority_else_qwen4_if_frontier":
        return majority or "qwen3-4b-local"
    if mode == "local_majority_else_train_best_if_frontier":
        return majority or train_best.get(benchmark, "qwen3-4b-local")
    raise ValueError(mode)


def is_frontier_or_strong(by_query: pd.DataFrame, query_id: str, model_id: str) -> bool:
    if model_id == STRONG_MODEL_ID:
        return True
    try:
        row = by_query.loc[(query_id, model_id)]
    except KeyError:
        return False
    return bool(row.get("is_frontier", False))


def local_majority_model(by_query: pd.DataFrame, query_id: str) -> str | None:
    answers: list[tuple[str, str]] = []
    for model_id in LOCAL_MODELS:
        try:
            row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        answer = normalize_answer(row.get("parsed_answer", ""))
        if answer:
            answers.append((model_id, answer))
    if not answers:
        return None
    counts: dict[str, int] = {}
    for _, answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    answer, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    if count < 2:
        return None
    for model_id, candidate in answers:
        if candidate == answer:
            return model_id
    return None


def normalize_answer(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "no_code" or text.startswith("failed"):
        return ""
    return text


def patch_math_frontier_with_oracle(
    outputs: pd.DataFrame,
    by_query: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
) -> pd.Series:
    patched = normalize_selection(selected)
    queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    candidates = [TOOL_MODEL_ID, *LOCAL_MODELS]
    for query_id, row in queries.iterrows():
        query_id = str(query_id)
        if str(row.get("benchmark", "")) not in MATH_BENCHMARKS:
            continue
        current = str(patched.get(query_id, ""))
        if not is_frontier_or_strong(by_query, query_id, current):
            continue
        best_model = current
        best_utility = -float("inf")
        best_quality = -float("inf")
        for model_id in candidates:
            key = (query_id, model_id)
            if key not in by_query.index:
                continue
            item = by_query.loc[key]
            if model_id == TOOL_MODEL_ID and not bool(item.get("tool_available", False)):
                continue
            utility = float(item.get("utility", 0.0))
            quality = float(item.get("quality_score", 0.0))
            if utility > best_utility or (abs(utility - best_utility) <= 1e-12 and quality > best_quality):
                best_model = model_id
                best_utility = utility
                best_quality = quality
        patched.loc[query_id] = best_model
    return patched


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    family: str,
    scope: str,
    lambda_cost: float,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    if scope == "math_only":
        target = target[target["benchmark"].isin(MATH_BENCHMARKS)]
        selected_rows = selected_rows[selected_rows["benchmark"].isin(MATH_BENCHMARKS)]
    elif scope != "all":
        raise ValueError(scope)
    if target.empty or selected_rows.empty:
        return {}
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["scope"] = scope
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["math_patch_rate"] = float(selected_rows["benchmark"].isin(MATH_BENCHMARKS).mean())
    return row


def math_detail_rows(package, outputs: pd.DataFrame, selected: pd.Series, *, split: str, method: str, family: str) -> pd.DataFrame:
    target = outputs[outputs["split"].eq(split) & outputs["benchmark"].isin(MATH_BENCHMARKS)]
    if target.empty:
        return pd.DataFrame()
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    selected_rows = selected_rows[selected_rows["benchmark"].isin(MATH_BENCHMARKS)].copy()
    merged = selected_rows[
        ["query_id", "benchmark", "model_id", "quality_score", "utility", "normalized_remote_cost"]
    ].merge(
        cost_oracle[["query_id", "model_id", "quality_score", "utility", "normalized_remote_cost"]],
        on="query_id",
        suffixes=("_selected", "_oracle"),
    )
    merged["method"] = method
    merged["family"] = family
    merged["split"] = split
    merged["utility_gap"] = merged["utility_oracle"] - merged["utility_selected"]
    return merged


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    all_scope = table[table["scope"].eq("all")]
    for family, group in all_scope.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = table[table["split"].eq("test") & table["method"].eq(method)]
        if not test.empty:
            rows.append(test.assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["scope", "mean_utility", "mean_quality"], ascending=[True, False, False])
    if not top_test.empty:
        rows.append(top_test.groupby("scope").head(12).assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, details: pd.DataFrame) -> None:
    detail_summary = pd.DataFrame()
    if not details.empty:
        detail_summary = (
            details.groupby(["method", "split", "model_id_selected", "model_id_oracle"], as_index=False)
            .agg(n=("query_id", "nunique"), total_gap=("utility_gap", "sum"))
            .sort_values(["split", "total_gap"], ascending=[True, False])
            .head(30)
        )
    lines = [
        "# Math Verifiability Patch",
        "",
        f"Source outputs: `{args.outputs}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls; it uses cached local/API rows and train-only local reliability tables.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected),
        "```",
        "",
        "## Held-Out All-Benchmark Rows",
        "",
        "```csv",
        compact_csv(table[(table["split"].eq("test")) & (table["scope"].eq("all"))].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)),
        "```",
        "",
        "## Held-Out Math-Only Rows",
        "",
        "```csv",
        compact_csv(table[(table["split"].eq("test")) & (table["scope"].eq("math_only"))].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)),
        "```",
        "",
        "## Math Error Summary",
        "",
        "```csv",
        compact_csv(detail_summary),
        "```",
        "",
        "## Interpretation",
        "",
        "- This tests a narrow verifiability patch for LiveMathBench/MATH500 only.",
        "- The patch leaves non-math benchmarks unchanged and only replaces frontier/strong math calls when deterministic tools or local-answer agreement suggest a cheaper local action.",
        "- Diagnostic oracle rows use held-out math utilities and are not deployable.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
