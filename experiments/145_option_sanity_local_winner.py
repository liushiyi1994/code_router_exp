from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


OPTION_BENCHMARKS = {"gpqa", "mmlupro"}
LOCAL_MODELS = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
]
STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a narrow GPQA/MMLUPro local option-sanity route patch.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--augmented-outputs",
        type=Path,
        default=Path("results/controlled/broad100_train_supervised_strong_gain_gate/model_outputs_with_gemini_strong_all_splits.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_option_sanity_local_winner"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")

    suites: list[tuple[str, pd.DataFrame]] = [
        ("base", package.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))),
    ]
    if args.augmented_outputs.exists():
        suites.append(("augmented_strong", load_precomputed_outputs(args.augmented_outputs, lambda_cost=float(args.lambda_cost))))

    all_eval: list[pd.DataFrame] = []
    all_selected: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_bench: list[pd.DataFrame] = []
    for matrix_name, outputs in suites:
        eval_table, selected, scores, bench = run_suite(package, outputs, matrix_name=matrix_name, lambda_cost=float(args.lambda_cost))
        all_eval.append(eval_table)
        all_selected.append(selected)
        all_scores.append(scores.assign(matrix_name=matrix_name))
        all_bench.append(bench.assign(matrix_name=matrix_name))

    eval_out = pd.concat(all_eval, ignore_index=True)
    selected_out = pd.concat(all_selected, ignore_index=True)
    scores_out = pd.concat(all_scores, ignore_index=True)
    bench_out = pd.concat(all_bench, ignore_index=True)

    eval_out.to_csv(args.output_dir / "table_option_sanity_all.csv", index=False)
    selected_out.to_csv(args.output_dir / "table_option_sanity_selected.csv", index=False)
    scores_out.to_csv(args.output_dir / "table_option_sanity_candidate_scores.csv", index=False)
    bench_out.to_csv(args.output_dir / "table_option_sanity_benchmark_gap.csv", index=False)
    write_memo(args.output_dir, eval_out, selected_out, bench_out)
    print(f"Wrote option-sanity local-winner results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_precomputed_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce").fillna(0.0)
    for column in ["cost_total_usd", "latency_s", "normalized_remote_cost"]:
        if column not in outputs:
            outputs[column] = 0.0
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    if "tool_available" not in outputs:
        outputs["tool_available"] = False
    return outputs


def run_suite(
    package,
    outputs: pd.DataFrame,
    *,
    matrix_name: str,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    score_table = fit_and_score_candidates(outputs)
    local_choice = {
        split: select_predicted_local(score_table, outputs, split=split)
        for split in ["train", "val", "test"]
    }
    base_methods = {
        "observable_local_state_v5": {
            split: normalize_selection(package.observable_local_state_selection(outputs, split=split))
            for split in ["val", "test"]
        },
        "tool_probe_profile_v4": {
            split: normalize_selection(package.profile_v4_selection_for_split(outputs, split=split))
            for split in ["val", "test"]
        },
    }

    methods: dict[str, dict[str, pd.Series]] = {}
    for base_name, by_split in base_methods.items():
        methods[base_name] = by_split
        methods[f"{base_name}_option_sanity_local_patch"] = {
            split: patch_option_benchmarks(outputs, by_split[split], local_choice[split], split=split)
            for split in ["val", "test"]
        }
        methods[f"{base_name}_option_local_oracle_patch"] = {
            split: patch_option_benchmarks(outputs, by_split[split], local_oracle_selection(outputs, split=split), split=split)
            for split in ["val", "test"]
        }
        if STRONG_MODEL_ID in set(outputs["model_id"].astype(str)):
            methods[f"{base_name}_option_local_strong_oracle_patch"] = {
                split: patch_option_benchmarks(
                    outputs,
                    by_split[split],
                    local_or_strong_oracle_selection(outputs, split=split),
                    split=split,
                )
                for split in ["val", "test"]
            }
            for threshold in candidate_thresholds(score_table[score_table["split"].eq("val")]["pred_correct"].to_numpy()):
                method = f"{base_name}_option_sanity_strong_if_conf_lt_{threshold:.3f}"
                methods[method] = {
                    split: patch_option_benchmarks(
                        outputs,
                        by_split[split],
                        local_or_strong_by_confidence(local_choice[split], threshold=threshold),
                        split=split,
                    )
                    for split in ["val", "test"]
                }

    eval_rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        target = outputs[outputs["split"].eq(split)]
        cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
        quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
        for method, by_split in methods.items():
            selected = by_split[split]
            selected_rows = package.selected_to_rows(outputs, selected, split=split)
            if selected_rows.empty:
                continue
            row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
            row["matrix_name"] = matrix_name
            row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean()) if STRONG_MODEL_ID in set(outputs["model_id"].astype(str)) else 0.0
            row["option_patch_rate"] = float(
                target.drop_duplicates("query_id")["benchmark"].astype(str).isin(OPTION_BENCHMARKS).mean()
            )
            eval_rows.append(row)

    eval_table = pd.DataFrame(eval_rows)
    selected = validation_selected(eval_table)
    bench = benchmark_gap_table(package, outputs, methods, lambda_cost=lambda_cost)
    return eval_table, selected, score_table, bench


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def fit_and_score_candidates(outputs: pd.DataFrame) -> pd.DataFrame:
    candidates = candidate_feature_table(outputs)
    train = candidates[candidates["split"].eq("train")].copy()
    if len(set(train["quality"].astype(int))) < 2:
        candidates["pred_correct"] = float(train["quality"].mean()) if len(train) else 0.0
        return candidates
    model = Pipeline(
        [
            (
                "pre",
                ColumnTransformer(
                    [
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            ["benchmark", "model_id", "answer", "answer_valid", "is_majority_answer"],
                        ),
                        (
                            "num",
                            StandardScaler(with_mean=False),
                            [
                                "answer_vote_count",
                                "local_valid_answer_count",
                                "local_unique_answer_count",
                                "local_max_vote",
                                "local_vote_margin",
                                "candidate_train_quality",
                                "candidate_benchmark_train_quality",
                                "answer_train_quality",
                                "answer_model_train_quality",
                            ],
                        ),
                    ]
                ),
            ),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)),
        ]
    )
    feature_columns = [
        "benchmark",
        "model_id",
        "answer",
        "answer_valid",
        "is_majority_answer",
        "answer_vote_count",
        "local_valid_answer_count",
        "local_unique_answer_count",
        "local_max_vote",
        "local_vote_margin",
        "candidate_train_quality",
        "candidate_benchmark_train_quality",
        "answer_train_quality",
        "answer_model_train_quality",
    ]
    model.fit(train[feature_columns], train["quality"].astype(int))
    candidates["pred_correct"] = model.predict_proba(candidates[feature_columns])[:, 1]
    return candidates


def candidate_feature_table(outputs: pd.DataFrame) -> pd.DataFrame:
    option_outputs = outputs[outputs["benchmark"].astype(str).isin(OPTION_BENCHMARKS)].copy()
    local = option_outputs[option_outputs["model_id"].astype(str).isin(LOCAL_MODELS)].copy()
    local["answer_norm"] = local["parsed_answer"].map(normalize_option)
    train = local[local["split"].eq("train")].copy()
    global_rel = train.groupby("model_id")["quality_score"].mean().to_dict()
    bench_rel = train.groupby(["benchmark", "model_id"])["quality_score"].mean().to_dict()
    answer_rel = train.groupby(["benchmark", "answer_norm"])["quality_score"].mean().to_dict()
    model_answer_rel = (
        train.groupby(["benchmark", "model_id", "answer_norm"])["quality_score"].mean().to_dict()
    )

    rows: list[dict[str, Any]] = []
    for query_id, group in local.groupby("query_id"):
        group = group.copy()
        query = group.iloc[0]
        answers = {str(row["model_id"]): normalize_option(row.get("parsed_answer", "")) for _, row in group.iterrows()}
        valid_answers = [answer for answer in answers.values() if answer in {"A", "B", "C", "D"}]
        counts = {answer: valid_answers.count(answer) for answer in sorted(set(valid_answers))}
        local_max_vote = max(counts.values()) if counts else 0
        sorted_counts = sorted(counts.values(), reverse=True)
        local_vote_margin = local_max_vote - (sorted_counts[1] if len(sorted_counts) > 1 else 0)
        for _, row in group.iterrows():
            model_id = str(row["model_id"])
            answer = answers.get(model_id, "INVALID")
            answer_valid = answer in {"A", "B", "C", "D"}
            rows.append(
                {
                    "query_id": str(query_id),
                    "split": str(row["split"]),
                    "benchmark": str(row["benchmark"]),
                    "model_id": model_id,
                    "answer": answer if answer_valid else "INVALID",
                    "answer_valid": "valid" if answer_valid else "invalid",
                    "answer_vote_count": float(counts.get(answer, 0) if answer_valid else 0),
                    "is_majority_answer": "yes" if answer_valid and counts.get(answer, 0) == local_max_vote and local_max_vote > 0 else "no",
                    "local_valid_answer_count": float(len(valid_answers)),
                    "local_unique_answer_count": float(len(counts)),
                    "local_max_vote": float(local_max_vote),
                    "local_vote_margin": float(local_vote_margin),
                    "candidate_train_quality": float(global_rel.get(model_id, train["quality_score"].mean() if len(train) else 0.0)),
                    "candidate_benchmark_train_quality": float(
                        bench_rel.get((str(row["benchmark"]), model_id), global_rel.get(model_id, 0.0))
                    ),
                    "answer_train_quality": float(answer_rel.get((str(row["benchmark"]), answer), train["quality_score"].mean() if len(train) else 0.0)),
                    "answer_model_train_quality": float(
                        model_answer_rel.get((str(row["benchmark"]), model_id, answer), bench_rel.get((str(row["benchmark"]), model_id), 0.0))
                    ),
                    "quality": float(row["quality_score"]),
                    "utility": float(row["utility"]),
                    "parsed_answer": row.get("parsed_answer", ""),
                    "query_text": str(query.get("query_text", "")),
                }
            )
    return pd.DataFrame(rows)


def normalize_option(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"A", "B", "C", "D"}:
        return text
    if len(text) >= 1 and text[0] in {"A", "B", "C", "D"}:
        return text[0]
    return "INVALID"


def select_predicted_local(score_table: pd.DataFrame, outputs: pd.DataFrame, *, split: str) -> pd.Series:
    selected: dict[str, str] = {}
    rows = score_table[score_table["split"].eq(split)].copy()
    for query_id, group in rows.groupby("query_id"):
        valid = group[group["answer_valid"].eq("valid")].copy()
        if valid.empty:
            valid = group.copy()
        picked = valid.sort_values(
            ["pred_correct", "answer_vote_count", "candidate_benchmark_train_quality", "candidate_train_quality"],
            ascending=[False, False, False, False],
        ).iloc[0]
        selected[str(query_id)] = str(picked["model_id"])
    return with_confidence(pd.Series(selected), rows)


def with_confidence(selected: pd.Series, rows: pd.DataFrame) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    conf = rows.sort_values("pred_correct", ascending=False).drop_duplicates("query_id").set_index("query_id")["pred_correct"]
    out.attrs["confidence"] = {str(query_id): float(value) for query_id, value in conf.items()}
    return out.astype(str)


def patch_option_benchmarks(outputs: pd.DataFrame, base: pd.Series, patch: pd.Series, *, split: str) -> pd.Series:
    selected = normalize_selection(base)
    query_info = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    for query_id, row in query_info.iterrows():
        query_id = str(query_id)
        if str(row["benchmark"]) in OPTION_BENCHMARKS and query_id in patch.index:
            selected.loc[query_id] = str(patch.loc[query_id])
    return selected.astype(str)


def local_oracle_selection(outputs: pd.DataFrame, *, split: str) -> pd.Series:
    candidates = outputs[
        outputs["split"].eq(split)
        & outputs["benchmark"].astype(str).isin(OPTION_BENCHMARKS)
        & outputs["model_id"].astype(str).isin(LOCAL_MODELS)
    ].copy()
    picked = candidates.loc[candidates.groupby("query_id")["utility"].idxmax()]
    return pd.Series({str(row["query_id"]): str(row["model_id"]) for _, row in picked.iterrows()})


def local_or_strong_oracle_selection(outputs: pd.DataFrame, *, split: str) -> pd.Series:
    models = set(LOCAL_MODELS)
    if STRONG_MODEL_ID in set(outputs["model_id"].astype(str)):
        models.add(STRONG_MODEL_ID)
    candidates = outputs[
        outputs["split"].eq(split)
        & outputs["benchmark"].astype(str).isin(OPTION_BENCHMARKS)
        & outputs["model_id"].astype(str).isin(models)
    ].copy()
    picked = candidates.loc[candidates.groupby("query_id")["utility"].idxmax()]
    return pd.Series({str(row["query_id"]): str(row["model_id"]) for _, row in picked.iterrows()})


def local_or_strong_by_confidence(local_choice: pd.Series, *, threshold: float) -> pd.Series:
    selected = local_choice.copy()
    selected.index = selected.index.astype(str)
    conf = selected.attrs.get("confidence", {})
    for query_id in selected.index:
        if float(conf.get(str(query_id), 0.0)) < float(threshold):
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected.astype(str)


def candidate_thresholds(values: np.ndarray) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    fixed = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
    if finite.size == 0:
        return fixed
    qs = np.quantile(finite, np.linspace(0.05, 0.95, 10)).tolist()
    return sorted(set(float(round(value, 6)) for value in [*fixed, *qs]))


def validation_selected(eval_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for matrix_name, matrix in eval_table.groupby("matrix_name"):
        val = matrix[matrix["split"].eq("val") & ~matrix["method"].astype(str).str.contains("oracle")].copy()
        if val.empty:
            continue
        candidates = {
            "val_best_utility": val.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1),
            "val_best_under_frontier_040": val[val["frontier_call_rate"].le(0.40)].sort_values(
                ["mean_utility", "mean_quality"], ascending=False
            ).head(1),
        }
        seen: set[str] = set()
        for rule, picked in candidates.items():
            if picked.empty:
                continue
            method = str(picked.iloc[0]["method"])
            if method in seen:
                continue
            seen.add(method)
            rows.append(picked.assign(selection_rule=rule))
            test = matrix[matrix["split"].eq("test") & matrix["method"].eq(method)]
            if not test.empty:
                rows.append(test.assign(selection_rule=f"{rule}_test"))
        top_test = matrix[matrix["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(6)
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def benchmark_gap_table(package, outputs: pd.DataFrame, methods: dict[str, dict[str, pd.Series]], *, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        target = outputs[outputs["split"].eq(split)]
        cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
        oracle_by_query = cost_oracle.set_index("query_id")
        for method, by_split in methods.items():
            selected_rows = package.selected_to_rows(outputs, by_split[split], split=split)
            merged = selected_rows.merge(
                oracle_by_query[["utility", "quality_score"]].rename(
                    columns={"utility": "oracle_utility", "quality_score": "oracle_quality"}
                ),
                left_on="query_id",
                right_index=True,
                how="left",
            )
            merged["utility_gap"] = merged["oracle_utility"] - merged["utility"]
            merged["quality_gap"] = merged["oracle_quality"] - merged["quality_score"]
            for benchmark, frame in merged.groupby("benchmark"):
                rows.append(
                    {
                        "method": method,
                        "split": split,
                        "benchmark": str(benchmark),
                        "n": int(frame["query_id"].nunique()),
                        "mean_quality": float(frame["quality_score"].mean()),
                        "mean_utility": float(frame["utility"].mean()),
                        "mean_oracle_utility": float(frame["oracle_utility"].mean()),
                        "mean_utility_gap": float(frame["utility_gap"].mean()),
                        "total_utility_gap": float(frame["utility_gap"].sum()),
                        "lambda_cost": float(lambda_cost),
                    }
                )
    return pd.DataFrame(rows)


def write_memo(output_dir: Path, eval_table: pd.DataFrame, selected: pd.DataFrame, bench: pd.DataFrame) -> None:
    lines = [
        "# Broad100 Option-Sanity Local Winner",
        "",
        "This experiment targets the largest current broad100 option-answer slices: GPQA and MMLUPro.",
        "It trains a local-candidate correctness model on train rows only, then patches only GPQA/MMLUPro decisions.",
        "It makes no GPT, Gemini, Claude, or vLLM calls; it uses cached candidate answers and cached Gemini-strong rows when present.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected),
        "```",
        "",
        "## Best Held-Out Test Rows",
        "",
        "```csv",
        compact_csv(eval_table[eval_table["split"].eq("test")].sort_values(["matrix_name", "mean_utility", "mean_quality"], ascending=[True, False, False]).head(30)),
        "```",
        "",
        "## Option-Slice Benchmark Gaps",
        "",
        "```csv",
        compact_csv(
            bench[
                bench["benchmark"].astype(str).isin(sorted(OPTION_BENCHMARKS))
                & bench["split"].eq("test")
            ].sort_values(["total_utility_gap"], ascending=False).head(30)
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- If the train-fitted local patch does not beat the base method on held-out test, cached local answer patterns are not enough for this slice.",
        "- The local-oracle and local+strong-oracle rows show how much headroom exists if the missing option-sanity signal were available.",
    ]
    (output_dir / "OPTION_SANITY_LOCAL_WINNER_MEMO.md").write_text("\n".join(lines), encoding="utf-8")


def compact_csv(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    out = frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
