from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from routecode.controlled.live_stage0 import normalize_answer


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train self-consistency feature gates for base/self/strong routing.")
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
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_self_consistency_feature_gate"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    outputs = load_outputs(args.outputs)
    probe = load_probe(args.probe_table)
    table = run_feature_gates(
        package,
        outputs,
        probe,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_self_consistency_feature_gate_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_self_consistency_feature_gate_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "SELF_CONSISTENCY_FEATURE_GATE_MEMO.md", args, probe, table, selected)
    print(f"Wrote self-consistency feature-gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_outputs(path: Path) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    for column in ["quality_score", "utility", "cost_total_usd", "normalized_remote_cost", "latency_s"]:
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    return outputs.drop_duplicates(["query_id", "model_id"], keep="last")


def load_probe(path: Path) -> pd.DataFrame:
    probe = pd.read_csv(path).copy()
    probe["query_id"] = probe["query_id"].astype(str)
    probe["split"] = probe["split"].astype(str)
    for column in [
        "n_samples",
        "valid_count",
        "top_vote_count",
        "vote_frac",
        "vote_margin",
        "vote_entropy",
        "latency_s",
        "input_tokens",
        "output_tokens",
    ]:
        probe[column] = pd.to_numeric(probe[column], errors="coerce").fillna(0.0)
    return probe


def run_feature_gates(
    package,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()
    base_specs = {
        "observable_local_state_v5": lambda split: package.observable_local_state_selection(outputs_no_self, split=split),
        "observable_local_state_v5_no_strong": lambda split: package.observable_local_state_selection(outputs_no_strong_self, split=split),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}
        ),
    }
    for base_name, builder in base_specs.items():
        base = {split: normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        for split in ["val", "test"]:
            rows.append(
                evaluate_selection(
                    package,
                    outputs,
                    base[split],
                    split=split,
                    method=base_name,
                    family="base",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
            rows.append(
                evaluate_selection(
                    package,
                    outputs,
                    oracle_between_actions(outputs, base[split], [self_model_id, STRONG_MODEL_ID]),
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
        rows.extend(
            run_models_for_base(
                package,
                outputs,
                probe,
                base,
                base_name=base_name,
                self_model_id=self_model_id,
                lambda_cost=lambda_cost,
                max_features=max_features,
            )
        )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def run_models_for_base(
    package,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    base_name: str,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train = build_feature_frame(outputs, probe, base["train"], split="train", self_model_id=self_model_id)
    val = build_feature_frame(outputs, probe, base["val"], split="val", self_model_id=self_model_id)
    test = build_feature_frame(outputs, probe, base["test"], split="test", self_model_id=self_model_id)
    if train.empty or val.empty or test.empty:
        return rows
    action_cols = ["utility_base", "utility_self", "utility_strong"]
    for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
        x_train, x_val, x_test = featurize(train, val, test, feature_view=feature_view, max_features=max_features)
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            predictions: dict[str, pd.DataFrame] = {}
            for action_col in action_cols:
                model = Ridge(alpha=float(alpha), solver="lsqr")
                model.fit(x_train, train[action_col].to_numpy(dtype=float))
                predictions[action_col] = pd.DataFrame(
                    {
                        "val": np.asarray(model.predict(x_val), dtype=float),
                        "test": np.asarray(model.predict(x_test), dtype=float),
                    }
                )
            val_scores = pd.DataFrame(
                {
                    "base": predictions["utility_base"]["val"].to_numpy(dtype=float),
                    "self": predictions["utility_self"]["val"].to_numpy(dtype=float),
                    "strong": predictions["utility_strong"]["val"].to_numpy(dtype=float),
                },
                index=val["query_id"].astype(str),
            )
            test_scores = pd.DataFrame(
                {
                    "base": predictions["utility_base"]["test"].to_numpy(dtype=float),
                    "self": predictions["utility_self"]["test"].to_numpy(dtype=float),
                    "strong": predictions["utility_strong"]["test"].to_numpy(dtype=float),
                },
                index=test["query_id"].astype(str),
            )
            rows.extend(
                selected_val_and_test_rows(
                    package,
                    outputs,
                    base,
                    val_scores,
                    test_scores,
                    method=f"{base_name}_self_feature_ridge_{feature_view}_alpha{alpha:g}",
                    family="self_feature_ridge",
                    self_model_id=self_model_id,
                    lambda_cost=lambda_cost,
                    feature_view=feature_view,
                    alpha=alpha,
                )
            )
    return rows


def build_feature_frame(
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    base_selection: pd.Series,
    *,
    split: str,
    self_model_id: str,
) -> pd.DataFrame:
    query_info = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    probe_by_query = probe[probe["split"].eq(split)].set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id, base_model in base_selection.items():
        query_id = str(query_id)
        if query_id not in query_info.index or query_id not in probe_by_query.index:
            continue
        base_model = str(base_model)
        if (query_id, base_model) not in by_query.index or (query_id, self_model_id) not in by_query.index or (
            query_id,
            STRONG_MODEL_ID,
        ) not in by_query.index:
            continue
        info = query_info.loc[query_id]
        probe_row = probe_by_query.loc[query_id]
        base_row = by_query.loc[(query_id, base_model)]
        self_row = by_query.loc[(query_id, self_model_id)]
        strong_row = by_query.loc[(query_id, STRONG_MODEL_ID)]
        majority_norm = normalize_answer(str(probe_row.get("majority_answer_norm", "")))
        base_norm = normalize_answer(str(base_row.get("parsed_answer", "")))
        norm_counts = parse_norm_counts(probe_row.get("answer_norms_json", "[]"))
        local_agree_count = local_answer_agreement_count(outputs, query_id, majority_norm)
        row = {
            "query_id": query_id,
            "query_text": str(info.get("query_text", "")),
            "benchmark": str(info.get("benchmark", "")),
            "domain": str(info.get("domain", "")),
            "metric": str(info.get("metric", "")),
            "base_model_id": base_model,
            "base_provider": str(base_row.get("provider", "")),
            "base_is_local": bool(base_row.get("is_local", False)),
            "base_is_frontier": bool(base_row.get("is_frontier", False)),
            "base_is_strong": base_model == STRONG_MODEL_ID,
            "base_answer_norm": base_norm,
            "majority_answer_norm": majority_norm,
            "base_equals_self_majority": bool(base_norm and majority_norm and base_norm == majority_norm),
            "n_samples": float(probe_row.get("n_samples", 0.0) or 0.0),
            "valid_count": float(probe_row.get("valid_count", 0.0) or 0.0),
            "top_vote_count": float(probe_row.get("top_vote_count", 0.0) or 0.0),
            "vote_frac": float(probe_row.get("vote_frac", 0.0) or 0.0),
            "vote_margin": float(probe_row.get("vote_margin", 0.0) or 0.0),
            "vote_entropy": float(probe_row.get("vote_entropy", 0.0) or 0.0),
            "all_samples_agree": bool(probe_row.get("all_samples_agree", False)),
            "unique_answer_count": float(len(norm_counts)),
            "local_agree_with_majority_count": float(local_agree_count),
            "majority_answer_len": float(len(majority_norm)),
            "base_answer_len": float(len(base_norm)),
            "probe_latency_s": float(probe_row.get("latency_s", 0.0) or 0.0),
            "probe_output_tokens": float(probe_row.get("output_tokens", 0.0) or 0.0),
            "utility_base": float(base_row["utility"]),
            "utility_self": float(self_row["utility"]),
            "utility_strong": float(strong_row["utility"]),
        }
        row["oracle_action"] = ["base", "self", "strong"][
            int(np.argmax([row["utility_base"], row["utility_self"], row["utility_strong"]]))
        ]
        row["feature_text"] = feature_text(row)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_norm_counts(value: object) -> Counter[str]:
    try:
        values = json.loads(str(value))
    except json.JSONDecodeError:
        values = []
    return Counter(normalize_answer(str(item)) for item in values if normalize_answer(str(item)))


def local_answer_agreement_count(outputs: pd.DataFrame, query_id: str, majority_norm: str) -> int:
    if not majority_norm:
        return 0
    local = outputs[
        outputs["query_id"].astype(str).eq(str(query_id))
        & outputs["is_local"].astype(bool)
        & ~outputs["model_id"].astype(str).isin(["deterministic_math_tool", DEFAULT_SELF_MODEL_ID])
    ]
    return int(sum(normalize_answer(str(value)) == majority_norm for value in local["parsed_answer"].fillna("")))


def feature_text(row: dict[str, Any]) -> str:
    pieces = [
        str(row.get("benchmark", "")),
        str(row.get("domain", "")),
        str(row.get("metric", "")),
        str(row.get("base_model_id", "")),
        f"base_answer={row.get('base_answer_norm', '')}",
        f"self_answer={row.get('majority_answer_norm', '')}",
        "base_equals_self" if row.get("base_equals_self_majority") else "base_differs_self",
        str(row.get("query_text", "")),
    ]
    return " ".join(pieces)


def featurize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_view: str,
    max_features: int,
):
    numeric_columns = [
        "n_samples",
        "valid_count",
        "top_vote_count",
        "vote_frac",
        "vote_margin",
        "vote_entropy",
        "unique_answer_count",
        "local_agree_with_majority_count",
        "majority_answer_len",
        "base_answer_len",
        "probe_latency_s",
        "probe_output_tokens",
    ]
    categorical_columns = [
        "benchmark",
        "domain",
        "metric",
        "base_model_id",
        "base_provider",
        "base_is_local",
        "base_is_frontier",
        "base_is_strong",
        "base_equals_self_majority",
        "all_samples_agree",
    ]
    vectorizer = DictVectorizer(sparse=True)
    train_dicts = frame_to_dicts(train, numeric_columns, categorical_columns)
    val_dicts = frame_to_dicts(val, numeric_columns, categorical_columns)
    test_dicts = frame_to_dicts(test, numeric_columns, categorical_columns)
    x_train = vectorizer.fit_transform(train_dicts)
    x_val = vectorizer.transform(val_dicts)
    x_test = vectorizer.transform(test_dicts)
    if feature_view == "metadata_numeric":
        return x_train, x_val, x_test
    if feature_view != "metadata_numeric_text":
        raise ValueError(feature_view)
    text = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_text = text.fit_transform(train["feature_text"].fillna("").astype(str))
    val_text = text.transform(val["feature_text"].fillna("").astype(str))
    test_text = text.transform(test["feature_text"].fillna("").astype(str))
    return hstack([x_train, train_text]), hstack([x_val, val_text]), hstack([x_test, test_text])


def frame_to_dicts(frame: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        features: dict[str, Any] = {}
        for column in numeric_columns:
            features[column] = float(row.get(column, 0.0) or 0.0)
        for column in categorical_columns:
            features[f"{column}={row.get(column, '')}"] = 1.0
        rows.append(features)
    return rows


def selected_val_and_test_rows(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    *,
    method: str,
    family: str,
    self_model_id: str,
    lambda_cost: float,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_selected = scores_to_selection(base["val"], val_scores, self_model_id=self_model_id)
    val_row = evaluate_selection(
        package,
        outputs,
        val_selected,
        split="val",
        method=method,
        family=family,
        lambda_cost=lambda_cost,
        self_model_id=self_model_id,
    )
    val_row.update(extra)
    rows.append(val_row)
    test_selected = scores_to_selection(base["test"], test_scores, self_model_id=self_model_id)
    test_row = evaluate_selection(
        package,
        outputs,
        test_selected,
        split="test",
        method=method,
        family=family,
        lambda_cost=lambda_cost,
        self_model_id=self_model_id,
    )
    test_row.update(extra)
    rows.append(test_row)
    return rows


def scores_to_selection(base: pd.Series, scores: pd.DataFrame, *, self_model_id: str) -> pd.Series:
    selected = normalize_selection(base)
    for query_id, row in scores.iterrows():
        action = str(row.astype(float).idxmax())
        if action == "self":
            selected.loc[str(query_id)] = self_model_id
        elif action == "strong":
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected


def oracle_between_actions(outputs: pd.DataFrame, base: pd.Series, extra_actions: list[str]) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    selected = normalize_selection(base)
    for query_id, base_model in base.items():
        best_model = str(base_model)
        best_utility = -float("inf")
        best_quality = -float("inf")
        for model_id in [str(base_model), *extra_actions]:
            key = (str(query_id), str(model_id))
            if key not in by_query.index:
                continue
            row = by_query.loc[key]
            utility = float(row["utility"])
            quality = float(row["quality_score"])
            if utility > best_utility or (abs(utility - best_utility) <= 1e-12 and quality > best_quality):
                best_model = str(model_id)
                best_utility = utility
                best_quality = quality
        selected.loc[str(query_id)] = best_model
    return selected


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
    self_model_id: str,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["self_action_rate"] = float(selected_rows["model_id"].eq(self_model_id).mean())
    return row


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)]
        if not test.empty:
            rows.append(test.head(1).assign(selection_rule="val_best_utility_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5d7f7a")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Self-Consistency Feature Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_self_consistency_feature_gate_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, probe: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    probe_summary = (
        probe.groupby(["split", "benchmark"], as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            self_quality=("majority_quality", "mean"),
            mean_vote_frac=("vote_frac", "mean"),
            mean_vote_entropy=("vote_entropy", "mean"),
            mean_latency_s=("latency_s", "mean"),
        )
        .sort_values(["split", "benchmark"])
    )
    lines = [
        "# Self-Consistency Feature Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        f"Self model action: `{args.self_model_id}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls; it trains on cached train rows and selects methods on validation.",
        "",
        "## Probe Summary",
        "",
        "```csv",
        compact_csv(probe_summary),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected),
        "```",
        "",
        "## Held-Out Test Rows",
        "",
        "```csv",
        compact_csv(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(30)),
        "```",
        "",
        "## Interpretation",
        "",
        "- This tests whether cached self-consistency features make base/self/strong action choice predictable from train labels.",
        "- Numeric features include vote count, vote margin, entropy, answer agreement, base action metadata, and probe latency/token counts.",
        "- The text feature view adds query text plus compact route/probe tags.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
