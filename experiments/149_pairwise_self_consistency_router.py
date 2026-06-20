from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
ACTIONS = ["base", "self", "strong"]
PAIRS = [("base", "self"), ("base", "strong"), ("self", "strong")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise preference router over cached self-consistency features.")
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
        default=Path("results/controlled/broad100_pairwise_self_consistency_router"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_feature_gate")
    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    table = run_pairwise_routers(
        package,
        self_gate,
        outputs,
        probe,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = self_gate.validation_selected_rows(table)
    pairwise_diag = pairwise_diagnostics(table)
    table.to_csv(args.output_dir / "table_pairwise_self_consistency_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_pairwise_self_consistency_selected.csv", index=False)
    pairwise_diag.to_csv(args.output_dir / "table_pairwise_self_consistency_summary.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "PAIRWISE_SELF_CONSISTENCY_MEMO.md", args, table, selected, pairwise_diag)
    print(f"Wrote pairwise self-consistency router results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_pairwise_routers(
    package,
    self_gate,
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
        base = {split: self_gate.normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        for split in ["val", "test"]:
            rows.append(
                self_gate.evaluate_selection(
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
                self_gate.evaluate_selection(
                    package,
                    outputs,
                    self_gate.oracle_between_actions(outputs, base[split], [self_model_id, STRONG_MODEL_ID]),
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )
        train_q = self_gate.build_feature_frame(outputs, probe, base["train"], split="train", self_model_id=self_model_id)
        val_q = self_gate.build_feature_frame(outputs, probe, base["val"], split="val", self_model_id=self_model_id)
        test_q = self_gate.build_feature_frame(outputs, probe, base["test"], split="test", self_model_id=self_model_id)
        if train_q.empty or val_q.empty or test_q.empty:
            continue
        train_pairs = build_pair_frame(train_q)
        val_pairs = build_pair_frame(val_q)
        test_pairs = build_pair_frame(test_q)
        for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
            x_train, x_val, x_test = featurize_pairs(
                train_pairs,
                val_pairs,
                test_pairs,
                feature_view=feature_view,
                max_features=max_features,
            )
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                model = Ridge(alpha=float(alpha), solver="lsqr")
                model.fit(x_train, train_pairs["target_margin"].to_numpy(dtype=float))
                val_scores = aggregate_pair_scores(val_pairs, np.asarray(model.predict(x_val), dtype=float))
                test_scores = aggregate_pair_scores(test_pairs, np.asarray(model.predict(x_test), dtype=float))
                rows.extend(
                    selected_val_and_test_rows(
                        package,
                        self_gate,
                        outputs,
                        base,
                        val_scores,
                        test_scores,
                        method=f"{base_name}_pairwise_self_ridge_{feature_view}_alpha{alpha:g}",
                        family="pairwise_self_ridge",
                        self_model_id=self_model_id,
                        lambda_cost=lambda_cost,
                        feature_view=feature_view,
                        alpha=alpha,
                    )
                )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def build_pair_frame(query_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in query_features.iterrows():
        utilities = {
            "base": float(row["utility_base"]),
            "self": float(row["utility_self"]),
            "strong": float(row["utility_strong"]),
        }
        for action_a, action_b in PAIRS:
            item = row.to_dict()
            item["action_a"] = action_a
            item["action_b"] = action_b
            item["pair_id"] = f"{action_a}_vs_{action_b}"
            item["target_margin"] = utilities[action_a] - utilities[action_b]
            item["preferred_action"] = action_a if item["target_margin"] >= 0 else action_b
            item["pair_feature_text"] = f"{item.get('feature_text', '')} pair={item['pair_id']} action_a={action_a} action_b={action_b}"
            rows.append(item)
    return pd.DataFrame(rows)


def featurize_pairs(
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
        "action_a",
        "action_b",
        "pair_id",
    ]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(frame_to_dicts(train, numeric_columns, categorical_columns))
    x_val = vectorizer.transform(frame_to_dicts(val, numeric_columns, categorical_columns))
    x_test = vectorizer.transform(frame_to_dicts(test, numeric_columns, categorical_columns))
    if feature_view == "metadata_numeric":
        return x_train, x_val, x_test
    if feature_view != "metadata_numeric_text":
        raise ValueError(feature_view)
    text = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_text = text.fit_transform(train["pair_feature_text"].fillna("").astype(str))
    val_text = text.transform(val["pair_feature_text"].fillna("").astype(str))
    test_text = text.transform(test["pair_feature_text"].fillna("").astype(str))
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


def aggregate_pair_scores(pair_frame: pd.DataFrame, predicted_margins: np.ndarray) -> pd.DataFrame:
    scores = pd.DataFrame(0.0, index=sorted(pair_frame["query_id"].astype(str).unique()), columns=ACTIONS)
    for (_, row), margin in zip(pair_frame.iterrows(), predicted_margins):
        query_id = str(row["query_id"])
        action_a = str(row["action_a"])
        action_b = str(row["action_b"])
        value = float(margin)
        scores.loc[query_id, action_a] += value
        scores.loc[query_id, action_b] -= value
    return scores


def selected_val_and_test_rows(
    package,
    self_gate,
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
    for split, scores in [("val", val_scores), ("test", test_scores)]:
        selected = self_gate.scores_to_selection(base[split], scores, self_model_id=self_model_id)
        row = self_gate.evaluate_selection(
            package,
            outputs,
            selected,
            split=split,
            method=method,
            family=family,
            lambda_cost=lambda_cost,
            self_model_id=self_model_id,
        )
        row.update(extra)
        rows.append(row)
    return rows


def pairwise_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    test = table[table["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    return (
        test.groupby("family", as_index=False)
        .agg(
            best_test_utility=("mean_utility", "max"),
            best_test_quality=("mean_quality", "max"),
            best_oracle_ratio=("oracle_utility_ratio", "max"),
        )
        .sort_values("best_test_utility", ascending=False)
    )


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
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#596f8f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Pairwise Self-Consistency Router")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pairwise_self_consistency_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Pairwise Self-Consistency Router",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls; it trains pairwise utility-margin predictors on cached train rows.",
        "",
        "## Family Summary",
        "",
        "```csv",
        compact_csv(summary),
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
        "- This tests RouteLLM-style pairwise preference routing over base/self/strong actions.",
        "- Pairwise scores are aggregated by adding predicted action margins across the three action pairs.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
