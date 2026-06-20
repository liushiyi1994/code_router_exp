from __future__ import annotations

import argparse
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
from sklearn.linear_model import LogisticRegression, Ridge

from routecode.controlled.live_stage0 import normalize_answer


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
LOCAL_MODELS = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local", DEFAULT_SELF_MODEL_ID]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a precision filter over the cached Qwen14 frontier-needed probe.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--self-probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/table_vllm_self_consistency_probe.csv"),
    )
    parser.add_argument(
        "--frontier-probe-table",
        type=Path,
        default=Path("results/controlled/broad100_vllm_frontier_need_probe_qwen14b/table_vllm_frontier_need_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_vllm_frontier_precision_filter"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    frontier = load_module("experiments/157_frontier_need_predictor.py", "frontier_need")

    outputs = frontier.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    self_probe = frontier.load_probe(args.self_probe_table)
    frontier_probe = load_frontier_probe(args.frontier_probe_table)
    frontier_ids = frontier.frontier_model_ids(outputs)
    local_outputs = outputs[~outputs["model_id"].isin(frontier_ids)].copy()
    base = {
        split: frontier.normalize_selection(package.observable_local_state_selection(local_outputs, split=split))
        for split in ["train", "val", "test"]
    }
    train_frame = build_frame(outputs, self_probe, frontier_probe, base["train"], frontier_ids, split="train")
    val_frame = build_frame(outputs, self_probe, frontier_probe, base["val"], frontier_ids, split="val")
    test_frame = build_frame(outputs, self_probe, frontier_probe, base["test"], frontier_ids, split="test")
    frontier_lookup = frontier_train_lookup(train_frame, frontier_ids)

    table = run_filters(
        package,
        frontier,
        outputs,
        base,
        train_frame,
        val_frame,
        test_frame,
        frontier_ids,
        frontier_lookup,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    train_frame.to_csv(args.output_dir / "table_vllm_frontier_precision_filter_features_train.csv", index=False)
    table.to_csv(args.output_dir / "table_vllm_frontier_precision_filter_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_frontier_precision_filter_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "VLLM_FRONTIER_PRECISION_FILTER_MEMO.md", args, train_frame, val_frame, test_frame, table, selected)
    print(f"Wrote vLLM frontier precision-filter results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_frontier_probe(path: Path) -> pd.DataFrame:
    probe = pd.read_csv(path).copy()
    probe["query_id"] = probe["query_id"].astype(str)
    probe["split"] = probe["split"].astype(str)
    probe["frontier_needed"] = probe["frontier_needed"].astype(bool)
    probe["confidence"] = pd.to_numeric(probe["confidence"], errors="coerce").fillna(0.0)
    return probe.drop_duplicates(["query_id", "split"], keep="last")


def build_frame(
    outputs: pd.DataFrame,
    self_probe: pd.DataFrame,
    frontier_probe: pd.DataFrame,
    base_selection: pd.Series,
    frontier_ids: list[str],
    *,
    split: str,
) -> pd.DataFrame:
    queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    self_by_query = self_probe[self_probe["split"].eq(split)].set_index("query_id") if not self_probe.empty else pd.DataFrame()
    frontier_by_query = frontier_probe[frontier_probe["split"].eq(split)].set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id, local_model in base_selection.items():
        query_id = str(query_id)
        local_model = str(local_model)
        if query_id not in queries.index or query_id not in frontier_by_query.index or (query_id, local_model) not in by_query.index:
            continue
        info = queries.loc[query_id]
        local_row = by_query.loc[(query_id, local_model)]
        self_row = self_by_query.loc[query_id] if not self_by_query.empty and query_id in self_by_query.index else pd.Series(dtype=object)
        probe_row = frontier_by_query.loc[query_id]
        local_answer = normalize_answer(str(local_row.get("parsed_answer", "")))
        self_answer = normalize_answer(str(self_row.get("majority_answer_norm", "")))
        local_answer_values = local_answers(by_query, query_id)
        best_frontier_model = ""
        best_frontier_utility = -float("inf")
        best_frontier_quality = -float("inf")
        for model_id in frontier_ids:
            key = (query_id, model_id)
            if key not in by_query.index:
                continue
            row = by_query.loc[key]
            utility = float(row["utility"])
            quality = float(row["quality_score"])
            if utility > best_frontier_utility or (abs(utility - best_frontier_utility) <= 1e-12 and quality > best_frontier_quality):
                best_frontier_model = model_id
                best_frontier_utility = utility
                best_frontier_quality = quality
        oracle_frontier_needed = best_frontier_utility > float(local_row["utility"]) + 1e-12
        row = {
            "query_id": query_id,
            "query_text": str(info.get("query_text", "")),
            "benchmark": str(info.get("benchmark", "")),
            "domain": str(info.get("domain", "")),
            "metric": str(info.get("metric", "")),
            "local_model_id": local_model,
            "local_answer": local_answer,
            "self_answer": self_answer,
            "qwen14_frontier_needed": bool(probe_row.get("frontier_needed", False)),
            "qwen14_confidence": float(probe_row.get("confidence", 0.0) or 0.0),
            "qwen14_frontier_model": str(probe_row.get("frontier_model", "")),
            "qwen14_reason": str(probe_row.get("reason", "")),
            "self_vote_frac": float(self_row.get("vote_frac", 0.0) or 0.0),
            "self_vote_margin": float(self_row.get("vote_margin", 0.0) or 0.0),
            "self_vote_entropy": float(self_row.get("vote_entropy", 0.0) or 0.0),
            "n_local_answers": float(len(local_answer_values)),
            "n_unique_local_answers": float(len(set(local_answer_values))),
            "local_agree_count": float(sum(answer == local_answer for answer in local_answer_values if answer and local_answer)),
            "local_answer_len": float(len(local_answer)),
            "self_answer_len": float(len(self_answer)),
            "local_equals_self": bool(local_answer and self_answer and local_answer == self_answer),
            "utility_local": float(local_row["utility"]),
            "quality_local": float(local_row["quality_score"]),
            "best_frontier_model": best_frontier_model,
            "best_frontier_utility": float(best_frontier_utility),
            "best_frontier_quality": float(best_frontier_quality),
            "frontier_gain": float(best_frontier_utility) - float(local_row["utility"]),
            "oracle_frontier_needed": bool(oracle_frontier_needed),
        }
        row["feature_text"] = feature_text(row)
        rows.append(row)
    return pd.DataFrame(rows)


def local_answers(by_query: pd.DataFrame, query_id: str) -> list[str]:
    answers: list[str] = []
    for model_id in LOCAL_MODELS:
        key = (query_id, model_id)
        if key not in by_query.index:
            continue
        answer = normalize_answer(str(by_query.loc[key].get("parsed_answer", "")))
        if answer:
            answers.append(answer)
    return answers


def feature_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("benchmark", "")),
            str(row.get("domain", "")),
            str(row.get("metric", "")),
            str(row.get("local_model_id", "")),
            str(row.get("qwen14_frontier_model", "")),
            str(row.get("qwen14_reason", "")),
            f"local_answer={row.get('local_answer', '')}",
            f"self_answer={row.get('self_answer', '')}",
            str(row.get("query_text", "")),
        ]
    )


def run_filters(
    package,
    frontier,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    frontier_ids: list[str],
    frontier_lookup: dict[str, str],
    *,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            frontier.evaluate_selection(
                package, outputs, base[split], split=split, method="local_observable_state", family="reference", lambda_cost=lambda_cost
            )
        )
        rows.append(
            frontier.evaluate_selection(
                package,
                outputs,
                frontier.oracle_between_local_and_frontier(outputs, base[split], frontier_ids),
                split=split,
                method="diagnostic_oracle_between_local_and_frontier",
                family="diagnostic_oracle",
                lambda_cost=lambda_cost,
            )
        )

    for feature_view in ["metadata", "metadata_text"]:
        x_train, x_val, x_test = featurize(train, val, test, feature_view=feature_view, max_features=max_features)
        y_binary = train["oracle_frontier_needed"].astype(int).to_numpy()
        if len(set(y_binary.tolist())) > 1:
            for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
                model = LogisticRegression(C=float(c_value), class_weight="balanced", solver="liblinear", max_iter=2000)
                model.fit(x_train, y_binary)
                val_score = pd.Series(model.predict_proba(x_val)[:, 1], index=val["query_id"].astype(str))
                test_score = pd.Series(model.predict_proba(x_test)[:, 1], index=test["query_id"].astype(str))
                rows.extend(
                    select_and_eval(
                        package,
                        frontier,
                        outputs,
                        base,
                        val,
                        test,
                        val_score,
                        test_score,
                        frontier_lookup,
                        lambda_cost=lambda_cost,
                        method_prefix=f"qwen14_precision_logistic_{feature_view}_C{c_value:g}",
                        family="qwen14_precision_logistic",
                        score_name="p_oracle_frontier",
                        extra={"feature_view": feature_view, "classifier_c": float(c_value)},
                    )
                )

        y_gain = train["frontier_gain"].to_numpy(dtype=float)
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(x_train, y_gain)
            val_score = pd.Series(np.asarray(model.predict(x_val), dtype=float), index=val["query_id"].astype(str))
            test_score = pd.Series(np.asarray(model.predict(x_test), dtype=float), index=test["query_id"].astype(str))
            rows.extend(
                select_and_eval(
                    package,
                    frontier,
                    outputs,
                    base,
                    val,
                    test,
                    val_score,
                    test_score,
                    frontier_lookup,
                    lambda_cost=lambda_cost,
                    method_prefix=f"qwen14_precision_gain_ridge_{feature_view}_alpha{alpha:g}",
                    family="qwen14_precision_gain_ridge",
                    score_name="pred_gain",
                    extra={"feature_view": feature_view, "alpha": float(alpha)},
                )
            )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def select_and_eval(
    package,
    frontier,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_score: pd.Series,
    test_score: pd.Series,
    frontier_lookup: dict[str, str],
    *,
    lambda_cost: float,
    method_prefix: str,
    family: str,
    score_name: str,
    extra: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in candidate_thresholds(val_score):
        for cap in [0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.50, 1.00]:
            selected = apply_filter(base["val"], val, val_score, frontier_lookup, threshold=threshold, cap=cap)
            method = f"{method_prefix}_thr{threshold:.4f}_cap{cap:.2f}"
            row = frontier.evaluate_selection(
                package, outputs, selected, split="val", method=method, family=family, lambda_cost=lambda_cost
            )
            row.update(extra)
            row.update({"threshold": float(threshold), "frontier_cap": float(cap), "score_name": score_name})
            candidates.append(row)
    if not candidates:
        return []
    best = sorted(candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]
    test_selected = apply_filter(base["test"], test, test_score, frontier_lookup, threshold=float(best["threshold"]), cap=float(best["frontier_cap"]))
    test_row = frontier.evaluate_selection(
        package, outputs, test_selected, split="test", method=str(best["method"]), family=family, lambda_cost=lambda_cost
    )
    test_row.update(extra)
    test_row.update({"threshold": float(best["threshold"]), "frontier_cap": float(best["frontier_cap"]), "score_name": score_name})
    return [best, test_row]


def apply_filter(
    base: pd.Series,
    frame: pd.DataFrame,
    score: pd.Series,
    frontier_lookup: dict[str, str],
    *,
    threshold: float,
    cap: float,
) -> pd.Series:
    selected = base.copy().astype(str)
    candidates = frame[frame["qwen14_frontier_needed"].astype(bool)].copy()
    candidates["filter_score"] = candidates["query_id"].astype(str).map(score).fillna(-float("inf"))
    eligible = candidates[candidates["filter_score"].astype(float) >= float(threshold)].sort_values("filter_score", ascending=False)
    if cap < 1.0:
        eligible = eligible.head(max(1, int(np.floor(float(cap) * len(selected)))))
    for row in eligible.itertuples(index=False):
        selected.loc[str(row.query_id)] = str(frontier_lookup.get(str(row.benchmark), row.best_frontier_model or STRONG_MODEL_ID))
    return selected


def candidate_thresholds(score: pd.Series) -> list[float]:
    values = np.asarray(score.dropna(), dtype=float)
    if values.size == 0:
        return [0.0]
    quantiles = np.quantile(values, [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    fixed = np.asarray([-0.10, -0.05, 0.0, 0.02, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90])
    return sorted({round(float(value), 6) for value in np.concatenate([quantiles, fixed])})


def featurize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_view: str,
    max_features: int,
):
    numeric_columns = [
        "qwen14_frontier_needed",
        "qwen14_confidence",
        "self_vote_frac",
        "self_vote_margin",
        "self_vote_entropy",
        "n_local_answers",
        "n_unique_local_answers",
        "local_agree_count",
        "local_answer_len",
        "self_answer_len",
        "local_equals_self",
    ]
    categorical_columns = ["benchmark", "domain", "metric", "local_model_id", "qwen14_frontier_model"]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(frame_to_dicts(train, numeric_columns, categorical_columns))
    x_val = vectorizer.transform(frame_to_dicts(val, numeric_columns, categorical_columns))
    x_test = vectorizer.transform(frame_to_dicts(test, numeric_columns, categorical_columns))
    if feature_view == "metadata":
        return x_train, x_val, x_test
    if feature_view != "metadata_text":
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
            value = row.get(column, 0.0)
            if isinstance(value, (bool, np.bool_)):
                features[column] = float(value)
            else:
                features[column] = float(value or 0.0)
        for column in categorical_columns:
            features[f"{column}={row.get(column, '')}"] = 1.0
        rows.append(features)
    return rows


def frontier_train_lookup(train: pd.DataFrame, frontier_ids: list[str]) -> dict[str, str]:
    rows = []
    for benchmark, group in train.groupby("benchmark"):
        for model_id in frontier_ids:
            mask = group["best_frontier_model"].astype(str).eq(model_id)
            if not mask.any():
                # use observed utility in all rows where this model is not best only if unavailable from best labels.
                mean_utility = -float("inf")
            else:
                mean_utility = float(group.loc[mask, "best_frontier_utility"].mean())
            rows.append({"benchmark": benchmark, "model_id": model_id, "mean_utility": mean_utility})
    table = pd.DataFrame(rows).sort_values(["benchmark", "mean_utility"], ascending=[True, False])
    return table.drop_duplicates("benchmark").set_index("benchmark")["model_id"].astype(str).to_dict()


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


def write_figure(output_dir: Path, table: pd.DataFrame) -> None:
    test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(test["method"].iloc[::-1], test["mean_utility"].iloc[::-1], color="#5f7367")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Qwen14 Frontier Precision Filter")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_vllm_frontier_precision_filter_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "utility_gap_to_oracle",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
    ]
    label_summary = pd.concat([train.assign(split="train"), val.assign(split="val"), test.assign(split="test")], ignore_index=True)
    label_summary = (
        label_summary.groupby(["split", "benchmark", "qwen14_frontier_needed", "oracle_frontier_needed"], as_index=False)
        .agg(n=("query_id", "nunique"), mean_qwen14_confidence=("qwen14_confidence", "mean"), mean_frontier_gain=("frontier_gain", "mean"))
        .sort_values(["split", "benchmark", "qwen14_frontier_needed", "oracle_frontier_needed"])
    )
    lines = [
        "# Qwen14 Frontier Precision Filter",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Frontier probe table: `{args.frontier_probe_table}`.",
        "",
        "This run makes no model/provider API calls. It trains a precision filter on train, selects thresholds/caps on validation, and reports held-out test.",
        "",
        "## Label Summary",
        "",
        "```csv",
        compact_csv(label_summary, max_rows=120),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[c for c in cols if c in selected.columns] + [c for c in ["selection_rule"] if c in selected.columns]], max_rows=28),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)[[c for c in cols if c in table.columns]], max_rows=24),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
