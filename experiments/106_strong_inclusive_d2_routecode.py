from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.predictors.classifiers import RouteCodeLabelClassifier


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
BASE_GPT = "gpt-5.5"
STRONG_GPT = "strong-gpt-5.5"
ALL_MODELS = LOCAL_MODELS + [GEMINI, BASE_GPT, STRONG_GPT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictability-constrained RouteCode on the strong-inclusive exact-math table."
    )
    parser.add_argument(
        "--query-table",
        default="results/controlled/strong_inclusive_oracle_audit/query_table_with_strong_inclusive_oracle.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/strong_inclusive_d2_routecode")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--cost-target", type=float, default=0.35)
    parser.add_argument("--k-values", default="4,8,16,32")
    parser.add_argument("--alpha-values", default="0,0.05,0.1,0.3,1,3,10,30")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--random-state", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.query_table)
    table = merge_cached_probe_features(table)
    table = add_text_features(table)
    query_info = table.set_index("query_id")[["dataset", "split", "query_text"]]

    k_values = parse_ints(args.k_values)
    alpha_values = parse_floats(args.alpha_values)
    rows: list[dict[str, object]] = []
    label_rows: list[pd.DataFrame] = []

    for feature_name, spec in feature_sets(table).items():
        embeddings = build_embeddings(table, spec, random_state=args.random_state)
        utility_by_split = {
            split: build_utility_matrix(frame, args.lambda_cost)
            for split, frame in table.groupby("split", sort=False)
        }
        train_ids = table.loc[table["split"].eq("train"), "query_id"].tolist()
        val_ids = table.loc[table["split"].eq("val"), "query_id"].tolist()
        test_ids = table.loc[table["split"].eq("test"), "query_id"].tolist()
        train_utility = utility_by_split["train"].loc[train_ids]

        for k in k_values:
            for alpha in alpha_values:
                codebook = PredictabilityConstrainedRouteCode(
                    n_labels=k,
                    alpha=alpha,
                    beta=args.beta,
                    random_state=args.random_state,
                    max_iter=50,
                    refinement_iter=15,
                    n_init=10,
                ).fit(query_info.loc[train_ids], train_utility, embeddings.loc[train_ids])

                for split, ids in [("val", val_ids), ("test", test_ids)]:
                    frame = table.set_index("query_id").loc[ids]
                    utility = utility_by_split[split].loc[ids]

                    joint_labels = codebook.predict_joint_labels(utility, embeddings.loc[ids])
                    rows.append(
                        evaluated_row(
                            frame,
                            codebook.predict_from_labels(joint_labels),
                            method="d2_joint_oracle_labels",
                            feature_set=feature_name,
                            k=k,
                            alpha=alpha,
                            beta=args.beta,
                            lambda_cost=args.lambda_cost,
                            predicted_labels=joint_labels,
                            target_labels=joint_labels,
                            feature_cost_columns=spec["feature_cost_columns"],
                        )
                    )

                    centroid_labels = codebook.predict_labels(embeddings.loc[ids])
                    rows.append(
                        evaluated_row(
                            frame,
                            codebook.predict_from_labels(centroid_labels),
                            method="d2_embedding_centroid",
                            feature_set=feature_name,
                            k=k,
                            alpha=alpha,
                            beta=args.beta,
                            lambda_cost=args.lambda_cost,
                            predicted_labels=centroid_labels,
                            target_labels=joint_labels,
                            feature_cost_columns=spec["feature_cost_columns"],
                        )
                    )

                    label_predictor = RouteCodeLabelClassifier(random_state=args.random_state).fit(
                        codebook,
                        embeddings.loc[train_ids],
                    )
                    classifier_labels = label_predictor.predict_labels(embeddings.loc[ids])
                    rows.append(
                        evaluated_row(
                            frame,
                            codebook.predict_from_labels(classifier_labels),
                            method="d2_logistic_label_predictor",
                            feature_set=feature_name,
                            k=k,
                            alpha=alpha,
                            beta=args.beta,
                            lambda_cost=args.lambda_cost,
                            predicted_labels=classifier_labels,
                            target_labels=joint_labels,
                            feature_cost_columns=spec["feature_cost_columns"],
                            mean_confidence=float(label_predictor.predict_confidence(embeddings.loc[ids]).mean()),
                        )
                    )

                labels = codebook.train_labels_.rename("route_label").reset_index()
                labels["feature_set"] = feature_name
                labels["k"] = k
                labels["alpha"] = alpha
                label_rows.append(labels)

    for split in ["val", "test"]:
        frame = table[table["split"].eq(split)].set_index("query_id")
        rows.extend(reference_rows(frame, args.lambda_cost))

    result = pd.DataFrame(rows)
    selected = selected_rows(result, args.quality_gap_target, args.cost_target)
    labels = pd.concat(label_rows, ignore_index=True) if label_rows else pd.DataFrame()

    table.to_csv(output_dir / "query_table_with_d2_features.csv", index=False)
    result.to_csv(output_dir / "table_strong_inclusive_d2_routecode.csv", index=False)
    selected.to_csv(output_dir / "table_strong_inclusive_d2_selected.csv", index=False)
    labels.to_csv(output_dir / "table_strong_inclusive_d2_train_labels.csv", index=False)
    write_memo(output_dir, args, result, selected)
    print(f"Wrote strong-inclusive D2 RouteCode audit to {output_dir}")


def merge_cached_probe_features(table: pd.DataFrame) -> pd.DataFrame:
    specs = [
        (
            "rescue",
            "results/controlled/gpt_rescue_score_gate/query_table_with_gpt_rescue_scores.csv",
            [
                "p_local_correct",
                "p_gemini_correct",
                "p_gpt_correct",
                "p_hopeless",
                "local_choice_model",
                "local_choice",
                "rescue_route_cost",
                "rescue_latency_s",
                "rescue_input_tokens",
                "rescue_output_tokens",
            ],
        ),
        (
            "selector",
            "results/controlled/qwen32_local_selector_gate/query_table_with_qwen32_selector.csv",
            [
                "choice",
                "selector_confidence",
                "selector_need_frontier",
                "selector_local_quality",
                "selector_latency_s",
                "selector_input_tokens",
                "selector_output_tokens",
            ],
        ),
        (
            "gptadj",
            "results/controlled/answer_adjudicator_gpt_with_gpt/query_table_with_answer_adjudications.csv",
            [
                "selected_model",
                "selected_confidence",
                "adjudicator_cost",
                "adjudicator_latency_s",
                "adjudicator_input_tokens",
                "adjudicator_output_tokens",
            ],
        ),
        (
            "gemadj",
            "results/controlled/answer_adjudicator_gemini_with_gpt/query_table_with_answer_adjudications.csv",
            [
                "selected_model",
                "selected_confidence",
                "adjudicator_cost",
                "adjudicator_latency_s",
                "adjudicator_input_tokens",
                "adjudicator_output_tokens",
            ],
        ),
    ]
    merged = table.copy()
    for prefix, path, columns in specs:
        probe_path = Path(path)
        if not probe_path.exists():
            continue
        probe = pd.read_csv(probe_path)
        keep = ["query_id"] + [column for column in columns if column in probe.columns]
        probe = probe[keep].rename(columns={column: f"{prefix}_{column}" for column in keep if column != "query_id"})
        merged = merged.merge(probe, on="query_id", how="left")
    return merged


def add_text_features(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["text_query"] = table["query_text"].fillna("").astype(str) + " dataset=" + table["dataset"].fillna("").astype(str)
    table["text_local"] = table["text_query"]
    for column in sorted(table.columns):
        if column.endswith("_answer_norm") and BASE_GPT not in column and GEMINI not in column:
            table["text_local"] += " " + column + "=" + table[column].fillna("").astype(str)
    table["text_gemini"] = table["text_local"] + " gemini=" + table[f"{GEMINI}_answer_norm"].fillna("").astype(str)
    table["text_base"] = table["text_gemini"] + " base_gpt=" + table[f"{BASE_GPT}_answer_norm"].fillna("").astype(str)
    return table


def feature_sets(table: pd.DataFrame) -> dict[str, dict[str, object]]:
    agreement_cols = [
        column
        for column in table.columns
        if column.startswith("agree__") or column.endswith("_gemini_agree") or column.endswith("_base_agree")
    ]
    base_cat = unique(
        [
            "dataset",
            "qwen8_gemini_agree",
            "gemini_gpt_agree",
            "qwen8_4b_agree",
            "qwen8_06b_agree",
            "small_pair_agree",
            "all_three_agree",
            "base_gemini_agree",
            "verifier_verdict",
        ]
        + agreement_cols
    )
    base_num = unique(
        [
            "query_len",
            "number_count",
            "latex_count",
            "frac_count",
            "sqrt_count",
            "qwen3-8b-local_answer_len",
            "gemini-3.5-flash_answer_len",
            "gpt-5.5_answer_len",
            "qwen3-4b-local_answer_len",
            "qwen3-0.6b-probe_answer_len",
            "qwen3-14b-awq-local_answer_len",
            "local_max_vote",
            "local_ensemble_votes",
            "answer_len_gap_qwen_gemini",
            "gemini_prompt_tokens",
            "gemini_candidate_tokens",
            "gemini_thoughts_tokens",
            "gemini_total_tokens",
        ]
    )
    probe_cat = ["rescue_local_choice_model", "rescue_local_choice", "selector_choice", "selector_selector_need_frontier"]
    probe_num = [
        "rescue_p_local_correct",
        "rescue_p_gemini_correct",
        "rescue_p_gpt_correct",
        "rescue_p_hopeless",
        "selector_selector_confidence",
        "selector_selector_latency_s",
    ]
    remote_cat = ["gptadj_selected_model", "gemadj_selected_model"]
    remote_num = [
        "gptadj_selected_confidence",
        "gemadj_selected_confidence",
        "rescue_rescue_input_tokens",
        "rescue_rescue_output_tokens",
        "gptadj_adjudicator_input_tokens",
        "gptadj_adjudicator_output_tokens",
        "gemadj_adjudicator_input_tokens",
        "gemadj_adjudicator_output_tokens",
    ]
    return {
        "query_only": {
            "text_col": "text_query",
            "cat_cols": ["dataset"],
            "num_cols": ["query_len", "number_count", "latex_count", "frac_count", "sqrt_count"],
            "feature_cost_columns": [],
        },
        "local_answer_probe": {
            "text_col": "text_local",
            "cat_cols": base_cat,
            "num_cols": base_num,
            "feature_cost_columns": [],
        },
        "local_plus_qwen32_selector": {
            "text_col": "text_local",
            "cat_cols": unique(base_cat + probe_cat),
            "num_cols": unique(base_num + probe_num),
            "feature_cost_columns": [],
        },
        "remote_route_probe_diagnostic": {
            "text_col": "text_base",
            "cat_cols": unique(base_cat + probe_cat + remote_cat),
            "num_cols": unique(base_num + probe_num + remote_num),
            "feature_cost_columns": [
                f"{GEMINI}_cost",
                f"{BASE_GPT}_cost",
                "rescue_rescue_route_cost",
                "gptadj_adjudicator_cost",
                "gemadj_adjudicator_cost",
            ],
        },
    }


def build_embeddings(table: pd.DataFrame, spec: dict[str, object], random_state: int) -> pd.DataFrame:
    text_col = str(spec["text_col"])
    cat_cols = [column for column in spec["cat_cols"] if column in table.columns]
    num_cols = [column for column in spec["num_cols"] if column in table.columns]
    work = table.copy()
    for column in cat_cols:
        work[column] = work[column].fillna("NA").astype(str)
    for column in num_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    preprocessor = ColumnTransformer(
        [
            ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=4000, min_df=1), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ]
    )
    train = work[work["split"].eq("train")]
    all_features = preprocessor.fit(train).transform(work)
    max_components = max(2, min(48, all_features.shape[0] - 1, all_features.shape[1] - 1))
    reducer = make_pipeline(
        TruncatedSVD(n_components=max_components, random_state=random_state),
        StandardScaler(),
    )
    train_features = preprocessor.transform(train)
    reducer.fit(train_features)
    dense = reducer.transform(all_features)
    columns = [f"emb_{idx:02d}" for idx in range(dense.shape[1])]
    return pd.DataFrame(dense, index=work["query_id"], columns=columns)


def build_utility_matrix(frame: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    frame = frame.set_index("query_id")
    norm = max(float(frame["strong_cost"].sum()), 1e-12)
    rows = {}
    for model_id in ALL_MODELS:
        quality, cost = quality_and_cost(frame, model_id)
        rows[model_id] = quality - lambda_cost * cost * len(frame) / norm
    return pd.DataFrame(rows, index=frame.index)


def quality_and_cost(frame: pd.DataFrame, model_id: str) -> tuple[pd.Series, pd.Series]:
    if model_id == STRONG_GPT:
        return frame["strong_quality"].astype(float).fillna(0.0), frame["strong_cost"].astype(float).fillna(0.0)
    if model_id in LOCAL_MODELS:
        return frame[f"{model_id}_quality"].astype(float).fillna(0.0), pd.Series(0.0, index=frame.index)
    return frame[f"{model_id}_quality"].astype(float).fillna(0.0), frame[f"{model_id}_cost"].astype(float).fillna(0.0)


def evaluated_row(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    feature_set: str,
    k: int | str,
    alpha: float | str,
    beta: float | str,
    lambda_cost: float,
    predicted_labels: pd.Series | None = None,
    target_labels: pd.Series | None = None,
    feature_cost_columns: Iterable[str] = (),
    mean_confidence: float | str = "",
) -> dict[str, object]:
    qualities = []
    costs = []
    for query_id, row in frame.iterrows():
        action = str(actions.loc[query_id])
        quality, cost = row_quality_cost(row, action)
        cost += sum(float(row.get(column, 0.0) or 0.0) for column in feature_cost_columns)
        qualities.append(quality)
        costs.append(cost)
    strong_norm = max(float(frame["strong_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    normalized_cost = float(np.sum(costs) / strong_norm)
    mean_utility = float(mean_quality - lambda_cost * normalized_cost)
    oracle_quality = float(frame["strong_inclusive_cost_oracle_quality"].mean())
    oracle_utility = float(frame["strong_inclusive_cost_oracle_utility"].mean())
    label_accuracy = ""
    empirical_h = ""
    if predicted_labels is not None:
        counts = predicted_labels.value_counts(normalize=True)
        empirical_h = float(-(counts * np.log2(counts)).sum())
    if predicted_labels is not None and target_labels is not None:
        label_accuracy = float((predicted_labels.astype(int) == target_labels.astype(int)).mean())
    return {
        "method": method,
        "feature_set": feature_set,
        "split": str(frame["split"].iloc[0]),
        "k": k,
        "alpha": alpha,
        "beta": beta,
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "strong_inclusive_cost_oracle_quality": oracle_quality,
        "quality_gap_to_strong_inclusive_cost_oracle": float(oracle_quality - mean_quality),
        "normalized_remote_cost_vs_all_strong_gpt": normalized_cost,
        "mean_utility": mean_utility,
        "utility_ratio_to_strong_inclusive_cost_oracle": float(mean_utility / oracle_utility) if oracle_utility else np.nan,
        "label_accuracy_vs_joint_oracle": label_accuracy,
        "empirical_H_Z": empirical_h,
        "mean_confidence": mean_confidence,
        "feature_cost_columns": ",".join(feature_cost_columns),
        "action_counts": json.dumps({str(key): int(value) for key, value in actions.value_counts().to_dict().items()}),
    }


def row_quality_cost(row: pd.Series, action: str) -> tuple[float, float]:
    if action == STRONG_GPT:
        return float(row["strong_quality"]), float(row["strong_cost"])
    if action in LOCAL_MODELS:
        return float(row[f"{action}_quality"]), 0.0
    if action == GEMINI:
        return float(row[f"{GEMINI}_quality"]), float(row[f"{GEMINI}_cost"])
    if action == BASE_GPT:
        return float(row[f"{BASE_GPT}_quality"]), float(row[f"{BASE_GPT}_cost"])
    raise ValueError(f"Unknown action: {action}")


def reference_rows(frame: pd.DataFrame, lambda_cost: float) -> list[dict[str, object]]:
    rows = []
    for action in ALL_MODELS:
        actions = pd.Series(action, index=frame.index)
        rows.append(
            evaluated_row(
                frame,
                actions,
                method=f"all_{action}",
                feature_set="reference",
                k="",
                alpha="",
                beta="",
                lambda_cost=lambda_cost,
            )
        )
    rows.append(
        evaluated_row(
            frame,
            frame["strong_inclusive_cost_oracle_model"],
            method="strong_inclusive_cost_oracle",
            feature_set="reference",
            k="",
            alpha="",
            beta="",
            lambda_cost=lambda_cost,
        )
    )
    return rows


def selected_rows(table: pd.DataFrame, quality_gap_target: float, cost_target: float) -> pd.DataFrame:
    val = table[
        table["split"].eq("val")
        & ~table["method"].isin(["d2_joint_oracle_labels", "strong_inclusive_cost_oracle"])
    ].copy()
    rows = []
    for selection_rule, candidates in [
        ("val_best_utility_under_cost_target", val[val["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)]),
        (
            "val_feasible_quality_cost_target",
            val[
                val["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)
                & val["quality_gap_to_strong_inclusive_cost_oracle"].le(quality_gap_target)
            ],
        ),
        ("val_best_utility_any_cost", val),
    ]:
        if candidates.empty:
            continue
        picked = candidates.sort_values(
            ["utility_ratio_to_strong_inclusive_cost_oracle", "quality_gap_to_strong_inclusive_cost_oracle"],
            ascending=[False, True],
        ).head(1)
        method = str(picked.iloc[0]["method"])
        feature_set = str(picked.iloc[0]["feature_set"])
        k = picked.iloc[0]["k"]
        alpha = picked.iloc[0]["alpha"]
        matches = table[
            table["method"].eq(method)
            & table["feature_set"].eq(feature_set)
            & table["k"].astype(str).eq(str(k))
            & table["alpha"].astype(str).eq(str(alpha))
        ].copy()
        matches["selection_rule"] = selection_rule
        rows.append(matches)

    test = table[
        table["split"].eq("test")
        & ~table["method"].isin(["d2_joint_oracle_labels", "strong_inclusive_cost_oracle"])
    ].copy()
    diagnostic = test[
        test["normalized_remote_cost_vs_all_strong_gpt"].le(cost_target)
        & test["quality_gap_to_strong_inclusive_cost_oracle"].le(quality_gap_target)
    ]
    if not diagnostic.empty:
        picked = diagnostic.sort_values("utility_ratio_to_strong_inclusive_cost_oracle", ascending=False).head(1).copy()
        picked["selection_rule"] = "test_diagnostic_feasible_quality_cost"
        rows.append(picked)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_memo(output_dir: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    key_cols = [
        "selection_rule",
        "method",
        "feature_set",
        "split",
        "k",
        "alpha",
        "mean_quality",
        "quality_gap_to_strong_inclusive_cost_oracle",
        "normalized_remote_cost_vs_all_strong_gpt",
        "utility_ratio_to_strong_inclusive_cost_oracle",
        "label_accuracy_vs_joint_oracle",
        "action_counts",
    ]
    best_test = table[table["split"].eq("test")].sort_values(
        ["utility_ratio_to_strong_inclusive_cost_oracle", "quality_gap_to_strong_inclusive_cost_oracle"],
        ascending=[False, True],
    ).head(12)
    memo = f"""# Strong-Inclusive D2 RouteCode Audit

Input table: `{args.query_table}`

This run applies predictability-constrained RouteCode to the strong-inclusive exact-math model pool.
Codebooks are fit on train rows only. Validation rows select deployable rows; held-out test rows are
reported separately. No model/API calls are made.

## Selected Policies

{markdown_table(selected, [col for col in key_cols if col in selected.columns])}

## Best Held-Out Rows By Utility

{markdown_table(best_test, [col for col in key_cols if col in best_test.columns])}

## Interpretation

The `d2_joint_oracle_labels` rows are diagnostic upper bounds because they use held-out utilities to
assign labels. The deployable rows are `d2_embedding_centroid` and `d2_logistic_label_predictor`.
If the selected deployable rows do not meet the quality and cost gates, this is evidence that D2
alone does not close the exact-math observability gap for the current model pool.
"""
    (output_dir / "STRONG_INCLUSIVE_D2_ROUTECODE_MEMO.md").write_text(memo, encoding="utf-8")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


if __name__ == "__main__":
    main()
