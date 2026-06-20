from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
SMALL_LOCAL_MODELS = ("qwen3-4b-local", "qwen3-0.6b-probe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate cached Gemini-metadata gates for the mixed exact-math slice."
    )
    parser.add_argument(
        "--query-table",
        default="results/controlled/mixed_local_consensus_gate/query_table_with_small_locals.csv",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[
            "results/controlled/math500_qwen8_live_pilot_1024",
            "results/controlled/livemathbench_live_pilot_1024",
            "results/controlled/aime_qwen8_live_pilot_1024",
        ],
    )
    parser.add_argument("--output-dir", default="results/controlled/gemini_metadata_gate")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--target-quality-gap", type=float, default=0.03)
    parser.add_argument("--threshold-grid-size", type=int, default=18)
    return parser.parse_args()


def json_safe_counts(series: pd.Series) -> str:
    counts = {str(key): int(value) for key, value in series.value_counts().to_dict().items()}
    return json.dumps(counts, sort_keys=True)


def extract_usage_metadata(raw_path: object) -> dict[str, float]:
    path = Path(str(raw_path))
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
    if not isinstance(usage, dict):
        return {}
    return {
        "gemini_prompt_tokens": float(usage.get("promptTokenCount", 0) or 0),
        "gemini_candidate_tokens": float(usage.get("candidatesTokenCount", 0) or 0),
        "gemini_thoughts_tokens": float(usage.get("thoughtsTokenCount", 0) or 0),
        "gemini_total_tokens": float(usage.get("totalTokenCount", 0) or 0),
    }


def load_gemini_metadata(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[frame["status"].eq("success") & frame["model_id"].eq(GEMINI_MODEL)].copy()
        for _, row in frame.iterrows():
            usage = extract_usage_metadata(row.get("raw_output_path", ""))
            rows.append(
                {
                    "query_id": str(row["query_id"]),
                    "gemini_raw_output_path": str(row.get("raw_output_path", "")),
                    "gemini_meta_latency_s": float(row.get("latency_s", 0.0) or 0.0),
                    "gemini_prompt_tokens": usage.get("gemini_prompt_tokens", float(row.get("input_tokens", 0) or 0)),
                    "gemini_candidate_tokens": usage.get(
                        "gemini_candidate_tokens", float(row.get("output_tokens", 0) or 0)
                    ),
                    "gemini_thoughts_tokens": usage.get("gemini_thoughts_tokens", 0.0),
                    "gemini_total_tokens": usage.get("gemini_total_tokens", 0.0),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "query_id",
                "gemini_raw_output_path",
                "gemini_meta_latency_s",
                "gemini_prompt_tokens",
                "gemini_candidate_tokens",
                "gemini_thoughts_tokens",
                "gemini_total_tokens",
            ]
        )
    return pd.DataFrame(rows).drop_duplicates("query_id", keep="last")


def add_features(table: pd.DataFrame, gemini_metadata: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["query_id"] = table["query_id"].astype(str)
    table = table.merge(gemini_metadata, on="query_id", how="left")
    for column in [
        "gemini_meta_latency_s",
        "gemini_prompt_tokens",
        "gemini_candidate_tokens",
        "gemini_thoughts_tokens",
        "gemini_total_tokens",
        "verifier_cost",
        "self_cost",
    ]:
        if column not in table.columns:
            table[column] = 0.0
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)

    for column in ["qwen8_gemini_agree", "self_gemini_agree", "self_qwen8_agree"]:
        if column not in table.columns:
            table[column] = False
        table[column] = table[column].fillna(False).astype(bool)

    for model_id in SMALL_LOCAL_MODELS:
        answer_col = f"{model_id}_answer"
        if answer_col not in table.columns:
            table[answer_col] = ""
        table[answer_col] = table[answer_col].fillna("").astype(str)
        table[f"{model_id}_answer_len"] = table[answer_col].str.len()

    qwen_answer = table[f"{LOCAL_MODEL}_answer"].fillna("").astype(str)
    qwen4_answer = table["qwen3-4b-local_answer"].fillna("").astype(str)
    qwen06_answer = table["qwen3-0.6b-probe_answer"].fillna("").astype(str)
    table["qwen8_4b_agree"] = qwen_answer.eq(qwen4_answer) & qwen4_answer.ne("")
    table["qwen8_06b_agree"] = qwen_answer.eq(qwen06_answer) & qwen06_answer.ne("")
    table["small_pair_agree"] = (
        table["qwen8_4b_agree"] | table["qwen8_06b_agree"] | (qwen4_answer.eq(qwen06_answer) & qwen4_answer.ne(""))
    )
    table["all_three_agree"] = table["qwen8_4b_agree"] & table["qwen8_06b_agree"]
    table["answer_len_gap_qwen_gemini"] = (
        pd.to_numeric(table[f"{LOCAL_MODEL}_answer_len"], errors="coerce").fillna(0.0)
        - pd.to_numeric(table[f"{GEMINI_MODEL}_answer_len"], errors="coerce").fillna(0.0)
    ).abs()
    table["verifier_verdict"] = table.get("verifier_verdict", "missing").fillna("missing").astype(str)

    for column in [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        f"{LOCAL_MODEL}_answer_len",
        f"{GEMINI_MODEL}_answer_len",
        "verifier_input_tokens",
        "verifier_output_tokens",
        "self_input_tokens",
        "self_output_tokens",
    ]:
        if column not in table.columns:
            table[column] = 0.0
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table


def feature_sets() -> dict[str, dict[str, Any]]:
    return {
        "local_only": {
            "base_cost": "none",
            "cat": ["dataset", "qwen8_4b_agree", "qwen8_06b_agree", "small_pair_agree", "all_three_agree"],
            "num": [
                "query_len",
                "number_count",
                "latex_count",
                "frac_count",
                "sqrt_count",
                f"{LOCAL_MODEL}_answer_len",
                "qwen3-4b-local_answer_len",
                "qwen3-0.6b-probe_answer_len",
            ],
        },
        "gemini_metadata": {
            "base_cost": "gemini",
            "cat": [
                "dataset",
                "qwen8_gemini_agree",
                "qwen8_4b_agree",
                "qwen8_06b_agree",
                "small_pair_agree",
                "all_three_agree",
            ],
            "num": [
                "query_len",
                "number_count",
                "latex_count",
                "frac_count",
                "sqrt_count",
                f"{LOCAL_MODEL}_answer_len",
                f"{GEMINI_MODEL}_answer_len",
                "answer_len_gap_qwen_gemini",
                "gemini_prompt_tokens",
                "gemini_candidate_tokens",
                "gemini_thoughts_tokens",
                "gemini_total_tokens",
                "gemini_meta_latency_s",
            ],
        },
        "gemini_verifier_self": {
            "base_cost": "gemini_full",
            "cat": [
                "dataset",
                "verifier_verdict",
                "qwen8_gemini_agree",
                "self_gemini_agree",
                "self_qwen8_agree",
                "qwen8_4b_agree",
                "qwen8_06b_agree",
                "small_pair_agree",
                "all_three_agree",
            ],
            "num": [
                "query_len",
                "number_count",
                "latex_count",
                "frac_count",
                "sqrt_count",
                f"{LOCAL_MODEL}_answer_len",
                f"{GEMINI_MODEL}_answer_len",
                "answer_len_gap_qwen_gemini",
                "gemini_prompt_tokens",
                "gemini_candidate_tokens",
                "gemini_thoughts_tokens",
                "gemini_total_tokens",
                "verifier_input_tokens",
                "verifier_output_tokens",
                "self_input_tokens",
                "self_output_tokens",
            ],
        },
    }


def make_classifier(model_name: str, spec: dict[str, Any]):
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), spec["cat"]),
            ("num", StandardScaler(), spec["num"]),
        ]
    )
    if model_name == "logistic":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced")
    elif model_name == "rf":
        estimator = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
    elif model_name == "gb":
        estimator = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError(model_name)
    return make_pipeline(preprocessor, estimator)


def oracle_action_labels(frame: pd.DataFrame) -> pd.Series:
    utility_cols = [
        f"{LOCAL_MODEL}_utility_selected_cost",
        f"{GEMINI_MODEL}_utility_selected_cost",
        f"{GPT_MODEL}_utility_selected_cost",
    ]
    mapping = {
        f"{LOCAL_MODEL}_utility_selected_cost": "local",
        f"{GEMINI_MODEL}_utility_selected_cost": "gemini",
        f"{GPT_MODEL}_utility_selected_cost": "gpt",
    }
    return frame[utility_cols].idxmax(axis=1).map(mapping)


def action_from_probs(probs: pd.DataFrame, *, gpt_threshold: float, local_threshold: float) -> pd.Series:
    actions = pd.Series("gemini", index=probs.index)
    if "local" in probs.columns:
        actions.loc[probs["local"].ge(local_threshold)] = "local"
    if "gpt" in probs.columns:
        actions.loc[probs["gpt"].ge(gpt_threshold)] = "gpt"
    return actions


def predict_probabilities(clf: Any, frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    probs = clf.predict_proba(frame[spec["cat"] + spec["num"]])
    return pd.DataFrame(probs, index=frame.index, columns=list(clf.classes_))


def base_observation_cost(row: pd.Series, base_cost: str) -> tuple[float, bool]:
    cost = 0.0
    gemini_observed = False
    if base_cost in {"gemini", "gemini_verifier", "gemini_self", "gemini_full"}:
        cost += float(row[f"{GEMINI_MODEL}_cost"])
        gemini_observed = True
    if base_cost in {"gemini_verifier", "gemini_full"}:
        cost += float(row.get("verifier_cost", 0.0) or 0.0)
    if base_cost in {"gemini_self", "gemini_full"}:
        cost += float(row.get("self_cost", 0.0) or 0.0)
    return cost, gemini_observed


def evaluate_actions(
    table: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    split: str,
    feature_set: str,
    model_name: str,
    gpt_threshold: float,
    local_threshold: float,
    base_cost: str,
    lambda_cost: float,
    cost_norm: float,
    selection_role: str,
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    local_final: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        base_cost_value, gemini_observed = base_observation_cost(row, base_cost)
        if action == "local":
            quality = float(row[f"{LOCAL_MODEL}_quality"])
            cost = base_cost_value
            latency = float(row[f"{LOCAL_MODEL}_latency"])
            gpt = False
            gemini = gemini_observed
            local = True
        elif action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = base_cost_value if gemini_observed else float(row[f"{GEMINI_MODEL}_cost"])
            latency = max(float(row[f"{LOCAL_MODEL}_latency"]), float(row[f"{GEMINI_MODEL}_latency"]))
            gpt = False
            gemini = True
            local = False
        elif action == "gpt":
            quality = (
                float(row[f"{GPT_MODEL}_quality"])
                if bool(row.get("gpt_answer_available", True))
                else float(row[f"{GEMINI_MODEL}_quality"])
            )
            cost = base_cost_value + float(row[f"{GPT_MODEL}_cost"]) if gemini_observed else float(row[f"{GPT_MODEL}_cost"])
            latency = (
                max(float(row[f"{LOCAL_MODEL}_latency"]), float(row[f"{GEMINI_MODEL}_latency"]))
                if gemini_observed
                else float(row[f"{LOCAL_MODEL}_latency"])
            ) + float(row[f"{GPT_MODEL}_latency"])
            gpt = True
            gemini = gemini_observed
            local = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        latencies.append(latency)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
        local_final.append(local)

    quality_oracle = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    cost_oracle = table[
        [
            f"{LOCAL_MODEL}_utility_selected_cost",
            f"{GEMINI_MODEL}_utility_selected_cost",
            f"{GPT_MODEL}_utility_selected_cost",
        ]
    ].max(axis=1)
    sequential_oracle = table.get("sequential_cost_oracle_utility", cost_oracle).astype(float)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(mean_quality - lambda_cost * (np.mean(costs) / cost_norm))
    return {
        "selection_role": selection_role,
        "method": method,
        "split": split,
        "feature_set": feature_set,
        "model": model_name,
        "base_cost_mode": base_cost,
        "gpt_threshold": float(gpt_threshold),
        "local_threshold": float(local_threshold),
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_oracle_mean_quality": float(quality_oracle.mean()),
        "cost_oracle_mean_utility": float(cost_oracle.mean()),
        "sequential_oracle_mean_utility": float(sequential_oracle.mean()),
        "quality_gap_to_oracle": float(quality_oracle.mean() - mean_quality),
        "utility_ratio_to_cost_oracle": float(mean_utility / cost_oracle.mean())
        if abs(float(cost_oracle.mean())) > 1e-12
        else np.nan,
        "utility_ratio_to_sequential_oracle": float(mean_utility / sequential_oracle.mean())
        if abs(float(sequential_oracle.mean())) > 1e-12
        else np.nan,
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum())
        if float(table[f"{GPT_MODEL}_cost"].astype(float).sum()) > 0
        else np.nan,
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gemini_calls, gpt_calls)])),
        "gemini_call_rate": float(np.mean(gemini_calls)),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "local_final_rate": float(np.mean(local_final)),
        "remote_cost_total_usd": float(np.sum(costs)),
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
        "action_counts": json_safe_counts(actions),
    }


def evaluate_baselines(table: pd.DataFrame, lambda_cost: float, cost_norm: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        policies = {
            "all_qwen3-8b-local": pd.Series("local", index=frame.index),
            "all_gemini-3.5-flash": pd.Series("gemini", index=frame.index),
            "all_gpt-5.5": pd.Series("gpt", index=frame.index),
            "qwen8_gemini_agreement_else_gpt": pd.Series("gpt", index=frame.index),
        }
        policies["qwen8_gemini_agreement_else_gpt"].loc[frame["qwen8_gemini_agree"].astype(bool)] = "gemini"
        for method, actions in policies.items():
            rows.append(
                evaluate_actions(
                    frame,
                    actions,
                    method=method,
                    split=str(split),
                    feature_set="baseline",
                    model_name="rule",
                    gpt_threshold=np.nan,
                    local_threshold=np.nan,
                    base_cost="none",
                    lambda_cost=lambda_cost,
                    cost_norm=cost_norm,
                    selection_role="baseline",
                )
            )
    return rows


def sweep_rows(
    table: pd.DataFrame,
    *,
    threshold_grid: np.ndarray,
    lambda_cost: float,
    cost_norm: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any]]:
    train = table[table["split"].eq("train")].copy()
    val = table[table["split"].eq("val")].copy()
    test = table[table["split"].eq("test")].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError("Need non-empty train, val, and test splits.")

    rows: list[dict[str, object]] = []
    trained: dict[tuple[str, str], Any] = {}
    y_train = oracle_action_labels(train)
    for feature_set_name, spec in feature_sets().items():
        features = spec["cat"] + spec["num"]
        for model_name in ["logistic", "rf", "extra_trees", "gb"]:
            clf = make_classifier(model_name, spec)
            clf.fit(train[features], y_train)
            trained[(feature_set_name, model_name)] = clf
            probs_by_split = {
                "val": predict_probabilities(clf, val, spec),
                "test": predict_probabilities(clf, test, spec),
            }
            frames_by_split = {"val": val, "test": test}
            for gpt_threshold in threshold_grid:
                for local_threshold in threshold_grid:
                    method = (
                        f"{feature_set_name}_{model_name}_"
                        f"tg{gpt_threshold:.2f}_tl{local_threshold:.2f}"
                    )
                    for split_name, frame in frames_by_split.items():
                        actions = action_from_probs(
                            probs_by_split[split_name],
                            gpt_threshold=float(gpt_threshold),
                            local_threshold=float(local_threshold),
                        )
                        rows.append(
                            evaluate_actions(
                                frame,
                                actions,
                                method=method,
                                split=split_name,
                                feature_set=feature_set_name,
                                model_name=model_name,
                                gpt_threshold=float(gpt_threshold),
                                local_threshold=float(local_threshold),
                                base_cost=str(spec["base_cost"]),
                                lambda_cost=lambda_cost,
                                cost_norm=cost_norm,
                                selection_role="candidate",
                            )
                        )
    return pd.DataFrame(rows), trained


def refit_selected_on_train_val(
    table: pd.DataFrame,
    selected: pd.Series,
    *,
    lambda_cost: float,
    cost_norm: float,
) -> dict[str, object]:
    feature_set_name = str(selected["feature_set"])
    model_name = str(selected["model"])
    spec = feature_sets()[feature_set_name]
    train_val = table[table["split"].isin(["train", "val"])].copy()
    test = table[table["split"].eq("test")].copy()
    clf = make_classifier(model_name, spec)
    clf.fit(train_val[spec["cat"] + spec["num"]], oracle_action_labels(train_val))
    probs = predict_probabilities(clf, test, spec)
    actions = action_from_probs(
        probs,
        gpt_threshold=float(selected["gpt_threshold"]),
        local_threshold=float(selected["local_threshold"]),
    )
    return evaluate_actions(
        test,
        actions,
        method=f"{selected['method']}__refit_train_val",
        split="test",
        feature_set=feature_set_name,
        model_name=model_name,
        gpt_threshold=float(selected["gpt_threshold"]),
        local_threshold=float(selected["local_threshold"]),
        base_cost=str(spec["base_cost"]),
        lambda_cost=lambda_cost,
        cost_norm=cost_norm,
        selection_role="selected_test_refit_train_val",
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values: list[str] = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.query_table)
    metadata = load_gemini_metadata([Path(path) for path in args.run_dirs])
    table = add_features(table, metadata)
    table_path = output_dir / "query_table_with_gemini_metadata.csv"
    table.to_csv(table_path, index=False)

    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    grid = np.linspace(0.10, 0.95, max(2, int(args.threshold_grid_size)))
    candidate_rows, _ = sweep_rows(table, threshold_grid=grid, lambda_cost=args.lambda_cost, cost_norm=cost_norm)
    baseline_rows = pd.DataFrame(evaluate_baselines(table, args.lambda_cost, cost_norm))
    results = pd.concat([baseline_rows, candidate_rows], ignore_index=True)
    results = results.sort_values(["split", "selection_role", "mean_utility", "mean_quality"], ascending=[True, True, False, False])
    results_path = output_dir / "table_gemini_metadata_gate.csv"
    results.to_csv(results_path, index=False)

    val_candidates = candidate_rows[candidate_rows["split"].eq("val")].copy()
    val_feasible = val_candidates[val_candidates["quality_gap_to_oracle"].le(args.target_quality_gap + 1e-12)]
    val_pool = val_feasible if not val_feasible.empty else val_candidates
    selected_val = val_pool.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]
    selected_test = candidate_rows[
        candidate_rows["split"].eq("test") & candidate_rows["method"].eq(str(selected_val["method"]))
    ].iloc[0]
    selected_refit = refit_selected_on_train_val(
        table,
        selected_val,
        lambda_cost=args.lambda_cost,
        cost_norm=cost_norm,
    )
    test_candidates = candidate_rows[candidate_rows["split"].eq("test")].copy()
    test_feasible = test_candidates[test_candidates["quality_gap_to_oracle"].le(args.target_quality_gap + 5e-4)]
    best_test_diagnostic = (
        test_feasible if not test_feasible.empty else test_candidates
    ).sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]
    diagnostic_val = candidate_rows[
        candidate_rows["split"].eq("val") & candidate_rows["method"].eq(str(best_test_diagnostic["method"]))
    ].iloc[0]
    selected_rows = pd.DataFrame(
        [
            {**selected_val.to_dict(), "selection_role": "selected_val"},
            {**selected_test.to_dict(), "selection_role": "selected_test_train_only"},
            selected_refit,
            {**diagnostic_val.to_dict(), "selection_role": "best_test_diagnostic_validation_counterpart"},
            {**best_test_diagnostic.to_dict(), "selection_role": "best_test_diagnostic_not_selected"},
        ]
    )
    selected_path = output_dir / "table_gemini_metadata_selected.csv"
    selected_rows.to_csv(selected_path, index=False)

    memo_columns = [
        "selection_role",
        "method",
        "split",
        "mean_quality",
        "mean_utility",
        "quality_gap_to_oracle",
        "utility_ratio_to_cost_oracle",
        "utility_ratio_to_sequential_oracle",
        "normalized_remote_cost_vs_all_gpt",
        "frontier_call_rate",
        "gemini_call_rate",
        "gpt_call_rate",
        "local_final_rate",
        "remote_cost_total_usd",
        "action_counts",
    ]
    memo = [
        "# Gemini Metadata Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Run dirs: `{', '.join(args.run_dirs)}`.",
        f"Rows: `{len(table)}`. This script makes no API calls; it reads cached parquet rows and Gemini raw JSON metadata.",
        "",
        "The deployable accounting is intentionally conservative: a gate that uses Gemini answer/metadata pays the Gemini call even if it finally returns the local answer. Verifier/self-consistency feature sets also pay their cached probe costs on every row where those features are used.",
        "",
        "Selection protocol: train classifiers on train, choose thresholds on validation under the target quality gap, then report the matching held-out test row. The diagnostic test row is selected on test and is not a valid method-selection result.",
        "",
        "## Selected And Diagnostic Rows",
        "",
        markdown_table(selected_rows[memo_columns]),
        "",
        "## Best Held-Out Test Candidates",
        "",
        markdown_table(
            test_candidates.sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)[memo_columns]
        ),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{selected_path}`",
        f"- `{table_path}`",
    ]
    memo_path = output_dir / "GEMINI_METADATA_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {selected_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
