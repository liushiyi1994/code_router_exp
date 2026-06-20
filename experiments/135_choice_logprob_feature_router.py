from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


FRONTIER_MODELS = ["gemini-3.5-flash", "gpt-5.5"]
CHOICE_NUMERIC_COLUMNS = [
    "choice_logprob_a",
    "choice_logprob_b",
    "choice_logprob_c",
    "choice_logprob_d",
    "choice_logprob_margin",
    "choice_entropy",
    "choice_seen_count",
    "choice_missing_count",
    "choice_pred_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen choice-logprob features as cached routing probes.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--choice-table",
        type=Path,
        default=Path("results/controlled/broad100_qwen4_choice_logprob_probe/table_vllm_choice_logprob_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_choice_logprob_feature_router"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    sandbox = load_module("experiments/130_probe_signal_cached_sandbox.py", "probe_signal_cached")
    outputs = package.load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    choice = load_choice_table(args.choice_table)
    outputs = outputs[outputs["query_id"].astype(str).isin(set(choice.index.astype(str)))].copy()

    reference = run_reference_policies(package, outputs, lambda_cost=float(args.lambda_cost))
    utility = run_utility_regression(
        package,
        sandbox,
        outputs,
        choice,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    frontier = run_frontier_gain_gate(
        package,
        sandbox,
        outputs,
        choice,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    combined = pd.concat([reference, utility, frontier], ignore_index=True)
    selected = validation_selected_rows(combined)

    reference.to_csv(args.output_dir / "table_choice_logprob_reference_policies.csv", index=False)
    utility.to_csv(args.output_dir / "table_choice_logprob_utility_regression.csv", index=False)
    frontier.to_csv(args.output_dir / "table_choice_logprob_frontier_gain.csv", index=False)
    combined.to_csv(args.output_dir / "table_choice_logprob_feature_router_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_choice_logprob_feature_router_selected.csv", index=False)
    write_figure(args.output_dir, combined)
    write_memo(args.output_dir / "CHOICE_LOGPROB_FEATURE_ROUTER_MEMO.md", args.outputs, args.choice_table, outputs, choice, combined, selected)
    print(f"Wrote choice-logprob feature router experiment to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_choice_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["query_id"] = frame["query_id"].astype(str)
    frame["choice_pred"] = frame["choice_pred"].fillna("").astype(str)
    frame["choice_pred_index"] = frame["choice_pred"].map({"A": 0, "B": 1, "C": 2, "D": 3}).fillna(-1)
    for column in CHOICE_NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index("query_id").sort_index()


def run_reference_policies(package, outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        target_ids = split_query_ids(outputs, split)
        for method, selected in [
            ("tool_probe_profile_v4_choice_subset", package.profile_v4_selection_for_split(outputs, split=split).loc[target_ids]),
            (
                "observable_local_state_v5_choice_subset",
                package.observable_local_state_selection(outputs, split=split).loc[target_ids],
            ),
        ]:
            row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
            row.update(
                {
                    "method": method,
                    "family": "reference_policy",
                    "feature_view": "cached_policy",
                    "choice_mode": "none",
                    "alpha": np.nan,
                    "threshold": np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def run_utility_regression(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_view in ["query_text_with_benchmark", "query_text_with_local_answers"]:
        for choice_mode in ["none", "tokens", "numeric", "numeric_and_tokens"]:
            for alpha in [0.1, 1.0, 10.0, 100.0]:
                bundle = fit_utility_regressor(
                    package,
                    sandbox,
                    outputs,
                    choice,
                    feature_view=feature_view,
                    choice_mode=choice_mode,
                    alpha=alpha,
                    max_features=max_features,
                )
                for split in ["val", "test"]:
                    selected = predict_utility_regression(package, sandbox, outputs, choice, bundle=bundle, split=split)
                    row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
                    row.update(
                        {
                            "method": f"ridge_utility_{feature_view}_{choice_mode}_alpha{alpha:g}",
                            "family": "multioutput_utility_regression",
                            "feature_view": feature_view,
                            "choice_mode": choice_mode,
                            "alpha": float(alpha),
                            "threshold": np.nan,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def fit_utility_regressor(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    feature_view: str,
    choice_mode: str,
    alpha: float,
    max_features: int,
) -> dict[str, Any]:
    train_ids = split_query_ids(outputs, "train")
    feature_bundle = fit_feature_bundle(
        package,
        sandbox,
        outputs,
        choice,
        query_ids=train_ids,
        feature_view=feature_view,
        choice_mode=choice_mode,
        max_features=max_features,
    )
    train_x = transform_features(package, sandbox, outputs, choice, feature_bundle, train_ids)
    candidate_models = candidate_model_ids(package, outputs)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
    train_y = utility.loc[train_ids, candidate_models].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha)).fit(train_x, train_y)
    return {
        "model": model,
        "candidate_models": candidate_models,
        "feature_bundle": feature_bundle,
        "feature_view": feature_view,
        "choice_mode": choice_mode,
        "alpha": float(alpha),
    }


def predict_utility_regression(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    split: str,
) -> pd.Series:
    target_ids = split_query_ids(outputs, split)
    target_x = transform_features(package, sandbox, outputs, choice, bundle["feature_bundle"], target_ids)
    pred = np.asarray(bundle["model"].predict(target_x), dtype=float)
    candidate_models = list(bundle["candidate_models"])
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for row_index, query_id in enumerate(target_ids):
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        selected[query_id] = candidate_models[int(np.argmax(pred[row_index]))]
    return pd.Series(selected)


def run_frontier_gain_gate(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_view in ["query_text_with_benchmark", "query_text_with_local_answers"]:
        for choice_mode in ["none", "tokens", "numeric", "numeric_and_tokens"]:
            for alpha in [0.1, 1.0, 10.0, 100.0]:
                bundle = fit_frontier_gain_models(
                    package,
                    sandbox,
                    outputs,
                    choice,
                    feature_view=feature_view,
                    choice_mode=choice_mode,
                    alpha=alpha,
                    max_features=max_features,
                )
                val_candidates: list[dict[str, Any]] = []
                for threshold in candidate_thresholds(bundle["val_gain_pred"]):
                    selected = frontier_gain_selection(
                        package,
                        sandbox,
                        outputs,
                        choice,
                        bundle=bundle,
                        split="val",
                        threshold=threshold,
                    )
                    row = evaluate_selection(package, outputs, selected, split="val", lambda_cost=lambda_cost)
                    row.update(
                        {
                            "method": f"frontier_gain_{feature_view}_{choice_mode}_alpha{alpha:g}_thr{threshold:.4f}",
                            "family": "budgeted_frontier_gain_gate",
                            "feature_view": feature_view,
                            "choice_mode": choice_mode,
                            "alpha": float(alpha),
                            "threshold": float(threshold),
                        }
                    )
                    val_candidates.append(row)
                feasible = [row for row in val_candidates if float(row["frontier_call_rate"]) <= 0.40]
                pool = feasible or val_candidates
                best = sorted(pool, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]
                rows.append(best)
                selected = frontier_gain_selection(
                    package,
                    sandbox,
                    outputs,
                    choice,
                    bundle=bundle,
                    split="test",
                    threshold=float(best["threshold"]),
                )
                test_row = evaluate_selection(package, outputs, selected, split="test", lambda_cost=lambda_cost)
                test_row.update(
                    {
                        "method": str(best["method"]),
                        "family": "budgeted_frontier_gain_gate",
                        "feature_view": feature_view,
                        "choice_mode": choice_mode,
                        "alpha": float(alpha),
                        "threshold": float(best["threshold"]),
                    }
                )
                rows.append(test_row)
    return pd.DataFrame(rows)


def fit_frontier_gain_models(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    feature_view: str,
    choice_mode: str,
    alpha: float,
    max_features: int,
) -> dict[str, Any]:
    train_ids = split_query_ids(outputs, "train")
    feature_bundle = fit_feature_bundle(
        package,
        sandbox,
        outputs,
        choice,
        query_ids=train_ids,
        feature_view=feature_view,
        choice_mode=choice_mode,
        max_features=max_features,
    )
    train_x = transform_features(package, sandbox, outputs, choice, feature_bundle, train_ids)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
    base_train = package.profile_v4_selection_for_split(outputs, split="train").loc[train_ids]
    base_utility = selected_utility(outputs, base_train).loc[train_ids].to_numpy(dtype=float)
    frontier_utility = utility.loc[train_ids, FRONTIER_MODELS].to_numpy(dtype=float)
    gain_y = frontier_utility.max(axis=1) - base_utility
    gain_model = Ridge(alpha=float(alpha)).fit(train_x, gain_y)
    frontier_model = Ridge(alpha=float(alpha)).fit(train_x, frontier_utility)
    val_ids = split_query_ids(outputs, "val")
    val_gain_pred = np.asarray(gain_model.predict(transform_features(package, sandbox, outputs, choice, feature_bundle, val_ids)))
    return {
        "feature_bundle": feature_bundle,
        "gain_model": gain_model,
        "frontier_model": frontier_model,
        "val_gain_pred": val_gain_pred,
        "feature_view": feature_view,
        "choice_mode": choice_mode,
        "alpha": float(alpha),
    }


def frontier_gain_selection(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    split: str,
    threshold: float,
) -> pd.Series:
    target_ids = split_query_ids(outputs, split)
    selected = package.profile_v4_selection_for_split(outputs, split=split).loc[target_ids].copy()
    target_x = transform_features(package, sandbox, outputs, choice, bundle["feature_bundle"], target_ids)
    gains = np.asarray(bundle["gain_model"].predict(target_x), dtype=float)
    frontier_pred = np.asarray(bundle["frontier_model"].predict(target_x), dtype=float)
    for row_index, query_id in enumerate(target_ids):
        if gains[row_index] > float(threshold):
            selected.loc[query_id] = FRONTIER_MODELS[int(np.argmax(frontier_pred[row_index]))]
    return selected


def fit_feature_bundle(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    *,
    query_ids: list[str],
    feature_view: str,
    choice_mode: str,
    max_features: int,
) -> dict[str, Any]:
    texts = build_texts(package, sandbox, outputs, choice, query_ids, feature_view, choice_mode)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    vectorizer.fit(texts)
    numeric_mean = None
    numeric_scale = None
    if "numeric" in choice_mode:
        numeric = numeric_features(choice, query_ids)
        numeric_mean = numeric.mean(axis=0)
        numeric_scale = numeric.std(axis=0)
        numeric_scale[numeric_scale < 1e-8] = 1.0
    return {
        "vectorizer": vectorizer,
        "numeric_mean": numeric_mean,
        "numeric_scale": numeric_scale,
        "feature_view": feature_view,
        "choice_mode": choice_mode,
    }


def transform_features(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    bundle: dict[str, Any],
    query_ids: list[str],
):
    texts = build_texts(package, sandbox, outputs, choice, query_ids, str(bundle["feature_view"]), str(bundle["choice_mode"]))
    text_x = bundle["vectorizer"].transform(texts)
    if "numeric" not in str(bundle["choice_mode"]):
        return text_x
    numeric = numeric_features(choice, query_ids)
    numeric = (numeric - bundle["numeric_mean"]) / bundle["numeric_scale"]
    return hstack([text_x, csr_matrix(numeric)], format="csr")


def build_texts(
    package,
    sandbox,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    query_ids: list[str],
    feature_view: str,
    choice_mode: str,
) -> list[str]:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    texts = []
    for query_id in query_ids:
        text = sandbox.feature_text(str(query_id), query_info.loc[query_id], by_query, local_models, feature_view)
        if "tokens" in choice_mode:
            text = f"{text} {choice_tokens(choice.loc[query_id])}"
        texts.append(text)
    return texts


def choice_tokens(row: pd.Series) -> str:
    pred = str(row.get("choice_pred", "")).strip().upper() or "missing"
    return " ".join(
        [
            f"choice_pred_{pred}",
            f"choice_margin_bin_{numeric_bin(row.get('choice_logprob_margin'), [0.25, 0.75, 1.5, 3.0])}",
            f"choice_entropy_bin_{numeric_bin(row.get('choice_entropy'), [0.5, 1.0, 1.5])}",
            f"choice_seen_{int(row.get('choice_seen_count', 0) or 0)}",
        ]
    )


def numeric_bin(value: object, thresholds: list[float]) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "missing"
    for index, threshold in enumerate(thresholds):
        if numeric <= threshold:
            return str(index)
    return str(len(thresholds))


def numeric_features(choice: pd.DataFrame, query_ids: list[str]) -> np.ndarray:
    numeric = choice.loc[query_ids, CHOICE_NUMERIC_COLUMNS].to_numpy(dtype=float)
    return np.nan_to_num(numeric, nan=-50.0, posinf=0.0, neginf=-50.0)


def split_query_ids(outputs: pd.DataFrame, split: str) -> list[str]:
    return (
        outputs[outputs["split"].eq(split)]
        .drop_duplicates("query_id")
        .sort_values(["benchmark", "query_id"])["query_id"]
        .astype(str)
        .tolist()
    )


def candidate_model_ids(package, outputs: pd.DataFrame) -> list[str]:
    return [
        model_id
        for model_id in sorted(outputs["model_id"].astype(str).unique())
        if model_id != package.TOOL_MODEL
    ]


def selected_utility(outputs: pd.DataFrame, selected: pd.Series) -> pd.Series:
    rows = pd.DataFrame({"query_id": selected.index.astype(str), "model_id": selected.values.astype(str)})
    merged = rows.merge(outputs[["query_id", "model_id", "utility"]], on=["query_id", "model_id"], how="left")
    return pd.Series(merged["utility"].to_numpy(dtype=float), index=merged["query_id"].astype(str))


def candidate_thresholds(values: np.ndarray) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return [0.0]
    qs = np.quantile(finite, np.linspace(0.0, 0.95, 20)).tolist()
    return sorted(set(float(value) for value in [-1.0, 0.0, *qs]))


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    lambda_cost: float,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row("candidate", selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["benchmarks_json"] = selected_rows["benchmark"].value_counts().sort_index().to_json()
    return row


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[group["split"].eq("test") & group["method"].eq(best["method"])]
        if not test.empty:
            rows.append(test.iloc[0])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["selection_rule"] = "validation_best_mean_utility_by_family"
    return out


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    labels = (
        plot["family"].str.replace("_", " ", regex=False)
        + " / "
        + plot["feature_view"].astype(str)
        + " / "
        + plot["choice_mode"].astype(str)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], plot["mean_utility"].iloc[::-1], color="#4c78a8")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Choice-Logprob Feature Routing On GPQA/MMLUPro")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_choice_logprob_feature_router_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    outputs_path: Path,
    choice_path: Path,
    outputs: pd.DataFrame,
    choice: pd.DataFrame,
    combined: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    split_counts = (
        outputs.drop_duplicates("query_id")
        .groupby(["split", "benchmark"])
        .size()
        .rename("n_queries")
        .reset_index()
    )
    best_test = combined[combined["split"].eq("test")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    ).head(8)
    choice_summary = choice.groupby(["split", "benchmark"]).agg(
        n=("model_id", "size"),
        mean_probe_quality=("quality_score", "mean"),
        mean_margin=("choice_logprob_margin", "mean"),
        mean_entropy=("choice_entropy", "mean"),
    )
    probe_model = str(choice["model_id"].dropna().iloc[0]) if not choice["model_id"].dropna().empty else "choice-logprob probe"
    lines = [
        "# Choice-Logprob Feature Router",
        "",
        f"Source outputs: `{outputs_path}`.",
        f"Choice-logprob probe table: `{choice_path}`.",
        "",
        f"This run makes no external model or provider API calls. It uses cached `{probe_model}` vLLM choice-token logprob probe outputs.",
        "",
        "## Scope",
        "",
        "The choice-logprob probe currently applies to GPQA and MMLUPro multiple-choice rows only.",
        "",
        markdown_table(split_counts),
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Choice Probe Diagnostics",
        "",
        markdown_table(choice_summary.reset_index()),
        "",
        "## Interpretation",
        "",
        "- This is the constrained option-token version of the logprob idea. It fixes the free-generation truncation issue.",
        f"- If choice features still do not improve validation-selected held-out utility, `{probe_model}` confidence is not enough observability for these broad100 misses.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
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
