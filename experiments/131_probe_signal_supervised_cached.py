from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge


FRONTIER_MODELS = ["gemini-3.5-flash", "gpt-5.5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached supervised probe-signal predictors for broad100 routing.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_probe_signal_supervised_cached"),
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

    utility_table = run_utility_regression(
        package,
        sandbox,
        outputs,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    frontier_table = run_frontier_gain_gate(
        package,
        sandbox,
        outputs,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    combined = pd.concat([utility_table, frontier_table], ignore_index=True)
    selected = validation_selected_rows(combined)

    utility_table.to_csv(args.output_dir / "table_supervised_utility_regression.csv", index=False)
    frontier_table.to_csv(args.output_dir / "table_frontier_gain_gate.csv", index=False)
    combined.to_csv(args.output_dir / "table_probe_signal_supervised_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_probe_signal_supervised_selected.csv", index=False)
    write_figure(args.output_dir, combined)
    write_memo(args.output_dir / "PROBE_SIGNAL_SUPERVISED_CACHED_MEMO.md", args.outputs, combined, selected)
    print(f"Wrote supervised cached probe-signal sandbox to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_utility_regression(
    package,
    sandbox,
    outputs: pd.DataFrame,
    *,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_view in ["query_text", "query_text_with_benchmark", "query_text_with_local_answers"]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            model, vectorizer, candidate_models = fit_utility_regressor(
                package,
                sandbox,
                outputs,
                feature_view=feature_view,
                alpha=alpha,
                max_features=max_features,
            )
            for split in ["val", "test"]:
                selected = predict_utility_regression_selection(
                    package,
                    sandbox,
                    outputs,
                    model=model,
                    vectorizer=vectorizer,
                    candidate_models=candidate_models,
                    feature_view=feature_view,
                    split=split,
                )
                row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
                row.update(
                    {
                        "method": f"ridge_utility_{feature_view}_alpha{alpha:g}",
                        "family": "multioutput_utility_regression",
                        "feature_view": feature_view,
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
    *,
    feature_view: str,
    alpha: float,
    max_features: int,
):
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    train_queries = query_info[query_info["split"].eq("train")].copy()
    train_ids = train_queries.index.astype(str).tolist()
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    candidate_models = [
        model_id
        for model_id in sorted(outputs["model_id"].astype(str).unique())
        if model_id != package.TOOL_MODEL
    ]
    texts = [
        sandbox.feature_text(str(query_id), train_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in train_ids
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_x = vectorizer.fit_transform(texts)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
    train_y = utility.loc[train_ids, candidate_models].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha))
    model.fit(train_x, train_y)
    return model, vectorizer, candidate_models


def predict_utility_regression_selection(
    package,
    sandbox,
    outputs: pd.DataFrame,
    *,
    model,
    vectorizer,
    candidate_models: list[str],
    feature_view: str,
    split: str,
) -> pd.Series:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    target_queries = query_info[query_info["split"].eq(split)].copy()
    target_ids = target_queries.index.astype(str).tolist()
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    texts = [
        sandbox.feature_text(str(query_id), target_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in target_ids
    ]
    pred = np.asarray(model.predict(vectorizer.transform(texts)), dtype=float)
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
    *,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_view in ["query_text", "query_text_with_benchmark", "query_text_with_local_answers"]:
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            bundle = fit_frontier_gain_models(
                package,
                sandbox,
                outputs,
                feature_view=feature_view,
                alpha=alpha,
                max_features=max_features,
            )
            val_candidates: list[dict[str, Any]] = []
            for threshold in candidate_thresholds(bundle["val_gain_pred"]):
                selected = frontier_gain_selection(package, sandbox, outputs, bundle=bundle, split="val", threshold=threshold)
                row = evaluate_selection(package, outputs, selected, split="val", lambda_cost=lambda_cost)
                row.update(
                    {
                        "method": f"frontier_gain_{feature_view}_alpha{alpha:g}_thr{threshold:.4f}",
                        "family": "budgeted_frontier_gain_gate",
                        "feature_view": feature_view,
                        "alpha": float(alpha),
                        "threshold": float(threshold),
                    }
                )
                val_candidates.append(row)
            feasible = [row for row in val_candidates if float(row["frontier_call_rate"]) <= 0.40]
            pool = feasible or val_candidates
            best = sorted(pool, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]
            rows.append(best)
            test_selected = frontier_gain_selection(
                package,
                sandbox,
                outputs,
                bundle=bundle,
                split="test",
                threshold=float(best["threshold"]),
            )
            test_row = evaluate_selection(package, outputs, test_selected, split="test", lambda_cost=lambda_cost)
            test_row.update(
                {
                    "method": str(best["method"]),
                    "family": "budgeted_frontier_gain_gate",
                    "feature_view": feature_view,
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
    *,
    feature_view: str,
    alpha: float,
    max_features: int,
) -> dict[str, Any]:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    train_queries = query_info[query_info["split"].eq("train")].copy()
    train_ids = train_queries.index.astype(str).tolist()
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    texts = [
        sandbox.feature_text(str(query_id), train_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in train_ids
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_x = vectorizer.fit_transform(texts)
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
    base_train = package.profile_v4_selection_for_split(outputs, split="train")
    base_utility = selected_utility(outputs, base_train).loc[train_ids].to_numpy(dtype=float)
    frontier_utility = utility.loc[train_ids, FRONTIER_MODELS].to_numpy(dtype=float)
    gain_y = frontier_utility.max(axis=1) - base_utility
    gain_model = Ridge(alpha=float(alpha)).fit(train_x, gain_y)
    frontier_model = Ridge(alpha=float(alpha)).fit(train_x, frontier_utility)
    val_gain_pred = predict_gain_values(package, sandbox, outputs, vectorizer, gain_model, feature_view, split="val")
    return {
        "feature_view": feature_view,
        "vectorizer": vectorizer,
        "gain_model": gain_model,
        "frontier_model": frontier_model,
        "val_gain_pred": val_gain_pred,
    }


def predict_gain_values(package, sandbox, outputs: pd.DataFrame, vectorizer, gain_model, feature_view: str, *, split: str) -> np.ndarray:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    target_queries = query_info[query_info["split"].eq(split)].copy()
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    texts = [
        sandbox.feature_text(str(query_id), target_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in target_queries.index.astype(str)
    ]
    return np.asarray(gain_model.predict(vectorizer.transform(texts)), dtype=float)


def frontier_gain_selection(
    package,
    sandbox,
    outputs: pd.DataFrame,
    *,
    bundle: dict[str, Any],
    split: str,
    threshold: float,
) -> pd.Series:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    target_queries = query_info[query_info["split"].eq(split)].copy()
    target_ids = target_queries.index.astype(str).tolist()
    base = package.profile_v4_selection_for_split(outputs, split=split)
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    texts = [
        sandbox.feature_text(str(query_id), target_queries.loc[query_id], by_query, local_models, str(bundle["feature_view"]))
        for query_id in target_ids
    ]
    target_x = bundle["vectorizer"].transform(texts)
    gains = np.asarray(bundle["gain_model"].predict(target_x), dtype=float)
    frontier_pred = np.asarray(bundle["frontier_model"].predict(target_x), dtype=float)
    selected = base.copy()
    for row_index, query_id in enumerate(target_ids):
        if gains[row_index] > float(threshold):
            selected.loc[query_id] = FRONTIER_MODELS[int(np.argmax(frontier_pred[row_index]))]
    return selected


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
    return package.evaluation_row("candidate", selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for family, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[
            group["split"].eq("test")
            & group["feature_view"].eq(best["feature_view"])
            & group["alpha"].eq(best["alpha"])
            & group["threshold"].fillna(-9999).eq(best["threshold"] if pd.notna(best["threshold"]) else -9999)
        ]
        if test.empty:
            test = group[
                group["split"].eq("test")
                & group["feature_view"].eq(best["feature_view"])
                & group["alpha"].eq(best["alpha"])
            ].head(1)
        if not test.empty:
            rows.append(test.iloc[0])
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["selection_rule"] = "validation_best_mean_utility_by_family"
    return out


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["feature_view"].astype(str) + " a=" + plot[
        "alpha"
    ].astype(str)
    ax.barh(labels[::-1], plot["mean_utility"].iloc[::-1], color="#4c78a8")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Cached Supervised Probe-Signal Routing")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_probe_signal_supervised_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, outputs_path: Path, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(5)
    lines = [
        "# Supervised Cached Probe-Signal Sandbox",
        "",
        f"Source outputs: `{outputs_path}`.",
        "",
        "This run makes no model or provider API calls. Predictors are trained on train rows and selected on validation rows.",
        "",
        "## Ideas Tested",
        "",
        "- Multi-output Ridge utility regression over candidate models.",
        "- Budgeted frontier-gain gate: start from `tool_probe_profile_v4`, predict when Gemini/GPT escalation has positive utility value, and select the validation threshold under a 0.40 frontier-rate cap when possible.",
        "- Feature views: query text, query text plus benchmark tags, and query text plus cached local probe answers.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Interpretation",
        "",
        "- These are deployable-feature experiments, not oracle rows: training uses train labels, validation selects hyperparameters, and test is held out for reporting.",
        "- If the selected rows do not beat `observable_local_state_v5` utility `0.6756`, cached surface features are still not enough for the broad100 target.",
        "- Next probe-note item after this is vLLM logprob/confidence probes, because cached parsed answers lack uncertainty information.",
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
