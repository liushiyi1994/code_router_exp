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


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict cached Gemini-strong gain over broad100 base policies.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_gemini_strong_solver/model_outputs_with_gemini_strong.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_cached_strong_gain_regressor_gate"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    sandbox = load_module("experiments/130_probe_signal_cached_sandbox.py", "probe_signal_cached")
    ttc = load_module("experiments/138_cached_test_time_compute_router.py", "cached_ttc")
    outputs = ttc.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()

    table = run_regressor_gates(
        package,
        sandbox,
        outputs,
        outputs_no_strong,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_cached_strong_gain_regressor_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_cached_strong_gain_regressor_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "CACHED_STRONG_GAIN_REGRESSOR_MEMO.md", args.outputs, table, selected)
    print(f"Wrote cached strong-gain regressor results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_regressor_gates(
    package,
    sandbox,
    outputs: pd.DataFrame,
    outputs_no_strong: pd.DataFrame,
    *,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for base_name in ["tool_probe_profile_v4_no_strong", "observable_local_state_v5_no_strong"]:
        base = {
            "val": base_selection(package, outputs_no_strong, base_name=base_name, split="val"),
            "test": base_selection(package, outputs_no_strong, base_name=base_name, split="test"),
        }
        for split in ["val", "test"]:
            rows.append(evaluate_selection(package, outputs, base[split], split=split, lambda_cost=lambda_cost, method=base_name, family="base"))
        diagnostic = {
            split: oracle_between_base_and_strong(outputs, base[split])
            for split in ["val", "test"]
        }
        for split in ["val", "test"]:
            rows.append(
                evaluate_selection(
                    package,
                    outputs,
                    diagnostic[split],
                    split=split,
                    lambda_cost=lambda_cost,
                    method=f"{base_name}_oracle_between_base_and_strong",
                    family="diagnostic_oracle",
                )
            )
        for feature_view in ["query_text", "query_text_with_benchmark", "query_text_with_local_answers"]:
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                bundle = fit_gain_regressor(
                    package,
                    sandbox,
                    outputs,
                    outputs_no_strong,
                    base_selection_val=base["val"],
                    feature_view=feature_view,
                    alpha=alpha,
                    max_features=max_features,
                )
                val_candidates = []
                for threshold in candidate_thresholds(bundle["val_pred"]):
                    selected = apply_gain_gate(
                        package,
                        sandbox,
                        outputs_no_strong,
                        base["val"],
                        bundle=bundle,
                        split="val",
                        threshold=threshold,
                    )
                    row = evaluate_selection(
                        package,
                        outputs,
                        selected,
                        split="val",
                        lambda_cost=lambda_cost,
                        method=f"{base_name}_strong_gain_{feature_view}_alpha{alpha:g}_thr{threshold:.4f}",
                        family="strong_gain_regressor_gate",
                    )
                    row.update({"base_method": base_name, "feature_view": feature_view, "alpha": float(alpha), "threshold": float(threshold)})
                    val_candidates.append(row)
                pool = sorted(val_candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)
                best = pool[0]
                rows.append(best)
                test_selected = apply_gain_gate(
                    package,
                    sandbox,
                    outputs_no_strong,
                    base["test"],
                    bundle=bundle,
                    split="test",
                    threshold=float(best["threshold"]),
                )
                test_row = evaluate_selection(
                    package,
                    outputs,
                    test_selected,
                    split="test",
                    lambda_cost=lambda_cost,
                    method=str(best["method"]),
                    family="strong_gain_regressor_gate",
                )
                test_row.update(
                    {
                        "base_method": base_name,
                        "feature_view": feature_view,
                        "alpha": float(alpha),
                        "threshold": float(best["threshold"]),
                    }
                )
                rows.append(test_row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def base_selection(package, outputs_no_strong: pd.DataFrame, *, base_name: str, split: str) -> pd.Series:
    if base_name == "tool_probe_profile_v4_no_strong":
        return normalize_selection(package.profile_v4_selection_for_split(outputs_no_strong, split=split, exclude_models={STRONG_MODEL_ID}))
    if base_name == "observable_local_state_v5_no_strong":
        return normalize_selection(package.observable_local_state_selection(outputs_no_strong, split=split))
    raise ValueError(f"Unknown base_name: {base_name}")


def fit_gain_regressor(
    package,
    sandbox,
    outputs: pd.DataFrame,
    outputs_no_strong: pd.DataFrame,
    *,
    base_selection_val: pd.Series,
    feature_view: str,
    alpha: float,
    max_features: int,
) -> dict[str, Any]:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    val_queries = query_info[query_info["split"].eq("val")].copy()
    val_ids = val_queries.index.astype(str).tolist()
    by_query = outputs_no_strong.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs_no_strong)
    texts = [
        sandbox.feature_text(str(query_id), val_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in val_ids
    ]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    val_x = vectorizer.fit_transform(texts)
    y = strong_gain_targets(outputs, base_selection_val).loc[val_ids].to_numpy(dtype=float)
    model = Ridge(alpha=float(alpha), solver="lsqr")
    model.fit(val_x, y)
    val_pred = np.asarray(model.predict(val_x), dtype=float)
    return {
        "feature_view": feature_view,
        "vectorizer": vectorizer,
        "model": model,
        "val_pred": val_pred,
    }


def apply_gain_gate(
    package,
    sandbox,
    outputs_no_strong: pd.DataFrame,
    base_selection_for_split: pd.Series,
    *,
    bundle: dict[str, Any],
    split: str,
    threshold: float,
) -> pd.Series:
    query_info = outputs_no_strong.drop_duplicates("query_id").set_index("query_id")
    target_queries = query_info[query_info["split"].eq(split)].copy()
    target_ids = target_queries.index.astype(str).tolist()
    by_query = outputs_no_strong.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs_no_strong)
    texts = [
        sandbox.feature_text(str(query_id), target_queries.loc[query_id], by_query, local_models, str(bundle["feature_view"]))
        for query_id in target_ids
    ]
    pred = np.asarray(bundle["model"].predict(bundle["vectorizer"].transform(texts)), dtype=float)
    selected = base_selection_for_split.copy()
    for row_index, query_id in enumerate(target_ids):
        if pred[row_index] > float(threshold):
            selected.loc[query_id] = STRONG_MODEL_ID
    return selected


def strong_gain_targets(outputs: pd.DataFrame, base_selection_val: pd.Series) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    gains: dict[str, float] = {}
    for query_id, model_id in base_selection_val.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) not in by_query.index or (query_id, str(model_id)) not in by_query.index:
            continue
        strong = by_query.loc[(query_id, STRONG_MODEL_ID)]
        base = by_query.loc[(query_id, str(model_id))]
        gains[query_id] = float(strong["utility"]) - float(base["utility"])
    return pd.Series(gains)


def oracle_between_base_and_strong(outputs: pd.DataFrame, base_selection_for_split: pd.Series) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    selected = base_selection_for_split.copy()
    for query_id, model_id in base_selection_for_split.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) not in by_query.index or (query_id, str(model_id)) not in by_query.index:
            continue
        strong = by_query.loc[(query_id, STRONG_MODEL_ID)]
        base = by_query.loc[(query_id, str(model_id))]
        if float(strong["utility"]) > float(base["utility"]):
            selected.loc[query_id] = STRONG_MODEL_ID
    return selected


def candidate_thresholds(values: np.ndarray) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return [0.0]
    qs = np.quantile(finite, np.linspace(0.0, 0.95, 20)).tolist()
    return sorted(set(float(value) for value in [-1.0, 0.0, 0.01, 0.03, 0.05, *qs]))


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    lambda_cost: float,
    method: str,
    family: str,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    return row


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for family, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[group["split"].eq("test") & group["method"].eq(best["method"])]
        if not test.empty:
            rows.append(test.iloc[0])
    return pd.DataFrame(rows)


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#7c6bb0")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Cached Gemini-Strong Gain Prediction")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cached_strong_gain_regressor_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, outputs_path: Path, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    lines = [
        "# Cached Strong-Gain Regressor Gate",
        "",
        f"Source outputs: `{outputs_path}`.",
        "",
        "This run makes no provider API calls. It trains on validation rows where cached Gemini strong-solve exists, then applies the learned strong-gain threshold to held-out test rows.",
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
        "- This is an exploratory action-value predictor for deciding when strong test-time compute is worth cost.",
        "- It is not a final deployable benchmark protocol because strong-solve labels are available only on validation/test in the current cache; use it to decide whether better gain targets are promising.",
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
