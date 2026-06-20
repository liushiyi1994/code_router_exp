from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a train/val/test strong-gain gate over cached Gemini strong rows.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--strong-csvs",
        nargs="+",
        type=Path,
        default=[
            Path("results/controlled/broad100_gemini_strong_solver_train/table_broad_gemini_strong_outputs.csv"),
            Path("results/controlled/broad100_gemini_strong_solver/table_broad_gemini_strong_outputs.csv"),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_train_supervised_strong_gain_gate"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    sandbox = load_module("experiments/130_probe_signal_cached_sandbox.py", "probe_signal_cached")
    strong_runner = load_module("experiments/129_broad_gemini_strong_solver.py", "gemini_strong_runner")

    base_outputs = package.load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    strong = load_strong_rows(args.strong_csvs)
    strong.to_csv(args.output_dir / "table_gemini_strong_outputs_all_splits.csv", index=False)

    outputs = strong_runner.append_strong_rows(base_outputs, strong, lambda_cost=float(args.lambda_cost))
    outputs.to_parquet(args.output_dir / "model_outputs_with_gemini_strong_all_splits.parquet", index=False)
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()

    table = run_train_supervised_gates(
        package,
        sandbox,
        outputs,
        outputs_no_strong,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_train_supervised_strong_gain_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_train_supervised_strong_gain_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "TRAIN_SUPERVISED_STRONG_GAIN_MEMO.md", args, strong, table, selected)
    print(f"Wrote train-supervised strong-gain gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_strong_rows(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    strong = pd.concat(frames, ignore_index=True)
    strong["query_id"] = strong["query_id"].astype(str)
    strong = strong.sort_values(["split", "query_id"]).drop_duplicates("query_id", keep="last")
    return strong


def run_train_supervised_gates(
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
        base = {split: base_selection(package, outputs_no_strong, base_name=base_name, split=split) for split in ["train", "val", "test"]}
        for split in ["val", "test"]:
            rows.append(evaluate_selection(package, outputs, base[split], split=split, lambda_cost=lambda_cost, method=base_name, family="base"))

        diagnostic = {split: oracle_between_base_and_strong(outputs, base[split]) for split in ["val", "test"]}
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

        rows.extend(run_prior_gates(package, outputs, base, base_name=base_name, lambda_cost=lambda_cost))
        rows.extend(
            run_text_model_gates(
                package,
                sandbox,
                outputs,
                outputs_no_strong,
                base,
                base_name=base_name,
                lambda_cost=lambda_cost,
                max_features=max_features,
            )
        )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def base_selection(package, outputs_no_strong: pd.DataFrame, *, base_name: str, split: str) -> pd.Series:
    if base_name == "tool_probe_profile_v4_no_strong":
        return normalize_selection(package.profile_v4_selection_for_split(outputs_no_strong, split=split, exclude_models={STRONG_MODEL_ID}))
    if base_name == "observable_local_state_v5_no_strong":
        return normalize_selection(package.observable_local_state_selection(outputs_no_strong, split=split))
    raise ValueError(f"Unknown base_name: {base_name}")


def run_prior_gates(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    base_name: str,
    lambda_cost: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_gain = strong_gain_targets(outputs, base["train"]).rename("gain")
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    train_meta = query_info.loc[train_gain.index, ["benchmark", "domain", "metric"]].copy()
    train_meta["base_model"] = base["train"].loc[train_gain.index].astype(str)
    train_meta["gain"] = train_gain.astype(float)

    for keys in [["benchmark"], ["benchmark", "metric"], ["benchmark", "base_model"], ["domain", "metric"]]:
        pred = {
            split: prior_predictions(train_meta, query_info, base[split], keys=keys)
            for split in ["val", "test"]
        }
        best = select_best_threshold(package, outputs, base["val"], pred["val"], split="val", lambda_cost=lambda_cost)
        method = f"{base_name}_train_prior_gain_{'_'.join(keys)}_thr{best['threshold']:.4f}"
        best.update({"method": method, "family": "train_prior_gain_gate", "base_method": base_name, "feature_view": "+".join(keys)})
        rows.append(best)
        test_selected = apply_gain_gate(base["test"], pred["test"], threshold=float(best["threshold"]))
        test_row = evaluate_selection(package, outputs, test_selected, split="test", lambda_cost=lambda_cost, method=method, family="train_prior_gain_gate")
        test_row.update({"base_method": base_name, "feature_view": "+".join(keys), "threshold": float(best["threshold"])})
        rows.append(test_row)
    return rows


def prior_predictions(
    train_meta: pd.DataFrame,
    query_info: pd.DataFrame,
    base_for_split: pd.Series,
    *,
    keys: list[str],
) -> pd.Series:
    table = train_meta.groupby(keys)["gain"].mean()
    global_mean = float(train_meta["gain"].mean())
    preds: dict[str, float] = {}
    for query_id, base_model in base_for_split.items():
        row = query_info.loc[str(query_id)]
        lookup = []
        for key in keys:
            lookup.append(str(base_model) if key == "base_model" else row.get(key, ""))
        value = table.get(tuple(lookup) if len(lookup) > 1 else lookup[0], global_mean)
        preds[str(query_id)] = float(value)
    return pd.Series(preds)


def run_text_model_gates(
    package,
    sandbox,
    outputs: pd.DataFrame,
    outputs_no_strong: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    base_name: str,
    lambda_cost: float,
    max_features: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_gain = strong_gain_targets(outputs, base["train"])
    y_train = train_gain.to_numpy(dtype=float)
    if len(y_train) == 0:
        return rows

    for feature_view in ["query_text", "query_text_with_benchmark", "query_text_with_local_answers"]:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
        x_train = vectorizer.fit_transform(feature_texts(package, sandbox, outputs_no_strong, train_gain.index.tolist(), feature_view))
        x_val = vectorizer.transform(feature_texts(package, sandbox, outputs_no_strong, base["val"].index.astype(str).tolist(), feature_view))
        x_test = vectorizer.transform(feature_texts(package, sandbox, outputs_no_strong, base["test"].index.astype(str).tolist(), feature_view))

        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            model = Ridge(alpha=float(alpha), solver="lsqr")
            model.fit(x_train, y_train)
            val_pred = pd.Series(np.asarray(model.predict(x_val), dtype=float), index=base["val"].index.astype(str))
            test_pred = pd.Series(np.asarray(model.predict(x_test), dtype=float), index=base["test"].index.astype(str))
            rows.extend(
                selected_val_and_test_rows(
                    package,
                    outputs,
                    base,
                    val_pred,
                    test_pred,
                    split_family="train_text_ridge_gain_gate",
                    method_prefix=f"{base_name}_train_ridge_gain_{feature_view}_alpha{alpha:g}",
                    base_name=base_name,
                    feature_view=feature_view,
                    lambda_cost=lambda_cost,
                )
            )

        y_binary = (y_train > 0.0).astype(int)
        if len(set(y_binary.tolist())) > 1:
            for c_value in [0.1, 1.0, 10.0]:
                clf = LogisticRegression(C=float(c_value), class_weight="balanced", max_iter=2000)
                clf.fit(x_train, y_binary)
                val_pred = pd.Series(clf.predict_proba(x_val)[:, 1], index=base["val"].index.astype(str))
                test_pred = pd.Series(clf.predict_proba(x_test)[:, 1], index=base["test"].index.astype(str))
                rows.extend(
                    selected_val_and_test_rows(
                        package,
                        outputs,
                        base,
                        val_pred,
                        test_pred,
                        split_family="train_text_logistic_gain_gate",
                        method_prefix=f"{base_name}_train_logistic_strong_wins_{feature_view}_C{c_value:g}",
                        base_name=base_name,
                        feature_view=feature_view,
                        lambda_cost=lambda_cost,
                    )
                )
    return rows


def selected_val_and_test_rows(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    val_pred: pd.Series,
    test_pred: pd.Series,
    *,
    split_family: str,
    method_prefix: str,
    base_name: str,
    feature_view: str,
    lambda_cost: float,
) -> list[dict[str, Any]]:
    best = select_best_threshold(package, outputs, base["val"], val_pred, split="val", lambda_cost=lambda_cost)
    method = f"{method_prefix}_thr{best['threshold']:.4f}"
    best.update({"method": method, "family": split_family, "base_method": base_name, "feature_view": feature_view})
    test_selected = apply_gain_gate(base["test"], test_pred, threshold=float(best["threshold"]))
    test_row = evaluate_selection(package, outputs, test_selected, split="test", lambda_cost=lambda_cost, method=method, family=split_family)
    test_row.update({"base_method": base_name, "feature_view": feature_view, "threshold": float(best["threshold"])})
    return [best, test_row]


def select_best_threshold(
    package,
    outputs: pd.DataFrame,
    base_for_val: pd.Series,
    val_pred: pd.Series,
    *,
    split: str,
    lambda_cost: float,
) -> dict[str, Any]:
    candidates = []
    for threshold in candidate_thresholds(val_pred.to_numpy(dtype=float)):
        selected = apply_gain_gate(base_for_val, val_pred, threshold=threshold)
        row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost, method="candidate", family="candidate")
        row["threshold"] = float(threshold)
        candidates.append(row)
    return sorted(candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]


def feature_texts(package, sandbox, outputs_no_strong: pd.DataFrame, query_ids: list[str], feature_view: str) -> list[str]:
    query_info = outputs_no_strong.drop_duplicates("query_id").set_index("query_id")
    by_query = outputs_no_strong.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs_no_strong)
    return [
        sandbox.feature_text(str(query_id), query_info.loc[str(query_id)], by_query, local_models, feature_view)
        for query_id in query_ids
    ]


def strong_gain_targets(outputs: pd.DataFrame, base_selection_for_split: pd.Series) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    gains: dict[str, float] = {}
    for query_id, model_id in base_selection_for_split.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) not in by_query.index or (query_id, str(model_id)) not in by_query.index:
            continue
        strong = by_query.loc[(query_id, STRONG_MODEL_ID)]
        base = by_query.loc[(query_id, str(model_id))]
        gains[query_id] = float(strong["utility"]) - float(base["utility"])
    return pd.Series(gains).sort_index()


def apply_gain_gate(base_selection_for_split: pd.Series, pred: pd.Series, *, threshold: float) -> pd.Series:
    selected = base_selection_for_split.copy()
    selected.index = selected.index.astype(str)
    for query_id, value in pred.items():
        if float(value) > float(threshold):
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected


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
    qs = np.quantile(finite, np.linspace(0.0, 0.98, 32)).tolist()
    fixed = [-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5]
    return sorted(set(float(value) for value in [*fixed, *qs]))


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
        if family == "diagnostic_oracle":
            continue
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
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#506b8f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Train-Supervised Gemini-Strong Gain Gates")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_train_supervised_strong_gain_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, strong: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    by_split = strong.groupby("split").agg(rows=("query_id", "size"), mean_quality=("quality_score", "mean"), cost_usd=("cost_total_usd", "sum")).reset_index()
    lines = [
        "# Train-Supervised Strong-Gain Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Strong CSVs: `{', '.join(str(path) for path in args.strong_csvs)}`.",
        "Claude is not used. The only provider rows added here are cached Gemini strong-solve rows.",
        "",
        "## Strong Rows",
        "",
        markdown_table(by_split),
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
        "- This is the first train/val/test-correct strong-gain gate for broad100: strong-gain targets are fit on train and thresholds are chosen on validation.",
        "- It tests whether the previous negative strong-gain result was mainly caused by not having train strong labels.",
        "- The benchmark/domain prior rows are diagnostic but still train-derived; text rows use query text, benchmark tags, and cached local answer features.",
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
