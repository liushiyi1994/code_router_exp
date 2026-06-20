from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train high-precision strong/self need classifiers over cached probe features.")
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
        default=Path("results/controlled/broad100_strong_need_classifier_gate"),
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
    fast_gate = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_action_gate")
    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    table = run_classifier_gates(
        package,
        self_gate,
        fast_gate,
        outputs,
        probe,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_strong_need_classifier_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_strong_need_classifier_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "STRONG_NEED_CLASSIFIER_GATE_MEMO.md", args, table, selected)
    print(f"Wrote strong-need classifier gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_classifier_gates(
    package,
    self_gate,
    fast_gate,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()
    base_specs = {
        "observable_local_state_v5_no_strong": lambda split: fast_gate.fast_observable_local_state_selection(
            package, outputs_no_strong_self, split=split
        ),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}
        ),
    }
    rows: list[dict[str, Any]] = []
    for base_name, builder in base_specs.items():
        print(f"running strong/self classifiers for {base_name}")
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
                    self_gate.oracle_between_actions(outputs, base[split], [self_model_id, STRONG_MODEL_ID]),
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                    self_model_id=self_model_id,
                )
            )

        train = self_gate.build_feature_frame(outputs, probe, base["train"], split="train", self_model_id=self_model_id)
        val = self_gate.build_feature_frame(outputs, probe, base["val"], split="val", self_model_id=self_model_id)
        test = self_gate.build_feature_frame(outputs, probe, base["test"], split="test", self_model_id=self_model_id)
        if train.empty or val.empty or test.empty:
            continue
        for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
            x_train, x_val, x_test = self_gate.featurize(
                train, val, test, feature_view=feature_view, max_features=max_features
            )
            for strong_margin in [0.0, 0.05, 0.10]:
                y_strong = (
                    train["utility_strong"].to_numpy(dtype=float)
                    > np.maximum(train["utility_base"].to_numpy(dtype=float), train["utility_self"].to_numpy(dtype=float))
                    + float(strong_margin)
                ).astype(int)
                for self_margin in [0.0, 0.05]:
                    y_self = (
                        train["utility_self"].to_numpy(dtype=float)
                        > train["utility_base"].to_numpy(dtype=float) + float(self_margin)
                    ).astype(int)
                    for c_value in [0.1, 1.0, 10.0]:
                        p_strong_val, p_strong_test = fit_predict_binary(x_train, y_strong, x_val, x_test, c_value=c_value)
                        p_self_val, p_self_test = fit_predict_binary(x_train, y_self, x_val, x_test, c_value=c_value)
                        rows.extend(
                            threshold_grid_rows(
                                package,
                                outputs,
                                base,
                                val,
                                test,
                                p_strong_val,
                                p_strong_test,
                                p_self_val,
                                p_self_test,
                                base_name=base_name,
                                feature_view=feature_view,
                                c_value=c_value,
                                strong_margin=strong_margin,
                                self_margin=self_margin,
                                lambda_cost=lambda_cost,
                                self_model_id=self_model_id,
                            )
                        )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def fit_predict_binary(x_train, y_train: np.ndarray, x_val, x_test, *, c_value: float) -> tuple[np.ndarray, np.ndarray]:
    unique = sorted(set(int(x) for x in y_train.tolist()))
    if len(unique) < 2:
        constant = float(unique[0]) if unique else 0.0
        return np.full(x_val.shape[0], constant, dtype=float), np.full(x_test.shape[0], constant, dtype=float)
    model = LogisticRegression(C=float(c_value), class_weight="balanced", solver="liblinear", max_iter=3000)
    model.fit(x_train, y_train)
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return (
        np.asarray(model.predict_proba(x_val)[:, positive_index], dtype=float),
        np.asarray(model.predict_proba(x_test)[:, positive_index], dtype=float),
    )


def threshold_grid_rows(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    p_strong_val: np.ndarray,
    p_strong_test: np.ndarray,
    p_self_val: np.ndarray,
    p_self_test: np.ndarray,
    *,
    base_name: str,
    feature_view: str,
    c_value: float,
    strong_margin: float,
    self_margin: float,
    lambda_cost: float,
    self_model_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for action_order in ["strong_first", "self_first"]:
        for strong_threshold in [0.20, 0.35, 0.50, 0.65, 0.80]:
            for self_threshold in [0.20, 0.50, 0.80]:
                method = (
                    f"{base_name}_need_logreg_{feature_view}_C{c_value:g}"
                    f"_sm{strong_margin:g}_xm{self_margin:g}"
                    f"_st{strong_threshold:g}_xt{self_threshold:g}_{action_order}"
                )
                val_selected = apply_probability_policy(
                    base["val"],
                    val_frame["query_id"].astype(str).tolist(),
                    p_strong_val,
                    p_self_val,
                    strong_threshold=strong_threshold,
                    self_threshold=self_threshold,
                    action_order=action_order,
                    self_model_id=self_model_id,
                )
                test_selected = apply_probability_policy(
                    base["test"],
                    test_frame["query_id"].astype(str).tolist(),
                    p_strong_test,
                    p_self_test,
                    strong_threshold=strong_threshold,
                    self_threshold=self_threshold,
                    action_order=action_order,
                    self_model_id=self_model_id,
                )
                for split, selected in [("val", val_selected), ("test", test_selected)]:
                    row = evaluate_selection(
                        package,
                        outputs,
                        selected,
                        split=split,
                        method=method,
                        family="strong_self_need_logreg",
                        lambda_cost=lambda_cost,
                        self_model_id=self_model_id,
                    )
                    row.update(
                        {
                            "base_method": base_name,
                            "feature_view": feature_view,
                            "C": float(c_value),
                            "strong_margin": float(strong_margin),
                            "self_margin": float(self_margin),
                            "strong_threshold": float(strong_threshold),
                            "self_threshold": float(self_threshold),
                            "action_order": action_order,
                        }
                    )
                    rows.append(row)
    return rows


def apply_probability_policy(
    base: pd.Series,
    query_ids: list[str],
    p_strong: np.ndarray,
    p_self: np.ndarray,
    *,
    strong_threshold: float,
    self_threshold: float,
    action_order: str,
    self_model_id: str,
) -> pd.Series:
    selected = normalize_selection(base)
    for query_id, strong_prob, self_prob in zip(query_ids, p_strong, p_self):
        query_id = str(query_id)
        use_strong = float(strong_prob) >= float(strong_threshold)
        use_self = float(self_prob) >= float(self_threshold)
        if action_order == "strong_first":
            if use_strong:
                selected.loc[query_id] = STRONG_MODEL_ID
            elif use_self:
                selected.loc[query_id] = self_model_id
        elif action_order == "self_first":
            if use_self:
                selected.loc[query_id] = self_model_id
            elif use_strong:
                selected.loc[query_id] = STRONG_MODEL_ID
        else:
            raise ValueError(action_order)
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
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
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
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#6f7d4f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Strong/Self Need Classifier Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_strong_need_classifier_gate.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "cost_oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
        "family",
    ]
    lines = [
        "# Strong/Self Need Classifier Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls; it uses cached self-consistency probe rows.",
        "It trains binary train-only classifiers for strong-needed and self-needed actions, then validation-selects probability thresholds.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[cols + [c for c in selected.columns if c in {"base_method", "feature_view", "C", "strong_margin", "self_margin", "strong_threshold", "self_threshold", "action_order", "selection_rule"}]], max_rows=24),
        "```",
        "",
        "## Best Held-Out Test Rows",
        "",
        "```csv",
        compact_csv(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)[cols].head(16)),
        "```",
        "",
        "## Interpretation",
        "",
        "- This tests whether a high-precision frontier/strong-need classifier works better than scalar utility regression.",
        "- A positive result must beat the global self-consistency feature gate while preserving a reasonable frontier-call rate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
