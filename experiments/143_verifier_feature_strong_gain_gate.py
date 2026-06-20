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
    parser = argparse.ArgumentParser(description="Use local verifier outputs as features for a train-supervised strong-gain gate.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_train_supervised_strong_gain_gate/model_outputs_with_gemini_strong_all_splits.parquet"),
    )
    parser.add_argument(
        "--verifier-table",
        type=Path,
        default=Path("results/controlled/broad100_qwen32_answer_verifier_strong_gate/table_vllm_answer_verifier_probe.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_verifier_feature_strong_gain_gate"),
    )
    parser.add_argument("--base-method", default="observable_local_state_v5_no_strong")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    train_gate = load_module("experiments/141_train_supervised_strong_gain_gate.py", "train_strong_gain_gate")
    outputs = load_outputs(args.outputs)
    verifier = load_verifier(args.verifier_table)
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()
    base = {
        split: train_gate.base_selection(package, outputs_no_strong, base_name=str(args.base_method), split=split)
        for split in ["train", "val", "test"]
    }
    table = run_verifier_feature_gates(
        package,
        train_gate,
        outputs,
        outputs_no_strong,
        verifier,
        base,
        base_method=str(args.base_method),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = train_gate.validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_verifier_feature_strong_gain_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_verifier_feature_strong_gain_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "VERIFIER_FEATURE_STRONG_GAIN_MEMO.md", args, verifier, table, selected)
    print(f"Wrote verifier-feature strong-gain gate results to {args.output_dir}")


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
    return outputs


def load_verifier(path: Path) -> pd.DataFrame:
    verifier = pd.read_csv(path).copy()
    verifier["query_id"] = verifier["query_id"].astype(str)
    verifier["verdict"] = verifier["verdict"].fillna("unknown").astype(str)
    verifier["confidence"] = pd.to_numeric(verifier["confidence"], errors="coerce").fillna(0.5)
    verifier["escalate_score"] = pd.to_numeric(verifier["escalate_score"], errors="coerce").fillna(0.5)
    verifier["reason"] = verifier["reason"].fillna("").astype(str)
    return verifier.drop_duplicates("query_id").set_index("query_id")


def run_verifier_feature_gates(
    package,
    train_gate,
    outputs: pd.DataFrame,
    outputs_no_strong: pd.DataFrame,
    verifier: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    base_method: str,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            train_gate.evaluate_selection(
                package,
                outputs,
                base[split],
                split=split,
                lambda_cost=lambda_cost,
                method=f"{base_method}_base",
                family="base",
            )
        )
        rows.append(
            train_gate.evaluate_selection(
                package,
                outputs,
                train_gate.oracle_between_base_and_strong(outputs, base[split]),
                split=split,
                lambda_cost=lambda_cost,
                method=f"{base_method}_oracle_between_base_and_strong",
                family="diagnostic_oracle",
            )
        )

    rows.extend(run_prior_gates(package, train_gate, outputs, verifier, base, base_method=base_method, lambda_cost=lambda_cost))
    rows.extend(
        run_text_gates(
            package,
            train_gate,
            outputs,
            outputs_no_strong,
            verifier,
            base,
            base_method=base_method,
            lambda_cost=lambda_cost,
            max_features=max_features,
        )
    )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def run_prior_gates(package, train_gate, outputs: pd.DataFrame, verifier: pd.DataFrame, base: dict[str, pd.Series], *, base_method: str, lambda_cost: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_gain = train_gate.strong_gain_targets(outputs, base["train"]).rename("gain")
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    train_meta = query_info.loc[train_gain.index, ["benchmark", "domain", "metric"]].copy()
    train_meta["base_model"] = base["train"].loc[train_gain.index].astype(str)
    train_meta = train_meta.join(verifier[["verdict", "confidence", "escalate_score"]], how="left")
    train_meta["verdict"] = train_meta["verdict"].fillna("unknown")
    train_meta["confidence_bin"] = confidence_bin(train_meta["confidence"])
    train_meta["gain"] = train_gain.astype(float)
    for keys in [["benchmark", "verdict"], ["benchmark", "base_model", "verdict"], ["benchmark", "verdict", "confidence_bin"]]:
        pred = {
            split: prior_predictions(train_meta, query_info, verifier, base[split], keys=keys)
            for split in ["val", "test"]
        }
        best = select_best_threshold(train_gate, package, outputs, base["val"], pred["val"], split="val", lambda_cost=lambda_cost)
        method = f"{base_method}_verifier_prior_gain_{'_'.join(keys)}_thr{best['threshold']:.4f}"
        best.update({"method": method, "family": "verifier_prior_gain_gate", "base_method": base_method, "feature_view": "+".join(keys)})
        rows.append(best)
        test_selected = train_gate.apply_gain_gate(base["test"], pred["test"], threshold=float(best["threshold"]))
        test_row = train_gate.evaluate_selection(package, outputs, test_selected, split="test", lambda_cost=lambda_cost, method=method, family="verifier_prior_gain_gate")
        test_row.update({"base_method": base_method, "feature_view": "+".join(keys), "threshold": float(best["threshold"])})
        rows.append(test_row)
    return rows


def prior_predictions(
    train_meta: pd.DataFrame,
    query_info: pd.DataFrame,
    verifier: pd.DataFrame,
    base_for_split: pd.Series,
    *,
    keys: list[str],
) -> pd.Series:
    table = train_meta.groupby(keys)["gain"].mean()
    global_mean = float(train_meta["gain"].mean())
    preds: dict[str, float] = {}
    for query_id, base_model in base_for_split.items():
        qid = str(query_id)
        row = query_info.loc[qid]
        probe = verifier.loc[qid] if qid in verifier.index else pd.Series(dtype=object)
        lookup = []
        for key in keys:
            if key == "base_model":
                lookup.append(str(base_model))
            elif key == "verdict":
                lookup.append(str(probe.get("verdict", "unknown")))
            elif key == "confidence_bin":
                lookup.append(str(confidence_bin(pd.Series([probe.get("confidence", 0.5)])).iloc[0]))
            else:
                lookup.append(str(row.get(key, "")))
        preds[qid] = float(table.get(tuple(lookup) if len(lookup) > 1 else lookup[0], global_mean))
    return pd.Series(preds)


def run_text_gates(
    package,
    train_gate,
    outputs: pd.DataFrame,
    outputs_no_strong: pd.DataFrame,
    verifier: pd.DataFrame,
    base: dict[str, pd.Series],
    *,
    base_method: str,
    lambda_cost: float,
    max_features: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_gain = train_gate.strong_gain_targets(outputs, base["train"])
    y_train = train_gain.to_numpy(dtype=float)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    x_train = vectorizer.fit_transform(verifier_texts(outputs_no_strong, verifier, base["train"].loc[train_gain.index], train_gain.index.tolist()))
    x_val = vectorizer.transform(verifier_texts(outputs_no_strong, verifier, base["val"], base["val"].index.astype(str).tolist()))
    x_test = vectorizer.transform(verifier_texts(outputs_no_strong, verifier, base["test"], base["test"].index.astype(str).tolist()))

    for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
        model = Ridge(alpha=float(alpha), solver="lsqr")
        model.fit(x_train, y_train)
        val_pred = pd.Series(np.asarray(model.predict(x_val), dtype=float), index=base["val"].index.astype(str))
        test_pred = pd.Series(np.asarray(model.predict(x_test), dtype=float), index=base["test"].index.astype(str))
        rows.extend(
            val_selected_rows(
                train_gate,
                package,
                outputs,
                base,
                val_pred,
                test_pred,
                method_prefix=f"{base_method}_verifier_text_ridge_alpha{alpha:g}",
                family="verifier_text_ridge_gain_gate",
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
                val_selected_rows(
                    train_gate,
                    package,
                    outputs,
                    base,
                    val_pred,
                    test_pred,
                    method_prefix=f"{base_method}_verifier_text_logistic_C{c_value:g}",
                    family="verifier_text_logistic_gain_gate",
                    lambda_cost=lambda_cost,
                )
            )
    return rows


def verifier_texts(outputs_no_strong: pd.DataFrame, verifier: pd.DataFrame, base_for_split: pd.Series, query_ids: list[str]) -> list[str]:
    query_info = outputs_no_strong.drop_duplicates("query_id").set_index("query_id")
    texts = []
    for query_id in query_ids:
        qid = str(query_id)
        row = query_info.loc[qid]
        probe = verifier.loc[qid] if qid in verifier.index else pd.Series(dtype=object)
        confidence = float(probe.get("confidence", 0.5) or 0.5)
        score = float(probe.get("escalate_score", 0.5) or 0.5)
        parts = [
            str(row.get("query_text", "")),
            f"benchmark_{row.get('benchmark', '')}",
            f"domain_{row.get('domain', '')}",
            f"metric_{row.get('metric', '')}",
            f"base_model_{base_for_split.loc[qid]}",
            f"verdict_{probe.get('verdict', 'unknown')}",
            f"confidence_bin_{confidence_bucket(confidence)}",
            f"escalate_score_bin_{confidence_bucket(score)}",
            f"reason_{probe.get('reason', '')}",
        ]
        texts.append(" ".join(parts))
    return texts


def val_selected_rows(train_gate, package, outputs: pd.DataFrame, base: dict[str, pd.Series], val_pred: pd.Series, test_pred: pd.Series, *, method_prefix: str, family: str, lambda_cost: float) -> list[dict[str, Any]]:
    best = select_best_threshold(train_gate, package, outputs, base["val"], val_pred, split="val", lambda_cost=lambda_cost)
    method = f"{method_prefix}_thr{best['threshold']:.4f}"
    best.update({"method": method, "family": family})
    test_selected = train_gate.apply_gain_gate(base["test"], test_pred, threshold=float(best["threshold"]))
    test_row = train_gate.evaluate_selection(package, outputs, test_selected, split="test", lambda_cost=lambda_cost, method=method, family=family)
    test_row["threshold"] = float(best["threshold"])
    return [best, test_row]


def select_best_threshold(train_gate, package, outputs: pd.DataFrame, base_for_val: pd.Series, val_pred: pd.Series, *, split: str, lambda_cost: float) -> dict[str, Any]:
    candidates = []
    for threshold in train_gate.candidate_thresholds(val_pred.to_numpy(dtype=float)):
        selected = train_gate.apply_gain_gate(base_for_val, val_pred, threshold=threshold)
        row = train_gate.evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost, method="candidate", family="candidate")
        row["threshold"] = float(threshold)
        candidates.append(row)
    return sorted(candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]


def confidence_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "missing"
    return str(int(min(max(float(value), 0.0), 0.999) * 10))


def confidence_bin(values: pd.Series) -> pd.Series:
    return values.fillna(0.5).map(confidence_bucket).astype(str)


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(14)
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#74628f")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Verifier-Feature Strong-Gain Gates")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_verifier_feature_strong_gain_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, verifier: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    lines = [
        "# Verifier-Feature Strong-Gain Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Verifier table: `{args.verifier_table}`.",
        f"Base method: `{args.base_method}`.",
        "This run makes no model or provider calls; it reuses cached Qwen32 answer-verifier outputs.",
        f"Verifier rows: `{len(verifier)}`.",
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
        "- This tests whether the local verifier output is useful as a supervised signal, even though the direct verdict gate was poor.",
        "- Thresholds are selected on validation; held-out test rows are reported separately.",
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
