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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate local action-compare probe outputs as routing features.")
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
        "--action-probe-table",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_action_compare_probe_qwen4b_conservative/"
            "table_vllm_action_compare_probe.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_action_compare_feature_gate"),
    )
    parser.add_argument("--base-method", default="tool_probe_profile_v4_no_strong")
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
    action_probe = load_action_probe(args.action_probe_table)
    table = run_action_feature_gate(
        package,
        self_gate,
        fast_gate,
        outputs,
        probe,
        action_probe,
        base_method=str(args.base_method),
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = self_gate.validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_action_compare_feature_gate_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_action_compare_feature_gate_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "ACTION_COMPARE_FEATURE_GATE_MEMO.md", args, action_probe, table, selected)
    print(f"Wrote action-compare feature-gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_action_probe(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path).copy()
    table["query_id"] = table["query_id"].astype(str)
    table["split"] = table["split"].astype(str)
    table["action"] = table["action"].fillna("base").astype(str)
    for column in ["confidence", "latency_s"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    return table.drop_duplicates(["query_id", "split"], keep="last")


def run_action_feature_gate(
    package,
    self_gate,
    fast_gate,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    action_probe: pd.DataFrame,
    *,
    base_method: str,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()
    base = {
        split: base_selection(package, fast_gate, outputs_no_strong_self, base_method=base_method, split=split)
        for split in ["train", "val", "test"]
    }
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(
            self_gate.evaluate_selection(
                package,
                outputs,
                base[split],
                split=split,
                method=base_method,
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
                method=f"{base_method}_oracle_between_base_self_strong",
                family="diagnostic_oracle",
                lambda_cost=lambda_cost,
                self_model_id=self_model_id,
            )
        )

    frames = {
        split: add_action_features(
            self_gate.build_feature_frame(outputs, probe, base[split], split=split, self_model_id=self_model_id),
            action_probe[action_probe["split"].eq(split)],
        )
        for split in ["train", "val", "test"]
    }
    if frames["train"].empty or frames["val"].empty or frames["test"].empty:
        return pd.DataFrame(rows)

    for feature_view in ["action_numeric", "action_numeric_text"]:
        x_train, x_val, x_test = featurize_action_frames(
            frames["train"],
            frames["val"],
            frames["test"],
            feature_view=feature_view,
            max_features=max_features,
        )
        for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            scores = fit_action_utility_scores(frames, x_train, x_val, x_test, alpha=float(alpha))
            for margin in [0.0, 0.01, 0.02, 0.05, 0.10]:
                method = f"{base_method}_action_compare_ridge_{feature_view}_alpha{alpha:g}_margin{margin:g}"
                rows.extend(
                    eval_scores(
                        package,
                        self_gate,
                        outputs,
                        base,
                        scores,
                        method=method,
                        family="action_compare_feature_ridge",
                        self_model_id=self_model_id,
                        lambda_cost=lambda_cost,
                        feature_view=feature_view,
                        alpha=alpha,
                        margin=margin,
                    )
                )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def base_selection(package, fast_gate, outputs: pd.DataFrame, *, base_method: str, split: str) -> pd.Series:
    if base_method == "tool_probe_profile_v4_no_strong":
        return normalize_selection(package.profile_v4_selection_for_split(outputs, split=split, exclude_models={STRONG_MODEL_ID}))
    if base_method == "observable_local_state_v5_no_strong":
        return normalize_selection(fast_gate.fast_observable_local_state_selection(package, outputs, split=split))
    raise ValueError(f"Unknown base method: {base_method}")


def add_action_features(frame: pd.DataFrame, action_probe: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or action_probe.empty:
        return pd.DataFrame()
    action = action_probe.set_index("query_id")
    merged = frame[frame["query_id"].astype(str).isin(set(action.index.astype(str)))].copy()
    if merged.empty:
        return merged
    merged = merged.join(action[["action", "confidence", "latency_s", "reason"]], on="query_id", rsuffix="_action")
    latency_column = "latency_s_action" if "latency_s_action" in merged.columns else "latency_s"
    merged["action_compare_action"] = merged["action"].fillna("base").astype(str)
    merged["action_compare_confidence"] = pd.to_numeric(merged["confidence"], errors="coerce").fillna(0.0)
    merged["action_compare_latency_s"] = pd.to_numeric(merged[latency_column], errors="coerce").fillna(0.0)
    merged["action_compare_reason"] = merged["reason"].fillna("").astype(str)
    for action_name in ["base", "self", "strong"]:
        merged[f"action_compare_is_{action_name}"] = merged["action_compare_action"].eq(action_name)
    merged["feature_text"] = (
        merged["feature_text"].fillna("").astype(str)
        + " action_compare="
        + merged["action_compare_action"].astype(str)
        + " action_reason="
        + merged["action_compare_reason"].astype(str)
    )
    return merged.drop(columns=["action", "confidence", "latency_s_action", "reason"], errors="ignore")


def featurize_action_frames(
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
        "action_compare_confidence",
        "action_compare_latency_s",
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
        "action_compare_action",
        "action_compare_is_base",
        "action_compare_is_self",
        "action_compare_is_strong",
    ]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(frame_to_dicts(train, numeric_columns, categorical_columns))
    x_val = vectorizer.transform(frame_to_dicts(val, numeric_columns, categorical_columns))
    x_test = vectorizer.transform(frame_to_dicts(test, numeric_columns, categorical_columns))
    if feature_view == "action_numeric":
        return x_train, x_val, x_test
    if feature_view != "action_numeric_text":
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
            features[column] = float(row.get(column, 0.0) or 0.0)
        for column in categorical_columns:
            features[f"{column}={row.get(column, '')}"] = 1.0
        rows.append(features)
    return rows


def fit_action_utility_scores(frames: dict[str, pd.DataFrame], x_train, x_val, x_test, *, alpha: float) -> dict[str, dict[str, pd.Series]]:
    out: dict[str, dict[str, pd.Series]] = {}
    for action_col, action_name in [
        ("utility_base", "base"),
        ("utility_self", "self"),
        ("utility_strong", "strong"),
    ]:
        model = Ridge(alpha=float(alpha), solver="lsqr")
        model.fit(x_train, frames["train"][action_col].to_numpy(dtype=float))
        out[action_name] = {
            "val": pd.Series(
                np.asarray(model.predict(x_val), dtype=float),
                index=frames["val"]["query_id"].astype(str).tolist(),
            ),
            "test": pd.Series(
                np.asarray(model.predict(x_test), dtype=float),
                index=frames["test"]["query_id"].astype(str).tolist(),
            ),
        }
    return out


def eval_scores(
    package,
    self_gate,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    scores: dict[str, dict[str, pd.Series]],
    *,
    method: str,
    family: str,
    self_model_id: str,
    lambda_cost: float,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        score_frame = pd.DataFrame(
            {
                "base": scores["base"][split],
                "self": scores["self"][split],
                "strong": scores["strong"][split],
            }
        )
        target_ids = score_frame.index.tolist()
        selected = scores_to_selection(base[split], score_frame.loc[target_ids], self_model_id=self_model_id, margin=float(extra["margin"]))
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


def scores_to_selection(base: pd.Series, scores: pd.DataFrame, *, self_model_id: str, margin: float) -> pd.Series:
    selected = normalize_selection(base)
    for query_id, row in scores.iterrows():
        values = row.astype(float)
        base_score = float(values["base"])
        best_action = str(values.idxmax())
        if float(values[best_action]) < base_score + float(margin):
            continue
        if best_action == "self":
            selected.loc[str(query_id)] = self_model_id
        elif best_action == "strong":
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    out = out.astype(str)
    return out


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    if table.empty:
        return
    plot = table[table["split"].eq("test")].sort_values("mean_utility", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(plot["method"], plot["mean_utility"])
    ax.invert_yaxis()
    ax.set_xlabel("Held-out mean utility")
    ax.set_title("Action-Compare Feature Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_action_compare_feature_gate_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, action_probe: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    summary = (
        action_probe.groupby(["split", "action"])
        .size()
        .reset_index(name="n")
        .to_csv(index=False)
        .strip()
    )
    selected_csv = selected[
        [
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
            "selection_rule",
        ]
    ].to_csv(index=False, float_format="%.4f")
    path.write_text(
        "\n".join(
            [
                "# Action-Compare Feature Gate",
                "",
                f"Outputs: `{args.outputs}`.",
                f"Self-consistency probe: `{args.probe_table}`.",
                f"Action probe: `{args.action_probe_table}`.",
                "This run uses cached local vLLM action-probe rows only; no provider API calls are made.",
                "",
                "## Action Probe Summary",
                "",
                "```csv",
                summary,
                "```",
                "",
                "## Validation-Selected And Diagnostics",
                "",
                "```csv",
                selected_csv.strip(),
                "```",
                "",
                "## Interpretation",
                "",
                "- This is a train-only calibration of local action-probe features.",
                "- Success requires held-out utility to improve over the base policy and approach the action-set oracle.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
