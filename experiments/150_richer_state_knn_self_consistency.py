from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
ACTIONS = ["base", "self", "strong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="kNN router over cached rich self-consistency probe state.")
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
        default=Path("results/controlled/broad100_richer_state_knn_self_consistency"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=20000)
    parser.add_argument("--neighbors", default="1,3,5,10,20,40")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_feature_gate")
    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    neighbors = [int(item) for item in str(args.neighbors).split(",") if str(item).strip()]
    table = run_knn(
        package,
        self_gate,
        outputs,
        probe,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
        neighbors=neighbors,
    )
    selected = self_gate.validation_selected_rows(table)
    summary = family_summary(table)
    table.to_csv(args.output_dir / "table_richer_state_knn_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_richer_state_knn_selected.csv", index=False)
    summary.to_csv(args.output_dir / "table_richer_state_knn_summary.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "RICHER_STATE_KNN_SELF_CONSISTENCY_MEMO.md", args, table, selected, summary)
    print(f"Wrote richer-state kNN results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_knn(
    package,
    self_gate,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
    neighbors: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([STRONG_MODEL_ID, self_model_id])].copy()
    base_specs = {
        "observable_local_state_v5": lambda split: package.observable_local_state_selection(outputs_no_self, split=split),
        "observable_local_state_v5_no_strong": lambda split: package.observable_local_state_selection(outputs_no_strong_self, split=split),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}
        ),
    }
    for base_name, builder in base_specs.items():
        base = {split: self_gate.normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        for split in ["val", "test"]:
            rows.append(
                evaluate_selection(
                    package,
                    self_gate,
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
                    self_gate,
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
        train_model_values, model_ids = model_utility_matrix(outputs, train["query_id"].astype(str).tolist())
        for feature_view in ["query_text", "query_metadata", "query_probe_state", "probe_state_only"]:
            x_train, x_val, x_test = featurize(train, val, test, feature_view=feature_view, max_features=max_features)
            index = fit_index(x_train, max(neighbors))
            action_train_values = train[["utility_base", "utility_self", "utility_strong"]].to_numpy(dtype=float)
            for k in neighbors:
                action_val_scores = neighbor_mean_scores(index, x_val, action_train_values, ACTIONS, k=k)
                action_test_scores = neighbor_mean_scores(index, x_test, action_train_values, ACTIONS, k=k)
                rows.extend(
                    action_val_and_test_rows(
                        package,
                        self_gate,
                        outputs,
                        base,
                        action_val_scores,
                        action_test_scores,
                        method=f"{base_name}_richer_state_action_knn_{feature_view}_k{k}",
                        family="richer_state_action_knn",
                        self_model_id=self_model_id,
                        lambda_cost=lambda_cost,
                        feature_view=feature_view,
                        k=k,
                    )
                )
                model_val_scores = neighbor_mean_scores(index, x_val, train_model_values, model_ids, k=k)
                model_test_scores = neighbor_mean_scores(index, x_test, train_model_values, model_ids, k=k)
                rows.extend(
                    model_val_and_test_rows(
                        package,
                        self_gate,
                        outputs,
                        model_val_scores,
                        model_test_scores,
                        method=f"{base_name}_richer_state_model_knn_{feature_view}_k{k}",
                        family="richer_state_model_knn",
                        self_model_id=self_model_id,
                        lambda_cost=lambda_cost,
                        feature_view=feature_view,
                        k=k,
                    )
                )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def featurize(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_view: str,
    max_features: int,
):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_text = feature_texts(train, feature_view)
    val_text = feature_texts(val, feature_view)
    test_text = feature_texts(test, feature_view)
    return vectorizer.fit_transform(train_text), vectorizer.transform(val_text), vectorizer.transform(test_text)


def feature_texts(frame: pd.DataFrame, feature_view: str) -> list[str]:
    texts: list[str] = []
    for _, row in frame.iterrows():
        query = str(row.get("query_text", ""))
        metadata = " ".join(
            [
                f"benchmark={row.get('benchmark', '')}",
                f"domain={row.get('domain', '')}",
                f"metric={row.get('metric', '')}",
            ]
        )
        state = probe_state_text(row)
        if feature_view == "query_text":
            texts.append(query)
        elif feature_view == "query_metadata":
            texts.append(" ".join([metadata, query]))
        elif feature_view == "query_probe_state":
            texts.append(" ".join([metadata, state, query]))
        elif feature_view == "probe_state_only":
            texts.append(" ".join([metadata, state]))
        else:
            raise ValueError(feature_view)
    return texts


def probe_state_text(row: pd.Series) -> str:
    majority = short_answer(row.get("majority_answer_norm", ""))
    base_answer = short_answer(row.get("base_answer_norm", ""))
    return " ".join(
        [
            f"base_model={row.get('base_model_id', '')}",
            f"base_provider={row.get('base_provider', '')}",
            f"base_local={bool(row.get('base_is_local', False))}",
            f"base_frontier={bool(row.get('base_is_frontier', False))}",
            f"base_self_relation={'same' if bool(row.get('base_equals_self_majority', False)) else 'different'}",
            f"base_answer={base_answer}",
            f"self_answer={majority}",
            f"vote_frac={bucket(float(row.get('vote_frac', 0.0) or 0.0), [0.34, 0.67, 0.99])}",
            f"vote_margin={bucket(float(row.get('vote_margin', 0.0) or 0.0), [0.1, 0.5, 0.9])}",
            f"vote_entropy={bucket(float(row.get('vote_entropy', 0.0) or 0.0), [0.1, 0.8, 1.3])}",
            f"unique_answers={count_bucket(float(row.get('unique_answer_count', 0.0) or 0.0))}",
            f"local_agree={count_bucket(float(row.get('local_agree_with_majority_count', 0.0) or 0.0))}",
            f"all_samples_agree={bool(row.get('all_samples_agree', False))}",
            f"probe_tokens={bucket(float(row.get('probe_output_tokens', 0.0) or 0.0), [64, 128, 256])}",
        ]
    )


def short_answer(value: object, *, limit: int = 80) -> str:
    text = str(value or "").strip().replace("\n", " ")
    return text[:limit]


def bucket(value: float, cuts: list[float]) -> str:
    for index, cut in enumerate(cuts):
        if value <= cut:
            return f"b{index}"
    return f"b{len(cuts)}"


def count_bucket(value: float) -> str:
    if value <= 0:
        return "zero"
    if value <= 1:
        return "one"
    if value <= 2:
        return "two"
    return "many"


def fit_index(x_train, max_k: int) -> NearestNeighbors:
    n_neighbors = min(int(max_k), int(x_train.shape[0]))
    return NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute").fit(x_train)


def neighbor_mean_scores(
    index: NearestNeighbors,
    x_eval,
    train_values: np.ndarray,
    columns: list[str],
    *,
    k: int,
) -> pd.DataFrame:
    effective_k = min(int(k), int(train_values.shape[0]))
    neighbor_idx = index.kneighbors(x_eval, n_neighbors=effective_k, return_distance=False)
    means = np.vstack([train_values[row].mean(axis=0) for row in neighbor_idx])
    return pd.DataFrame(means, columns=columns)


def model_utility_matrix(outputs: pd.DataFrame, query_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    model_ids = sorted(outputs["model_id"].astype(str).unique())
    pivot = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="last")
    matrix = pivot.reindex(query_ids)[model_ids].fillna(-1e6).to_numpy(dtype=float)
    return matrix, model_ids


def action_val_and_test_rows(
    package,
    self_gate,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    *,
    method: str,
    family: str,
    self_model_id: str,
    lambda_cost: float,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, scores in [("val", val_scores), ("test", test_scores)]:
        scores = scores.copy()
        scores.index = base[split].index.astype(str)
        selected = self_gate.scores_to_selection(base[split], scores, self_model_id=self_model_id)
        row = evaluate_selection(
            package,
            self_gate,
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


def model_val_and_test_rows(
    package,
    self_gate,
    outputs: pd.DataFrame,
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    *,
    method: str,
    family: str,
    self_model_id: str,
    lambda_cost: float,
    **extra: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, scores in [("val", val_scores), ("test", test_scores)]:
        query_ids = sorted(outputs[outputs["split"].eq(split)]["query_id"].astype(str).unique())
        scores = scores.copy()
        scores.index = query_ids
        selected = scores.idxmax(axis=1).astype(str)
        row = evaluate_selection(
            package,
            self_gate,
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


def evaluate_selection(
    package,
    self_gate,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
    self_model_id: str,
) -> dict[str, Any]:
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
    row.update(oracle_recall_metrics(package, outputs, selected, split=split, self_model_id=self_model_id))
    return row


def oracle_recall_metrics(package, outputs: pd.DataFrame, selected: pd.Series, *, split: str, self_model_id: str) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    selected_rows = package.selected_to_rows(outputs, selected, split=split).set_index("query_id")
    oracle_rows = target.loc[target.groupby("query_id")["utility"].idxmax()].set_index("query_id")
    common = selected_rows.index.intersection(oracle_rows.index)
    metrics: dict[str, Any] = {}
    if len(common) == 0:
        return {
            "cost_oracle_model_match_rate": float("nan"),
            "strong_oracle_recall": float("nan"),
            "self_oracle_recall": float("nan"),
            "gpt_oracle_recall": float("nan"),
            "gemini_flash_oracle_recall": float("nan"),
            "tool_oracle_recall": float("nan"),
        }
    selected_models = selected_rows.loc[common, "model_id"].astype(str)
    oracle_models = oracle_rows.loc[common, "model_id"].astype(str)
    metrics["cost_oracle_model_match_rate"] = float((selected_models == oracle_models).mean())
    for column, model_id in [
        ("strong_oracle_recall", STRONG_MODEL_ID),
        ("self_oracle_recall", self_model_id),
        ("gpt_oracle_recall", "gpt-5.5"),
        ("gemini_flash_oracle_recall", "gemini-3.5-flash"),
        ("tool_oracle_recall", "deterministic_math_tool"),
    ]:
        mask = oracle_models.eq(model_id)
        metrics[column] = float(selected_models[mask].eq(model_id).mean()) if bool(mask.any()) else float("nan")
    return metrics


def family_summary(table: pd.DataFrame) -> pd.DataFrame:
    test = table[table["split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    return (
        test.groupby("family", as_index=False)
        .agg(
            best_test_utility=("mean_utility", "max"),
            best_test_quality=("mean_quality", "max"),
            best_oracle_ratio=("oracle_utility_ratio", "max"),
            best_cost_oracle_model_match=("cost_oracle_model_match_rate", "max"),
            best_strong_oracle_recall=("strong_oracle_recall", "max"),
            best_self_oracle_recall=("self_oracle_recall", "max"),
        )
        .sort_values("best_test_utility", ascending=False)
    )


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#667f5d")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Richer-State kNN With Self-Consistency Features")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_richer_state_knn_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# Richer-State kNN With Self-Consistency Features",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls; it uses cached probe rows and fits TF-IDF/kNN on train only.",
        "",
        "## Family Summary",
        "",
        "```csv",
        compact_csv(summary),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected),
        "```",
        "",
        "## Held-Out Test Rows",
        "",
        "```csv",
        compact_csv(table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(35)),
        "```",
        "",
        "## Interpretation",
        "",
        "- Action kNN averages train utilities for base/self/strong actions among nearest train probe states.",
        "- Model kNN averages train utilities for every cached model ID among nearest train probe states.",
        "- Feature views range from raw query text to query plus benchmark, base action, self-consistency answer, vote margin, entropy, and answer-agreement tags.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
