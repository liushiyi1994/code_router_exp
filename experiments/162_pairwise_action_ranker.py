from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression, Ridge


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
ACTIONS = ["base", "self", "strong"]
PAIRS = [("base", "self"), ("base", "strong"), ("self", "strong")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pairwise decision-aware base/self/strong action ranker.")
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
        "--embedding-cache-dir",
        type=Path,
        default=Path("results/controlled/broad100_embedding_self_action_gate/embedding_cache"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_pairwise_action_ranker"),
    )
    parser.add_argument("--self-model-id", default=DEFAULT_SELF_MODEL_ID)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    self_gate = load_module("experiments/148_self_consistency_feature_gate.py", "self_consistency_gate")
    calibrated = load_module("experiments/152_calibrated_self_consistency_action_gate.py", "calibrated_gate")

    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    table = run_pairwise_rankers(
        package,
        self_gate,
        calibrated,
        outputs,
        probe,
        embedding_cache_dir=args.embedding_cache_dir,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_pairwise_action_ranker_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_pairwise_action_ranker_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "PAIRWISE_ACTION_RANKER_MEMO.md", args, table, selected)
    print(f"Wrote pairwise action-ranker results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_pairwise_rankers(
    package,
    self_gate,
    calibrated,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    embedding_cache_dir: Path,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    local_agree = calibrated.local_agreement_counts(outputs, self_model_id=self_model_id)
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([self_model_id, STRONG_MODEL_ID])].copy()
    base_specs = {
        "observable_local_state_v5": lambda split: package.observable_local_state_selection(outputs_no_self, split=split),
        "observable_local_state_v5_no_strong": lambda split: package.observable_local_state_selection(
            outputs_no_strong_self, split=split
        ),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split, exclude_models={STRONG_MODEL_ID}
        ),
    }
    oracle_stats = split_oracle_stats(outputs)
    for base_name, builder in base_specs.items():
        base = {split: self_gate.normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        frames = {
            split: calibrated.build_feature_frame_fast(
                outputs,
                probe,
                base[split],
                split=split,
                self_model_id=self_model_id,
                local_agree=local_agree,
            )
            for split in ["train", "val", "test"]
        }
        action_metrics = {
            split: build_action_metrics(outputs, frames[split], split=split, self_model_id=self_model_id)
            for split in ["val", "test"]
        }
        for split in ["val", "test"]:
            rows.append(
                evaluate_action_indices(
                    action_metrics[split],
                    np.zeros(len(frames[split]), dtype=int),
                    split=split,
                    method=base_name,
                    family="base",
                    lambda_cost=lambda_cost,
                    oracle_stats=oracle_stats[split],
                )
            )
            rows.append(
                evaluate_action_indices(
                    action_metrics[split],
                    np.argmax(frames[split][["utility_base", "utility_self", "utility_strong"]].to_numpy(dtype=float), axis=1),
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                    oracle_stats=oracle_stats[split],
                )
            )
        if frames["train"].empty or frames["val"].empty or frames["test"].empty:
            continue
        rows.extend(
            run_base_pairwise_models(
                calibrated,
                frames,
                action_metrics,
                oracle_stats,
                base_name=base_name,
                embedding_cache_dir=embedding_cache_dir,
                lambda_cost=lambda_cost,
                max_features=max_features,
            )
        )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def run_base_pairwise_models(
    calibrated,
    frames: dict[str, pd.DataFrame],
    action_metrics: dict[str, dict[str, np.ndarray]],
    oracle_stats: dict[str, dict[str, np.ndarray]],
    *,
    base_name: str,
    embedding_cache_dir: Path,
    lambda_cost: float,
    max_features: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_sets = build_feature_sets(calibrated, frames, base_name, embedding_cache_dir, max_features=max_features)
    for feature_view, matrices in feature_sets.items():
        for learner in ["ridge", "logistic"]:
            hp_values = [0.1, 1.0, 10.0, 100.0] if learner == "ridge" else [0.1, 1.0, 10.0]
            for hp_value in hp_values:
                score_by_split = fit_pairwise_scores(frames, matrices, learner=learner, hp_value=float(hp_value))
                for self_bias in [-0.05, 0.0, 0.05, 0.10]:
                    for strong_bias in [-0.20, -0.10, -0.05, 0.0, 0.05]:
                        for strong_cap in [None, 0.25, 0.30, 0.35, 0.40]:
                            method = method_name(
                                base_name,
                                feature_view,
                                learner,
                                hp_value,
                                self_bias,
                                strong_bias,
                                strong_cap,
                            )
                            for split in ["val", "test"]:
                                actions = scores_to_action_indices(
                                    score_by_split[split],
                                    self_bias=float(self_bias),
                                    strong_bias=float(strong_bias),
                                    strong_cap=strong_cap,
                                )
                                row = evaluate_action_indices(
                                    action_metrics[split],
                                    actions,
                                    split=split,
                                    method=method,
                                    family=f"pairwise_{learner}",
                                    lambda_cost=lambda_cost,
                                    oracle_stats=oracle_stats[split],
                                )
                                row.update(
                                    {
                                        "base_name": base_name,
                                        "feature_view": feature_view,
                                        "learner": learner,
                                        "hp_value": float(hp_value),
                                        "self_bias": float(self_bias),
                                        "strong_bias": float(strong_bias),
                                        "strong_cap": np.nan if strong_cap is None else float(strong_cap),
                                    }
                                )
                                rows.append(row)
    return rows


def build_feature_sets(
    calibrated,
    frames: dict[str, pd.DataFrame],
    base_name: str,
    embedding_cache_dir: Path,
    *,
    max_features: int,
) -> dict[str, dict[str, Any]]:
    x_train_num, x_val_num, x_test_num = calibrated.featurize(
        frames["train"],
        frames["val"],
        frames["test"],
        feature_view="metadata_numeric",
        max_features=max_features,
    )
    x_train_text, x_val_text, x_test_text = calibrated.featurize(
        frames["train"],
        frames["val"],
        frames["test"],
        feature_view="metadata_numeric_text",
        max_features=max_features,
    )
    feature_sets: dict[str, dict[str, Any]] = {
        "metadata_numeric": {"train": x_train_num, "val": x_val_num, "test": x_test_num},
        "metadata_numeric_text": {"train": x_train_text, "val": x_val_text, "test": x_test_text},
    }
    embedding_paths = {
        split: embedding_cache_dir / f"intfloat_e5-small-v2__{base_name}__{split}__{len(frames[split])}.npy"
        for split in ["train", "val", "test"]
    }
    if all(path.exists() for path in embedding_paths.values()):
        embeddings = {split: csr_matrix(np.load(path)) for split, path in embedding_paths.items()}
        feature_sets["meta_embed"] = {
            "train": hstack([x_train_num, embeddings["train"]]),
            "val": hstack([x_val_num, embeddings["val"]]),
            "test": hstack([x_test_num, embeddings["test"]]),
        }
    return feature_sets


def fit_pairwise_scores(
    frames: dict[str, pd.DataFrame],
    matrices: dict[str, Any],
    *,
    learner: str,
    hp_value: float,
) -> dict[str, np.ndarray]:
    scores = {
        "val": np.zeros((len(frames["val"]), len(ACTIONS)), dtype=float),
        "test": np.zeros((len(frames["test"]), len(ACTIONS)), dtype=float),
    }
    action_index = {action: idx for idx, action in enumerate(ACTIONS)}
    for action_a, action_b in PAIRS:
        diff = (
            frames["train"][f"utility_{action_a}"].to_numpy(dtype=float)
            - frames["train"][f"utility_{action_b}"].to_numpy(dtype=float)
        )
        if learner == "ridge":
            model = Ridge(alpha=float(hp_value), solver="lsqr")
            model.fit(matrices["train"], diff)
            for split in ["val", "test"]:
                pred = np.asarray(model.predict(matrices[split]), dtype=float)
                scores[split][:, action_index[action_a]] += pred
                scores[split][:, action_index[action_b]] -= pred
        elif learner == "logistic":
            target = (diff > 1e-9).astype(int)
            if len(set(target.tolist())) < 2:
                continue
            weights = np.maximum(np.abs(diff), 0.05)
            model = LogisticRegression(C=float(hp_value), solver="liblinear", class_weight="balanced", max_iter=2000)
            model.fit(matrices["train"], target, sample_weight=weights)
            for split in ["val", "test"]:
                pred = np.asarray(model.predict_proba(matrices[split])[:, 1], dtype=float) - 0.5
                scores[split][:, action_index[action_a]] += pred
                scores[split][:, action_index[action_b]] -= pred
        else:
            raise ValueError(learner)
    return scores


def scores_to_action_indices(
    scores: np.ndarray,
    *,
    self_bias: float,
    strong_bias: float,
    strong_cap: float | None,
) -> np.ndarray:
    adjusted = np.asarray(scores, dtype=float).copy()
    adjusted[:, 1] += float(self_bias)
    adjusted[:, 2] += float(strong_bias)
    actions = np.argmax(adjusted, axis=1)
    if strong_cap is not None:
        max_strong = int(np.floor(float(strong_cap) * len(actions)))
        strong_idx = np.where(actions == 2)[0]
        if len(strong_idx) > max_strong:
            margins = adjusted[strong_idx, 2] - np.maximum(adjusted[strong_idx, 0], adjusted[strong_idx, 1])
            keep = set(strong_idx[np.argsort(margins)[::-1][:max_strong]].tolist())
            for idx in strong_idx:
                if int(idx) not in keep:
                    actions[idx] = 0 if adjusted[idx, 0] >= adjusted[idx, 1] else 1
    return actions


def build_action_metrics(
    outputs: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    split: str,
    self_model_id: str,
) -> dict[str, np.ndarray]:
    target = outputs[outputs["split"].eq(split)].set_index(["query_id", "model_id"])
    model_columns = ["model_base", "model_self", "model_strong"]
    metric_columns = {
        "quality": "quality_score",
        "utility": "utility",
        "norm_cost": "normalized_remote_cost",
        "usd_cost": "cost_total_usd",
        "latency": "latency_s",
        "frontier": "is_frontier",
        "local": "is_local",
    }
    metrics: dict[str, list[list[Any]]] = {name: [] for name in metric_columns}
    selected_models: list[list[str]] = []
    for _, row in frame.iterrows():
        query_id = str(row["query_id"])
        models = [str(row[column]) for column in model_columns]
        selected_models.append(models)
        for metric_name, source_column in metric_columns.items():
            values = []
            for model_id in models:
                value = target.loc[(query_id, model_id), source_column] if (query_id, model_id) in target.index else 0.0
                values.append(bool(value) if metric_name in {"frontier", "local"} else float(value))
            metrics[metric_name].append(values)
    return {
        key: np.asarray(value, dtype=bool if key in {"frontier", "local"} else float)
        for key, value in metrics.items()
    } | {"selected_models": np.asarray(selected_models, dtype=object), "self_model_id": self_model_id}


def split_oracle_stats(outputs: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    stats: dict[str, dict[str, np.ndarray]] = {}
    for split in ["val", "test"]:
        target = outputs[outputs["split"].eq(split)]
        cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
        quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
        stats[split] = {
            "utility": cost_oracle["utility"].to_numpy(dtype=float),
            "quality": quality_oracle["quality_score"].to_numpy(dtype=float),
        }
    return stats


def evaluate_action_indices(
    metrics: dict[str, np.ndarray],
    action_indices: np.ndarray,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
    oracle_stats: dict[str, np.ndarray],
) -> dict[str, Any]:
    idx = np.arange(len(action_indices))
    quality = metrics["quality"][idx, action_indices]
    utility = metrics["utility"][idx, action_indices]
    norm_cost = metrics["norm_cost"][idx, action_indices]
    usd_cost = metrics["usd_cost"][idx, action_indices]
    latency = metrics["latency"][idx, action_indices]
    frontier = metrics["frontier"][idx, action_indices]
    local = metrics["local"][idx, action_indices]
    selected_models = metrics["selected_models"][idx, action_indices]
    model_counts = pd.Series(selected_models).value_counts().sort_index().to_dict()
    oracle_utility = oracle_stats["utility"]
    oracle_quality = oracle_stats["quality"]
    return {
        "method": method,
        "family": family,
        "split": split,
        "n_queries": int(len(action_indices)),
        "mean_quality": float(np.mean(quality)),
        "mean_utility": float(np.mean(utility)),
        "quality_oracle_mean_quality": float(np.mean(oracle_quality)),
        "cost_oracle_mean_utility": float(np.mean(oracle_utility)),
        "quality_gap_to_oracle": float(np.mean(oracle_quality) - np.mean(quality)),
        "utility_gap_to_oracle": float(np.mean(oracle_utility) - np.mean(utility)),
        "oracle_utility_ratio": float(np.mean(utility) / np.mean(oracle_utility)),
        "remote_cost_total_usd": float(np.sum(usd_cost)),
        "normalized_remote_cost_mean": float(np.mean(norm_cost)),
        "frontier_call_rate": float(np.mean(frontier)),
        "local_call_rate": float(np.mean(local)),
        "mean_latency_s": float(np.mean(latency)),
        "p95_latency_s": float(np.quantile(latency, 0.95)),
        "lambda_cost": float(lambda_cost),
        "selected_models_json": json.dumps({str(key): int(value) for key, value in model_counts.items()}, sort_keys=True),
        "strong_call_rate": float(np.mean(selected_models == STRONG_MODEL_ID)),
        "self_action_rate": float(np.mean(selected_models == metrics["self_model_id"])),
    }


def method_name(
    base_name: str,
    feature_view: str,
    learner: str,
    hp_value: float,
    self_bias: float,
    strong_bias: float,
    strong_cap: float | None,
) -> str:
    name = (
        f"{base_name}_pairwise_{learner}_{feature_view}_hp{hp_value:g}"
        f"_self{self_bias:.2f}_strong{strong_bias:.2f}"
    )
    if strong_cap is not None:
        name += f"_cap{strong_cap:.2f}"
    return name


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
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


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
    fig, ax = plt.subplots(figsize=(10, 6.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#586f86")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Pairwise Action Ranker")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pairwise_action_ranker_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_call_rate",
        "self_action_rate",
        "base_name",
        "feature_view",
        "learner",
        "hp_value",
        "self_bias",
        "strong_bias",
        "strong_cap",
        "selection_rule",
    ]
    lines = [
        "# Pairwise Action Ranker",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        f"Embedding cache: `{args.embedding_cache_dir}`.",
        "",
        "This evaluator makes no provider API or vLLM calls. It trains pairwise base/self/strong preference models on cached train utilities and selects hyperparameters on validation.",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[column for column in cols if column in selected.columns]], max_rows=40),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(
            table[table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)[[column for column in cols if column in table.columns]],
            max_rows=32,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- Pairwise ranking tests the decision-aware idea from the probe-signal notes: learn action preferences rather than three absolute utilities.",
        "- If validation-selected held-out utility does not beat the current E5 self-action gate, pairwise ranking is not yet the missing probe signal.",
        "- Top held-out diagnostic rows are not deployable claims because they are selected with test labels.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
