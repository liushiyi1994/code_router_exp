from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from routecode.controlled.live_stage0 import normalize_answer


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
DEFAULT_SELF_MODEL_ID = "qwen3-32b-awq-selfconsistency-n3-local"
ACTIONS = ["base", "self", "strong"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-calibrated base/self/strong action gate.")
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
        default=Path("results/controlled/broad100_calibrated_self_consistency_action_gate"),
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
    outputs = self_gate.load_outputs(args.outputs)
    probe = self_gate.load_probe(args.probe_table)
    table = run_calibrated_gates(
        package,
        outputs,
        probe,
        self_model_id=str(args.self_model_id),
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    selected = validation_selected_rows(table)
    table.to_csv(args.output_dir / "table_calibrated_action_gate_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_calibrated_action_gate_selected.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "CALIBRATED_ACTION_GATE_MEMO.md", args, table, selected)
    print(f"Wrote calibrated action-gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_calibrated_gates(
    package,
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    *,
    self_model_id: str,
    lambda_cost: float,
    max_features: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outputs_no_self = outputs[~outputs["model_id"].eq(self_model_id)].copy()
    outputs_no_strong_self = outputs[~outputs["model_id"].isin([self_model_id, STRONG_MODEL_ID])].copy()
    base_specs = {
        "observable_local_state_v5": lambda split: fast_observable_local_state_selection(package, outputs_no_self, split=split),
        "observable_local_state_v5_no_strong": lambda split: fast_observable_local_state_selection(
            package,
            outputs_no_strong_self,
            split=split,
        ),
        "tool_probe_profile_v4": lambda split: package.profile_v4_selection_for_split(outputs_no_self, split=split),
        "tool_probe_profile_v4_no_strong": lambda split: package.profile_v4_selection_for_split(
            outputs_no_strong_self, split=split,
            exclude_models={STRONG_MODEL_ID},
        ),
    }
    matrices = split_matrices(outputs, self_model_id=self_model_id)
    local_agree = local_agreement_counts(outputs, self_model_id=self_model_id)
    for base_name, builder in base_specs.items():
        print(f"running base={base_name}", flush=True)
        base = {split: normalize_selection(builder(split)) for split in ["train", "val", "test"]}
        frames = {
            split: build_feature_frame_fast(
                outputs,
                probe,
                base[split],
                split=split,
                self_model_id=self_model_id,
                local_agree=local_agree,
            )
            for split in ["train", "val", "test"]
        }
        for split in ["val", "test"]:
            rows.append(
                evaluate_selection_fast(
                    frames[split],
                    matrices[split],
                    np.full(len(frames[split]), "base", dtype=object),
                    split=split,
                    method=base_name,
                    family="base",
                    lambda_cost=lambda_cost,
                )
            )
            oracle_actions = oracle_action_labels(frames[split])
            rows.append(
                evaluate_selection_fast(
                    frames[split],
                    matrices[split],
                    oracle_actions,
                    split=split,
                    method=f"{base_name}_oracle_between_base_self_strong",
                    family="diagnostic_oracle",
                    lambda_cost=lambda_cost,
                )
            )
        if frames["train"].empty or frames["val"].empty or frames["test"].empty:
            continue
        for feature_view in ["metadata_numeric", "metadata_numeric_text"]:
            print(f"  feature_view={feature_view}", flush=True)
            x_train, x_val, x_test = featurize(
                frames["train"],
                frames["val"],
                frames["test"],
                feature_view=feature_view,
                max_features=max_features,
            )
            for alpha in [0.1, 1.0, 10.0, 100.0, 1000.0]:
                print(f"    alpha={alpha:g}", flush=True)
                val_scores, test_scores = fit_action_scores(
                    x_train,
                    x_val,
                    x_test,
                    frames["train"],
                    alpha=float(alpha),
                )
                rows.extend(
                    calibrated_rows(
                        frames,
                        matrices,
                        val_scores,
                        test_scores,
                        base_name=base_name,
                        feature_view=feature_view,
                        alpha=float(alpha),
                        lambda_cost=lambda_cost,
                    )
                )
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def split_matrices(outputs: pd.DataFrame, *, self_model_id: str) -> dict[str, dict[str, Any]]:
    matrices: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        target = outputs[outputs["split"].eq(split)].copy()
        query_ids = target.drop_duplicates("query_id")["query_id"].astype(str).tolist()
        by_model = target.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="last")
        quality = target.pivot_table(index="query_id", columns="model_id", values="quality_score", aggfunc="last")
        norm_cost = target.pivot_table(index="query_id", columns="model_id", values="normalized_remote_cost", aggfunc="last")
        usd_cost = target.pivot_table(index="query_id", columns="model_id", values="cost_total_usd", aggfunc="last")
        latency = target.pivot_table(index="query_id", columns="model_id", values="latency_s", aggfunc="last")
        frontier = target.pivot_table(index="query_id", columns="model_id", values="is_frontier", aggfunc="last")
        local = target.pivot_table(index="query_id", columns="model_id", values="is_local", aggfunc="last")
        matrices[split] = {
            "query_ids": query_ids,
            "utility": by_model,
            "quality": quality,
            "norm_cost": norm_cost,
            "usd_cost": usd_cost,
            "latency": latency,
            "frontier": frontier,
            "local": local,
            "cost_oracle_utility": by_model.max(axis=1).reindex(query_ids).to_numpy(dtype=float),
            "quality_oracle": quality.max(axis=1).reindex(query_ids).to_numpy(dtype=float),
            "self_model_id": self_model_id,
        }
    return matrices


def fast_observable_local_state_selection(package, outputs: pd.DataFrame, *, split: str, min_support: int = 2) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    actions = ["deterministic_math_tool", *local_models, "gemini-3.5-flash", "gpt-5.5", STRONG_MODEL_ID]
    available = set(outputs["model_id"].astype(str).unique())
    actions = [model_id for model_id in actions if model_id in available]
    trainval_queries = outputs[outputs["split"].isin(["train", "val"])].drop_duplicates("query_id").set_index("query_id")
    target_queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="last")
    quality = outputs.pivot_table(index="query_id", columns="model_id", values="quality_score", aggfunc="last")
    cost = outputs.pivot_table(index="query_id", columns="model_id", values="normalized_remote_cost", aggfunc="last")
    tool_available = {
        str(query_id): bool(package.deterministic_tool_choice(by_query, str(query_id)))
        for query_id in outputs["query_id"].astype(str).unique()
    }

    groups: dict[tuple[Any, ...], list[str]] = {}
    for query_id, row in trainval_queries.iterrows():
        query_id = str(query_id)
        state = package.observable_local_state(query_id, row, by_query, local_models)
        groups.setdefault(state, []).append(query_id)

    chosen_by_state: dict[tuple[Any, ...], str] = {}
    for state, query_ids in groups.items():
        if len(query_ids) < int(min_support):
            continue
        candidates: list[dict[str, Any]] = []
        for model_id in actions:
            valid_ids = query_ids
            if model_id == "deterministic_math_tool":
                valid_ids = [query_id for query_id in query_ids if tool_available.get(str(query_id), False)]
            if not valid_ids or model_id not in utility.columns:
                continue
            utilities = utility.loc[valid_ids, model_id].dropna()
            if utilities.empty:
                continue
            qualities = quality.loc[valid_ids, model_id].dropna() if model_id in quality.columns else pd.Series(dtype=float)
            costs = cost.loc[valid_ids, model_id].dropna() if model_id in cost.columns else pd.Series(dtype=float)
            candidates.append(
                {
                    "model_id": model_id,
                    "mean_utility": float(utilities.mean()),
                    "mean_quality": float(qualities.mean()) if not qualities.empty else 0.0,
                    "mean_cost": float(costs.mean()) if not costs.empty else 0.0,
                }
            )
        if candidates:
            best = sorted(candidates, key=lambda row: (row["mean_utility"], row["mean_quality"], -row["mean_cost"]), reverse=True)[0]
            chosen_by_state[state] = str(best["model_id"])

    fallback_by_benchmark = package.benchmark_utility_fallback(outputs)
    selected: dict[str, str] = {}
    for query_id, row in target_queries.iterrows():
        query_id = str(query_id)
        if tool_available.get(query_id, False):
            selected[query_id] = "deterministic_math_tool"
            continue
        state = package.observable_local_state(query_id, row, by_query, local_models)
        model_id = chosen_by_state.get(state)
        if not model_id or model_id == "deterministic_math_tool":
            model_id = fallback_by_benchmark.get(str(row["benchmark"]), "qwen3-14b-awq-local")
        selected[query_id] = str(model_id)
    return pd.Series(selected)


def local_agreement_counts(outputs: pd.DataFrame, *, self_model_id: str) -> dict[str, int]:
    local = outputs[
        outputs["is_local"].astype(bool)
        & ~outputs["model_id"].astype(str).isin(["deterministic_math_tool", self_model_id])
    ].copy()
    local["answer_norm"] = local["parsed_answer"].map(lambda value: normalize_answer(str(value)))
    counts: dict[str, int] = {}
    for query_id, group in local.groupby("query_id"):
        answer_counts = group["answer_norm"].value_counts()
        counts[str(query_id)] = int(answer_counts.iloc[0]) if not answer_counts.empty and str(answer_counts.index[0]) else 0
    return counts


def build_feature_frame_fast(
    outputs: pd.DataFrame,
    probe: pd.DataFrame,
    base_selection: pd.Series,
    *,
    split: str,
    self_model_id: str,
    local_agree: dict[str, int],
) -> pd.DataFrame:
    query_info = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    probe_by_query = probe[probe["split"].eq(split)].set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id, base_model in base_selection.items():
        query_id = str(query_id)
        base_model = str(base_model)
        if query_id not in query_info.index or query_id not in probe_by_query.index:
            continue
        if (query_id, base_model) not in by_query.index or (query_id, self_model_id) not in by_query.index:
            continue
        if (query_id, STRONG_MODEL_ID) not in by_query.index:
            continue
        info = query_info.loc[query_id]
        probe_row = probe_by_query.loc[query_id]
        base_row = by_query.loc[(query_id, base_model)]
        self_row = by_query.loc[(query_id, self_model_id)]
        strong_row = by_query.loc[(query_id, STRONG_MODEL_ID)]
        majority_norm = normalize_answer(str(probe_row.get("majority_answer_norm", "")))
        base_norm = normalize_answer(str(base_row.get("parsed_answer", "")))
        row = {
            "query_id": query_id,
            "query_text": str(info.get("query_text", "")),
            "benchmark": str(info.get("benchmark", "")),
            "domain": str(info.get("domain", "")),
            "metric": str(info.get("metric", "")),
            "base_model_id": base_model,
            "base_provider": str(base_row.get("provider", "")),
            "base_is_local": bool(base_row.get("is_local", False)),
            "base_is_frontier": bool(base_row.get("is_frontier", False)),
            "base_answer_norm": base_norm,
            "majority_answer_norm": majority_norm,
            "base_equals_self_majority": bool(base_norm and majority_norm and base_norm == majority_norm),
            "n_samples": float(probe_row.get("n_samples", 0.0) or 0.0),
            "valid_count": float(probe_row.get("valid_count", 0.0) or 0.0),
            "top_vote_count": float(probe_row.get("top_vote_count", 0.0) or 0.0),
            "vote_frac": float(probe_row.get("vote_frac", 0.0) or 0.0),
            "vote_margin": float(probe_row.get("vote_margin", 0.0) or 0.0),
            "vote_entropy": float(probe_row.get("vote_entropy", 0.0) or 0.0),
            "all_samples_agree": bool(probe_row.get("all_samples_agree", False)),
            "local_agree_with_majority_count": float(local_agree.get(query_id, 0)),
            "majority_answer_len": float(len(majority_norm)),
            "base_answer_len": float(len(base_norm)),
            "probe_latency_s": float(probe_row.get("latency_s", 0.0) or 0.0),
            "probe_output_tokens": float(probe_row.get("output_tokens", 0.0) or 0.0),
            "model_base": base_model,
            "model_self": self_model_id,
            "model_strong": STRONG_MODEL_ID,
            "utility_base": float(base_row["utility"]),
            "utility_self": float(self_row["utility"]),
            "utility_strong": float(strong_row["utility"]),
        }
        row["feature_text"] = feature_text(row)
        rows.append(row)
    return pd.DataFrame(rows)


def feature_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("benchmark", "")),
            str(row.get("domain", "")),
            str(row.get("metric", "")),
            str(row.get("base_model_id", "")),
            f"base_answer={row.get('base_answer_norm', '')}",
            f"self_answer={row.get('majority_answer_norm', '')}",
            "base_equals_self" if row.get("base_equals_self_majority") else "base_differs_self",
            str(row.get("query_text", "")),
        ]
    )


def featurize(
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
        "local_agree_with_majority_count",
        "majority_answer_len",
        "base_answer_len",
        "probe_latency_s",
        "probe_output_tokens",
    ]
    categorical_columns = [
        "benchmark",
        "domain",
        "metric",
        "base_model_id",
        "base_provider",
        "base_is_local",
        "base_is_frontier",
        "base_equals_self_majority",
        "all_samples_agree",
    ]
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(frame_to_dicts(train, numeric_columns, categorical_columns))
    x_val = vectorizer.transform(frame_to_dicts(val, numeric_columns, categorical_columns))
    x_test = vectorizer.transform(frame_to_dicts(test, numeric_columns, categorical_columns))
    if feature_view == "metadata_numeric":
        return x_train, x_val, x_test
    if feature_view != "metadata_numeric_text":
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


def fit_action_scores(x_train, x_val, x_test, train: pd.DataFrame, *, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    for action, column in [("base", "utility_base"), ("self", "utility_self"), ("strong", "utility_strong")]:
        model = Ridge(alpha=float(alpha), solver="lsqr")
        model.fit(x_train, train[column].to_numpy(dtype=float))
        val_scores[action] = np.asarray(model.predict(x_val), dtype=float)
        test_scores[action] = np.asarray(model.predict(x_test), dtype=float)
    return pd.DataFrame(val_scores), pd.DataFrame(test_scores)


def calibrated_rows(
    frames: dict[str, pd.DataFrame],
    matrices: dict[str, dict[str, Any]],
    val_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    *,
    base_name: str,
    feature_view: str,
    alpha: float,
    lambda_cost: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, frontier_cap in [
        ("calibrated_action_gate", None),
        ("calibrated_action_gate_cap25", 0.25),
        ("calibrated_action_gate_cap30", 0.30),
        ("calibrated_action_gate_cap35", 0.35),
        ("calibrated_action_gate_cap40", 0.40),
    ]:
        best: dict[str, Any] | None = None
        for self_bias in np.linspace(-0.30, 0.30, 13):
            for strong_bias in np.linspace(-0.70, 0.30, 21):
                val_actions = score_to_actions(val_scores, self_bias=self_bias, strong_bias=strong_bias, frontier_cap=frontier_cap)
                row = evaluate_selection_fast(
                    frames["val"],
                    matrices["val"],
                    val_actions,
                    split="val",
                    method="candidate",
                    family=family,
                    lambda_cost=lambda_cost,
                )
                row.update(
                    {
                        "base_method": base_name,
                        "feature_view": feature_view,
                        "alpha": float(alpha),
                        "self_bias": float(self_bias),
                        "strong_bias": float(strong_bias),
                        "frontier_cap": np.nan if frontier_cap is None else float(frontier_cap),
                    }
                )
                if best is None or (row["mean_utility"], row["mean_quality"]) > (best["mean_utility"], best["mean_quality"]):
                    best = row
        if best is None:
            continue
        method = (
            f"{base_name}_{family}_{feature_view}_alpha{alpha:g}"
            f"_self{best['self_bias']:.2f}_strong{best['strong_bias']:.2f}"
        )
        if not pd.isna(best["frontier_cap"]):
            method += f"_cap{best['frontier_cap']:.2f}"
        best["method"] = method
        rows.append(best)
        test_actions = score_to_actions(
            test_scores,
            self_bias=float(best["self_bias"]),
            strong_bias=float(best["strong_bias"]),
            frontier_cap=None if pd.isna(best["frontier_cap"]) else float(best["frontier_cap"]),
        )
        test_row = evaluate_selection_fast(
            frames["test"],
            matrices["test"],
            test_actions,
            split="test",
            method=method,
            family=family,
            lambda_cost=lambda_cost,
        )
        test_row.update(
            {
                "base_method": base_name,
                "feature_view": feature_view,
                "alpha": float(alpha),
                "self_bias": float(best["self_bias"]),
                "strong_bias": float(best["strong_bias"]),
                "frontier_cap": best["frontier_cap"],
            }
        )
        rows.append(test_row)
    return rows


def score_to_actions(
    scores: pd.DataFrame,
    *,
    self_bias: float,
    strong_bias: float,
    frontier_cap: float | None,
) -> np.ndarray:
    adjusted = scores[ACTIONS].to_numpy(dtype=float).copy()
    adjusted[:, 1] += float(self_bias)
    adjusted[:, 2] += float(strong_bias)
    actions = np.asarray(ACTIONS, dtype=object)[np.argmax(adjusted, axis=1)]
    if frontier_cap is not None:
        max_strong = int(np.floor(float(frontier_cap) * len(actions)))
        strong_idx = np.where(actions == "strong")[0]
        if len(strong_idx) > max_strong:
            margins = adjusted[strong_idx, 2] - np.maximum(adjusted[strong_idx, 0], adjusted[strong_idx, 1])
            keep = set(strong_idx[np.argsort(margins)[::-1][:max_strong]].tolist())
            for idx in strong_idx:
                if int(idx) not in keep:
                    actions[idx] = "base" if adjusted[idx, 0] >= adjusted[idx, 1] else "self"
    return actions


def oracle_action_labels(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["utility_base", "utility_self", "utility_strong"]].to_numpy(dtype=float)
    return np.asarray(ACTIONS, dtype=object)[np.argmax(values, axis=1)]


def evaluate_selection_fast(
    frame: pd.DataFrame,
    matrix: dict[str, Any],
    actions: np.ndarray,
    *,
    split: str,
    method: str,
    family: str,
    lambda_cost: float,
) -> dict[str, Any]:
    selected_models = action_models(frame, actions)
    query_ids = frame["query_id"].astype(str).tolist()
    selected = gather_selected(matrix, query_ids, selected_models)
    quality = selected["quality"]
    utility = selected["utility"]
    norm_cost = selected["norm_cost"]
    usd_cost = selected["usd_cost"]
    latency = selected["latency"]
    frontier = selected["frontier"]
    local = selected["local"]
    oracle_utility = matrix["cost_oracle_utility"]
    oracle_quality = matrix["quality_oracle"]
    model_counts = pd.Series(selected_models).value_counts().sort_index().to_dict()
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(query_ids)),
        "mean_quality": float(np.mean(quality)),
        "mean_utility": float(np.mean(utility)),
        "quality_oracle_mean_quality": float(np.mean(oracle_quality)),
        "cost_oracle_mean_utility": float(np.mean(oracle_utility)),
        "quality_gap_to_oracle": float(np.mean(oracle_quality) - np.mean(quality)),
        "utility_gap_to_oracle": float(np.mean(oracle_utility) - np.mean(utility)),
        "oracle_utility_ratio": float(np.mean(utility) / np.mean(oracle_utility)) if abs(float(np.mean(oracle_utility))) > 1e-12 else np.nan,
        "remote_cost_total_usd": float(np.sum(usd_cost)),
        "normalized_remote_cost_mean": float(np.mean(norm_cost)),
        "frontier_call_rate": float(np.mean(frontier)),
        "local_call_rate": float(np.mean(local)),
        "mean_latency_s": float(np.mean(latency)),
        "p95_latency_s": float(np.quantile(latency, 0.95)),
        "lambda_cost": float(lambda_cost),
        "selected_models_json": json.dumps({str(key): int(value) for key, value in model_counts.items()}, sort_keys=True),
        "family": family,
        "strong_call_rate": float(np.mean(np.asarray(selected_models, dtype=object) == STRONG_MODEL_ID)),
        "self_action_rate": float(np.mean(np.asarray(selected_models, dtype=object) == matrix["self_model_id"])),
    }


def action_models(frame: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
    models = []
    for action, (_, row) in zip(actions, frame.iterrows()):
        if action == "base":
            models.append(str(row["model_base"]))
        elif action == "self":
            models.append(str(row["model_self"]))
        elif action == "strong":
            models.append(str(row["model_strong"]))
        else:
            raise ValueError(str(action))
    return np.asarray(models, dtype=object)


def gather_selected(matrix: dict[str, Any], query_ids: list[str], selected_models: np.ndarray) -> dict[str, np.ndarray]:
    out = {"quality": [], "utility": [], "norm_cost": [], "usd_cost": [], "latency": [], "frontier": [], "local": []}
    maps = {
        "quality": matrix["quality"],
        "utility": matrix["utility"],
        "norm_cost": matrix["norm_cost"],
        "usd_cost": matrix["usd_cost"],
        "latency": matrix["latency"],
        "frontier": matrix["frontier"],
        "local": matrix["local"],
    }
    for query_id, model_id in zip(query_ids, selected_models):
        for key, pivot in maps.items():
            value = pivot.loc[query_id, model_id] if model_id in pivot.columns and query_id in pivot.index else 0.0
            if key in {"frontier", "local"}:
                out[key].append(bool(value))
            else:
                out[key].append(float(value) if not pd.isna(value) else 0.0)
    return {key: np.asarray(value) for key, value in out.items()}


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


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
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
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
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#7a6f52")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Calibrated Self-Consistency Action Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_calibrated_action_gate_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Calibrated Self-Consistency Action Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Probe table: `{args.probe_table}`.",
        "This evaluator makes no GPT, Gemini, Claude, or vLLM calls; it uses cached rows and selects action-score biases on validation.",
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
        "- The model predicts base/self/strong utilities from train rows, then validation selects self and strong score biases.",
        "- Frontier-cap variants keep only the highest-margin strong actions under validation-selected caps.",
        "- This is a deployable calibration layer over the existing self-consistency probe, not a held-out oracle.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
