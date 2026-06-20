from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_POOL = (
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
)
SELF_CONSISTENCY_MODEL = "qwen3-32b-awq-selfconsistency-n3-local"
ALPHAS = (0.1, 0.2, 0.3)
CONFIDENCE_THRESHOLDS = (0.55, 0.65, 0.75, 0.85)
MAX_SET_SIZES = (1,)
FALLBACK_KINDS = ("current_base",)


@dataclass(frozen=True)
class PolicyConfig:
    family: str
    score_column: str
    alpha: float | None
    confidence_threshold: float | None
    max_set_size: int
    fallback_kind: str

    @property
    def method(self) -> str:
        if self.alpha is not None:
            threshold = f"alpha{self.alpha:g}"
        else:
            threshold = f"conf{self.confidence_threshold:g}"
        return (
            f"{self.family}_{threshold}_set{self.max_set_size}"
            f"_fallback{self.fallback_kind}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cached Broad100 conformal answer-set and Confidence-Informed "
            "Self-Consistency probe policies."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--self-consistency",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "table_vllm_self_consistency_probe.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_conformal_answer_set_probe_policy"),
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_policy_variable_verifier_fusion/"
            "table_current_policy_variable_verifier_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default="base_current_policy")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = prepare_outputs(pd.read_parquet(args.outputs), lambda_cost=float(args.lambda_cost))
    self_consistency = load_self_consistency(args.self_consistency)
    matrix = build_matrix(outputs)
    base_series = load_current_base(args.base_query_choices, str(args.base_policy))

    standard_all, standard_selected, standard_choices = run_standard(outputs, self_consistency, matrix, base_series, args)
    heldout_all, heldout_selected = run_benchmark_heldout(outputs, self_consistency, matrix, base_series, args)
    heldout_summary = summarize_heldout(heldout_selected)
    state_table = build_answer_state_table(outputs, self_consistency)

    standard_all.to_csv(args.output_dir / "table_conformal_answer_set_policy_all.csv", index=False)
    standard_selected.to_csv(args.output_dir / "table_conformal_answer_set_policy_selected.csv", index=False)
    standard_choices.to_csv(args.output_dir / "table_conformal_answer_set_query_choices.csv", index=False)
    heldout_all.to_csv(args.output_dir / "table_conformal_answer_set_benchmark_heldout_all.csv", index=False)
    heldout_selected.to_csv(args.output_dir / "table_conformal_answer_set_benchmark_heldout_selected.csv", index=False)
    heldout_summary.to_csv(args.output_dir / "table_conformal_answer_set_benchmark_heldout_summary.csv", index=False)
    state_table.to_csv(args.output_dir / "table_answer_set_probe_states.csv", index=False)
    write_figure(args.output_dir, standard_selected, heldout_summary)
    write_memo(
        args.output_dir / "CONFORMAL_ANSWER_SET_PROBE_POLICY_MEMO.md",
        args,
        outputs,
        standard_selected,
        heldout_summary,
        heldout_selected,
    )
    print(f"Wrote conformal answer-set probe policy results to {args.output_dir}")


def prepare_outputs(outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    out = outputs.copy()
    out["query_id"] = out["query_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["answer_norm"] = out["parsed_answer"].map(normalized_answer)
    if "utility" not in out.columns:
        out["utility"] = (
            pd.to_numeric(out["quality_score"], errors="coerce").fillna(0.0)
            - float(lambda_cost)
            * pd.to_numeric(out["normalized_remote_cost"], errors="coerce").fillna(0.0)
        )
    for column in ["quality_score", "utility", "normalized_remote_cost", "latency_s", "cost_total_usd"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    for column in ["is_local", "is_frontier"]:
        out[column] = out[column].fillna(False).astype(bool)
    return out


def load_self_consistency(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["query_id", "answer_norms", "vote_frac"])
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        answer_norms = parse_json_list(getattr(row, "answer_norms_json", "[]"))
        sample_qualities = parse_json_list(getattr(row, "sample_qualities_json", "[]"))
        rows.append(
            {
                "query_id": str(row.query_id),
                "majority_answer_norm": normalized_answer(getattr(row, "majority_answer_norm", "")),
                "answer_norms": [normalized_answer(item) for item in answer_norms],
                "sample_qualities": [float(item) for item in sample_qualities if is_number(item)],
                "vote_frac": float(getattr(row, "vote_frac", 0.0) or 0.0),
                "vote_margin": float(getattr(row, "vote_margin", 0.0) or 0.0),
                "vote_entropy": float(getattr(row, "vote_entropy", 0.0) or 0.0),
                "n_samples": int(getattr(row, "n_samples", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def run_standard(
    outputs: pd.DataFrame,
    self_consistency: pd.DataFrame,
    matrix: dict[str, Any],
    base_series: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    query_meta = query_metadata(outputs)
    train_ids = query_meta.loc[query_meta["split"].eq("train"), "query_id"].tolist()
    val_ids = query_meta.loc[query_meta["split"].eq("val"), "query_id"].tolist()
    test_ids = query_meta.loc[query_meta["split"].eq("test"), "query_id"].tolist()
    all_rows, choices = evaluate_scenario(
        outputs,
        self_consistency,
        matrix,
        base_series,
        train_ids,
        val_ids,
        test_ids,
        args,
        scenario="standard",
        heldout_benchmark="",
    )
    selected = select_by_validation(all_rows)
    selected_choices = choices_for_selected(selected, choices, outputs)
    return all_rows, selected, selected_choices


def run_benchmark_heldout(
    outputs: pd.DataFrame,
    self_consistency: pd.DataFrame,
    matrix: dict[str, Any],
    base_series: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_meta = query_metadata(outputs)
    all_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    for heldout in sorted(query_meta["benchmark"].unique().tolist()):
        train_ids = query_meta.loc[
            query_meta["split"].eq("train") & query_meta["benchmark"].ne(heldout), "query_id"
        ].tolist()
        val_ids = query_meta.loc[
            query_meta["split"].eq("val") & query_meta["benchmark"].ne(heldout), "query_id"
        ].tolist()
        test_ids = query_meta.loc[
            query_meta["split"].eq("test") & query_meta["benchmark"].eq(heldout), "query_id"
        ].tolist()
        rows, _choices = evaluate_scenario(
            outputs,
            self_consistency,
            matrix,
            base_series,
            train_ids,
            val_ids,
            test_ids,
            args,
            scenario="benchmark_heldout",
            heldout_benchmark=str(heldout),
        )
        all_frames.append(rows)
        selected_frames.append(select_by_validation(rows))
    return pd.concat(all_frames, ignore_index=True), pd.concat(selected_frames, ignore_index=True)


def evaluate_scenario(
    outputs: pd.DataFrame,
    self_consistency: pd.DataFrame,
    matrix: dict[str, Any],
    base_series: pd.Series,
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    *,
    scenario: str,
    heldout_benchmark: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    calibrator = fit_calibrator(outputs, self_consistency, train_ids)
    configs = candidate_configs()
    rows: list[dict[str, Any]] = []
    choices: dict[tuple[str, str], pd.DataFrame] = {}
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        rows.extend(baseline_rows(matrix, base_series, train_ids, ids, split, scenario, heldout_benchmark, args))
        scored_groups = build_group_table(outputs, self_consistency, ids, calibrator["model_weights"])
        scored_groups = score_groups(
            scored_groups,
            calibrator["support_reliability"],
            fallback=float(calibrator["support_prior"]),
        )
        groups_by_query = {str(query_id): group.copy() for query_id, group in scored_groups.groupby("query_id", sort=False)}
        for config in configs:
            selected = apply_policy_from_groups(groups_by_query, ids, calibrator, config, base_series)
            selected_series = selected.set_index("query_id")["model_id"].astype(str)
            rows.append(metric_row(matrix, ids, selected_series, config.method, config.family, split, scenario, heldout_benchmark, args))
            choices[(config.method, split)] = selected
    table = pd.DataFrame(rows).sort_values(["scenario", "heldout_benchmark", "family", "method", "eval_split"])
    return table, choices


def candidate_configs() -> list[PolicyConfig]:
    configs: list[PolicyConfig] = []
    for fallback in FALLBACK_KINDS:
        for max_set_size in MAX_SET_SIZES:
            for alpha in ALPHAS:
                configs.append(
                    PolicyConfig(
                        family="conformal_answer_set",
                        score_column="local_cp_score",
                        alpha=float(alpha),
                        confidence_threshold=None,
                        max_set_size=max_set_size,
                        fallback_kind=fallback,
                    )
                )
                configs.append(
                    PolicyConfig(
                        family="cisc_conformal_answer_set",
                        score_column="cisc_score",
                        alpha=float(alpha),
                        confidence_threshold=None,
                        max_set_size=max_set_size,
                        fallback_kind=fallback,
                    )
                )
            for threshold in CONFIDENCE_THRESHOLDS:
                configs.append(
                    PolicyConfig(
                        family="cisc_confidence_threshold",
                        score_column="cisc_score",
                        alpha=None,
                        confidence_threshold=float(threshold),
                        max_set_size=max_set_size,
                        fallback_kind=fallback,
                    )
                )
                configs.append(
                    PolicyConfig(
                        family="self_consistency_majority_threshold",
                        score_column="sc_majority_score",
                        alpha=None,
                        confidence_threshold=float(threshold),
                        max_set_size=max_set_size,
                        fallback_kind=fallback,
                    )
                )
    return configs


def fit_calibrator(outputs: pd.DataFrame, self_consistency: pd.DataFrame, train_ids: list[str]) -> dict[str, Any]:
    train = outputs[outputs["query_id"].isin(train_ids)].copy()
    local_train = train[train["model_id"].isin(LOCAL_POOL)].copy()
    global_quality = float(local_train["quality_score"].mean()) if not local_train.empty else 0.0
    model_weights: dict[str, float] = {}
    for model_id in LOCAL_POOL:
        group = local_train[local_train["model_id"].eq(model_id)]
        model_weights[model_id] = float((group["quality_score"].sum() + 2.0 * global_quality) / (len(group) + 2.0))
    utility_prior = train.groupby("model_id")["utility"].mean().astype(float).to_dict()
    fallback = {
        "global_best": str(train.groupby("model_id")["utility"].mean().idxmax()),
        "frontier_best": best_model(train[train["is_frontier"]]),
        "local_best": best_model(local_train),
    }
    train_groups = build_group_table(outputs, self_consistency, train_ids, model_weights)
    support_prior = float(train_groups["correct"].mean()) if not train_groups.empty else global_quality
    support_reliability: dict[int, float] = {}
    for support, group in train_groups.groupby("support"):
        support_reliability[int(support)] = float((group["correct"].sum() + 2.0 * support_prior) / (len(group) + 2.0))
    train_groups = score_groups(train_groups, support_reliability)
    qhats = {
        "local_cp_score": conformal_thresholds(train_groups, "local_cp_score"),
        "cisc_score": conformal_thresholds(train_groups, "cisc_score"),
    }
    return {
        "model_weights": model_weights,
        "utility_prior": utility_prior,
        "support_reliability": support_reliability,
        "support_prior": support_prior,
        "fallback": fallback,
        "qhats": qhats,
    }


def apply_policy_from_groups(
    groups_by_query: dict[str, pd.DataFrame],
    ids: list[str],
    calibrator: dict[str, Any],
    config: PolicyConfig,
    base_series: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id in ids:
        group = groups_by_query.get(str(query_id), pd.DataFrame())
        if config.fallback_kind == "current_base" and str(query_id) in base_series.index:
            fallback = str(base_series.loc[str(query_id)])
        else:
            fallback = calibrator["fallback"].get(config.fallback_kind, calibrator["fallback"]["global_best"])
        selected_model = fallback
        selected_answer = ""
        selected_score = 0.0
        prediction_set_size = 0
        used_answer_set = False
        if not group.empty:
            ranked = group.sort_values(
                [config.score_column, "support", "cisc_score"], ascending=[False, False, False]
            )
            top = ranked.iloc[0]
            selected_score = float(top[config.score_column])
            selected_answer = str(top["answer_norm"])
            if config.alpha is not None:
                qhat = float(calibrator["qhats"].get(config.score_column, {}).get(float(config.alpha), 1.0))
                score_threshold = max(0.0, 1.0 - qhat)
            else:
                score_threshold = float(config.confidence_threshold or 0.0)
            prediction_set = ranked[ranked[config.score_column].astype(float) >= score_threshold]
            prediction_set_size = int(len(prediction_set))
            used_answer_set = bool(0 < prediction_set_size <= int(config.max_set_size) and selected_score >= score_threshold)
            if used_answer_set:
                selected_model = choose_local_action(top, calibrator["utility_prior"])
        rows.append(
            {
                "query_id": str(query_id),
                "model_id": str(selected_model),
                "selected_answer_norm": selected_answer,
                "answer_set_score": selected_score,
                "prediction_set_size": prediction_set_size,
                "used_answer_set": used_answer_set,
                "fallback_kind": config.fallback_kind,
            }
        )
    return pd.DataFrame(rows)


def build_group_table(
    outputs: pd.DataFrame,
    self_consistency: pd.DataFrame,
    ids: list[str],
    model_weights: dict[str, float],
) -> pd.DataFrame:
    local = outputs[outputs["query_id"].isin(ids) & outputs["model_id"].isin(LOCAL_POOL)].copy()
    sc_by_query = {str(row.query_id): row for row in self_consistency.itertuples(index=False)}
    rows: list[dict[str, Any]] = []
    for query_id, query_rows in local.groupby("query_id", sort=False):
        query_id = str(query_id)
        valid = query_rows[query_rows["answer_norm"].astype(str).ne("")]
        if valid.empty:
            continue
        total_local_weight = sum(float(model_weights.get(str(row.model_id), 0.0)) for row in valid.itertuples(index=False))
        sc_counts: dict[str, int] = defaultdict(int)
        sc = sc_by_query.get(query_id)
        if sc is not None:
            for answer in getattr(sc, "answer_norms", []):
                if answer:
                    sc_counts[str(answer)] += 1
        sc_total_weight = float(model_weights.get(SELF_CONSISTENCY_MODEL, 0.0))
        per_sample_weight = sc_total_weight / max(1, sum(sc_counts.values()))
        for answer, group in valid.groupby("answer_norm", sort=False):
            answer = str(answer)
            models = sorted(group["model_id"].astype(str).unique().tolist())
            local_weight = sum(float(model_weights.get(model, 0.0)) for model in models)
            sc_weight = float(sc_counts.get(answer, 0)) * per_sample_weight
            total_weight = max(total_local_weight + sc_total_weight, 1e-12)
            rows.append(
                {
                    "query_id": query_id,
                    "split": str(group.iloc[0]["split"]),
                    "benchmark": str(group.iloc[0]["benchmark"]),
                    "answer_norm": answer,
                    "support": int(len(models)),
                    "models_json": json.dumps(models),
                    "local_weight_frac": float(local_weight / max(total_local_weight, 1e-12)),
                    "cisc_weight_frac": float((local_weight + sc_weight) / total_weight),
                    "sc_sample_count": int(sc_counts.get(answer, 0)),
                    "sc_majority_score": float(getattr(sc, "vote_frac", 0.0) or 0.0)
                    if sc is not None and answer == str(getattr(sc, "majority_answer_norm", ""))
                    else 0.0,
                    "correct": float(group["quality_score"].max()),
                }
            )
    return pd.DataFrame(rows)


def score_groups(groups: pd.DataFrame, support_reliability: dict[int, float], fallback: float = 0.0) -> pd.DataFrame:
    if groups.empty:
        return groups
    out = groups.copy()
    out["support_reliability"] = [
        float(support_reliability.get(int(support), fallback)) for support in out["support"].tolist()
    ]
    out["local_cp_score"] = (
        0.7 * out["local_weight_frac"].astype(float)
        + 0.3 * out["support_reliability"].astype(float)
    ).clip(0.0, 1.0)
    out["cisc_score"] = (
        0.7 * out["cisc_weight_frac"].astype(float)
        + 0.3 * out["support_reliability"].astype(float)
    ).clip(0.0, 1.0)
    return out


def conformal_thresholds(train_groups: pd.DataFrame, score_column: str) -> dict[float, float]:
    if train_groups.empty:
        return {float(alpha): 1.0 for alpha in ALPHAS}
    nonconformity: list[float] = []
    for _query_id, group in train_groups.groupby("query_id", sort=False):
        correct = group[group["correct"].astype(float) > 0.5]
        score = float(correct[score_column].max()) if not correct.empty else 0.0
        nonconformity.append(float(1.0 - score))
    values = np.sort(np.asarray(nonconformity, dtype=float))
    n = len(values)
    qhats: dict[float, float] = {}
    for alpha in ALPHAS:
        rank = int(math.ceil((n + 1) * (1.0 - float(alpha)))) - 1
        rank = min(max(rank, 0), n - 1)
        qhats[float(alpha)] = float(values[rank])
    return qhats


def choose_local_action(row: pd.Series, utility_prior: dict[str, float]) -> str:
    models = json.loads(str(row["models_json"]))
    best = str(models[0])
    best_score = -1e9
    for model in models:
        score = float(utility_prior.get(str(model), -1e9))
        if score > best_score:
            best = str(model)
            best_score = score
    return best


def baseline_rows(
    matrix: dict[str, Any],
    base_series: pd.Series,
    train_ids: list[str],
    ids: list[str],
    split: str,
    scenario: str,
    heldout: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    train_utility = matrix["utility"].loc[train_ids]
    global_best = str(train_utility.mean(axis=0).idxmax())
    local_cols = [model for model in matrix["model_ids"] if model in LOCAL_POOL]
    frontier_cols = [model for model in matrix["model_ids"] if bool(matrix["frontier"][model].any())]
    local_best = str(train_utility[local_cols].mean(axis=0).idxmax())
    frontier_best = str(train_utility[frontier_cols].mean(axis=0).idxmax()) if frontier_cols else global_best
    rows = [
        metric_row(
            matrix,
            ids,
            base_series.reindex(ids).astype(str),
            "current_base",
            "current_base",
            split,
            scenario,
            heldout,
            args,
        ),
        metric_row(matrix, ids, pd.Series(global_best, index=ids), "global_best_single", "global_best_single", split, scenario, heldout, args),
        metric_row(matrix, ids, pd.Series(local_best, index=ids), "local_best_single", "local_best_single", split, scenario, heldout, args),
        metric_row(matrix, ids, pd.Series(frontier_best, index=ids), "frontier_best_single", "frontier_best_single", split, scenario, heldout, args),
    ]
    oracle_actions = matrix["oracle_action"].reindex(ids).astype(str)
    rows.append(metric_row(matrix, ids, oracle_actions, "query_oracle", "query_oracle_upper_bound", split, scenario, heldout, args, diagnostic=True))
    local_oracle = matrix["utility"].loc[ids, local_cols].idxmax(axis=1).astype(str)
    rows.append(metric_row(matrix, ids, local_oracle, "local_query_oracle", "local_query_oracle_upper_bound", split, scenario, heldout, args, diagnostic=True))
    return rows


def metric_row(
    matrix: dict[str, Any],
    ids: list[str],
    selected: pd.Series,
    method: str,
    family: str,
    split: str,
    scenario: str,
    heldout: str,
    args: argparse.Namespace,
    **extra: Any,
) -> dict[str, Any]:
    ids = [str(item) for item in ids]
    selected = selected.reindex(ids).astype(str)
    utility = matrix["utility"].reindex(ids)
    quality = matrix["quality"].reindex(ids)
    cost = matrix["cost"].reindex(ids)
    frontier = matrix["frontier"].reindex(ids)
    selected_utility = np.array([float(utility.loc[qid, model]) for qid, model in selected.items()])
    selected_quality = np.array([float(quality.loc[qid, model]) for qid, model in selected.items()])
    selected_cost = np.array([float(cost.loc[qid, model]) for qid, model in selected.items()])
    selected_frontier = np.array([bool(frontier.loc[qid, model]) for qid, model in selected.items()])
    oracle_utility = matrix["oracle_utility"].reindex(ids).astype(float).to_numpy()
    oracle_quality = matrix["oracle_quality"].reindex(ids).astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(selected_utility, int(args.bootstrap_samples), int(args.seed))
    return {
        "scenario": scenario,
        "heldout_benchmark": heldout,
        "method": method,
        "family": family,
        "eval_split": split,
        "n_queries": int(len(ids)),
        "mean_quality": float(selected_quality.mean()) if ids else float("nan"),
        "mean_utility": float(selected_utility.mean()) if ids else float("nan"),
        "mean_utility_ci_low": ci_low,
        "mean_utility_ci_high": ci_high,
        "mean_normalized_cost": float(selected_cost.mean()) if ids else float("nan"),
        "oracle_mean_quality": float(oracle_quality.mean()) if ids else float("nan"),
        "oracle_mean_utility": float(oracle_utility.mean()) if ids else float("nan"),
        "oracle_utility_ratio": float(selected_utility.mean() / max(float(oracle_utility.mean()), 1e-12)) if ids else float("nan"),
        "utility_gap_to_oracle": float(oracle_utility.mean() - selected_utility.mean()) if ids else float("nan"),
        "quality_gap_to_oracle": float(oracle_quality.mean() - selected_quality.mean()) if ids else float("nan"),
        "frontier_call_rate": float(selected_frontier.mean()) if ids else float("nan"),
        "selected_models_json": json.dumps(selected.value_counts().sort_index().to_dict(), sort_keys=True),
        **extra,
    }


def select_by_validation(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    group_cols = ["scenario", "heldout_benchmark", "family"]
    for (_scenario, _heldout, family), group in table.groupby(group_cols, dropna=False):
        if str(family).endswith("_upper_bound"):
            for split in ["val", "test"]:
                subset = group[group["eval_split"].eq(split)]
                if not subset.empty:
                    item = subset.sort_values("mean_utility", ascending=False).iloc[0].copy()
                    item["selection_rule"] = f"diagnostic_{split}"
                    rows.append(item)
            continue
        val = group[group["eval_split"].eq("val")].copy()
        test = group[group["eval_split"].eq("test")].copy()
        if val.empty:
            continue
        for rule, candidates in {
            "val_best_mean_utility": val,
            "val_best_utility_frontier_le_0.4": val[val["frontier_call_rate"] <= 0.4],
        }.items():
            if candidates.empty:
                continue
            best = candidates.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).iloc[0].copy()
            best["selection_rule"] = rule
            rows.append(best)
            match = test[test["method"].astype(str).eq(str(best["method"]))]
            if not match.empty:
                test_row = match.iloc[0].copy()
                test_row["selection_rule"] = f"{rule}_test"
                rows.append(test_row)
    return pd.DataFrame(rows).sort_values(["scenario", "heldout_benchmark", "family", "selection_rule", "eval_split"])


def choices_for_selected(selected: pd.DataFrame, choices: dict[tuple[str, str], pd.DataFrame], outputs: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    selected_methods = set(selected.loc[selected["eval_split"].eq("test"), "method"].astype(str).tolist())
    rows: list[pd.DataFrame] = []
    info = outputs.drop_duplicates("query_id")[["query_id", "query_text", "benchmark", "domain", "metric"]]
    for (method, split), frame in choices.items():
        if split != "test" or method not in selected_methods:
            continue
        merged = frame.merge(info, on="query_id", how="left")
        merged = merged.merge(
            outputs[["query_id", "model_id", "quality_score", "utility", "normalized_remote_cost", "is_frontier", "parsed_answer"]],
            on=["query_id", "model_id"],
            how="left",
        )
        merged["method"] = method
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_heldout(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    test = selected[selected["eval_split"].eq("test")].copy()
    if test.empty:
        return pd.DataFrame()
    return (
        test.groupby(["family", "selection_rule"], as_index=False)
        .agg(
            mean_heldout_quality=("mean_quality", "mean"),
            mean_heldout_utility=("mean_utility", "mean"),
            mean_heldout_oracle_ratio=("oracle_utility_ratio", "mean"),
            mean_normalized_cost=("mean_normalized_cost", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
            n_heldout_benchmarks=("heldout_benchmark", "nunique"),
        )
        .sort_values("mean_heldout_utility", ascending=False)
        .reset_index(drop=True)
    )


def build_answer_state_table(outputs: pd.DataFrame, self_consistency: pd.DataFrame) -> pd.DataFrame:
    ids = query_metadata(outputs)["query_id"].tolist()
    local = outputs[outputs["model_id"].isin(LOCAL_POOL)]
    weights = local.groupby("model_id")["quality_score"].mean().astype(float).to_dict()
    groups = build_group_table(outputs, self_consistency, ids, weights)
    groups = score_groups(groups, {})
    if groups.empty:
        return groups
    summary = (
        groups.sort_values(["query_id", "cisc_score"], ascending=[True, False])
        .groupby("query_id", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return summary


def build_matrix(outputs: pd.DataFrame) -> dict[str, Any]:
    model_ids = sorted(outputs["model_id"].unique().tolist())
    utility = outputs.pivot(index="query_id", columns="model_id", values="utility").reindex(columns=model_ids).fillna(-1e9)
    quality = outputs.pivot(index="query_id", columns="model_id", values="quality_score").reindex(columns=model_ids).fillna(0.0)
    cost = outputs.pivot(index="query_id", columns="model_id", values="normalized_remote_cost").reindex(columns=model_ids).fillna(0.0)
    frontier = outputs.pivot(index="query_id", columns="model_id", values="is_frontier").reindex(columns=model_ids).fillna(False).astype(bool)
    oracle_idx = utility.to_numpy().argmax(axis=1)
    oracle_action = pd.Series([model_ids[int(idx)] for idx in oracle_idx], index=utility.index)
    oracle_utility = pd.Series(utility.to_numpy()[np.arange(len(utility)), oracle_idx], index=utility.index)
    oracle_quality = pd.Series(quality.to_numpy()[np.arange(len(quality)), oracle_idx], index=quality.index)
    return {
        "model_ids": model_ids,
        "utility": utility,
        "quality": quality,
        "cost": cost,
        "frontier": frontier,
        "oracle_action": oracle_action,
        "oracle_utility": oracle_utility,
        "oracle_quality": oracle_quality,
    }


def load_current_base(path: Path, policy: str) -> pd.Series:
    frame = pd.read_csv(path)
    frame = frame[frame["policy"].astype(str).eq(policy)].copy()
    if frame.empty:
        raise RuntimeError(f"Base policy {policy!r} not found in {path}")
    selected_col = "selected_model" if "selected_model" in frame.columns else "selected_model_id"
    return frame.set_index("query_id")[selected_col].astype(str)


def query_metadata(outputs: pd.DataFrame) -> pd.DataFrame:
    return (
        outputs.sort_values(["query_id", "model_id"])
        .drop_duplicates("query_id")
        [["query_id", "query_text", "split", "benchmark", "domain", "metric"]]
        .reset_index(drop=True)
    )


def best_model(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    return str(frame.groupby("model_id")["utility"].mean().idxmax())


def normalized_answer(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text or text in {"nan", "none", "null", "no_code"} or text.startswith("failed"):
        return ""
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    return text.removeprefix("answer:").strip().strip("$").strip()


def parse_json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value))
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_figure(out_dir: Path, selected: pd.DataFrame, heldout_summary: pd.DataFrame) -> None:
    test = selected[
        selected["eval_split"].eq("test")
        & selected["selection_rule"].astype(str).str.endswith("_test")
        & ~selected["family"].astype(str).str.endswith("_upper_bound")
    ].copy()
    if not test.empty:
        plot = test.sort_values("mean_utility", ascending=False).head(16)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(plot["family"] + "\n" + plot["selection_rule"], plot["mean_utility"], color="#58706a")
        ax.set_xlabel("Held-out test utility")
        ax.set_title("Conformal/CISC Answer-Set Probe Policies")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_conformal_answer_set_standard_utility.pdf")
        plt.close(fig)
    if not heldout_summary.empty:
        plot = heldout_summary.sort_values("mean_heldout_utility", ascending=False).head(16)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(plot["family"] + "\n" + plot["selection_rule"], plot["mean_heldout_utility"], color="#6d6f91")
        ax.set_xlabel("Mean benchmark-heldout utility")
        ax.set_title("Benchmark-Heldout Transfer")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_conformal_answer_set_benchmark_heldout_utility.pdf")
        plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    outputs: pd.DataFrame,
    selected: pd.DataFrame,
    heldout_summary: pd.DataFrame,
    heldout_selected: pd.DataFrame,
) -> None:
    standard_cols = [
        "family",
        "method",
        "eval_split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "mean_normalized_cost",
        "frontier_call_rate",
        "selection_rule",
    ]
    heldout_cols = [
        "heldout_benchmark",
        "family",
        "method",
        "eval_split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "selection_rule",
    ]
    lines = [
        "# Conformal Answer-Set Probe Policy",
        "",
        "This cached Broad100 experiment ports two related uncertainty-routing ideas into ProbeCode:",
        "conformal/prediction-set routing and Confidence-Informed Self-Consistency.",
        "No provider calls, vLLM calls, fine-tuning, or benchmark-specific verifier rules are used.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/211_conformal_answer_set_probe_policy.py",
        "```",
        "",
        "## Inputs",
        "",
        f"- Outputs: `{args.outputs}`",
        f"- Self-consistency table: `{args.self_consistency}`",
        f"- Query rows: `{outputs['query_id'].nunique()}`",
        f"- Actions: `{outputs['model_id'].nunique()}`",
        "",
        "## Method",
        "",
        "- `conformal_answer_set`: calibrates a train-only nonconformity threshold over local answer-group confidence.",
        "- `cisc_conformal_answer_set`: uses the same conformal rule, but scores answer groups with calibrated local-model weights plus self-consistency sample support.",
        "- `cisc_confidence_threshold`: uses CISC-weighted top-answer confidence with validation-selected thresholds.",
        "- `self_consistency_majority_threshold`: baseline that trusts the cached self-consistency majority only above a validation-selected vote fraction.",
        "",
        "## Standard Selected Rows",
        "",
        markdown_table(selected[selected["eval_split"].isin(["val", "test"])][standard_cols]),
        "",
        "## Benchmark-Heldout Summary",
        "",
        markdown_table(heldout_summary),
        "",
        "## Benchmark-Heldout Selected Test Rows",
        "",
        markdown_table(heldout_selected[heldout_selected["eval_split"].eq("test")][heldout_cols]),
        "",
        "## Artifacts",
        "",
        f"- All standard rows: `{path.parent / 'table_conformal_answer_set_policy_all.csv'}`",
        f"- Selected standard rows: `{path.parent / 'table_conformal_answer_set_policy_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_conformal_answer_set_query_choices.csv'}`",
        f"- Heldout all rows: `{path.parent / 'table_conformal_answer_set_benchmark_heldout_all.csv'}`",
        f"- Heldout selected rows: `{path.parent / 'table_conformal_answer_set_benchmark_heldout_selected.csv'}`",
        f"- Heldout summary: `{path.parent / 'table_conformal_answer_set_benchmark_heldout_summary.csv'}`",
        f"- Answer state table: `{path.parent / 'table_answer_set_probe_states.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
