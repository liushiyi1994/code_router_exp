from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.data.text_features import build_hashing_embeddings
from routecode.probes.policies import (
    belief_entropy,
    belief_margin,
    expected_model_utility_from_belief,
    select_models_from_belief,
)


@dataclass(frozen=True)
class ProbeRoutePPConfig:
    k: int = 16
    alpha: float = 0.5
    beta: float = 0.0
    lambda_cost: float = 0.35
    lambda_latency: float = 0.05
    train_split: str = "train"
    eval_split: str = "test"
    embedding_features: int = 128
    state_temperature: float = 1.0
    probe_knn_k: int = 5
    probe_blend_weight: float = 0.65
    entropy_threshold: float | None = None
    margin_threshold: float = 0.20
    voi_min_gain: float = 0.0
    probe_cost_utility: float = 0.0
    probe_latency_s: float = 0.0
    random_state: int = 0


@dataclass(frozen=True)
class ProbeRoutePPArtifacts:
    state_model_utility_table: pd.DataFrame
    routing_decisions: pd.DataFrame
    main_eval: pd.DataFrame
    cost_latency_summary: pd.DataFrame
    calibration_table: pd.DataFrame
    before_beliefs: pd.DataFrame
    after_beliefs: pd.DataFrame
    metadata: dict[str, Any]


def build_proberoutepp_artifacts(scored_outputs: pd.DataFrame, config: ProbeRoutePPConfig) -> ProbeRoutePPArtifacts:
    scored = prepare_scored_outputs(scored_outputs, config)
    query_info = _query_info(scored)
    embeddings = build_hashing_embeddings(query_info, n_features=config.embedding_features)
    train_ids, eval_ids = _split_query_ids(scored, config)
    model_ids = sorted(scored["model_id"].astype(str).unique())

    train_scored = scored[scored["query_id"].isin(train_ids)].copy()
    eval_scored = scored[scored["query_id"].isin(eval_ids)].copy()
    train_utility = _pivot_metric(train_scored, "utility", model_ids)
    eval_utility = _pivot_metric(eval_scored, "utility", model_ids, fill_from=train_utility)
    eval_quality = _pivot_metric(eval_scored, "quality_score", model_ids, fill_from=train_utility)
    eval_cost = _pivot_metric(eval_scored, "cost_total_usd", model_ids, fill_from=train_utility, fill_value=0.0)
    eval_latency = _pivot_metric(eval_scored, "latency_s", model_ids, fill_from=train_utility, fill_value=0.0)

    codebook = PredictabilityConstrainedRouteCode(
        n_labels=config.k,
        alpha=config.alpha,
        beta=config.beta,
        random_state=config.random_state,
    ).fit(query_info.loc[train_utility.index], train_utility, embeddings.loc[train_utility.index])
    if codebook.train_labels_ is None:
        raise RuntimeError("ProbeRoute++ state learner did not produce train labels")
    state_labels = [_state_label(label) for label in range(codebook.effective_labels)]
    state_model_table = _build_state_model_table(
        train_scored=train_scored,
        train_labels=codebook.train_labels_.map(_state_label),
        state_labels=state_labels,
        model_ids=model_ids,
    )
    state_model_utility = _wide_state_table(state_model_table, "mean_utility", state_labels, model_ids)
    state_model_quality = _wide_state_table(state_model_table, "mean_quality", state_labels, model_ids)
    state_model_cost = _wide_state_table(state_model_table, "mean_remote_cost_usd", state_labels, model_ids)
    state_model_latency = _wide_state_table(state_model_table, "mean_latency_s", state_labels, model_ids)

    before_beliefs = _label_distribution_to_states(
        codebook.predict_label_distribution(embeddings.loc[eval_utility.index]),
        state_labels,
        temperature=config.state_temperature,
    )
    knn_beliefs = _knn_state_beliefs(
        train_embeddings=embeddings.loc[train_utility.index],
        train_labels=codebook.train_labels_.map(_state_label),
        target_embeddings=embeddings.loc[eval_utility.index],
        state_labels=state_labels,
        k=config.probe_knn_k,
    )
    after_beliefs = _blend_and_normalize(before_beliefs, knn_beliefs, config.probe_blend_weight)

    before_expected = expected_model_utility_from_belief(before_beliefs, state_model_utility)
    after_expected = expected_model_utility_from_belief(after_beliefs, state_model_utility)
    predicted_gain = after_expected.max(axis=1) - before_expected.max(axis=1)
    entropy = before_beliefs.apply(belief_entropy, axis=1)
    margin = before_beliefs.apply(belief_margin, axis=1)
    entropy_threshold = (
        float(config.entropy_threshold)
        if config.entropy_threshold is not None
        else 0.55 * math.log2(max(len(state_labels), 2))
    )
    threshold_probe = ((entropy >= entropy_threshold) | (margin <= float(config.margin_threshold))).rename(
        "threshold_probe"
    )
    voi_probe = ((predicted_gain - float(config.probe_cost_utility)) > float(config.voi_min_gain)).rename("voi_probe")

    policy_specs = {
        "proberoutepp_no_probe": pd.Series(False, index=eval_utility.index),
        "proberoutepp_threshold_probe": threshold_probe.reindex(eval_utility.index).fillna(False).astype(bool),
        "proberoutepp_voi_probe": voi_probe.reindex(eval_utility.index).fillna(False).astype(bool),
    }
    routing_decisions = pd.concat(
        [
            _routing_decisions_for_policy(
                method=method,
                probe_used=probe_used,
                before_beliefs=before_beliefs,
                after_beliefs=after_beliefs,
                state_model_utility=state_model_utility,
                state_model_quality=state_model_quality,
                state_model_cost=state_model_cost,
                state_model_latency=state_model_latency,
                eval_scored=eval_scored,
                probe_latency_s=config.probe_latency_s,
                probe_cost_utility=config.probe_cost_utility,
            )
            for method, probe_used in policy_specs.items()
        ],
        ignore_index=True,
    )
    main_eval = _build_main_eval(
        scored=scored,
        eval_scored=eval_scored,
        query_info=query_info.loc[eval_utility.index],
        train_scored=train_scored,
        eval_utility=eval_utility,
        before_beliefs=before_beliefs,
        after_beliefs=after_beliefs,
        knn_beliefs=knn_beliefs,
        state_model_utility=state_model_utility,
        routing_decisions=routing_decisions,
        config=config,
    )
    cost_latency = _cost_latency_summary(scored)
    calibration_table = _calibration_placeholder(state_labels)
    metadata = {
        "k_requested": int(config.k),
        "k_effective": int(codebook.effective_labels),
        "alpha": float(config.alpha),
        "beta": float(config.beta),
        "lambda_cost": float(config.lambda_cost),
        "lambda_latency": float(config.lambda_latency),
        "train_queries": int(len(train_utility)),
        "eval_queries": int(len(eval_utility)),
        "model_count": int(len(model_ids)),
        "state_belief_invariant": "query/probe -> latent route-state belief -> model",
        **codebook.objective_summary(),
    }
    return ProbeRoutePPArtifacts(
        state_model_utility_table=state_model_table,
        routing_decisions=routing_decisions,
        main_eval=main_eval,
        cost_latency_summary=cost_latency,
        calibration_table=calibration_table,
        before_beliefs=before_beliefs,
        after_beliefs=after_beliefs,
        metadata=metadata,
    )


def prepare_scored_outputs(scored_outputs: pd.DataFrame, config: ProbeRoutePPConfig) -> pd.DataFrame:
    required = ["query_id", "model_id", "quality_score", "cost_total_usd", "latency_s"]
    missing = [column for column in required if column not in scored_outputs.columns]
    if missing:
        raise ValueError(f"scored_outputs missing required columns: {missing}")
    scored = scored_outputs.copy()
    if "status" in scored.columns:
        scored = scored[scored["status"].astype(str).eq("success")].copy()
    if scored.empty:
        raise ValueError("No successful scored output rows available")
    scored["query_id"] = scored["query_id"].astype(str)
    scored["model_id"] = scored["model_id"].astype(str)
    if "query_text" not in scored.columns:
        scored["query_text"] = scored["query_id"]
    if "benchmark" not in scored.columns:
        scored["benchmark"] = scored.get("domain", "unknown")
    if "domain" not in scored.columns:
        scored["domain"] = scored["benchmark"]
    if "is_frontier" not in scored.columns:
        scored["is_frontier"] = scored["cost_total_usd"].astype(float) > 0.0
    if "is_local" not in scored.columns:
        scored["is_local"] = ~scored["is_frontier"].astype(bool)
    scored["quality_score"] = pd.to_numeric(scored["quality_score"], errors="coerce").fillna(0.0)
    scored["cost_total_usd"] = pd.to_numeric(scored["cost_total_usd"], errors="coerce").fillna(0.0)
    scored["latency_s"] = pd.to_numeric(scored["latency_s"], errors="coerce").fillna(0.0)
    scored = _ensure_train_eval_split(scored, config)
    cost_norm = _cost_normalizer(scored)
    latency_norm = _latency_normalizer(scored)
    scored["normalized_remote_cost"] = scored["cost_total_usd"] / cost_norm
    scored["normalized_latency"] = scored["latency_s"] / latency_norm
    scored["utility"] = (
        scored["quality_score"]
        - float(config.lambda_cost) * scored["normalized_remote_cost"]
        - float(config.lambda_latency) * scored["normalized_latency"]
    )
    return scored


def write_proberoutepp_outputs(artifacts: ProbeRoutePPArtifacts, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_model_utility_table": out_dir / "state_model_utility_table.parquet",
        "routing_decisions": out_dir / "routing_decisions.parquet",
        "table_main_eval": out_dir / "table_main_eval.csv",
        "cost_latency_summary": out_dir / "cost_latency_summary.csv",
        "table_calibration": out_dir / "table_calibration.csv",
        "run_report": out_dir / "RUN_REPORT.md",
        "metadata": out_dir / "proberoutepp_metadata.json",
        "fig_quality_cost_frontier": out_dir / "fig_quality_cost_frontier.pdf",
        "fig_latency_breakdown": out_dir / "fig_latency_breakdown.pdf",
    }
    artifacts.state_model_utility_table.to_parquet(paths["state_model_utility_table"], index=False)
    artifacts.routing_decisions.to_parquet(paths["routing_decisions"], index=False)
    artifacts.main_eval.to_csv(paths["table_main_eval"], index=False)
    artifacts.cost_latency_summary.to_csv(paths["cost_latency_summary"], index=False)
    artifacts.calibration_table.to_csv(paths["table_calibration"], index=False)
    paths["metadata"].write_text(json.dumps(artifacts.metadata, indent=2, sort_keys=True), encoding="utf-8")
    _write_plots(artifacts.main_eval, paths["fig_quality_cost_frontier"], paths["fig_latency_breakdown"])
    paths["run_report"].write_text(_run_report(artifacts), encoding="utf-8")
    return paths


def _query_info(scored: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ["query_id", "query_text", "benchmark", "domain", "split"] if column in scored.columns]
    info = scored[columns].drop_duplicates("query_id").set_index("query_id").sort_index()
    info.index = info.index.astype(str)
    return info


def _ensure_train_eval_split(scored: pd.DataFrame, config: ProbeRoutePPConfig) -> pd.DataFrame:
    scored = scored.copy()
    existing = set(scored.get("split", pd.Series(dtype=str)).astype(str).unique())
    if config.train_split in existing and config.eval_split in existing:
        scored["split"] = scored["split"].astype(str)
        return scored
    query_ids = sorted(scored["query_id"].astype(str).unique())
    n_queries = len(query_ids)
    if n_queries < 2:
        raise ValueError("ProbeRoute++ needs at least two queries to create train/eval splits")
    train_count = max(1, int(0.6 * n_queries))
    if train_count >= n_queries:
        train_count = n_queries - 1
    validation_count = int(0.2 * n_queries) if n_queries - train_count > 1 else 0
    split_map: dict[str, str] = {}
    for idx, query_id in enumerate(query_ids):
        if idx < train_count:
            split_map[query_id] = config.train_split
        elif idx < train_count + validation_count:
            split_map[query_id] = "validation"
        else:
            split_map[query_id] = config.eval_split
    scored["split"] = scored["query_id"].map(split_map)
    return scored


def _split_query_ids(scored: pd.DataFrame, config: ProbeRoutePPConfig) -> tuple[pd.Index, pd.Index]:
    query_split = scored[["query_id", "split"]].drop_duplicates("query_id").set_index("query_id")["split"].astype(str)
    train_ids = pd.Index(sorted(query_split[query_split.eq(config.train_split)].index.astype(str)), name="query_id")
    eval_ids = pd.Index(sorted(query_split[query_split.eq(config.eval_split)].index.astype(str)), name="query_id")
    if train_ids.empty or eval_ids.empty:
        raise ValueError(f"Need non-empty {config.train_split!r} and {config.eval_split!r} splits")
    return train_ids, eval_ids


def _cost_normalizer(scored: pd.DataFrame) -> float:
    gpt = scored[scored["model_id"].astype(str).str.contains("gpt", case=False, na=False)]
    if not gpt.empty:
        value = float(gpt.groupby("query_id")["cost_total_usd"].mean().mean())
    else:
        frontier = scored[scored["is_frontier"].astype(bool)]
        value = float(frontier.groupby("query_id")["cost_total_usd"].mean().mean()) if not frontier.empty else 0.0
    if value <= 0.0:
        value = float(scored["cost_total_usd"].max())
    return max(value, 1e-12)


def _latency_normalizer(scored: pd.DataFrame) -> float:
    gpt = scored[scored["model_id"].astype(str).str.contains("gpt", case=False, na=False)]
    if not gpt.empty:
        value = float(gpt["latency_s"].quantile(0.95))
    else:
        value = float(scored["latency_s"].quantile(0.95))
    return max(value, 1e-12)


def _pivot_metric(
    frame: pd.DataFrame,
    value_column: str,
    model_ids: list[str],
    *,
    fill_from: pd.DataFrame | None = None,
    fill_value: float | None = None,
) -> pd.DataFrame:
    matrix = (
        frame.pivot_table(index="query_id", columns="model_id", values=value_column, aggfunc="mean")
        .reindex(columns=model_ids)
        .sort_index()
    )
    matrix.index = matrix.index.astype(str)
    if fill_from is not None:
        matrix = matrix.fillna(fill_from.mean(axis=0))
    if fill_value is not None:
        matrix = matrix.fillna(float(fill_value))
    matrix = matrix.fillna(matrix.mean(axis=0)).fillna(0.0)
    matrix.index.name = "query_id"
    return matrix


def _build_state_model_table(
    *,
    train_scored: pd.DataFrame,
    train_labels: pd.Series,
    state_labels: list[str],
    model_ids: list[str],
) -> pd.DataFrame:
    labels = train_labels.rename("state_label").reset_index().rename(columns={"index": "query_id"})
    labels["query_id"] = labels["query_id"].astype(str)
    joined = train_scored.merge(labels, on="query_id", how="inner")
    grouped = joined.groupby(["state_label", "model_id"], dropna=False).agg(
        mean_quality=("quality_score", "mean"),
        mean_remote_cost_usd=("cost_total_usd", "mean"),
        mean_latency_s=("latency_s", "mean"),
        mean_utility=("utility", "mean"),
        n_train_examples=("query_id", "nunique"),
    )
    grid = pd.MultiIndex.from_product([state_labels, model_ids], names=["state_label", "model_id"])
    table = grouped.reindex(grid).reset_index()
    global_by_model = (
        train_scored.groupby("model_id")
        .agg(
            mean_quality=("quality_score", "mean"),
            mean_remote_cost_usd=("cost_total_usd", "mean"),
            mean_latency_s=("latency_s", "mean"),
            mean_utility=("utility", "mean"),
        )
        .reindex(model_ids)
        .fillna(0.0)
    )
    for column in ["mean_quality", "mean_remote_cost_usd", "mean_latency_s", "mean_utility"]:
        table[column] = table.apply(
            lambda row: global_by_model.loc[row["model_id"], column] if pd.isna(row[column]) else row[column],
            axis=1,
        )
    table["n_train_examples"] = table["n_train_examples"].fillna(0).astype(int)
    table["state_best_model"] = table.groupby("state_label")["mean_utility"].transform(
        lambda values: table.loc[values.idxmax(), "model_id"] if len(values) else ""
    )
    return table.sort_values(["state_label", "model_id"]).reset_index(drop=True)


def _wide_state_table(table: pd.DataFrame, value_column: str, state_labels: list[str], model_ids: list[str]) -> pd.DataFrame:
    wide = table.pivot(index="state_label", columns="model_id", values=value_column).reindex(index=state_labels, columns=model_ids)
    return wide.fillna(wide.mean(axis=0)).fillna(0.0)


def _label_distribution_to_states(distribution: pd.DataFrame, state_labels: list[str], *, temperature: float) -> pd.DataFrame:
    values = distribution.to_numpy(dtype=float)
    if temperature > 0 and abs(temperature - 1.0) > 1e-12:
        values = np.power(np.clip(values, 1e-12, 1.0), 1.0 / float(temperature))
        values = values / values.sum(axis=1, keepdims=True)
    beliefs = pd.DataFrame(values, index=distribution.index.astype(str), columns=state_labels)
    beliefs.index.name = "query_id"
    return beliefs


def _knn_state_beliefs(
    *,
    train_embeddings: pd.DataFrame,
    train_labels: pd.Series,
    target_embeddings: pd.DataFrame,
    state_labels: list[str],
    k: int,
) -> pd.DataFrame:
    train_matrix = train_embeddings.to_numpy(dtype=float)
    target_matrix = target_embeddings.to_numpy(dtype=float)
    label_values = train_labels.reindex(train_embeddings.index).astype(str).to_numpy()
    rows: list[np.ndarray] = []
    for vector in target_matrix:
        if len(train_matrix) == 0:
            rows.append(np.ones(len(state_labels), dtype=float) / len(state_labels))
            continue
        distances = np.linalg.norm(train_matrix - vector[None, :], axis=1)
        nearest = np.argsort(distances)[: max(1, min(int(k), len(distances)))]
        counts = pd.Series(label_values[nearest]).value_counts(normalize=True)
        rows.append(np.array([float(counts.get(label, 0.0)) for label in state_labels], dtype=float))
    beliefs = pd.DataFrame(rows, index=target_embeddings.index.astype(str), columns=state_labels)
    beliefs.index.name = "query_id"
    return beliefs.apply(_normalize_row, axis=1)


def _blend_and_normalize(before: pd.DataFrame, update: pd.DataFrame, blend_weight: float) -> pd.DataFrame:
    weight = min(max(float(blend_weight), 0.0), 1.0)
    blended = (1.0 - weight) * before + weight * update.reindex(before.index).fillna(0.0)
    return blended.apply(_normalize_row, axis=1)


def _normalize_row(row: pd.Series) -> pd.Series:
    total = float(row.sum())
    if total <= 1e-12:
        return pd.Series(1.0 / len(row), index=row.index)
    return row / total


def _routing_decisions_for_policy(
    *,
    method: str,
    probe_used: pd.Series,
    before_beliefs: pd.DataFrame,
    after_beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    state_model_quality: pd.DataFrame,
    state_model_cost: pd.DataFrame,
    state_model_latency: pd.DataFrame,
    eval_scored: pd.DataFrame,
    probe_latency_s: float,
    probe_cost_utility: float,
) -> pd.DataFrame:
    before_selected = select_models_from_belief(before_beliefs, state_model_utility)
    after_selected = select_models_from_belief(after_beliefs, state_model_utility)
    expected_utility_before = expected_model_utility_from_belief(before_beliefs, state_model_utility)
    expected_utility_after = expected_model_utility_from_belief(after_beliefs, state_model_utility)
    expected_quality_before = expected_model_utility_from_belief(before_beliefs, state_model_quality)
    expected_quality_after = expected_model_utility_from_belief(after_beliefs, state_model_quality)
    expected_cost_before = expected_model_utility_from_belief(before_beliefs, state_model_cost)
    expected_cost_after = expected_model_utility_from_belief(after_beliefs, state_model_cost)
    expected_latency_before = expected_model_utility_from_belief(before_beliefs, state_model_latency)
    expected_latency_after = expected_model_utility_from_belief(after_beliefs, state_model_latency)
    probe_used = probe_used.reindex(before_beliefs.index).fillna(False).astype(bool)
    selected = before_selected.where(~probe_used, after_selected)
    selected_beliefs = before_beliefs.copy()
    selected_beliefs.loc[probe_used[probe_used].index] = after_beliefs.loc[probe_used[probe_used].index]
    actual = _selected_rows(eval_scored, selected).set_index("query_id")
    rows: list[dict[str, Any]] = []
    for query_id in before_beliefs.index:
        model_id = str(selected.loc[query_id])
        belief = selected_beliefs.loc[query_id]
        before_json = _belief_json(before_beliefs.loc[query_id])
        after_json = _belief_json(after_beliefs.loc[query_id])
        expected_utility_frame = expected_utility_after if probe_used.loc[query_id] else expected_utility_before
        expected_quality_frame = expected_quality_after if probe_used.loc[query_id] else expected_quality_before
        expected_cost_frame = expected_cost_after if probe_used.loc[query_id] else expected_cost_before
        expected_latency_frame = expected_latency_after if probe_used.loc[query_id] else expected_latency_before
        rows.append(
            {
                "query_id": query_id,
                "method": method,
                "selected_model": model_id,
                "state_distribution_before_probe_json": before_json,
                "state_distribution_after_probe_json": after_json if bool(probe_used.loc[query_id]) else "",
                "selected_state": str(belief.idxmax()),
                "state_confidence": float(belief.max()),
                "probe_used": bool(probe_used.loc[query_id]),
                "probe_type": "non_generative_knn_state_probe" if bool(probe_used.loc[query_id]) else None,
                "probe_cost_utility": float(probe_cost_utility) if bool(probe_used.loc[query_id]) else 0.0,
                "probe_latency_s": float(probe_latency_s) if bool(probe_used.loc[query_id]) else 0.0,
                "expected_utility": float(expected_utility_frame.loc[query_id, model_id]),
                "expected_quality": float(expected_quality_frame.loc[query_id, model_id]),
                "expected_remote_cost_usd": float(expected_cost_frame.loc[query_id, model_id]),
                "expected_latency_ms": float(expected_latency_frame.loc[query_id, model_id] * 1000.0),
                "actual_quality": float(actual.loc[query_id, "quality_score"]),
                "actual_remote_cost_usd": float(actual.loc[query_id, "cost_total_usd"]),
                "actual_latency_s": float(actual.loc[query_id, "latency_s"]),
                "decision_reason": _decision_reason(bool(probe_used.loc[query_id])),
            }
        )
    return pd.DataFrame(rows)


def _build_main_eval(
    *,
    scored: pd.DataFrame,
    eval_scored: pd.DataFrame,
    query_info: pd.DataFrame,
    train_scored: pd.DataFrame,
    eval_utility: pd.DataFrame,
    before_beliefs: pd.DataFrame,
    after_beliefs: pd.DataFrame,
    knn_beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    routing_decisions: pd.DataFrame,
    config: ProbeRoutePPConfig,
) -> pd.DataFrame:
    model_ids = list(eval_utility.columns)
    eval_ids = eval_utility.index
    train_mean = train_scored.groupby("model_id")["utility"].mean().sort_values(ascending=False)
    best_single = str(train_mean.index[0])
    local_train = train_scored[train_scored["is_local"].astype(bool)]
    best_local = str(local_train.groupby("model_id")["utility"].mean().idxmax()) if not local_train.empty else best_single
    frontier_train = train_scored[train_scored["is_frontier"].astype(bool)]
    best_frontier = str(frontier_train.groupby("model_id")["utility"].mean().idxmax()) if not frontier_train.empty else best_single
    selections: dict[str, tuple[pd.Series, pd.Series, str]] = {
        "best_single_overall": (
            pd.Series(best_single, index=eval_ids),
            pd.Series(False, index=eval_ids),
            "Best train utility model.",
        ),
        "best_local": (
            pd.Series(best_local, index=eval_ids),
            pd.Series(False, index=eval_ids),
            "Best train utility local model.",
        ),
        "query_oracle": (
            _pivot_metric(eval_scored, "quality_score", model_ids).idxmax(axis=1),
            pd.Series(False, index=eval_ids),
            "Diagnostic observed-quality oracle.",
        ),
        "cost_aware_oracle": (
            eval_utility.idxmax(axis=1),
            pd.Series(False, index=eval_ids),
            "Diagnostic observed utility oracle.",
        ),
        "dataset_lookup": (
            _best_group_lookup(train_scored, query_info, default_model=best_single),
            pd.Series(False, index=eval_ids),
            "Benchmark/domain lookup fit on train only.",
        ),
        "embedding_cluster_lookup": (
            select_models_from_belief(before_beliefs, state_model_utility),
            pd.Series(False, index=eval_ids),
            "Query embedding centroid state lookup.",
        ),
        "kNN_router": (
            select_models_from_belief(knn_beliefs, state_model_utility),
            pd.Series(False, index=eval_ids),
            "kNN state-belief lookup using train states.",
        ),
        "confidence_cascade": (
            _confidence_cascade(before_beliefs, best_local=best_local, best_frontier=best_frontier),
            _low_confidence_mask(before_beliefs),
            "Escalate low-confidence state beliefs to best frontier model.",
        ),
    }
    if "gpt-5.5" in model_ids:
        selections["all_gpt_frontier"] = (
            pd.Series("gpt-5.5", index=eval_ids),
            pd.Series(False, index=eval_ids),
            "All queries routed to GPT-family frontier model.",
        )
    gemini_ids = [model_id for model_id in model_ids if "gemini" in model_id.lower()]
    if gemini_ids:
        selections["all_gemini_frontier"] = (
            pd.Series(gemini_ids[0], index=eval_ids),
            pd.Series(False, index=eval_ids),
            "All queries routed to Gemini-family frontier model.",
        )
    claude_ids = [model_id for model_id in model_ids if "claude" in model_id.lower()]
    if claude_ids:
        selections["all_claude_frontier"] = (
            pd.Series(claude_ids[0], index=eval_ids),
            pd.Series(False, index=eval_ids),
            "All queries routed to Claude-family frontier model.",
        )
    for method, group in routing_decisions.groupby("method", sort=False):
        selected = group.set_index("query_id")["selected_model"].reindex(eval_ids)
        probe_used = group.set_index("query_id")["probe_used"].reindex(eval_ids).fillna(False).astype(bool)
        selections[method] = (selected, probe_used, "ProbeRoute++ state-belief policy.")
    oracle_utility = eval_utility.max(axis=1)
    rows = [
        _evaluate_selection(
            method=method,
            selected=selected,
            eval_scored=eval_scored,
            scored=scored,
            oracle_utility=oracle_utility,
            probe_used=probe_used,
            probe_cost_utility=config.probe_cost_utility,
            probe_latency_s=config.probe_latency_s,
            notes=notes,
        )
        for method, (selected, probe_used, notes) in selections.items()
    ]
    return pd.DataFrame(rows).sort_values(["utility_cost_latency_aware", "quality_mean"], ascending=False)


def _selected_rows(scored: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    key = pd.DataFrame({"query_id": selected.index.astype(str), "model_id": selected.astype(str).values})
    rows = key.merge(scored, on=["query_id", "model_id"], how="left")
    if rows["quality_score"].isna().any():
        missing = rows.loc[rows["quality_score"].isna(), ["query_id", "model_id"]].head().to_dict("records")
        raise ValueError(f"Selected model rows missing from eval scored outputs: {missing}")
    return rows


def _best_group_lookup(train_scored: pd.DataFrame, query_info: pd.DataFrame, *, default_model: str) -> pd.Series:
    group_col = "benchmark" if "benchmark" in train_scored.columns else "domain"
    table = (
        train_scored.groupby([group_col, "model_id"], dropna=False)["utility"]
        .mean()
        .reset_index()
        .sort_values("utility", ascending=False)
    )
    best = table.drop_duplicates(group_col).set_index(group_col)["model_id"].to_dict()
    if group_col not in query_info.columns:
        return pd.Series(default_model, index=query_info.index)
    return query_info[group_col].map(lambda value: best.get(value, default_model)).rename("selected_model")


def _confidence_cascade(before_beliefs: pd.DataFrame, *, best_local: str, best_frontier: str) -> pd.Series:
    low_confidence = _low_confidence_mask(before_beliefs)
    selected = pd.Series(best_local, index=before_beliefs.index)
    selected.loc[low_confidence] = best_frontier
    return selected


def _low_confidence_mask(before_beliefs: pd.DataFrame) -> pd.Series:
    entropy_threshold = 0.55 * math.log2(max(before_beliefs.shape[1], 2))
    entropy = before_beliefs.apply(belief_entropy, axis=1)
    margin = before_beliefs.apply(belief_margin, axis=1)
    return ((entropy >= entropy_threshold) | (margin <= 0.20)).rename("low_confidence")


def _evaluate_selection(
    *,
    method: str,
    selected: pd.Series,
    eval_scored: pd.DataFrame,
    scored: pd.DataFrame,
    oracle_utility: pd.Series,
    probe_used: pd.Series,
    probe_cost_utility: float,
    probe_latency_s: float,
    notes: str,
) -> dict[str, Any]:
    selected = selected.reindex(oracle_utility.index)
    probe_used = probe_used.reindex(oracle_utility.index).fillna(False).astype(bool)
    chosen = _selected_rows(eval_scored, selected)
    utility = chosen["utility"].astype(float).to_numpy() - probe_used.to_numpy(dtype=float) * float(probe_cost_utility)
    quality = chosen["quality_score"].astype(float)
    remote_cost = chosen["cost_total_usd"].astype(float)
    latency = chosen["latency_s"].astype(float) + probe_used.to_numpy(dtype=float) * float(probe_latency_s)
    all_gpt_cost = _all_gpt_cost(scored)
    regret = oracle_utility.reindex(chosen["query_id"]).to_numpy(dtype=float) - utility
    return {
        "method": method,
        "n_queries": int(len(chosen)),
        "quality_mean": float(quality.mean()),
        "utility_cost_latency_aware": float(np.mean(utility)),
        "oracle_regret": float(np.mean(regret)),
        "remote_cost_per_query": float(remote_cost.mean()),
        "remote_cost_per_1k_queries": float(remote_cost.mean() * 1000.0),
        "normalized_remote_cost_vs_all_gpt": float(remote_cost.mean() / all_gpt_cost),
        "frontier_call_rate": float(chosen["is_frontier"].astype(bool).mean()),
        "local_call_rate": float(chosen["is_local"].astype(bool).mean()),
        "probe_call_rate": float(probe_used.mean()),
        "latency_mean": float(latency.mean()),
        "latency_p50": float(latency.quantile(0.50)),
        "latency_p95": float(latency.quantile(0.95)),
        "latency_p99": float(latency.quantile(0.99)),
        "notes": notes,
    }


def _all_gpt_cost(scored: pd.DataFrame) -> float:
    gpt = scored[scored["model_id"].astype(str).str.contains("gpt", case=False, na=False)]
    if not gpt.empty:
        return max(float(gpt.groupby("query_id")["cost_total_usd"].mean().mean()), 1e-12)
    frontier = scored[scored["is_frontier"].astype(bool)]
    if not frontier.empty:
        return max(float(frontier.groupby("query_id")["cost_total_usd"].mean().mean()), 1e-12)
    return max(float(scored["cost_total_usd"].mean()), 1e-12)


def _cost_latency_summary(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.groupby(["model_id", "provider", "is_frontier"], dropna=False)
        .agg(
            n_calls=("query_id", "count"),
            mean_quality=("quality_score", "mean"),
            total_cost_usd=("cost_total_usd", "sum"),
            mean_cost_usd=("cost_total_usd", "mean"),
            latency_mean=("latency_s", "mean"),
            latency_p50=("latency_s", lambda s: float(s.quantile(0.50))),
            latency_p95=("latency_s", lambda s: float(s.quantile(0.95))),
            latency_p99=("latency_s", lambda s: float(s.quantile(0.99))),
        )
        .reset_index()
    )


def _calibration_placeholder(state_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": "active_latent_state_calibration",
                "status": "not_run_in_stage2",
                "state_count": len(state_labels),
                "examples_per_state": r,
                "planned_model_evaluations": len(state_labels) * r,
                "notes": "Stage 3 calibration will reuse these latent states and update state-to-model utilities.",
            }
            for r in [4, 8, 16, 32]
        ]
    )


def _write_plots(main_eval: pd.DataFrame, quality_cost_path: Path, latency_path: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.scatter(main_eval["normalized_remote_cost_vs_all_gpt"], main_eval["quality_mean"])
    for row in main_eval.itertuples(index=False):
        if str(row.method).startswith("proberoutepp") or row.method in {"best_local", "all_gpt_frontier", "cost_aware_oracle"}:
            plt.annotate(row.method, (row.normalized_remote_cost_vs_all_gpt, row.quality_mean), fontsize=7)
    plt.xlabel("Normalized remote cost vs all GPT")
    plt.ylabel("Quality")
    plt.tight_layout()
    plt.savefig(quality_cost_path)
    plt.close()

    subset = main_eval[main_eval["method"].isin(["best_local", "all_gpt_frontier", "proberoutepp_no_probe", "proberoutepp_voi_probe"])]
    plt.figure(figsize=(7, 4.5))
    plt.bar(subset["method"], subset["latency_p95"])
    plt.ylabel("p95 latency (s)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(latency_path)
    plt.close()


def _run_report(artifacts: ProbeRoutePPArtifacts) -> str:
    main = artifacts.main_eval
    oracle = _optional_method(main, "cost_aware_oracle")
    voi = _optional_method(main, "proberoutepp_voi_probe")
    target_line = ""
    if oracle is not None and voi is not None:
        target_line = (
            f"- ProbeRoute++ VOI quality gap to cost-aware oracle: "
            f"`{float(oracle['quality_mean'] - voi['quality_mean']):.4f}`.\n"
            f"- ProbeRoute++ VOI normalized remote cost vs all GPT: "
            f"`{float(voi['normalized_remote_cost_vs_all_gpt']):.4f}`.\n"
            f"- ProbeRoute++ VOI frontier call rate: `{float(voi['frontier_call_rate']):.4f}`.\n"
            f"- ProbeRoute++ VOI probe rate: `{float(voi['probe_call_rate']):.4f}`."
        )
    table_lines = "\n".join(
        f"| {row.method} | {row.quality_mean:.4f} | {row.utility_cost_latency_aware:.4f} | "
        f"{row.normalized_remote_cost_vs_all_gpt:.4f} | {row.frontier_call_rate:.4f} | "
        f"{row.probe_call_rate:.4f} | {row.latency_p95:.4f} |"
        for row in main.head(12).itertuples(index=False)
    )
    return f"""# ProbeRoute++ Stage 2 Report

This run evaluates the state-mediated ProbeRoute++ router:

```text
query/probe -> latent route-state belief -> selected model
```

The state codebook and state-to-model table are fit on train queries only. Held-out routing uses query text hashing features and an optional non-generative kNN state probe over train states. This is a working Stage 2 method artifact, not a SOTA claim.

## Metadata

```json
{json.dumps(artifacts.metadata, indent=2, sort_keys=True)}
```

## Target Snapshot

{target_line}

## Main Evaluation

| method | quality | utility | normalized remote cost | frontier rate | probe rate | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table_lines}

## Outputs

- `state_model_utility_table.parquet`
- `routing_decisions.parquet`
- `table_main_eval.csv`
- `cost_latency_summary.csv`
- `table_calibration.csv`
- `fig_quality_cost_frontier.pdf`
- `fig_latency_breakdown.pdf`

Closed-source provider families remain in the plan: OpenAI GPT-family, Anthropic Claude-family, and Google Gemini-family. This run only evaluates providers present in the input outcome matrix.
"""


def _optional_method(table: pd.DataFrame, method: str) -> pd.Series | None:
    rows = table[table["method"].eq(method)]
    return None if rows.empty else rows.iloc[0]


def _state_label(label: int) -> str:
    return f"z{int(label):02d}"


def _belief_json(belief: pd.Series) -> str:
    return json.dumps({str(key): float(value) for key, value in belief.items()}, sort_keys=True)


def _decision_reason(probed: bool) -> str:
    if probed:
        return "Non-generative uncertainty/kNN probe updated the latent state belief; selected model maximizes expected state utility."
    return "High-confidence query-only latent state belief; selected model maximizes expected state utility."


def _stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)
