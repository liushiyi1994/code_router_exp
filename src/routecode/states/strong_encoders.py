from __future__ import annotations

from collections.abc import Callable, Sequence
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.codes.routecode import RouteCodeCodebook
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.predictor_diagnostics import expected_calibration_error, label_accuracy
from routecode.matrix import Matrices
from routecode.metrics import bootstrap_mean_ci, recovered_gap, selected_values
from routecode.predictors.classifiers import MLPRouteCodeLabelClassifier, RouteCodeLabelClassifier
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


EmbeddingProvider = Callable[[pd.Series, pd.DataFrame], pd.DataFrame]


def evaluate_strong_encoder_state_observability(
    *,
    train: Matrices,
    test: Matrices,
    readiness_table: pd.DataFrame,
    embedding_provider: EmbeddingProvider | None,
    k: int = 16,
    alpha: float = 3.0,
    beta: float = 0.0,
    state_families: Sequence[str] = ("flat_routecode", "d2"),
    predictors: Sequence[str] = ("centroid", "knn", "logistic", "mlp"),
    random_state: int = 0,
    n_bootstrap: int = 100,
    ci: float = 0.95,
    n_neighbors: int = 15,
    max_iter: int = 200,
    refinement_iter: int = 10,
) -> pd.DataFrame:
    """Evaluate query->state->model routing with cached strong encoder embeddings."""

    if readiness_table.empty:
        return pd.DataFrame([_skipped_row(None, "no_readiness_rows")])

    runnable_mask = readiness_table["runnable_as_encoder_baseline"].map(_as_bool)
    runnable = readiness_table[runnable_mask].copy()
    non_runnable = readiness_table[~runnable_mask].copy()
    if runnable.empty:
        return pd.DataFrame([_skipped_row(row, "no_cached_encoder_candidate") for _, row in readiness_table.iterrows()])

    rows: list[dict[str, Any]] = []
    all_query_info = _combined_query_info(train, test)
    references = _reference_values(train, test)
    for encoder_index, (_, readiness_row) in enumerate(runnable.iterrows()):
        if embedding_provider is None:
            rows.append(_failed_row(readiness_row, "missing_embedding_provider"))
            continue
        started = time.perf_counter()
        try:
            embeddings = embedding_provider(readiness_row, all_query_info)
            _validate_embeddings(embeddings, train.utility.index, test.utility.index)
        except Exception as exc:  # pragma: no cover - defensive real-cache path.
            rows.append(_failed_row(readiness_row, f"embedding_extraction_failed:{type(exc).__name__}:{exc}"))
            continue
        embedding_seconds = time.perf_counter() - started
        train_embeddings = embeddings.loc[train.utility.index]
        test_embeddings = embeddings.loc[test.utility.index]
        for family_index, family in enumerate(state_families):
            try:
                state_bundle = _fit_state_bundle(
                    state_family=str(family),
                    train=train,
                    test=test,
                    embeddings=embeddings,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    random_state=int(random_state) + 100 * encoder_index + family_index,
                    max_iter=max_iter,
                    refinement_iter=refinement_iter,
                )
            except Exception as exc:  # pragma: no cover - defensive experiment path.
                failed = _failed_row(readiness_row, f"state_codebook_failed:{family}:{type(exc).__name__}:{exc}")
                failed["state_family"] = str(family)
                rows.append(failed)
                continue
            for predictor_index, predictor in enumerate(predictors):
                predictor_started = time.perf_counter()
                try:
                    predicted_labels, confidence = _predict_state_labels(
                        predictor=str(predictor),
                        codebook=state_bundle.codebook,
                        train_embeddings=train_embeddings,
                        test_embeddings=test_embeddings,
                        random_state=int(random_state) + 1000 * encoder_index + 100 * family_index + predictor_index,
                        max_iter=max_iter,
                        n_neighbors=n_neighbors,
                    )
                except Exception as exc:  # pragma: no cover - defensive experiment path.
                    failed = _failed_row(
                        readiness_row,
                        f"state_predictor_failed:{family}:{predictor}:{type(exc).__name__}:{exc}",
                    )
                    failed["state_family"] = str(family)
                    failed["state_predictor"] = str(predictor)
                    rows.append(failed)
                    continue
                predictor_seconds = time.perf_counter() - predictor_started
                rows.append(
                    _metric_row(
                        readiness_row=readiness_row,
                        state_bundle=state_bundle,
                        predicted_labels=predicted_labels,
                        confidence=confidence,
                        test=test,
                        references=references,
                        state_predictor=str(predictor),
                        embedding_dim=int(train_embeddings.shape[1]),
                        embedding_seconds=embedding_seconds,
                        predictor_seconds=predictor_seconds,
                        seed=int(random_state) + 10000 * encoder_index + 100 * family_index + predictor_index,
                        n_bootstrap=n_bootstrap,
                        ci=ci,
                    )
                )

    rows.extend(_skipped_row(row, "no_cached_encoder_candidate") for _, row in non_runnable.iterrows())
    return pd.DataFrame(rows)


class _StateBundle:
    def __init__(
        self,
        *,
        state_family: str,
        oracle_state_method: str,
        codebook: RouteCodeCodebook | PredictabilityConstrainedRouteCode,
        oracle_labels: pd.Series,
        oracle_selected: pd.Series,
        oracle_mean_utility: float,
        k: int,
        alpha: float,
    ) -> None:
        self.state_family = state_family
        self.oracle_state_method = oracle_state_method
        self.codebook = codebook
        self.oracle_labels = oracle_labels
        self.oracle_selected = oracle_selected
        self.oracle_mean_utility = float(oracle_mean_utility)
        self.k = int(k)
        self.alpha = float(alpha)


def _fit_state_bundle(
    *,
    state_family: str,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    k: int,
    alpha: float,
    beta: float,
    random_state: int,
    max_iter: int,
    refinement_iter: int,
) -> _StateBundle:
    normalized = state_family.lower()
    if normalized in {"flat", "flat_routecode", "routecode"}:
        codebook = RouteCodeCodebook(k, random_state=random_state, max_iter=max_iter).fit(
            train.query_info,
            train.utility,
            embeddings,
        )
        oracle_labels = codebook.predict_utility_labels(test.utility)
        oracle_method = "flat_routecode_utility_oracle"
        family = "flat_routecode"
    elif normalized in {"d2", "predictability_constrained", "d2_predictability_constrained"}:
        codebook = PredictabilityConstrainedRouteCode(
            k,
            alpha=alpha,
            beta=beta,
            random_state=random_state,
            max_iter=max_iter,
            refinement_iter=refinement_iter,
        ).fit(train.query_info, train.utility, embeddings)
        oracle_labels = codebook.predict_joint_labels(test.utility, embeddings.loc[test.utility.index])
        oracle_method = "d2_joint_oracle_labels"
        family = "d2_predictability_constrained"
    else:
        raise ValueError(f"Unknown state family: {state_family}")
    oracle_selected = codebook.predict_from_labels(oracle_labels)
    oracle_mean = float(selected_values(test.utility, oracle_selected).mean())
    return _StateBundle(
        state_family=family,
        oracle_state_method=oracle_method,
        codebook=codebook,
        oracle_labels=oracle_labels,
        oracle_selected=oracle_selected,
        oracle_mean_utility=oracle_mean,
        k=k,
        alpha=alpha if family == "d2_predictability_constrained" else np.nan,
    )


def _predict_state_labels(
    *,
    predictor: str,
    codebook: RouteCodeCodebook | PredictabilityConstrainedRouteCode,
    train_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
    random_state: int,
    max_iter: int,
    n_neighbors: int,
) -> tuple[pd.Series, pd.Series]:
    name = predictor.lower()
    if name == "centroid":
        labels = codebook.predict_labels(test_embeddings)
        confidence = _centroid_confidence(codebook, test_embeddings)
        return labels.astype(int), confidence
    if name == "logistic":
        classifier = RouteCodeLabelClassifier(random_state=random_state, max_iter=max_iter).fit(codebook, train_embeddings)
        labels = classifier.predict_labels(test_embeddings)
        confidence = classifier.predict_confidence(test_embeddings)
        return labels.astype(int), confidence
    if name == "mlp":
        classifier = MLPRouteCodeLabelClassifier(random_state=random_state, max_iter=max_iter).fit(codebook, train_embeddings)
        labels = classifier.predict_labels(test_embeddings)
        confidence = classifier.predict_confidence(test_embeddings)
        return labels.astype(int), confidence
    if name == "knn":
        if codebook.train_labels_ is None:
            raise RuntimeError("Codebook must have train labels before kNN prediction")
        y = codebook.train_labels_.loc[train_embeddings.index].astype(int)
        if y.nunique() == 1:
            labels = pd.Series([int(y.iloc[0])] * len(test_embeddings), index=test_embeddings.index, name="route_label")
            confidence = pd.Series(1.0, index=test_embeddings.index, name="route_label_confidence")
            return labels, confidence
        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_embeddings.to_numpy(dtype=float))
        x_test = scaler.transform(test_embeddings.to_numpy(dtype=float))
        model = KNeighborsClassifier(n_neighbors=min(max(1, int(n_neighbors)), len(train_embeddings)))
        model.fit(x_train, y.to_numpy())
        labels = pd.Series(model.predict(x_test).astype(int), index=test_embeddings.index, name="route_label")
        probabilities = model.predict_proba(x_test)
        confidence = pd.Series(probabilities.max(axis=1), index=test_embeddings.index, name="route_label_confidence")
        return labels, confidence
    raise ValueError(f"Unknown state predictor: {predictor}")


def _metric_row(
    *,
    readiness_row: pd.Series,
    state_bundle: _StateBundle,
    predicted_labels: pd.Series,
    confidence: pd.Series,
    test: Matrices,
    references: dict[str, float],
    state_predictor: str,
    embedding_dim: int,
    embedding_seconds: float,
    predictor_seconds: float,
    seed: int,
    n_bootstrap: int,
    ci: float,
) -> dict[str, Any]:
    selected = state_bundle.codebook.predict_from_labels(predicted_labels)
    evaluated = evaluate_selection(
        method=f"{state_bundle.state_family}_strong_encoder_{state_predictor}",
        selected_models=selected,
        matrices=test,
        baseline_mean=references["best_single"],
        learned_reference_mean=state_bundle.oracle_mean_utility,
        oracle_mean=references["query_oracle"],
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        k=state_bundle.k,
        labels=predicted_labels,
    )
    deployable_mean = float(evaluated["mean_utility"])
    deployable_ci_low = float(evaluated["utility_ci_low"])
    deployable_ci_high = float(evaluated["utility_ci_high"])
    oracle_state_values = selected_values(test.utility, state_bundle.oracle_selected)
    oracle_ci_low, oracle_ci_high = bootstrap_mean_ci(
        oracle_state_values,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed + 17,
    )
    correct = (predicted_labels.astype(int) == state_bundle.oracle_labels.astype(int)).astype(int)
    row = _empty_row()
    row.update(evaluated)
    row.update(_metadata(readiness_row))
    row.update(
        {
            "result_id": "",
            "comparison": f"{state_bundle.state_family}_strong_encoder_{state_predictor}",
            "state_family": state_bundle.state_family,
            "oracle_state_method": state_bundle.oracle_state_method,
            "deployable_state_method": f"strong_encoder_{state_predictor}",
            "K": state_bundle.k,
            "alpha": state_bundle.alpha,
            "encoder_family": "local_transformer",
            "strong_encoder_status": "executed",
            "status": "executed",
            "reason": "",
            "query_oracle_mean_utility": references["query_oracle"],
            "query_oracle_mean_utility_ci_low": references["query_oracle_ci_low"],
            "query_oracle_mean_utility_ci_high": references["query_oracle_ci_high"],
            "best_single_mean_utility": references["best_single"],
            "best_single_mean_utility_ci_low": references["best_single_ci_low"],
            "best_single_mean_utility_ci_high": references["best_single_ci_high"],
            "oracle_state_mean_utility": state_bundle.oracle_mean_utility,
            "oracle_state_mean_utility_ci_low": min(float(oracle_ci_low), state_bundle.oracle_mean_utility),
            "oracle_state_mean_utility_ci_high": max(float(oracle_ci_high), state_bundle.oracle_mean_utility),
            "deployable_state_mean_utility": deployable_mean,
            "deployable_state_mean_utility_ci_low": min(deployable_ci_low, deployable_mean),
            "deployable_state_mean_utility_ci_high": max(deployable_ci_high, deployable_mean),
            "state_observability_gap": float(state_bundle.oracle_mean_utility - deployable_mean),
            "state_observability_gap_ci_low": min(float(oracle_ci_low), state_bundle.oracle_mean_utility)
            - max(deployable_ci_high, deployable_mean),
            "state_observability_gap_ci_high": max(float(oracle_ci_high), state_bundle.oracle_mean_utility)
            - min(deployable_ci_low, deployable_mean),
            "query_oracle_gap": float(references["query_oracle"] - deployable_mean),
            "query_oracle_gap_ci_low": references["query_oracle_ci_low"] - max(deployable_ci_high, deployable_mean),
            "query_oracle_gap_ci_high": references["query_oracle_ci_high"] - min(deployable_ci_low, deployable_mean),
            "state_oracle_gap_vs_best_single": float(state_bundle.oracle_mean_utility - references["best_single"]),
            "deployable_gap_vs_best_single": float(deployable_mean - references["best_single"]),
            "state_gap_closed": recovered_gap(deployable_mean, references["best_single"], state_bundle.oracle_mean_utility),
            "full_gap_closed_vs_query_oracle": recovered_gap(
                deployable_mean,
                references["best_single"],
                references["query_oracle"],
            ),
            "oracle_state_recovered_gap_vs_oracle": recovered_gap(
                state_bundle.oracle_mean_utility,
                references["best_single"],
                references["query_oracle"],
            ),
            "deployable_recovered_gap_vs_oracle": recovered_gap(
                deployable_mean,
                references["best_single"],
                references["query_oracle"],
            ),
            "label_accuracy": label_accuracy(state_bundle.oracle_labels, predicted_labels),
            "mean_confidence": float(confidence.mean()) if len(confidence) else np.nan,
            "ece": expected_calibration_error(confidence, correct, n_bins=10),
            "evidence_source": "phase2_strong_encoder_state_predictor",
            "interpretation": _interpretation(float(state_bundle.oracle_mean_utility - deployable_mean), evaluated),
            "state_predictor": state_predictor,
            "embedding_source": "local_transformer",
            "embedding_dim": int(embedding_dim),
            "embedding_seconds": float(embedding_seconds),
            "predictor_seconds": float(predictor_seconds),
            "routing_invariant": "query_to_state_to_model",
        }
    )
    return row


def _centroid_confidence(
    codebook: RouteCodeCodebook | PredictabilityConstrainedRouteCode,
    embeddings: pd.DataFrame,
) -> pd.Series:
    if isinstance(codebook, PredictabilityConstrainedRouteCode):
        return codebook.predict_label_confidence(embeddings)
    if codebook.embedding_centroids_ is None:
        raise RuntimeError("Codebook must have embedding centroids")
    values = embeddings.to_numpy(dtype=float)
    centroids = codebook.embedding_centroids_.to_numpy(dtype=float)
    distances = ((values[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    logits = -distances
    logits = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    return pd.Series(probabilities.max(axis=1), index=embeddings.index, name="route_label_confidence")


def _reference_values(train: Matrices, test: Matrices) -> dict[str, float]:
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    query_oracle = OracleRouter().predict(test.utility)
    best_values = selected_values(test.utility, best_single)
    oracle_values = selected_values(test.utility, query_oracle)
    best_low, best_high = bootstrap_mean_ci(best_values, n_bootstrap=500, ci=0.95, seed=0)
    oracle_low, oracle_high = bootstrap_mean_ci(oracle_values, n_bootstrap=500, ci=0.95, seed=1)
    best_mean = float(best_values.mean())
    oracle_mean = float(oracle_values.mean())
    return {
        "best_single": best_mean,
        "best_single_ci_low": min(float(best_low), best_mean),
        "best_single_ci_high": max(float(best_high), best_mean),
        "query_oracle": oracle_mean,
        "query_oracle_ci_low": min(float(oracle_low), oracle_mean),
        "query_oracle_ci_high": max(float(oracle_high), oracle_mean),
    }


def _combined_query_info(train: Matrices, test: Matrices) -> pd.DataFrame:
    combined = pd.concat([train.query_info, test.query_info], axis=0)
    return combined.loc[~combined.index.duplicated(keep="first")]


def _validate_embeddings(embeddings: pd.DataFrame, train_index: pd.Index, test_index: pd.Index) -> None:
    required = pd.Index(train_index.tolist() + test_index.tolist())
    missing = required.difference(embeddings.index)
    if len(missing) > 0:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Strong encoder embeddings missing query rows: {preview}")
    if embeddings.shape[1] == 0:
        raise ValueError("Strong encoder embeddings have zero columns")


def _skipped_row(readiness_row: pd.Series | None, reason: str) -> dict[str, Any]:
    row = _empty_row()
    if readiness_row is not None:
        row.update(_metadata(readiness_row))
    row.update(
        {
            "method": "strong_encoder_state_predictor",
            "comparison": "strong_encoder_state_predictor",
            "strong_encoder_status": "skipped",
            "status": "skipped",
            "reason": reason,
            "encoder_family": "local_transformer",
            "embedding_source": "local_transformer",
            "evidence_source": "phase2_strong_encoder_state_predictor",
            "routing_invariant": "query_to_state_to_model",
        }
    )
    return row


def _failed_row(readiness_row: pd.Series, reason: str) -> dict[str, Any]:
    row = _empty_row()
    row.update(_metadata(readiness_row))
    row.update(
        {
            "method": "strong_encoder_state_predictor",
            "comparison": "strong_encoder_state_predictor",
            "strong_encoder_status": "failed",
            "status": "failed",
            "reason": reason,
            "encoder_family": "local_transformer",
            "embedding_source": "local_transformer",
            "evidence_source": "phase2_strong_encoder_state_predictor",
            "routing_invariant": "query_to_state_to_model",
        }
    )
    return row


def _empty_row() -> dict[str, Any]:
    return {
        "result_id": "",
        "comparison": "",
        "state_family": "",
        "oracle_state_method": "",
        "deployable_state_method": "",
        "K": np.nan,
        "alpha": np.nan,
        "encoder_family": "",
        "strong_encoder_status": "",
        "query_oracle_mean_utility": np.nan,
        "query_oracle_mean_utility_ci_low": np.nan,
        "query_oracle_mean_utility_ci_high": np.nan,
        "best_single_mean_utility": np.nan,
        "best_single_mean_utility_ci_low": np.nan,
        "best_single_mean_utility_ci_high": np.nan,
        "oracle_state_mean_utility": np.nan,
        "oracle_state_mean_utility_ci_low": np.nan,
        "oracle_state_mean_utility_ci_high": np.nan,
        "deployable_state_mean_utility": np.nan,
        "deployable_state_mean_utility_ci_low": np.nan,
        "deployable_state_mean_utility_ci_high": np.nan,
        "state_observability_gap": np.nan,
        "state_observability_gap_ci_low": np.nan,
        "state_observability_gap_ci_high": np.nan,
        "query_oracle_gap": np.nan,
        "query_oracle_gap_ci_low": np.nan,
        "query_oracle_gap_ci_high": np.nan,
        "state_oracle_gap_vs_best_single": np.nan,
        "deployable_gap_vs_best_single": np.nan,
        "state_gap_closed": np.nan,
        "full_gap_closed_vs_query_oracle": np.nan,
        "oracle_state_recovered_gap_vs_oracle": np.nan,
        "deployable_recovered_gap_vs_oracle": np.nan,
        "label_accuracy": np.nan,
        "mean_confidence": np.nan,
        "ece": np.nan,
        "evidence_source": "",
        "interpretation": "",
        "status": "",
        "reason": "",
        "method": "",
        "oracle_regret": np.nan,
        "mean_utility": np.nan,
        "mean_quality": np.nan,
        "normalized_cost": np.nan,
        "utility_ci_low": np.nan,
        "utility_ci_high": np.nan,
        "recovered_gap_vs_learned": np.nan,
        "recovered_gap_vs_oracle": np.nan,
        "selected_model_entropy": np.nan,
        "rate_log2K": np.nan,
        "empirical_H_Z": np.nan,
        "probe_scope": "strong_encoder_state_observability",
        "model_id": "",
        "cache_status": "",
        "readiness_reason": "",
        "architecture": "",
        "model_type": "",
        "hidden_size": "",
        "size_gb": np.nan,
        "local_path": "",
        "state_predictor": "",
        "embedding_source": "",
        "embedding_dim": np.nan,
        "embedding_seconds": np.nan,
        "predictor_seconds": np.nan,
        "routing_invariant": "",
    }


def _metadata(readiness_row: pd.Series) -> dict[str, Any]:
    return {
        "probe_scope": "strong_encoder_state_observability",
        "model_id": str(readiness_row.get("model_id", "")),
        "cache_status": str(readiness_row.get("cache_status", "")),
        "readiness_reason": str(readiness_row.get("reason", "")),
        "architecture": str(readiness_row.get("architecture", "")),
        "model_type": str(readiness_row.get("model_type", "")),
        "hidden_size": readiness_row.get("hidden_size", ""),
        "size_gb": readiness_row.get("size_gb", np.nan),
        "local_path": str(readiness_row.get("local_path", "")),
    }


def _interpretation(state_gap: float, evaluated: dict[str, Any]) -> str:
    if state_gap <= 0.01 and float(evaluated["recovered_gap_vs_oracle"]) >= 0.8:
        return "strong encoder makes route state mostly observable from query text"
    if state_gap <= 0.05:
        return "strong encoder closes most of the state-observability gap but query-oracle gap may remain"
    if float(evaluated["recovered_gap_vs_oracle"]) < 0.0:
        return "strong encoder state predictor remains below best-single baseline"
    return "strong encoder state predictor leaves a material observability gap"


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
