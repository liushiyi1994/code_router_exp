from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from routecode.metrics import bootstrap_mean_ci


PROBE_SIGNAL_COLUMNS = [
    "method",
    "status",
    "n_queries",
    "n_train",
    "n_test",
    "state_prediction_accuracy",
    "state_prediction_accuracy_ci_low",
    "state_prediction_accuracy_ci_high",
    "routing_utility",
    "observability_gap_closed",
    "mean_probe_cost_proxy",
    "regret_prediction_auc",
    "notes",
]

METHODS = [
    "query_only_state_predictor",
    "probe_only_state_predictor",
    "query_plus_probe_state_predictor",
    "query_plus_knn_uncertainty_state_predictor",
    "query_plus_confidence_state_predictor",
]

PROBE_NUMERIC_COLUMNS = [
    "self_confidence",
    "agreement_score",
    "knn_label_entropy",
    "knn_winner_entropy",
    "latency_sec",
    "input_tokens",
    "output_tokens",
    "probe_cost_proxy",
]

KNN_COLUMNS = ["knn_label_entropy", "knn_winner_entropy"]
CONFIDENCE_COLUMNS = ["self_confidence", "agreement_score"]


def analyze_probe_signal(
    *,
    probe_features: pd.DataFrame,
    state_targets: pd.DataFrame | None = None,
    query_features: pd.DataFrame | None = None,
    random_state: int = 0,
) -> pd.DataFrame:
    """Evaluate whether probe features predict latent route states."""

    if state_targets is None or state_targets.empty:
        return _blocked_table("blocked_missing_state_targets", "No route-state target table was supplied.")
    _require_columns(probe_features, ["query_id"], name="probe_features")
    _require_columns(state_targets, ["query_id", "state_label"], name="state_targets")

    probe_by_query = _aggregate_probe_features(probe_features)
    query_by_query = _normalise_query_features(query_features)
    target_columns = ["query_id", "state_label"] + (["split"] if "split" in state_targets.columns else [])
    target = state_targets[target_columns].drop_duplicates("query_id").copy()
    frame = target.merge(probe_by_query, on="query_id", how="inner")
    if not query_by_query.empty:
        frame = frame.merge(query_by_query, on="query_id", how="left")
    if frame.empty:
        return _blocked_table(
            "blocked_no_aligned_state_targets",
            "Probe query IDs do not overlap the route-state target table.",
        )
    if frame["state_label"].nunique() < 2:
        return _blocked_table(
            "blocked_insufficient_state_classes",
            "Aligned targets contain fewer than two route-state classes.",
            n_queries=len(frame),
        )

    query_columns = [column for column in query_by_query.columns if column != "query_id"]
    probe_columns = [column for column in PROBE_NUMERIC_COLUMNS if column in frame.columns]
    rows = [
        _evaluate_method(
            frame,
            method="query_only_state_predictor",
            feature_columns=query_columns,
            random_state=random_state,
        ),
        _evaluate_method(
            frame,
            method="probe_only_state_predictor",
            feature_columns=probe_columns,
            random_state=random_state,
        ),
        _evaluate_method(
            frame,
            method="query_plus_probe_state_predictor",
            feature_columns=query_columns + probe_columns,
            random_state=random_state,
        ),
        _evaluate_method(
            frame,
            method="query_plus_knn_uncertainty_state_predictor",
            feature_columns=query_columns + [column for column in KNN_COLUMNS if column in frame.columns],
            random_state=random_state,
        ),
        _evaluate_method(
            frame,
            method="query_plus_confidence_state_predictor",
            feature_columns=query_columns + [column for column in CONFIDENCE_COLUMNS if column in frame.columns],
            random_state=random_state,
        ),
    ]
    return pd.DataFrame(rows, columns=PROBE_SIGNAL_COLUMNS)


def _evaluate_method(
    frame: pd.DataFrame,
    *,
    method: str,
    feature_columns: list[str],
    random_state: int,
) -> dict[str, Any]:
    feature_columns = _dedupe(feature_columns)
    if not feature_columns:
        return _row(
            method=method,
            status="blocked_missing_features",
            n_queries=len(frame),
            notes="No feature columns are available for this method.",
        )
    if "split" in frame.columns and {"train", "test"}.issubset(set(frame["split"].astype(str))):
        train = frame[frame["split"].astype(str).eq("train")]
        test = frame[frame["split"].astype(str).eq("test")]
    else:
        class_counts = frame["state_label"].value_counts()
        stratify = frame["state_label"] if class_counts.min() >= 2 else None
        try:
            train, test = train_test_split(
                frame,
                test_size=0.33,
                random_state=random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            return _row(
                method=method,
                status="blocked_split_failed",
                n_queries=len(frame),
                notes=str(exc),
            )
    if train["state_label"].nunique() < 2 or test.empty:
        return _row(
            method=method,
            status="blocked_insufficient_split_classes",
            n_queries=len(frame),
            n_train=len(train),
            n_test=len(test),
            notes="Train/test split does not contain enough state classes.",
        )
    model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            feature_columns,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(train[feature_columns], train["state_label"].astype(str))
    predictions = model.predict(test[feature_columns])
    correct = (test["state_label"].astype(str).to_numpy() == predictions).astype(float)
    accuracy = float(accuracy_score(test["state_label"].astype(str), predictions))
    ci_low, ci_high = bootstrap_mean_ci(correct, n_bootstrap=500, ci=0.95, seed=random_state)
    return _row(
        method=method,
        status="executed",
        n_queries=len(frame),
        n_train=len(train),
        n_test=len(test),
        state_prediction_accuracy=accuracy,
        state_prediction_accuracy_ci_low=min(ci_low, accuracy),
        state_prediction_accuracy_ci_high=max(ci_high, accuracy),
        mean_probe_cost_proxy=_mean_probe_cost(frame),
        regret_prediction_auc=_regret_auc(model, test, feature_columns),
        notes="State prediction only; routing utility requires a state-to-model utility table.",
    )


def _aggregate_probe_features(probe_features: pd.DataFrame) -> pd.DataFrame:
    columns = ["query_id"] + [column for column in PROBE_NUMERIC_COLUMNS if column in probe_features.columns]
    frame = probe_features[columns].copy()
    for column in columns:
        if column != "query_id":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.groupby("query_id", as_index=False).mean(numeric_only=True)


def _normalise_query_features(query_features: pd.DataFrame | None) -> pd.DataFrame:
    if query_features is None or query_features.empty:
        return pd.DataFrame(columns=["query_id"])
    _require_columns(query_features, ["query_id"], name="query_features")
    frame = query_features.drop_duplicates("query_id").copy()
    numeric_columns = [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column != "query_id"
    ]
    return frame[["query_id"] + numeric_columns]


def _regret_auc(model: Pipeline, test: pd.DataFrame, feature_columns: list[str]) -> float:
    if "regret" not in test.columns:
        return math.nan
    regret = pd.to_numeric(test["regret"], errors="coerce")
    if regret.isna().all() or regret.nunique(dropna=True) < 2:
        return math.nan
    high_regret = regret >= regret.median()
    if high_regret.nunique(dropna=True) < 2:
        return math.nan
    try:
        probabilities = model.predict_proba(test[feature_columns]).max(axis=1)
        return float(roc_auc_score(high_regret.astype(int), 1.0 - probabilities))
    except Exception:
        return math.nan


def _blocked_table(status: str, notes: str, *, n_queries: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [_row(method=method, status=status, n_queries=n_queries, notes=notes) for method in METHODS],
        columns=PROBE_SIGNAL_COLUMNS,
    )


def _row(
    *,
    method: str,
    status: str,
    n_queries: int = 0,
    n_train: int = 0,
    n_test: int = 0,
    state_prediction_accuracy: float = math.nan,
    state_prediction_accuracy_ci_low: float = math.nan,
    state_prediction_accuracy_ci_high: float = math.nan,
    routing_utility: float = math.nan,
    observability_gap_closed: float = math.nan,
    mean_probe_cost_proxy: float = math.nan,
    regret_prediction_auc: float = math.nan,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "method": method,
        "status": status,
        "n_queries": int(n_queries),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "state_prediction_accuracy": state_prediction_accuracy,
        "state_prediction_accuracy_ci_low": state_prediction_accuracy_ci_low,
        "state_prediction_accuracy_ci_high": state_prediction_accuracy_ci_high,
        "routing_utility": routing_utility,
        "observability_gap_closed": observability_gap_closed,
        "mean_probe_cost_proxy": mean_probe_cost_proxy,
        "regret_prediction_auc": regret_prediction_auc,
        "notes": notes,
    }


def _mean_probe_cost(frame: pd.DataFrame) -> float:
    if "probe_cost_proxy" not in frame.columns:
        return math.nan
    return float(pd.to_numeric(frame["probe_cost_proxy"], errors="coerce").mean())


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _require_columns(frame: pd.DataFrame, columns: list[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")
