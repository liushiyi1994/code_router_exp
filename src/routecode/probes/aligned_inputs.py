from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.matrix import Matrices
from routecode.probes.policies import expected_model_utility_from_belief
from routecode.probes.probe_features import PROBE_FEATURE_COLUMNS, compute_knn_uncertainty


@dataclass(frozen=True)
class AlignedOfflineInputs:
    probe_features: pd.DataFrame
    state_targets: pd.DataFrame
    query_features: pd.DataFrame
    before_beliefs: pd.DataFrame
    after_beliefs: pd.DataFrame
    state_model_utility: pd.DataFrame
    query_model_utility: pd.DataFrame
    probe_cost: pd.Series
    predicted_gain: pd.Series


def build_aligned_offline_inputs(
    *,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    k: int = 16,
    alpha: float = 3.0,
    beta: float = 0.0,
    random_state: int = 0,
    max_iter: int = 25,
    refinement_iter: int = 10,
    n_neighbors: int = 15,
    probe_cost_proxy: float = 0.0001,
) -> AlignedOfflineInputs:
    """Build aligned offline probe/policy inputs from existing benchmark matrices.

    The generated probe features are train-only kNN uncertainty features, not
    true local model probes. They are useful for plumbing and split discipline.
    """

    all_query_ids = train.utility.index.append(test.utility.index)
    aligned_embeddings = embeddings.loc[all_query_ids]
    codebook = PredictabilityConstrainedRouteCode(
        n_labels=k,
        alpha=alpha,
        beta=beta,
        random_state=random_state,
        max_iter=max_iter,
        refinement_iter=refinement_iter,
    ).fit(train.query_info, train.utility, embeddings)
    if codebook.train_labels_ is None or codebook.label_utility_ is None:
        raise RuntimeError("Failed to fit route-state codebook")

    train_labels = codebook.train_labels_.astype(int)
    test_labels = codebook.predict_joint_labels(test.utility, embeddings.loc[test.utility.index]).astype(int)
    state_targets = pd.concat(
        [
            pd.DataFrame(
                {"query_id": train_labels.index, "state_label": train_labels.to_numpy(), "split": "train"}
            ),
            pd.DataFrame({"query_id": test_labels.index, "state_label": test_labels.to_numpy(), "split": "test"}),
        ],
        ignore_index=True,
    )
    query_features = aligned_embeddings.reset_index().rename(columns={aligned_embeddings.index.name or "index": "query_id"})
    probe_features = _offline_knn_probe_features(
        train=train,
        test=test,
        embeddings=aligned_embeddings,
        train_labels=train_labels,
        k_neighbors=n_neighbors,
        n_states=codebook.effective_labels,
        probe_cost_proxy=probe_cost_proxy,
    )
    state_names = [_state_name(label) for label in range(codebook.effective_labels)]
    state_model_utility = codebook.label_utility_.copy()
    state_model_utility.index = pd.Index(state_names, name="state_label")

    before_beliefs = _fit_belief_model(
        train_features=embeddings.loc[train.utility.index],
        train_labels=train_labels,
        test_features=embeddings.loc[test.utility.index],
        state_names=state_names,
        random_state=random_state,
    )
    train_probe = _probe_numeric_matrix(probe_features, train.utility.index)
    test_probe = _probe_numeric_matrix(probe_features, test.utility.index)
    after_beliefs = _fit_belief_model(
        train_features=pd.concat([embeddings.loc[train.utility.index], train_probe], axis=1),
        train_labels=train_labels,
        test_features=pd.concat([embeddings.loc[test.utility.index], test_probe], axis=1),
        state_names=state_names,
        random_state=random_state,
    )
    before_value = expected_model_utility_from_belief(before_beliefs, state_model_utility).max(axis=1)
    after_value = expected_model_utility_from_belief(after_beliefs, state_model_utility).max(axis=1)
    probe_cost = pd.Series(float(probe_cost_proxy), index=test.utility.index, name="probe_cost")
    predicted_gain = (after_value - before_value).rename("predicted_gain")
    return AlignedOfflineInputs(
        probe_features=probe_features,
        state_targets=state_targets,
        query_features=query_features,
        before_beliefs=before_beliefs,
        after_beliefs=after_beliefs,
        state_model_utility=state_model_utility,
        query_model_utility=test.utility.copy(),
        probe_cost=probe_cost,
        predicted_gain=predicted_gain,
    )


def _offline_knn_probe_features(
    *,
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    train_labels: pd.Series,
    k_neighbors: int,
    n_states: int,
    probe_cost_proxy: float,
) -> pd.DataFrame:
    train_label_table = pd.DataFrame(
        {
            "query_id": train_labels.index,
            "route_label": train_labels.to_numpy(),
            "winner_model_id": train.utility.idxmax(axis=1).astype(str).to_numpy(),
        }
    )
    uncertainty = compute_knn_uncertainty(
        embeddings=embeddings.reset_index().rename(columns={embeddings.index.name or "index": "query_id"}),
        labels=train_label_table,
        train_query_ids=train.utility.index,
        target_query_ids=train.utility.index.append(test.utility.index),
        k=k_neighbors,
    ).set_index("query_id")
    rows: list[dict[str, object]] = []
    created_at = datetime.now(UTC).isoformat()
    state_denominator = math.log2(max(int(n_states), 2))
    winner_denominator = math.log2(max(len(train.model_ids), 2))
    for query_id in train.utility.index.append(test.utility.index):
        label_entropy = float(uncertainty.loc[query_id, "knn_label_entropy"])
        winner_entropy = float(uncertainty.loc[query_id, "knn_winner_entropy"])
        rows.append(
            {
                "query_id": str(query_id),
                "probe_id": "offline_knn_uncertainty",
                "probe_type": "offline_knn_uncertainty",
                "probe_model_id": "train_embedding_knn",
                "prompt_template": "none",
                "generation_params_json": "{}",
                "raw_probe_output": "",
                "parsed_probe_answer": "",
                "self_confidence": _confidence_from_entropy(label_entropy, state_denominator),
                "logprob_mean": math.nan,
                "entropy_proxy": label_entropy,
                "agreement_score": _confidence_from_entropy(winner_entropy, winner_denominator),
                "knn_label_entropy": label_entropy,
                "knn_winner_entropy": winner_entropy,
                "latency_sec": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "probe_cost_proxy": float(probe_cost_proxy),
                "error_type": "",
                "error_message": "",
                "created_at": created_at,
            }
        )
    return pd.DataFrame(rows, columns=PROBE_FEATURE_COLUMNS)


def _fit_belief_model(
    *,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    test_features: pd.DataFrame,
    state_names: list[str],
    random_state: int,
) -> pd.DataFrame:
    labels = train_labels.loc[train_features.index].astype(int)
    if labels.nunique() == 1:
        only = _state_name(int(labels.iloc[0]))
        return pd.DataFrame(
            {state: [1.0 if state == only else 0.0] * len(test_features) for state in state_names},
            index=test_features.index,
        )
    feature_columns = list(train_features.columns)
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
    model.fit(train_features[feature_columns], labels.to_numpy())
    probabilities = model.predict_proba(test_features[feature_columns])
    result = pd.DataFrame(0.0, index=test_features.index, columns=state_names)
    for class_index, raw_label in enumerate(model.named_steps["classifier"].classes_):
        result[_state_name(int(raw_label))] = probabilities[:, class_index]
    row_sums = result.sum(axis=1).replace(0.0, 1.0)
    return result.div(row_sums, axis=0)


def _probe_numeric_matrix(probe_features: pd.DataFrame, query_ids: pd.Index) -> pd.DataFrame:
    columns = ["self_confidence", "agreement_score", "knn_label_entropy", "knn_winner_entropy", "probe_cost_proxy"]
    frame = probe_features.set_index("query_id")[columns].apply(pd.to_numeric, errors="coerce")
    return frame.loc[query_ids.astype(str)].set_index(query_ids)


def _confidence_from_entropy(entropy: float, denominator: float) -> float:
    if math.isnan(entropy) or denominator <= 0.0:
        return math.nan
    return float(max(0.0, min(1.0, 1.0 - entropy / denominator)))


def _state_name(label: int) -> str:
    return f"z{int(label)}"
