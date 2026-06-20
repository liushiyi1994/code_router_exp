from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.matrix import Matrices


@dataclass(frozen=True)
class RouteCodePolicyInputs:
    before_beliefs: pd.DataFrame
    after_beliefs: pd.DataFrame
    state_model_utility: pd.DataFrame
    query_model_utility: pd.DataFrame
    probe_cost: pd.Series
    predicted_gain: pd.Series
    metadata: dict[str, int | float | str]


def build_routecode_policy_inputs(
    *,
    train: Matrices,
    embeddings: pd.DataFrame,
    query_model_utility: pd.DataFrame,
    k: int,
    alpha: float,
    beta: float = 0.0,
    random_state: int = 0,
    max_iter: int = 25,
    refinement_iter: int = 10,
) -> RouteCodePolicyInputs:
    """Export deterministic RouteCode beliefs for the standard policy evaluator.

    The codebook is fit on train only. Evaluation labels are predicted from
    query embeddings, not from held-out query-model utility.
    """

    missing = query_model_utility.index.difference(embeddings.index)
    if len(missing):
        raise ValueError(f"Missing embeddings for {len(missing)} policy queries")
    codebook = PredictabilityConstrainedRouteCode(
        n_labels=int(k),
        alpha=float(alpha),
        beta=float(beta),
        random_state=int(random_state),
        max_iter=int(max_iter),
        refinement_iter=int(refinement_iter),
    ).fit(train.query_info, train.utility, embeddings)
    if codebook.label_utility_ is None:
        raise RuntimeError("Failed to fit RouteCode label utility table")
    labels = codebook.predict_labels(embeddings.loc[query_model_utility.index]).astype(int)
    state_count = int(codebook.effective_labels)
    state_names = [_state_name(label) for label in range(state_count)]
    beliefs = _one_hot_beliefs(labels, state_names)
    state_model_utility = codebook.label_utility_.copy()
    state_model_utility.index = pd.Index(state_names, name="state_label")
    probe_cost = pd.Series(0.0, index=query_model_utility.index, name="probe_cost")
    predicted_gain = pd.Series(0.0, index=query_model_utility.index, name="predicted_gain")
    metadata = {
        "k": int(k),
        "alpha": float(alpha),
        "beta": float(beta),
        "effective_labels": state_count,
        "train_rows": int(len(train.utility)),
        "policy_rows": int(len(query_model_utility)),
        "belief_type": "one_hot_routecode_embedding_predicted",
    }
    return RouteCodePolicyInputs(
        before_beliefs=beliefs,
        after_beliefs=beliefs.copy(),
        state_model_utility=state_model_utility,
        query_model_utility=query_model_utility.copy(),
        probe_cost=probe_cost,
        predicted_gain=predicted_gain,
        metadata=metadata,
    )


def _one_hot_beliefs(labels: pd.Series, state_names: list[str]) -> pd.DataFrame:
    beliefs = pd.DataFrame(0.0, index=labels.index.astype(str), columns=state_names)
    beliefs.index.name = "query_id"
    for query_id, label in labels.items():
        state = _state_name(int(label))
        if state not in beliefs.columns:
            raise ValueError(f"Predicted label {label} is outside exported state range")
        beliefs.loc[str(query_id), state] = 1.0
    if not np.allclose(beliefs.sum(axis=1).to_numpy(dtype=float), 1.0):
        raise RuntimeError("RouteCode belief rows are not normalized")
    return beliefs


def _state_name(label: int) -> str:
    return f"z{int(label)}"
