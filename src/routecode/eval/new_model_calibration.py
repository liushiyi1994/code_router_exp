from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class LabelCalibrationResult:
    label_to_model: dict[int, str]
    estimated_new_model_utility: pd.Series
    calibration_query_count: int


def sample_calibration_queries_per_label(
    labels: pd.Series,
    examples_per_label: int,
    seed: int = 0,
) -> pd.Index:
    """Sample up to r train queries per route label without replacement."""

    r = max(int(examples_per_label), 0)
    if r == 0 or labels.empty:
        return pd.Index([], name=labels.index.name)
    rng = np.random.default_rng(seed)
    sampled: list[object] = []
    for label in sorted(labels.dropna().unique()):
        query_ids = labels.index[labels == label].to_numpy()
        take = min(r, len(query_ids))
        if take == 0:
            continue
        chosen = rng.choice(query_ids, size=take, replace=False)
        sampled.extend(chosen.tolist())
    return pd.Index(sampled, name=labels.index.name)


def calibrate_new_model_by_label(
    labels: pd.Series,
    base_label_utility: pd.DataFrame,
    full_utility: pd.DataFrame,
    new_model_id: str,
    calibration_query_ids: pd.Index,
) -> LabelCalibrationResult:
    """Estimate held-out model utility by label and update label-to-model table.

    `full_utility` may contain the new model for all train rows, but this
    function reads the new-model column only for `calibration_query_ids`.
    """

    if new_model_id not in full_utility.columns:
        raise ValueError(f"Missing new model column: {new_model_id}")
    calibration_ids = pd.Index(calibration_query_ids).intersection(labels.index).intersection(full_utility.index)
    base_best = base_label_utility.idxmax(axis=1).astype(str).to_dict()
    estimates: dict[int, float] = {}
    label_to_model: dict[int, str] = {}
    global_base_best = str(base_label_utility.mean(axis=0).idxmax())

    for raw_label in base_label_utility.index:
        label = int(raw_label)
        label_ids = labels.index[labels == label]
        sampled_ids = calibration_ids.intersection(label_ids)
        if len(sampled_ids) == 0:
            estimate = float("nan")
        else:
            estimate = float(full_utility.loc[sampled_ids, new_model_id].mean())
        estimates[label] = estimate
        incumbent_model = str(base_best.get(raw_label, global_base_best))
        incumbent_utility = float(base_label_utility.loc[raw_label, incumbent_model])
        if not np.isnan(estimate) and estimate > incumbent_utility:
            label_to_model[label] = str(new_model_id)
        else:
            label_to_model[label] = incumbent_model

    return LabelCalibrationResult(
        label_to_model=label_to_model,
        estimated_new_model_utility=pd.Series(estimates, name=f"estimated_utility_{new_model_id}"),
        calibration_query_count=int(len(calibration_ids)),
    )


def budgeted_direct_oracle_labels(
    base_utility: pd.DataFrame,
    full_utility: pd.DataFrame,
    new_model_id: str,
    calibration_query_ids: pd.Index,
) -> pd.Series:
    """Training labels for a direct router under the same new-model budget.

    Old-model utilities are available for every train query. The new-model
    utility is available only on sampled calibration query IDs.
    """

    if new_model_id not in full_utility.columns:
        raise ValueError(f"Missing new model column: {new_model_id}")
    labels = base_utility.idxmax(axis=1).astype(str)
    calibration_ids = pd.Index(calibration_query_ids).intersection(base_utility.index).intersection(full_utility.index)
    if len(calibration_ids) == 0:
        return labels.rename("selected_model")
    candidate = pd.concat(
        [
            base_utility.loc[calibration_ids],
            full_utility.loc[calibration_ids, [new_model_id]],
        ],
        axis=1,
    )
    labels.loc[calibration_ids] = candidate.idxmax(axis=1).astype(str)
    return labels.rename("selected_model")


def selection_from_label_mapping(
    labels: pd.Series,
    label_to_model: dict[int, str],
    fallback_model: str,
) -> pd.Series:
    selected = [label_to_model.get(int(label), fallback_model) for label in labels]
    return pd.Series(selected, index=labels.index, name="selected_model")


def fit_predict_budgeted_direct_router(
    method: str,
    train_labels: pd.Series,
    train_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
    random_state: int = 0,
    max_iter: int = 1000,
    n_neighbors: int = 15,
) -> pd.Series:
    aligned_labels = train_labels.loc[train_embeddings.index].astype(str)
    if aligned_labels.nunique() == 1:
        predictions = [aligned_labels.iloc[0]] * len(test_embeddings)
        return pd.Series(predictions, index=test_embeddings.index, name="selected_model").astype(str)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_embeddings.to_numpy(dtype=float))
    x_test = scaler.transform(test_embeddings.to_numpy(dtype=float))
    method = method.lower()
    if method == "logistic":
        model = LogisticRegression(random_state=int(random_state), max_iter=int(max_iter))
    elif method == "svm":
        model = LinearSVC(random_state=int(random_state), max_iter=int(max_iter))
    elif method == "knn":
        model = KNeighborsClassifier(n_neighbors=min(max(1, int(n_neighbors)), len(train_embeddings)))
    elif method == "mlp":
        model = MLPClassifier(
            hidden_layer_sizes=(16,),
            solver="adam",
            learning_rate_init=0.01,
            n_iter_no_change=10,
            random_state=int(random_state),
            max_iter=min(int(max_iter), 200),
        )
    elif method in {"gradient_boosting", "gbt"}:
        model = GradientBoostingClassifier(
            random_state=int(random_state),
            n_estimators=50,
            max_depth=2,
        )
    else:
        raise ValueError(f"Unknown direct router method: {method}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(x_train, aligned_labels.to_numpy())
    predictions = model.predict(x_test)
    return pd.Series(predictions, index=test_embeddings.index, name="selected_model").astype(str)
