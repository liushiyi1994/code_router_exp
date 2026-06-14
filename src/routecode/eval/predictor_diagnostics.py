from __future__ import annotations

import numpy as np
import pandas as pd


def label_accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    actual, predicted = _align_labels(actual, predicted)
    if actual.empty:
        return 0.0
    return float((actual.astype(str) == predicted.astype(str)).mean())


def expected_calibration_error(confidence: pd.Series, correct: pd.Series, n_bins: int = 10) -> float:
    aligned = pd.concat(
        [
            confidence.rename("confidence").astype(float),
            correct.rename("correct").astype(float),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return 0.0
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (aligned["confidence"] >= low) & (aligned["confidence"] <= high)
        else:
            mask = (aligned["confidence"] >= low) & (aligned["confidence"] < high)
        bucket = aligned[mask]
        if bucket.empty:
            continue
        weight = len(bucket) / len(aligned)
        ece += weight * abs(float(bucket["confidence"].mean()) - float(bucket["correct"].mean()))
    return float(ece)


def calibration_curve_table(confidence: pd.Series, correct: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    aligned = pd.concat(
        [
            confidence.rename("confidence").astype(float),
            correct.rename("correct").astype(float),
        ],
        axis=1,
        join="inner",
    ).dropna()
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (aligned["confidence"] >= low) & (aligned["confidence"] <= high)
            label = f"[{low:.2f}, {high:.2f}]"
        else:
            mask = (aligned["confidence"] >= low) & (aligned["confidence"] < high)
            label = f"[{low:.2f}, {high:.2f})"
        bucket = aligned[mask]
        rows.append(
            {
                "bin": label,
                "bin_low": float(low),
                "bin_high": float(high),
                "count": int(len(bucket)),
                "mean_confidence": float(bucket["confidence"].mean()) if not bucket.empty else float("nan"),
                "accuracy": float(bucket["correct"].mean()) if not bucket.empty else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def utility_weighted_confusion(
    actual_labels: pd.Series,
    predicted_labels: pd.Series,
    utility: pd.DataFrame,
    label_to_model: dict[int | str, str],
) -> pd.DataFrame:
    actual, predicted = _align_labels(actual_labels, predicted_labels)
    rows = []
    for (true_label, predicted_label), query_ids in actual.groupby([actual, predicted]).groups.items():
        true_model = _label_model(label_to_model, true_label)
        predicted_model = _label_model(label_to_model, predicted_label)
        ids = list(query_ids)
        true_utility = utility.loc[ids, true_model].astype(float)
        predicted_utility = utility.loc[ids, predicted_model].astype(float)
        rows.append(
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "true_model": true_model,
                "predicted_model": predicted_model,
                "count": len(ids),
                "mean_true_label_utility": float(true_utility.mean()),
                "mean_predicted_label_utility": float(predicted_utility.mean()),
                "mean_regret": float((true_utility - predicted_utility).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["true_label", "predicted_label"]).reset_index(drop=True)


def _align_labels(actual: pd.Series, predicted: pd.Series) -> tuple[pd.Series, pd.Series]:
    aligned = pd.concat([actual.rename("actual"), predicted.rename("predicted")], axis=1, join="inner").dropna()
    return aligned["actual"], aligned["predicted"]


def _label_model(label_to_model: dict[int | str, str], label: int | str) -> str:
    if label in label_to_model:
        return label_to_model[label]
    try:
        return label_to_model[int(label)]
    except (KeyError, ValueError, TypeError):
        return label_to_model[str(label)]
