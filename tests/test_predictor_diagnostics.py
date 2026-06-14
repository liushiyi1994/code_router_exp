import numpy as np
import pandas as pd

from routecode.eval.predictor_diagnostics import (
    calibration_curve_table,
    expected_calibration_error,
    label_accuracy,
    utility_weighted_confusion,
)


def test_label_accuracy_counts_exact_label_matches():
    actual = pd.Series([0, 1, 1, 2], index=["q0", "q1", "q2", "q3"])
    predicted = pd.Series([0, 1, 2, 2], index=actual.index)

    assert label_accuracy(actual, predicted) == 0.75


def test_expected_calibration_error_bins_confidence_by_correctness():
    confidence = pd.Series([0.9, 0.8, 0.4, 0.3], index=["q0", "q1", "q2", "q3"])
    correct = pd.Series([1, 0, 1, 0], index=confidence.index)

    ece = expected_calibration_error(confidence, correct, n_bins=2)

    assert np.isclose(ece, 0.25)


def test_calibration_curve_table_reports_bin_confidence_and_accuracy():
    confidence = pd.Series([0.9, 0.8, 0.4, 0.3], index=["q0", "q1", "q2", "q3"])
    correct = pd.Series([1, 0, 1, 0], index=confidence.index)

    table = calibration_curve_table(confidence, correct, n_bins=2)

    assert list(table["bin"]) == ["[0.00, 0.50)", "[0.50, 1.00]"]
    assert np.isclose(table.loc[0, "mean_confidence"], 0.35)
    assert np.isclose(table.loc[0, "accuracy"], 0.5)


def test_utility_weighted_confusion_reports_regret_from_label_mistakes():
    utility = pd.DataFrame(
        {
            "cheap": [0.9, 0.1, 0.4],
            "strong": [0.2, 0.8, 0.7],
        },
        index=["q0", "q1", "q2"],
    )
    actual = pd.Series([0, 1, 1], index=utility.index)
    predicted = pd.Series([0, 0, 1], index=utility.index)
    label_to_model = {0: "cheap", 1: "strong"}

    table = utility_weighted_confusion(actual, predicted, utility, label_to_model)
    mistake = table[(table["true_label"] == 1) & (table["predicted_label"] == 0)].iloc[0]

    assert mistake["count"] == 1
    assert np.isclose(mistake["mean_true_label_utility"], 0.8)
    assert np.isclose(mistake["mean_predicted_label_utility"], 0.1)
    assert np.isclose(mistake["mean_regret"], 0.7)
