import numpy as np
import pandas as pd

from routecode.eval.residuals import residual_concentration_table, residual_query_table, residual_risk_coverage_table


def test_residual_concentration_table_reports_top_regret_mass():
    oracle = pd.Series([1.0, 1.0, 1.0, 1.0], index=["q0", "q1", "q2", "q3"])
    selected = pd.Series([0.0, 0.5, 1.0, 1.0], index=oracle.index)
    table = residual_concentration_table(oracle, selected, fractions=[0.25, 0.5])

    assert table.loc[table["top_fraction"] == 0.25, "regret_mass_fraction"].iloc[0] == 2 / 3
    assert table.loc[table["top_fraction"] == 0.5, "regret_mass_fraction"].iloc[0] == 1.0


def test_residual_query_table_includes_margin_and_centroid_distance():
    utility = pd.DataFrame(
        {"cheap": [0.9, 0.4], "strong": [0.5, 0.8]},
        index=["q0", "q1"],
    )
    selected_models = pd.Series(["cheap", "cheap"], index=utility.index)
    labels = pd.Series([0, 0], index=utility.index)
    embeddings = pd.DataFrame([[0.0, 0.0], [3.0, 4.0]], index=utility.index)
    table = residual_query_table(utility, selected_models, labels, embeddings)

    assert np.isclose(table.loc["q1", "regret"], 0.4)
    assert np.isclose(table.loc["q1", "oracle_margin"], 0.4)
    assert "distance_to_label_centroid" in table.columns


def test_residual_risk_coverage_table_reports_regret_capture_and_auc():
    residuals = pd.DataFrame(
        {
            "regret": [5.0, 4.0, 0.0, 0.0],
            "useful_score": [0.9, 0.8, 0.2, 0.1],
            "inverted_score": [0.1, 0.2, 0.8, 0.9],
        },
        index=["q0", "q1", "q2", "q3"],
    )

    table = residual_risk_coverage_table(
        residuals,
        score_columns=["useful_score", "inverted_score"],
        top_fractions=[0.5],
    )
    useful = table[table["score"] == "useful_score"].iloc[0]
    inverted = table[table["score"] == "inverted_score"].iloc[0]

    assert useful["n_flagged"] == 2
    assert useful["regret_mass_fraction"] == 1.0
    assert useful["positive_regret_recall"] == 1.0
    assert useful["auc_regret_positive"] == 1.0
    assert inverted["regret_mass_fraction"] == 0.0
    assert inverted["auc_regret_positive"] == 0.0
