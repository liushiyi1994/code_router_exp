from __future__ import annotations

import math

import pandas as pd

from routecode.states.observability import compute_observability_gap_table


def _row(method: str, utility: float, **extra: object) -> dict[str, object]:
    row = {
        "method": method,
        "mean_utility": utility,
        "K": extra.pop("K", float("nan")),
        "alpha": extra.pop("alpha", float("nan")),
        "label_accuracy": extra.pop("label_accuracy", float("nan")),
        "mean_confidence": extra.pop("mean_confidence", float("nan")),
        "recovered_gap_vs_oracle": extra.pop("recovered_gap_vs_oracle", float("nan")),
    }
    row.update(extra)
    return row


def test_observability_gap_separates_oracle_state_from_deployable_assignment():
    recovered_gap = pd.DataFrame(
        [
            _row("best_single", 0.50),
            _row("query_oracle", 0.90),
        ]
    )
    predictability = pd.DataFrame(
        [
            _row("flat_routecode_utility_oracle", 0.86, K=16, recovered_gap_vs_oracle=0.90),
            _row("flat_routecode_logistic_label_predictor", 0.62, K=16, recovered_gap_vs_oracle=0.30),
            _row("d2_joint_oracle_labels", 0.74, K=16, alpha=3.0, recovered_gap_vs_oracle=0.60),
            _row(
                "d2_embedding_centroid",
                0.72,
                K=16,
                alpha=3.0,
                label_accuracy=0.95,
                mean_confidence=0.96,
                recovered_gap_vs_oracle=0.55,
            ),
        ]
    )

    table = compute_observability_gap_table(
        recovered_gap,
        predictability,
        result_id="toy",
    )
    by_comparison = table.set_index("comparison")

    flat = by_comparison.loc["flat_routecode_logistic_label_predictor"]
    assert flat["state_family"] == "flat_routecode"
    assert math.isclose(flat["state_observability_gap"], 0.24)
    assert math.isclose(flat["query_oracle_gap"], 0.28)
    assert math.isclose(flat["state_gap_closed"], 1 / 3)
    assert math.isclose(flat["full_gap_closed_vs_query_oracle"], 0.30)
    assert math.isclose(flat["oracle_state_recovered_gap_vs_oracle"], 0.90)
    assert "observable" in flat["interpretation"]

    recovered_gap_with_ci = recovered_gap.assign(
        utility_ci_low=[0.48, 0.88],
        utility_ci_high=[0.52, 0.92],
    )
    predictability_with_ci = predictability.assign(
        utility_ci_low=[0.84, 0.60, 0.72, 0.70],
        utility_ci_high=[0.88, 0.64, 0.76, 0.74],
    )
    ci_table = compute_observability_gap_table(
        recovered_gap_with_ci,
        predictability_with_ci,
        result_id="toy",
    ).set_index("comparison")
    ci_flat = ci_table.loc["flat_routecode_logistic_label_predictor"]
    assert math.isclose(ci_flat["oracle_state_mean_utility_ci_low"], 0.84)
    assert math.isclose(ci_flat["deployable_state_mean_utility_ci_high"], 0.64)
    assert math.isclose(ci_flat["state_observability_gap_ci_low"], 0.20)
    assert math.isclose(ci_flat["state_observability_gap_ci_high"], 0.28)
    assert math.isclose(ci_flat["query_oracle_gap_ci_low"], 0.24)
    assert math.isclose(ci_flat["query_oracle_gap_ci_high"], 0.32)

    d2 = by_comparison.loc["d2_embedding_centroid_alpha_3"]
    assert d2["state_family"] == "d2_predictability_constrained"
    assert math.isclose(d2["state_observability_gap"], 0.02)
    assert math.isclose(d2["state_gap_closed"], 11 / 12)
    assert math.isclose(d2["label_accuracy"], 0.95)


def test_observability_gap_requires_best_single_and_query_oracle_rows():
    recovered_gap = pd.DataFrame([_row("best_single", 0.50)])
    predictability = pd.DataFrame([_row("flat_routecode_utility_oracle", 0.80, K=4)])

    try:
        compute_observability_gap_table(recovered_gap, predictability, result_id="bad")
    except ValueError as exc:
        assert "query_oracle" in str(exc)
    else:
        raise AssertionError("Expected missing query_oracle row to raise ValueError")
