from __future__ import annotations

import math

import pandas as pd

from routecode.probes.policies import (
    AlwaysProbePolicy,
    EntropyThresholdPolicy,
    MarginThresholdPolicy,
    NeverProbePolicy,
    OracleProbePolicy,
    PROBEROUTE_POLICY_COLUMNS,
    VOIProbePolicy,
    belief_entropy,
    evaluate_proberoute_policies,
    select_models_from_belief,
)


def _beliefs() -> tuple[pd.DataFrame, pd.DataFrame]:
    before = pd.DataFrame(
        {
            "z0": [0.9, 0.6],
            "z1": [0.1, 0.4],
        },
        index=pd.Index(["q0", "q1"], name="query_id"),
    )
    after = pd.DataFrame(
        {
            "z0": [0.85, 0.05],
            "z1": [0.15, 0.95],
        },
        index=before.index,
    )
    return before, after


def _state_utility() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cheap": [0.8, 0.1],
            "strong": [0.2, 0.9],
        },
        index=pd.Index(["z0", "z1"], name="state_label"),
    )


def _query_utility() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cheap": [0.75, 0.05],
            "strong": [0.3, 0.95],
        },
        index=pd.Index(["q0", "q1"], name="query_id"),
    )


def test_state_mediated_selection_uses_belief_and_state_utility():
    before, _ = _beliefs()

    selected = select_models_from_belief(before, _state_utility())

    assert selected.to_dict() == {"q0": "cheap", "q1": "cheap"}


def test_probe_policies_make_expected_decisions():
    before, after = _beliefs()
    probe_cost = pd.Series({"q0": 0.02, "q1": 0.02})
    predicted_gain = pd.Series({"q0": 0.01, "q1": 0.25})
    before_value = pd.Series({"q0": 0.74, "q1": 0.49})
    after_value = pd.Series({"q0": 0.71, "q1": 0.86})

    assert NeverProbePolicy().decide(before, after, probe_cost=probe_cost).to_dict() == {"q0": False, "q1": False}
    assert AlwaysProbePolicy().decide(before, after, probe_cost=probe_cost).to_dict() == {"q0": True, "q1": True}
    assert EntropyThresholdPolicy(threshold=0.9).decide(before, after, probe_cost=probe_cost).to_dict() == {
        "q0": False,
        "q1": True,
    }
    assert MarginThresholdPolicy(threshold=0.2).decide(before, after, probe_cost=probe_cost).to_dict() == {
        "q0": False,
        "q1": True,
    }
    assert VOIProbePolicy().decide(
        before,
        after,
        probe_cost=probe_cost,
        predicted_gain=predicted_gain,
    ).to_dict() == {"q0": False, "q1": True}
    assert OracleProbePolicy().decide(
        before,
        after,
        probe_cost=probe_cost,
        before_value=before_value,
        after_value=after_value,
    ).to_dict() == {"q0": False, "q1": True}
    assert math.isclose(belief_entropy(before.loc["q0"]), 0.4689955935892812)


def test_evaluate_proberoute_policies_accounts_for_probe_cost_and_oracle_regret():
    before, after = _beliefs()
    probe_cost = pd.Series({"q0": 0.02, "q1": 0.02})
    predicted_gain = pd.Series({"q0": -0.01, "q1": 0.50})
    policies = [
        NeverProbePolicy(),
        AlwaysProbePolicy(),
        VOIProbePolicy(),
        OracleProbePolicy(),
    ]

    table = evaluate_proberoute_policies(
        policies=policies,
        before_beliefs=before,
        after_beliefs=after,
        state_model_utility=_state_utility(),
        query_model_utility=_query_utility(),
        probe_cost=probe_cost,
        predicted_gain=predicted_gain,
        baseline_mean_utility=0.40,
        oracle_reference_mean_utility=0.85,
    )

    assert list(table.columns) == PROBEROUTE_POLICY_COLUMNS
    assert set(table["status"]) == {"executed"}
    never = table[table["policy"] == "never_probe"].iloc[0]
    always = table[table["policy"] == "always_probe"].iloc[0]
    voi = table[table["policy"] == "voi_probe"].iloc[0]
    oracle = table[table["policy"] == "oracle_probe"].iloc[0]

    assert never["fraction_probed"] == 0.0
    assert never["mean_net_utility_ci_low"] <= never["mean_net_utility"] <= never["mean_net_utility_ci_high"]
    assert always["fraction_probed"] == 1.0
    assert voi["fraction_probed"] == 0.5
    assert voi["mean_utility_ci_low"] <= voi["mean_utility"] <= voi["mean_utility_ci_high"]
    assert voi["mean_net_utility_ci_low"] <= voi["mean_net_utility"] <= voi["mean_net_utility_ci_high"]
    assert voi["observability_gap_closed_ci_low"] <= voi["observability_gap_closed"] <= voi["observability_gap_closed_ci_high"]
    assert oracle["fraction_probed"] == 0.5
    assert voi["mean_net_utility"] > never["mean_net_utility"]
    assert voi["mean_oracle_regret"] < never["mean_oracle_regret"]
