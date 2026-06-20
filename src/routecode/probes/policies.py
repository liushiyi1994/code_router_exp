from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
import pandas as pd

from routecode.metrics import bootstrap_mean_ci


PROBEROUTE_POLICY_COLUMNS = [
    "policy",
    "status",
    "n_queries",
    "mean_utility",
    "mean_utility_ci_low",
    "mean_utility_ci_high",
    "mean_net_utility",
    "mean_net_utility_ci_low",
    "mean_net_utility_ci_high",
    "mean_quality",
    "mean_model_cost",
    "mean_probe_cost_proxy",
    "fraction_probed",
    "mean_oracle_regret",
    "observability_gap_closed",
    "observability_gap_closed_ci_low",
    "observability_gap_closed_ci_high",
    "mean_latency_sec",
    "notes",
]


class ProbePolicy(Protocol):
    name: str

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        """Return a boolean probe decision indexed by query_id."""


@dataclass(frozen=True)
class NeverProbePolicy:
    name: str = "never_probe"

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        del after_beliefs, probe_cost, predicted_gain, before_value, after_value
        return pd.Series(False, index=before_beliefs.index, name=self.name)


@dataclass(frozen=True)
class AlwaysProbePolicy:
    name: str = "always_probe"

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        del after_beliefs, probe_cost, predicted_gain, before_value, after_value
        return pd.Series(True, index=before_beliefs.index, name=self.name)


@dataclass(frozen=True)
class EntropyThresholdPolicy:
    threshold: float
    name: str = "entropy_threshold"

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        del after_beliefs, probe_cost, predicted_gain, before_value, after_value
        entropy = before_beliefs.apply(belief_entropy, axis=1)
        return (entropy >= float(self.threshold)).rename(self.name)


@dataclass(frozen=True)
class MarginThresholdPolicy:
    threshold: float
    name: str = "margin_threshold"

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        del after_beliefs, probe_cost, predicted_gain, before_value, after_value
        margins = before_beliefs.apply(belief_margin, axis=1)
        return (margins <= float(self.threshold)).rename(self.name)


@dataclass(frozen=True)
class VOIProbePolicy:
    min_net_gain: float = 0.0
    name: str = "voi_probe"

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        del after_beliefs, before_value, after_value
        if predicted_gain is None:
            raise ValueError("VOIProbePolicy requires predicted_gain")
        costs = _aligned_series(probe_cost, before_beliefs.index, fill_value=0.0)
        gains = _aligned_series(predicted_gain, before_beliefs.index, fill_value=0.0)
        return ((gains - costs) > float(self.min_net_gain)).rename(self.name)


@dataclass(frozen=True)
class OracleProbePolicy:
    name: str = "oracle_probe"

    def decide(
        self,
        before_beliefs: pd.DataFrame,
        after_beliefs: pd.DataFrame | None = None,
        *,
        probe_cost: pd.Series | None = None,
        predicted_gain: pd.Series | None = None,
        before_value: pd.Series | None = None,
        after_value: pd.Series | None = None,
    ) -> pd.Series:
        del after_beliefs, predicted_gain
        if before_value is None or after_value is None:
            raise ValueError("OracleProbePolicy requires before_value and after_value")
        costs = _aligned_series(probe_cost, before_beliefs.index, fill_value=0.0)
        before = _aligned_series(before_value, before_beliefs.index, fill_value=0.0)
        after = _aligned_series(after_value, before_beliefs.index, fill_value=0.0)
        return ((after - before - costs) > 0.0).rename(self.name)


def belief_entropy(belief: pd.Series | np.ndarray) -> float:
    values = np.asarray(belief, dtype=float)
    values = values[np.isfinite(values) & (values > 0.0)]
    total = float(values.sum())
    if total <= 0.0:
        return 0.0
    probs = values / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return 0.0 if abs(entropy) < 1e-12 else entropy


def belief_margin(belief: pd.Series | np.ndarray) -> float:
    values = np.sort(np.asarray(belief, dtype=float))[::-1]
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(values[0] - values[1])


def expected_model_utility_from_belief(beliefs: pd.DataFrame, state_model_utility: pd.DataFrame) -> pd.DataFrame:
    state_columns = _state_columns(beliefs, state_model_utility)
    utilities = beliefs[state_columns].to_numpy(dtype=float) @ state_model_utility.loc[state_columns].to_numpy(dtype=float)
    return pd.DataFrame(utilities, index=beliefs.index, columns=state_model_utility.columns)


def select_models_from_belief(beliefs: pd.DataFrame, state_model_utility: pd.DataFrame) -> pd.Series:
    expected = expected_model_utility_from_belief(beliefs, state_model_utility)
    return expected.idxmax(axis=1).rename("selected_model")


def evaluate_proberoute_policies(
    *,
    policies: list[ProbePolicy],
    before_beliefs: pd.DataFrame,
    after_beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    query_model_utility: pd.DataFrame,
    probe_cost: pd.Series | None = None,
    predicted_gain: pd.Series | None = None,
    query_model_quality: pd.DataFrame | None = None,
    query_model_cost: pd.DataFrame | None = None,
    latency_sec: pd.Series | None = None,
    baseline_mean_utility: float | None = None,
    oracle_reference_mean_utility: float | None = None,
) -> pd.DataFrame:
    """Evaluate ProbeRoute++ policies through state beliefs, not direct probe-to-model labels."""

    _validate_policy_inputs(before_beliefs, after_beliefs, state_model_utility, query_model_utility)
    index = before_beliefs.index
    probe_cost = _aligned_series(probe_cost, index, fill_value=0.0)
    latency = _aligned_series(latency_sec, index, fill_value=0.0)
    before_selected = select_models_from_belief(before_beliefs, state_model_utility)
    after_selected = select_models_from_belief(after_beliefs, state_model_utility)
    before_value = _selected_values(query_model_utility, before_selected)
    after_value = _selected_values(query_model_utility, after_selected)
    oracle_value = query_model_utility.max(axis=1).reindex(index)

    rows: list[dict[str, object]] = []
    for policy_index, policy in enumerate(policies):
        try:
            decisions = policy.decide(
                before_beliefs,
                after_beliefs,
                probe_cost=probe_cost,
                predicted_gain=predicted_gain,
                before_value=before_value,
                after_value=after_value,
            ).reindex(index).fillna(False).astype(bool)
        except Exception as exc:
            rows.append(
                _blocked_policy_row(
                    policy.name,
                    "blocked_policy_failed",
                    n_queries=len(index),
                    notes=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        selected = before_selected.where(~decisions, after_selected)
        utility = _selected_values(query_model_utility, selected)
        net_utility = utility - probe_cost.where(decisions, 0.0)
        quality = _selected_values(query_model_quality, selected) if query_model_quality is not None else None
        model_cost = _selected_values(query_model_cost, selected) if query_model_cost is not None else None
        utility_ci_low, utility_ci_high = _mean_ci(utility, seed=policy_index)
        net_ci_low, net_ci_high = _mean_ci(net_utility, seed=1000 + policy_index)
        gap_ci_low, gap_ci_high = _gap_ci(
            net_ci_low,
            net_ci_high,
            baseline_mean_utility,
            oracle_reference_mean_utility,
        )
        rows.append(
            {
                "policy": policy.name,
                "status": "executed",
                "n_queries": int(len(index)),
                "mean_utility": float(utility.mean()),
                "mean_utility_ci_low": min(utility_ci_low, float(utility.mean())),
                "mean_utility_ci_high": max(utility_ci_high, float(utility.mean())),
                "mean_net_utility": float(net_utility.mean()),
                "mean_net_utility_ci_low": min(net_ci_low, float(net_utility.mean())),
                "mean_net_utility_ci_high": max(net_ci_high, float(net_utility.mean())),
                "mean_quality": _mean_or_nan(quality),
                "mean_model_cost": _mean_or_nan(model_cost),
                "mean_probe_cost_proxy": float(probe_cost.where(decisions, 0.0).mean()),
                "fraction_probed": float(decisions.mean()),
                "mean_oracle_regret": float((oracle_value - net_utility).mean()),
                "observability_gap_closed": _gap_closed(
                    float(net_utility.mean()),
                    baseline_mean_utility,
                    oracle_reference_mean_utility,
                ),
                "observability_gap_closed_ci_low": gap_ci_low,
                "observability_gap_closed_ci_high": gap_ci_high,
                "mean_latency_sec": float(latency.where(decisions, 0.0).mean()),
                "notes": "Routed through state belief and state-model utility.",
            }
        )
    return pd.DataFrame(rows, columns=PROBEROUTE_POLICY_COLUMNS)


def blocked_policy_table(status: str, notes: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _blocked_policy_row("never_probe", status, notes=notes),
            _blocked_policy_row("always_probe", status, notes=notes),
            _blocked_policy_row("entropy_threshold", status, notes=notes),
            _blocked_policy_row("margin_threshold", status, notes=notes),
            _blocked_policy_row("voi_probe", status, notes=notes),
            _blocked_policy_row("oracle_probe", status, notes=notes),
        ],
        columns=PROBEROUTE_POLICY_COLUMNS,
    )


def default_policy_set() -> list[ProbePolicy]:
    return [
        NeverProbePolicy(),
        AlwaysProbePolicy(),
        EntropyThresholdPolicy(threshold=1.0),
        MarginThresholdPolicy(threshold=0.2),
        VOIProbePolicy(),
        OracleProbePolicy(),
    ]


def _validate_policy_inputs(
    before_beliefs: pd.DataFrame,
    after_beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    query_model_utility: pd.DataFrame,
) -> None:
    if before_beliefs.empty or after_beliefs.empty:
        raise ValueError("before_beliefs and after_beliefs must be non-empty")
    if not before_beliefs.index.equals(after_beliefs.index):
        raise ValueError("before_beliefs and after_beliefs must share query_id index")
    if not before_beliefs.index.isin(query_model_utility.index).all():
        raise ValueError("query_model_utility must contain every belief query_id")
    _state_columns(before_beliefs, state_model_utility)


def _state_columns(beliefs: pd.DataFrame, state_model_utility: pd.DataFrame) -> list[str]:
    state_columns = [column for column in beliefs.columns if column in state_model_utility.index]
    if not state_columns:
        raise ValueError("No belief state columns match state_model_utility index")
    return state_columns


def _selected_values(matrix: pd.DataFrame | None, selected: pd.Series) -> pd.Series:
    if matrix is None:
        return pd.Series(math.nan, index=selected.index)
    values = [float(matrix.loc[query_id, model_id]) for query_id, model_id in selected.items()]
    return pd.Series(values, index=selected.index)


def _aligned_series(series: pd.Series | None, index: pd.Index, *, fill_value: float) -> pd.Series:
    if series is None:
        return pd.Series(float(fill_value), index=index)
    return pd.Series(series, dtype=float).reindex(index).fillna(float(fill_value))


def _mean_or_nan(values: pd.Series | None) -> float:
    if values is None:
        return math.nan
    return float(values.mean())


def _gap_closed(method: float, baseline: float | None, reference: float | None) -> float:
    if baseline is None or reference is None:
        return math.nan
    denominator = float(reference) - float(baseline)
    if denominator <= 1e-12:
        return 0.0
    return float((method - float(baseline)) / denominator)


def _gap_ci(
    low: float,
    high: float,
    baseline: float | None,
    reference: float | None,
) -> tuple[float, float]:
    low_gap = _gap_closed(low, baseline, reference)
    high_gap = _gap_closed(high, baseline, reference)
    if math.isnan(low_gap) or math.isnan(high_gap):
        return math.nan, math.nan
    return min(low_gap, high_gap), max(low_gap, high_gap)


def _mean_ci(values: pd.Series, *, seed: int) -> tuple[float, float]:
    if values.empty:
        return math.nan, math.nan
    return bootstrap_mean_ci(values, n_bootstrap=500, ci=0.95, seed=seed)


def _blocked_policy_row(policy: str, status: str, *, n_queries: int = 0, notes: str = "") -> dict[str, object]:
    return {
        "policy": policy,
        "status": status,
        "n_queries": int(n_queries),
        "mean_utility": math.nan,
        "mean_utility_ci_low": math.nan,
        "mean_utility_ci_high": math.nan,
        "mean_net_utility": math.nan,
        "mean_net_utility_ci_low": math.nan,
        "mean_net_utility_ci_high": math.nan,
        "mean_quality": math.nan,
        "mean_model_cost": math.nan,
        "mean_probe_cost_proxy": math.nan,
        "fraction_probed": math.nan,
        "mean_oracle_regret": math.nan,
        "observability_gap_closed": math.nan,
        "observability_gap_closed_ci_low": math.nan,
        "observability_gap_closed_ci_high": math.nan,
        "mean_latency_sec": math.nan,
        "notes": notes,
    }
