from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class LabelCalibrationResult:
    label_to_model: dict[int, str]
    estimated_new_model_utility: pd.Series
    calibration_query_count: int


@dataclass(frozen=True)
class ActiveStateCalibrationResult:
    label_to_model: dict[object, str]
    posterior: pd.DataFrame
    table_update: pd.DataFrame
    selected_queries: pd.DataFrame
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


def sample_random_calibration_queries(
    labels: pd.Series,
    total_budget: int,
    seed: int = 0,
) -> pd.Index:
    """Sample a fixed number of calibration queries without label stratification."""

    budget = max(int(total_budget), 0)
    if budget == 0 or labels.empty:
        return pd.Index([], name=labels.index.name)
    rng = np.random.default_rng(seed)
    query_ids = labels.index.to_numpy()
    take = min(budget, len(query_ids))
    chosen = rng.choice(query_ids, size=take, replace=False)
    return pd.Index(chosen.tolist(), name=labels.index.name)


def sample_dataset_stratified_calibration_queries(
    labels: pd.Series,
    query_info: pd.DataFrame,
    total_budget: int,
    seed: int = 0,
    dataset_column: str = "dataset",
) -> pd.Index:
    """Sample calibration queries by cycling through train dataset groups."""

    if dataset_column not in query_info.columns:
        return sample_random_calibration_queries(labels, total_budget=total_budget, seed=seed)
    aligned = query_info.reindex(labels.index)
    groups = aligned[dataset_column].fillna("unknown").astype(str)
    return _round_robin_sample_by_group(groups, total_budget=total_budget, seed=seed)


def sample_embedding_cluster_calibration_queries(
    labels: pd.Series,
    embeddings: pd.DataFrame,
    total_budget: int,
    seed: int = 0,
    n_clusters: int = 16,
) -> pd.Index:
    """Fit train-only embedding clusters and sample calibration queries across them."""

    budget = max(int(total_budget), 0)
    aligned = embeddings.reindex(labels.index).dropna(axis=0, how="any")
    if budget == 0 or aligned.empty:
        return pd.Index([], name=labels.index.name)
    cluster_count = min(max(1, int(n_clusters)), len(aligned))
    if cluster_count == 1:
        groups = pd.Series(["0"] * len(aligned), index=aligned.index)
    else:
        model = KMeans(n_clusters=cluster_count, random_state=int(seed), n_init=10)
        groups = pd.Series(model.fit_predict(aligned.to_numpy(dtype=float)).astype(str), index=aligned.index)
    return _round_robin_sample_by_group(groups, total_budget=budget, seed=seed)


def active_calibration_priority_by_label(labels: pd.Series, base_label_utility: pd.DataFrame) -> pd.Series:
    """Rank route states for new-model calibration without reading held-out utilities."""

    if labels.empty or base_label_utility.empty:
        return pd.Series(dtype=float, name="active_calibration_priority")
    traffic = labels.value_counts(normalize=True)
    priorities: dict[int, float] = {}
    for raw_label in base_label_utility.index:
        label = int(raw_label)
        utilities = base_label_utility.loc[raw_label].dropna().astype(float).sort_values(ascending=False)
        if utilities.empty:
            margin = 1.0
        elif len(utilities) == 1:
            margin = float(utilities.iloc[0])
        else:
            margin = float(utilities.iloc[0] - utilities.iloc[1])
        traffic_mass = float(traffic.get(raw_label, traffic.get(label, 0.0)))
        priorities[label] = traffic_mass / (max(margin, 0.0) + 1e-6)
    return pd.Series(priorities, name="active_calibration_priority").sort_values(ascending=False)


def sample_active_calibration_queries_by_label(
    labels: pd.Series,
    base_label_utility: pd.DataFrame,
    total_budget: int,
    seed: int = 0,
) -> pd.Index:
    """Sample calibration queries from high-value route states under a fixed total budget."""

    budget = max(int(total_budget), 0)
    if budget == 0 or labels.empty:
        return pd.Index([], name=labels.index.name)
    priority = active_calibration_priority_by_label(labels, base_label_utility)
    if priority.empty:
        return pd.Index([], name=labels.index.name)
    rng = np.random.default_rng(seed)
    selected: list[object] = []
    used: set[object] = set()
    ranked_labels = [int(label) for label in priority.index]
    while len(selected) < budget:
        made_progress = False
        for label in ranked_labels:
            if len(selected) >= budget:
                break
            candidates = [query_id for query_id in labels.index[labels == label].tolist() if query_id not in used]
            if not candidates:
                continue
            chosen = rng.choice(np.asarray(candidates, dtype=object), size=1, replace=False)[0]
            used.add(chosen)
            selected.append(chosen)
            made_progress = True
        if not made_progress:
            break
    return pd.Index(selected, name=labels.index.name)


def active_state_calibration_priority(
    labels: pd.Series,
    base_state_utility: pd.DataFrame,
    observations: pd.DataFrame | None = None,
    eval_cost: float | pd.Series | dict[object, float] = 1.0,
    prior_mean: float = 0.5,
    prior_strength: float = 2.0,
    state_column: str = "state_label",
    utility_column: str = "utility",
) -> pd.DataFrame:
    """Score route states by expected value of calibrating a new model.

    The posterior is a light-weight bounded-utility approximation: observed
    utilities are clipped to [0, 1] and accumulated as fractional Beta
    successes. This keeps the module dependency-free while still exposing the
    state-level uncertainty needed for value-of-calibration sampling.
    """

    columns = [
        "state_label",
        "traffic_mass",
        "n_observed",
        "posterior_mean",
        "posterior_variance",
        "posterior_std",
        "current_best_model",
        "current_best_utility",
        "prob_new_beats_current",
        "expected_positive_gain",
        "eval_cost",
        "value_of_calibration",
    ]
    if labels.empty or base_state_utility.empty:
        return pd.DataFrame(columns=columns)

    aligned_labels = labels.dropna()
    traffic = aligned_labels.value_counts(normalize=True)
    observation_stats = _state_observation_stats(
        labels=labels,
        observations=observations,
        state_column=state_column,
        utility_column=utility_column,
    )
    best_by_state = _current_best_by_state(base_state_utility)
    states = _sorted_states(set(best_by_state.index).union(set(aligned_labels.unique())))
    alpha0, beta0 = _beta_prior(prior_mean=prior_mean, prior_strength=prior_strength)
    rows: list[dict[str, object]] = []

    for state in states:
        n_observed = int(observation_stats.loc[state, "n_observed"]) if state in observation_stats.index else 0
        utility_sum = float(observation_stats.loc[state, "utility_sum"]) if state in observation_stats.index else 0.0
        alpha = alpha0 + utility_sum
        beta = beta0 + max(float(n_observed) - utility_sum, 0.0)
        posterior_mean = alpha / (alpha + beta)
        posterior_variance = alpha * beta / (((alpha + beta) ** 2) * (alpha + beta + 1.0))
        posterior_std = math.sqrt(max(posterior_variance, 0.0))
        current_best_model = str(best_by_state.loc[state, "current_best_model"])
        current_best_utility = float(best_by_state.loc[state, "current_best_utility"])
        prob_win, expected_gain = _normal_positive_gain(
            mean=posterior_mean,
            std=posterior_std,
            threshold=current_best_utility,
        )
        traffic_mass = float(traffic.get(state, 0.0))
        cost = max(_state_eval_cost(eval_cost, state), 1e-12)
        value = traffic_mass * prob_win * expected_gain * posterior_std / cost
        rows.append(
            {
                "state_label": state,
                "traffic_mass": traffic_mass,
                "n_observed": n_observed,
                "posterior_mean": posterior_mean,
                "posterior_variance": posterior_variance,
                "posterior_std": posterior_std,
                "current_best_model": current_best_model,
                "current_best_utility": current_best_utility,
                "prob_new_beats_current": prob_win,
                "expected_positive_gain": expected_gain,
                "eval_cost": cost,
                "value_of_calibration": value,
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(
        ["value_of_calibration", "traffic_mass", "posterior_variance"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def sample_active_state_calibration_queries(
    labels: pd.Series,
    base_state_utility: pd.DataFrame,
    total_budget: int,
    observations: pd.DataFrame | None = None,
    query_features: pd.DataFrame | None = None,
    scout_per_state: int = 1,
    eval_cost: float | pd.Series | dict[object, float] = 1.0,
    prior_mean: float = 0.5,
    prior_strength: float = 2.0,
    state_column: str = "state_label",
    utility_column: str = "utility",
) -> pd.DataFrame:
    """Select new-model calibration queries with scout and VOC phases."""

    columns = [
        "query_id",
        "state_label",
        "selection_phase",
        "selection_rank",
        "query_score",
        "value_of_calibration",
    ]
    budget = max(int(total_budget), 0)
    if budget == 0 or labels.empty:
        return pd.DataFrame(columns=columns)

    query_scores = _active_query_scores(labels, query_features)
    selected_rows: list[dict[str, object]] = []
    used: set[object] = _observed_query_ids(observations)

    def append_query(state: object, phase: str, value: float) -> bool:
        query_id = _best_query_in_state(labels=labels, state=state, query_scores=query_scores, used=used)
        if query_id is None:
            return False
        used.add(query_id)
        selected_rows.append(
            {
                "query_id": query_id,
                "state_label": state,
                "selection_phase": phase,
                "selection_rank": len(selected_rows),
                "query_score": float(query_scores.loc[query_id]),
                "value_of_calibration": float(value),
            }
        )
        return True

    scout_count = max(int(scout_per_state), 0)
    if scout_count > 0:
        for state in _sorted_states(labels.dropna().unique()):
            for _ in range(scout_count):
                if len(selected_rows) >= budget:
                    break
                append_query(state, "scout", value=float("nan"))
            if len(selected_rows) >= budget:
                break

    while len(selected_rows) < budget:
        priority = active_state_calibration_priority(
            labels=labels,
            base_state_utility=base_state_utility,
            observations=observations,
            eval_cost=eval_cost,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            state_column=state_column,
            utility_column=utility_column,
        )
        made_progress = False
        for row in priority.to_dict(orient="records"):
            state = row["state_label"]
            if append_query(state, "active", value=float(row["value_of_calibration"])):
                made_progress = True
                break
        if not made_progress:
            break

    return pd.DataFrame(selected_rows, columns=columns)


def conservative_state_model_update(
    base_state_utility: pd.DataFrame,
    posterior: pd.DataFrame,
    new_model_id: str,
    delta: float = 0.01,
    tau: float = 0.90,
    state_column: str = "state_label",
) -> pd.DataFrame:
    """Update the state-to-model table only when posterior evidence is strong."""

    columns = [
        "state_label",
        "current_best_model",
        "current_best_utility",
        "new_model_id",
        "posterior_mean",
        "prob_new_beats_current",
        "expected_gain",
        "switch_to_new_model",
        "selected_model",
    ]
    if base_state_utility.empty:
        return pd.DataFrame(columns=columns)

    best_by_state = _current_best_by_state(base_state_utility)
    posterior_by_state = posterior.set_index(state_column) if state_column in posterior.columns else pd.DataFrame()
    states = _sorted_states(set(best_by_state.index).union(set(posterior_by_state.index)))
    rows: list[dict[str, object]] = []
    for state in states:
        current_best_model = str(best_by_state.loc[state, "current_best_model"])
        current_best_utility = float(best_by_state.loc[state, "current_best_utility"])
        posterior_mean = _posterior_value(posterior_by_state, state, "posterior_mean", default=float("nan"))
        prob_win = _posterior_value(posterior_by_state, state, "prob_new_beats_current", default=0.0)
        expected_gain = posterior_mean - current_best_utility if not math.isnan(posterior_mean) else float("nan")
        switch = (
            not math.isnan(expected_gain)
            and expected_gain > float(delta)
            and float(prob_win) > float(tau)
        )
        rows.append(
            {
                "state_label": state,
                "current_best_model": current_best_model,
                "current_best_utility": current_best_utility,
                "new_model_id": str(new_model_id),
                "posterior_mean": posterior_mean,
                "prob_new_beats_current": prob_win,
                "expected_gain": expected_gain,
                "switch_to_new_model": bool(switch),
                "selected_model": str(new_model_id) if switch else current_best_model,
            }
        )
    table = pd.DataFrame(rows, columns=columns)
    table["switch_to_new_model"] = table["switch_to_new_model"].astype(object)
    return table


def calibrate_new_model_by_active_state(
    labels: pd.Series,
    base_state_utility: pd.DataFrame,
    full_utility: pd.DataFrame,
    new_model_id: str,
    total_budget: int,
    observations: pd.DataFrame | None = None,
    query_features: pd.DataFrame | None = None,
    scout_per_state: int = 1,
    eval_cost: float | pd.Series | dict[object, float] = 1.0,
    prior_mean: float = 0.5,
    prior_strength: float = 2.0,
    delta: float = 0.01,
    tau: float = 0.90,
    state_column: str = "state_label",
    utility_column: str = "utility",
) -> ActiveStateCalibrationResult:
    """Run cached Active State Calibration for a held-out new model.

    `full_utility` can contain the new model for all train rows, but this
    function reads that column only for selected calibration query IDs.
    """

    if new_model_id not in full_utility.columns:
        raise ValueError(f"Missing new model column: {new_model_id}")
    budget = max(int(total_budget), 0)
    eligible_labels = labels.loc[labels.index.intersection(full_utility.index)].dropna()
    selected_columns = [
        "query_id",
        "state_label",
        "selection_phase",
        "selection_rank",
        "query_score",
        "value_of_calibration",
        "observed_utility",
    ]
    observation_history = _initial_observation_history(
        labels=eligible_labels,
        observations=observations,
        state_column=state_column,
        utility_column=utility_column,
    )
    query_scores = _active_query_scores(eligible_labels, query_features)
    selected_rows: list[dict[str, object]] = []
    used: set[object] = _observed_query_ids(observation_history)

    def append_observation(state: object, phase: str, value: float) -> bool:
        query_id = _best_query_in_state(
            labels=eligible_labels,
            state=state,
            query_scores=query_scores,
            used=used,
        )
        if query_id is None:
            return False
        used.add(query_id)
        observed_utility = pd.to_numeric(pd.Series([full_utility.loc[query_id, new_model_id]]), errors="coerce").iloc[0]
        if pd.isna(observed_utility):
            return False
        row = {
            "query_id": query_id,
            "state_label": state,
            "selection_phase": phase,
            "selection_rank": len(selected_rows),
            "query_score": float(query_scores.loc[query_id]),
            "value_of_calibration": float(value),
            "observed_utility": float(observed_utility),
        }
        selected_rows.append(row)
        observation_history.loc[len(observation_history)] = {
            "query_id": query_id,
            state_column: state,
            utility_column: float(observed_utility),
        }
        return True

    scout_count = max(int(scout_per_state), 0)
    if scout_count > 0:
        for state in _sorted_states(eligible_labels.unique()):
            for _ in range(scout_count):
                if len(selected_rows) >= budget:
                    break
                append_observation(state, "scout", value=float("nan"))
            if len(selected_rows) >= budget:
                break

    while len(selected_rows) < budget:
        priority = active_state_calibration_priority(
            labels=eligible_labels,
            base_state_utility=base_state_utility,
            observations=observation_history,
            eval_cost=eval_cost,
            prior_mean=prior_mean,
            prior_strength=prior_strength,
            state_column=state_column,
            utility_column=utility_column,
        )
        made_progress = False
        for row in priority.to_dict(orient="records"):
            if append_observation(
                state=row["state_label"],
                phase="active",
                value=float(row["value_of_calibration"]),
            ):
                made_progress = True
                break
        if not made_progress:
            break

    posterior = active_state_calibration_priority(
        labels=eligible_labels,
        base_state_utility=base_state_utility,
        observations=observation_history,
        eval_cost=eval_cost,
        prior_mean=prior_mean,
        prior_strength=prior_strength,
        state_column=state_column,
        utility_column=utility_column,
    )
    table_update = conservative_state_model_update(
        base_state_utility=base_state_utility,
        posterior=posterior,
        new_model_id=new_model_id,
        delta=delta,
        tau=tau,
        state_column=state_column,
    )
    label_to_model = {
        row["state_label"]: str(row["selected_model"])
        for row in table_update.to_dict(orient="records")
    }
    selected_queries = pd.DataFrame(selected_rows, columns=selected_columns)
    return ActiveStateCalibrationResult(
        label_to_model=label_to_model,
        posterior=posterior,
        table_update=table_update,
        selected_queries=selected_queries,
        calibration_query_count=int(len(selected_queries)),
    )


def _round_robin_sample_by_group(groups: pd.Series, total_budget: int, seed: int) -> pd.Index:
    budget = max(int(total_budget), 0)
    groups = groups.dropna()
    if budget == 0 or groups.empty:
        return pd.Index([], name=groups.index.name)
    rng = np.random.default_rng(seed)
    remaining: dict[str, list[object]] = {}
    for group in sorted(groups.astype(str).unique()):
        query_ids = groups.index[groups.astype(str) == group].to_numpy(dtype=object)
        if len(query_ids) == 0:
            continue
        shuffled = rng.permutation(query_ids).tolist()
        remaining[str(group)] = shuffled
    selected: list[object] = []
    while len(selected) < budget:
        made_progress = False
        for group in sorted(remaining):
            if len(selected) >= budget:
                break
            if not remaining[group]:
                continue
            selected.append(remaining[group].pop(0))
            made_progress = True
        if not made_progress:
            break
    return pd.Index(selected, name=groups.index.name)


def _sorted_states(states: object) -> list[object]:
    return sorted(list(states), key=lambda value: (str(type(value)), str(value)))


def _beta_prior(prior_mean: float, prior_strength: float) -> tuple[float, float]:
    mean = min(max(float(prior_mean), 1e-6), 1.0 - 1e-6)
    strength = max(float(prior_strength), 1e-6)
    return mean * strength, (1.0 - mean) * strength


def _current_best_by_state(base_state_utility: pd.DataFrame) -> pd.DataFrame:
    numeric = base_state_utility.apply(pd.to_numeric, errors="coerce")
    global_model = str(numeric.mean(axis=0).idxmax())
    global_utility = float(numeric[global_model].mean())
    rows: list[dict[str, object]] = []
    for state in numeric.index:
        utilities = numeric.loc[state].dropna()
        if utilities.empty:
            rows.append(
                {
                    "state_label": state,
                    "current_best_model": global_model,
                    "current_best_utility": global_utility,
                }
            )
        else:
            best_model = str(utilities.idxmax())
            rows.append(
                {
                    "state_label": state,
                    "current_best_model": best_model,
                    "current_best_utility": float(utilities.loc[best_model]),
                }
            )
    return pd.DataFrame(rows).set_index("state_label")


def _state_observation_stats(
    labels: pd.Series,
    observations: pd.DataFrame | None,
    state_column: str,
    utility_column: str,
) -> pd.DataFrame:
    if observations is None or observations.empty or utility_column not in observations.columns:
        return pd.DataFrame(columns=["n_observed", "utility_sum"])
    observed = observations.copy()
    if state_column not in observed.columns:
        if "query_id" not in observed.columns:
            return pd.DataFrame(columns=["n_observed", "utility_sum"])
        observed[state_column] = observed["query_id"].map(labels)
    utilities = pd.to_numeric(observed[utility_column], errors="coerce").clip(lower=0.0, upper=1.0)
    observed = observed.assign(_bounded_utility=utilities).dropna(subset=[state_column, "_bounded_utility"])
    if observed.empty:
        return pd.DataFrame(columns=["n_observed", "utility_sum"])
    grouped = observed.groupby(state_column, sort=False)["_bounded_utility"].agg(["count", "sum"])
    grouped = grouped.rename(columns={"count": "n_observed", "sum": "utility_sum"})
    return grouped


def _initial_observation_history(
    labels: pd.Series,
    observations: pd.DataFrame | None,
    state_column: str,
    utility_column: str,
) -> pd.DataFrame:
    columns = ["query_id", state_column, utility_column]
    if observations is None or observations.empty:
        return pd.DataFrame(columns=columns)
    history = observations.copy()
    if "query_id" not in history.columns:
        history["query_id"] = pd.NA
    if state_column not in history.columns:
        history[state_column] = history["query_id"].map(labels)
    if utility_column not in history.columns:
        history[utility_column] = pd.NA
    return history.loc[:, columns].copy().reset_index(drop=True)


def _state_eval_cost(eval_cost: float | pd.Series | dict[object, float], state: object) -> float:
    if isinstance(eval_cost, pd.Series):
        return float(eval_cost.get(state, eval_cost.mean() if len(eval_cost) else 1.0))
    if isinstance(eval_cost, dict):
        return float(eval_cost.get(state, 1.0))
    return float(eval_cost)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_positive_gain(mean: float, std: float, threshold: float) -> tuple[float, float]:
    if std <= 1e-12:
        prob_win = 1.0 if mean > threshold else 0.0
        return prob_win, max(0.0, mean - threshold)
    z_score = (mean - threshold) / std
    prob_win = _normal_cdf(z_score)
    expected_gain = std * _normal_pdf(z_score) + (mean - threshold) * prob_win
    return prob_win, max(0.0, expected_gain)


def _observed_query_ids(observations: pd.DataFrame | None) -> set[object]:
    if observations is None or observations.empty or "query_id" not in observations.columns:
        return set()
    return set(observations["query_id"].dropna().tolist())


def _active_query_scores(labels: pd.Series, query_features: pd.DataFrame | None) -> pd.Series:
    scores = pd.Series(0.0, index=labels.index, name="query_score")
    if query_features is None or query_features.empty:
        return scores
    aligned = query_features.reindex(labels.index)
    weights = {
        "representativeness": 0.5,
        "uncertainty": 0.3,
        "routing_impact": 0.2,
    }
    for column, weight in weights.items():
        if column in aligned.columns:
            scores = scores.add(pd.to_numeric(aligned[column], errors="coerce").fillna(0.0) * weight, fill_value=0.0)
    return scores.astype(float)


def _best_query_in_state(
    labels: pd.Series,
    state: object,
    query_scores: pd.Series,
    used: set[object],
) -> object | None:
    candidates = [query_id for query_id in labels.index[labels == state].tolist() if query_id not in used]
    if not candidates:
        return None
    scored = pd.DataFrame(
        {
            "query_id": candidates,
            "query_score": [float(query_scores.get(query_id, 0.0)) for query_id in candidates],
            "stable_key": [str(query_id) for query_id in candidates],
        }
    )
    scored = scored.sort_values(["query_score", "stable_key"], ascending=[False, True])
    return scored.iloc[0]["query_id"]


def _posterior_value(posterior_by_state: pd.DataFrame, state: object, column: str, default: float) -> float:
    if posterior_by_state.empty or state not in posterior_by_state.index or column not in posterior_by_state.columns:
        return float(default)
    value = pd.to_numeric(pd.Series([posterior_by_state.loc[state, column]]), errors="coerce").iloc[0]
    return float(value) if not pd.isna(value) else float(default)


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
    logistic_solver: str = "lbfgs",
    svm_backend: str = "linear_svc",
    tol: float = 1e-4,
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
        model = LogisticRegression(
            random_state=int(random_state),
            max_iter=int(max_iter),
            solver=str(logistic_solver),
            tol=float(tol),
        )
    elif method == "svm":
        if str(svm_backend).lower() == "sgd":
            model = SGDClassifier(
                loss="hinge",
                random_state=int(random_state),
                max_iter=int(max_iter),
                tol=float(tol),
            )
        else:
            model = LinearSVC(random_state=int(random_state), max_iter=int(max_iter), tol=float(tol))
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
