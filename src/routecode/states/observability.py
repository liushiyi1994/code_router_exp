from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from routecode.metrics import recovered_gap


PHASE1_TABLES = {
    "recovered_gap": "table_recovered_gap.csv",
    "predictability_constrained": "table_predictability_constrained.csv",
}


def load_phase1_observability_inputs(result_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Phase 1 tables needed for the Phase 2 M0 observability recap."""
    result_path = Path(result_dir)
    recovered_path = result_path / PHASE1_TABLES["recovered_gap"]
    predictability_path = result_path / PHASE1_TABLES["predictability_constrained"]
    missing = [str(path) for path in [recovered_path, predictability_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Phase 1 observability input table(s): " + ", ".join(missing))
    return pd.read_csv(recovered_path), pd.read_csv(predictability_path)


def compute_observability_gap_table(
    recovered_gap_table: pd.DataFrame,
    predictability_table: pd.DataFrame,
    result_id: str,
) -> pd.DataFrame:
    """Compare route-state oracle performance with deployable query-to-state assignment.

    The key Phase 2 quantity is the utility loss from observing the route state
    only through query features instead of assigning the utility-optimal state.
    """
    best_single_row = _require_method_row(recovered_gap_table, "best_single")
    query_oracle_row = _require_method_row(recovered_gap_table, "query_oracle")
    best_single = float(best_single_row["mean_utility"])
    query_oracle = float(query_oracle_row["mean_utility"])
    rows: list[dict[str, object]] = []

    rows.extend(_flat_routecode_rows(predictability_table, result_id, best_single_row, query_oracle_row))
    rows.extend(_d2_rows(predictability_table, result_id, best_single_row, query_oracle_row))

    columns = [
        "result_id",
        "comparison",
        "state_family",
        "oracle_state_method",
        "deployable_state_method",
        "K",
        "alpha",
        "encoder_family",
        "strong_encoder_status",
        "query_oracle_mean_utility",
        "query_oracle_mean_utility_ci_low",
        "query_oracle_mean_utility_ci_high",
        "best_single_mean_utility",
        "best_single_mean_utility_ci_low",
        "best_single_mean_utility_ci_high",
        "oracle_state_mean_utility",
        "oracle_state_mean_utility_ci_low",
        "oracle_state_mean_utility_ci_high",
        "deployable_state_mean_utility",
        "deployable_state_mean_utility_ci_low",
        "deployable_state_mean_utility_ci_high",
        "state_observability_gap",
        "state_observability_gap_ci_low",
        "state_observability_gap_ci_high",
        "query_oracle_gap",
        "query_oracle_gap_ci_low",
        "query_oracle_gap_ci_high",
        "state_oracle_gap_vs_best_single",
        "deployable_gap_vs_best_single",
        "state_gap_closed",
        "full_gap_closed_vs_query_oracle",
        "oracle_state_recovered_gap_vs_oracle",
        "deployable_recovered_gap_vs_oracle",
        "label_accuracy",
        "mean_confidence",
        "evidence_source",
        "interpretation",
    ]
    return pd.DataFrame(rows, columns=columns)


def compute_observability_gap_for_result_dir(result_dir: str | Path) -> pd.DataFrame:
    recovered, predictability = load_phase1_observability_inputs(result_dir)
    return compute_observability_gap_table(recovered, predictability, result_id=Path(result_dir).name)


def _flat_routecode_rows(
    table: pd.DataFrame,
    result_id: str,
    best_single_row: pd.Series,
    query_oracle_row: pd.Series,
) -> list[dict[str, object]]:
    oracle = _first_method_row(table, "flat_routecode_utility_oracle")
    deployable = _first_method_row(table, "flat_routecode_logistic_label_predictor")
    if oracle is None or deployable is None:
        return []
    return [
        _comparison_row(
            result_id=result_id,
            comparison="flat_routecode_logistic_label_predictor",
            state_family="flat_routecode",
            oracle_state_method="flat_routecode_utility_oracle",
            deployable_state_method="flat_routecode_logistic_label_predictor",
            oracle_row=oracle,
            deployable_row=deployable,
            best_single_row=best_single_row,
            query_oracle_row=query_oracle_row,
            encoder_family="phase1_logistic_label_predictor",
        )
    ]


def _d2_rows(
    table: pd.DataFrame,
    result_id: str,
    best_single_row: pd.Series,
    query_oracle_row: pd.Series,
) -> list[dict[str, object]]:
    if "alpha" not in table.columns:
        return []
    oracle_rows = table[table["method"] == "d2_joint_oracle_labels"].copy()
    rows: list[dict[str, object]] = []
    for _, oracle in oracle_rows.iterrows():
        alpha = _as_float(oracle.get("alpha"))
        k_value = _as_float(oracle.get("K"))
        for deployable_method, encoder_family in [
            ("d2_embedding_centroid", "phase1_embedding_centroid"),
            ("d2_logistic_label_predictor", "phase1_logistic_label_predictor"),
        ]:
            candidates = table[table["method"] == deployable_method].copy()
            if "alpha" in candidates.columns:
                candidates = candidates[np.isclose(pd.to_numeric(candidates["alpha"], errors="coerce"), alpha)]
            if "K" in candidates.columns and not np.isnan(k_value):
                candidates = candidates[np.isclose(pd.to_numeric(candidates["K"], errors="coerce"), k_value)]
            if candidates.empty:
                continue
            deployable = candidates.iloc[0]
            rows.append(
                _comparison_row(
                    result_id=result_id,
                    comparison=f"{deployable_method}_alpha_{_format_alpha(alpha)}",
                    state_family="d2_predictability_constrained",
                    oracle_state_method="d2_joint_oracle_labels",
                    deployable_state_method=deployable_method,
                    oracle_row=oracle,
                    deployable_row=deployable,
                    best_single_row=best_single_row,
                    query_oracle_row=query_oracle_row,
                    encoder_family=encoder_family,
                )
            )
    return rows


def _comparison_row(
    *,
    result_id: str,
    comparison: str,
    state_family: str,
    oracle_state_method: str,
    deployable_state_method: str,
    oracle_row: pd.Series,
    deployable_row: pd.Series,
    best_single_row: pd.Series,
    query_oracle_row: pd.Series,
    encoder_family: str,
) -> dict[str, object]:
    best_single = float(best_single_row["mean_utility"])
    query_oracle = float(query_oracle_row["mean_utility"])
    oracle_mean = float(oracle_row["mean_utility"])
    deployable_mean = float(deployable_row["mean_utility"])
    best_low, best_high = _utility_ci(best_single_row)
    query_low, query_high = _utility_ci(query_oracle_row)
    oracle_low, oracle_high = _utility_ci(oracle_row)
    deployable_low, deployable_high = _utility_ci(deployable_row)
    alpha = _as_float(deployable_row.get("alpha", oracle_row.get("alpha", np.nan)))
    k_value = _as_float(deployable_row.get("K", oracle_row.get("K", np.nan)))
    state_gap = float(oracle_mean - deployable_mean)
    full_gap = float(query_oracle - deployable_mean)
    state_oracle_gain = float(oracle_mean - best_single)
    deployable_gain = float(deployable_mean - best_single)
    state_gap_closed = recovered_gap(deployable_mean, best_single, oracle_mean)
    full_gap_closed = recovered_gap(deployable_mean, best_single, query_oracle)
    oracle_state_recovered = recovered_gap(oracle_mean, best_single, query_oracle)
    deployable_recovered = recovered_gap(deployable_mean, best_single, query_oracle)
    return {
        "result_id": result_id,
        "comparison": comparison,
        "state_family": state_family,
        "oracle_state_method": oracle_state_method,
        "deployable_state_method": deployable_state_method,
        "K": k_value,
        "alpha": alpha,
        "encoder_family": encoder_family,
        "strong_encoder_status": "not_run_in_m0",
        "query_oracle_mean_utility": query_oracle,
        "query_oracle_mean_utility_ci_low": query_low,
        "query_oracle_mean_utility_ci_high": query_high,
        "best_single_mean_utility": best_single,
        "best_single_mean_utility_ci_low": best_low,
        "best_single_mean_utility_ci_high": best_high,
        "oracle_state_mean_utility": oracle_mean,
        "oracle_state_mean_utility_ci_low": oracle_low,
        "oracle_state_mean_utility_ci_high": oracle_high,
        "deployable_state_mean_utility": deployable_mean,
        "deployable_state_mean_utility_ci_low": deployable_low,
        "deployable_state_mean_utility_ci_high": deployable_high,
        "state_observability_gap": state_gap,
        "state_observability_gap_ci_low": oracle_low - deployable_high,
        "state_observability_gap_ci_high": oracle_high - deployable_low,
        "query_oracle_gap": full_gap,
        "query_oracle_gap_ci_low": query_low - deployable_high,
        "query_oracle_gap_ci_high": query_high - deployable_low,
        "state_oracle_gap_vs_best_single": state_oracle_gain,
        "deployable_gap_vs_best_single": deployable_gain,
        "state_gap_closed": state_gap_closed,
        "full_gap_closed_vs_query_oracle": full_gap_closed,
        "oracle_state_recovered_gap_vs_oracle": _row_float(
            oracle_row, "recovered_gap_vs_oracle", default=oracle_state_recovered
        ),
        "deployable_recovered_gap_vs_oracle": _row_float(
            deployable_row, "recovered_gap_vs_oracle", default=deployable_recovered
        ),
        "label_accuracy": _row_float(deployable_row, "label_accuracy"),
        "mean_confidence": _row_float(deployable_row, "mean_confidence"),
        "evidence_source": "phase1_existing_result_tables",
        "interpretation": _interpretation(state_gap, full_gap_closed),
    }


def _require_mean_utility(table: pd.DataFrame, method: str) -> float:
    return float(_require_method_row(table, method)["mean_utility"])


def _require_method_row(table: pd.DataFrame, method: str) -> pd.Series:
    row = _first_method_row(table, method)
    if row is None:
        raise ValueError(f"Required Phase 1 method row is missing: {method}")
    return row


def _first_method_row(table: pd.DataFrame, method: str) -> pd.Series | None:
    if "method" not in table.columns:
        raise ValueError("Expected input table to include a method column")
    rows = table[table["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0]


def _row_float(row: pd.Series, column: str, default: float = np.nan) -> float:
    if column not in row.index:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return float(value)


def _utility_ci(row: pd.Series) -> tuple[float, float]:
    mean = float(row["mean_utility"])
    low = _row_float(row, "utility_ci_low", default=mean)
    high = _row_float(row, "utility_ci_high", default=mean)
    return min(low, mean), max(high, mean)


def _as_float(value: object) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def _format_alpha(alpha: float) -> str:
    if np.isnan(alpha):
        return "nan"
    if float(alpha).is_integer():
        return str(int(alpha))
    return f"{alpha:g}".replace(".", "p")


def _interpretation(state_gap: float, full_gap_closed: float) -> str:
    if state_gap <= 0.01 and full_gap_closed >= 0.8:
        return "route state is mostly observable from current query features"
    if state_gap <= 0.05:
        return "route state is partly observable but still below the query-oracle frontier"
    if full_gap_closed < 0.0:
        return "query-to-state assignment is worse than best-single baseline"
    return "route state is not fully observable from current query features; material observability gap remains"
