from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CLAIM_ROWS = [
    {
        "claim_id": "low_rate_oracle_codes",
        "claim": "Useful low-rate utility route codes exist.",
    },
    {
        "claim_id": "small_inferred_labels",
        "claim": "Small inferred route labels recover most routing performance.",
    },
    {
        "claim_id": "model_pool_transfer",
        "claim": "Route labels transfer across model pools better than same-budget direct retraining.",
    },
    {
        "claim_id": "new_model_calibration",
        "claim": "New models can be integrated with fewer calibration examples than direct retraining.",
    },
    {
        "claim_id": "benchmark_diagnosis",
        "claim": "Benchmark routing results expose compressibility or split-design artifacts.",
    },
    {
        "claim_id": "adaptive_refinement",
        "claim": "Adaptive refinement improves cost-quality by refining uncertain queries.",
    },
]


def audit_claims(out_dir: str | Path) -> pd.DataFrame:
    out_path = Path(out_dir)
    tables = {
        "rate_distortion": _read_table(out_path / "table_rate_distortion.csv"),
        "recovered_gap": _read_table(out_path / "table_recovered_gap.csv"),
        "predictability": _read_table(out_path / "table_predictability_constrained.csv"),
        "transfer": _read_table(out_path / "table_model_pool_transfer.csv"),
        "new_model": _read_table(out_path / "table_new_model_integration.csv"),
        "stronger_direct": _read_table(out_path / "table_stronger_direct_router_probe.csv"),
        "split_rank": _read_table(out_path / "table_split_rank_correlation.csv"),
        "residual_risk": _read_table(out_path / "table_residual_risk.csv"),
        "adaptive": _read_table(out_path / "table_adaptive_refinement.csv"),
    }
    rows = [
        _low_rate_oracle_codes(tables, out_path),
        _small_inferred_labels(tables, out_path),
        _model_pool_transfer(tables, out_path),
        _new_model_calibration(tables, out_path),
        _benchmark_diagnosis(tables, out_path),
        _adaptive_refinement(tables, out_path),
    ]
    return pd.DataFrame(rows)


def _low_rate_oracle_codes(tables: dict[str, pd.DataFrame], out_path: Path) -> dict[str, object]:
    candidates = _concat_existing([tables["rate_distortion"], tables["predictability"]])
    oracle_rows = _method_filter(
        candidates,
        include_any=["routecode_oracle", "regret_routecode_oracle", "d2_joint_oracle"],
    )
    oracle_rows = _numeric_filter(oracle_rows, "K", max_value=16)
    row = _best_row(oracle_rows, "recovered_gap_vs_oracle")
    if row is None:
        return _missing_row(
            "low_rate_oracle_codes",
            "Missing low-rate RouteCode oracle rows in rate-distortion or D2 tables.",
            ["table_rate_distortion.csv", "table_predictability_constrained.csv"],
        )
    value = float(row["recovered_gap_vs_oracle"])
    status = "diagnostic_supported" if value >= 0.8 else "not_supported"
    return _claim_row(
        "low_rate_oracle_codes",
        status=status,
        primary_metric="best_low_rate_oracle_recovered_gap_vs_oracle",
        primary_value=value,
        threshold="diagnostic if >= 0.80 by K<=16",
        evidence=_evidence(row, out_path, _source_file(row, "rate_distortion_or_predictability")),
        interpretation=(
            "Low-rate oracle code labels preserve most oracle routing gain."
            if status == "diagnostic_supported"
            else "Low-rate oracle code labels do not yet preserve enough oracle routing gain."
        ),
    )


def _small_inferred_labels(tables: dict[str, pd.DataFrame], out_path: Path) -> dict[str, object]:
    candidates = _concat_existing([tables["predictability"], tables["recovered_gap"], tables["rate_distortion"]])
    inferred = _method_filter(
        candidates,
        include_any=[
            "predicted_topic",
            "routecode_predicted",
            "regret_routecode_predicted",
            "flat_routecode_logistic",
            "d2_embedding_centroid",
            "d2_logistic",
        ],
    )
    if "method" in inferred.columns:
        inferred = inferred[~inferred["method"].astype(str).str.contains("oracle", case=False, na=False)].copy()
    row = _best_row(inferred, "recovered_gap_vs_oracle")
    if row is None:
        return _missing_row(
            "small_inferred_labels",
            "Missing predicted-topic or predicted-code recovery rows.",
            [
                "table_predictability_constrained.csv",
                "table_recovered_gap.csv",
                "table_rate_distortion.csv",
            ],
        )
    value = float(row["recovered_gap_vs_oracle"])
    status = "supported" if value >= 0.85 else "not_supported"
    if status == "supported" and not _has_supported_recovery_ci(row):
        status = "missing_evidence"
    return _claim_row(
        "small_inferred_labels",
        status=status,
        primary_metric="best_inferred_recovered_gap_vs_oracle",
        primary_value=value,
        threshold="supported only if recovered gap >= 0.85 and lower bootstrap CI >= 0.80",
        evidence=_evidence(row, out_path, _source_file(row, "predictability_or_recovered_gap")),
        interpretation=(
            "Current inferred labels pass the high-recovery threshold."
            if status == "supported"
            else "Do not claim that small inferred labels recover most routing performance."
        ),
    )


def _model_pool_transfer(tables: dict[str, pd.DataFrame], out_path: Path) -> dict[str, object]:
    table = tables["transfer"]
    if table.empty:
        return _missing_row(
            "model_pool_transfer",
            "Missing model-pool transfer table.",
            ["table_model_pool_transfer.csv"],
        )
    transfer_rows = _method_filter(table, include_any=["source_d2_label_transfer"])
    direct_rows = _method_filter(table, include_any=["target_direct_"])
    transfer_best = _best_row(transfer_rows, "recovered_gap_vs_oracle")
    direct_best = _best_row(direct_rows, "recovered_gap_vs_oracle")
    if transfer_best is None or direct_best is None:
        return _missing_row(
            "model_pool_transfer",
            "Transfer or same-budget direct retraining rows are missing.",
            ["table_model_pool_transfer.csv"],
        )
    matched = _matched_best_difference(
        routecode_rows=transfer_rows,
        direct_rows=direct_rows,
        group_columns=["transfer_scenario"],
        metric="recovered_gap_vs_oracle",
    )
    value = matched[0] if matched is not None else float(transfer_best["recovered_gap_vs_oracle"] - direct_best["recovered_gap_vs_oracle"])
    primary_metric = (
        "mean_matched_transfer_minus_direct_recovered_gap"
        if matched is not None
        else "best_transfer_minus_best_direct_recovered_gap"
    )
    status = "diagnostic_alive" if value > 0 else "not_supported"
    return _claim_row(
        "model_pool_transfer",
        status=status,
        primary_metric=primary_metric,
        primary_value=value,
        threshold="diagnostic if > 0; paper-level support requires broader split/model coverage",
        evidence=matched[1]
        if matched is not None
        else (
            _evidence(transfer_best, out_path, "table_model_pool_transfer.csv")
            + "; direct_baseline="
            + _evidence(direct_best, out_path, "table_model_pool_transfer.csv")
        ),
        interpretation=(
            "Transfer remains alive as a bounded diagnostic."
            if status == "diagnostic_alive"
            else "Current transfer rows do not beat direct retraining."
        ),
    )


def _new_model_calibration(tables: dict[str, pd.DataFrame], out_path: Path) -> dict[str, object]:
    table = _concat_existing([tables["new_model"], tables["stronger_direct"]])
    if table.empty:
        return _missing_row(
            "new_model_calibration",
            "Missing new-model calibration or stronger direct-router probe tables.",
            ["table_new_model_integration.csv", "table_stronger_direct_router_probe.csv"],
        )
    routecode_rows = _method_filter(table, include_any=["routecode_label_calibration"])
    direct_rows = _method_filter(table, include_any=["direct_retraining"])
    routecode_best = _best_row(routecode_rows, "recovered_gap_vs_oracle")
    direct_best = _best_row(direct_rows, "recovered_gap_vs_oracle")
    if routecode_best is None or direct_best is None:
        return _missing_row(
            "new_model_calibration",
            "RouteCode calibration or direct retraining rows are missing.",
            ["table_new_model_integration.csv", "table_stronger_direct_router_probe.csv"],
        )
    matched = _matched_best_difference(
        routecode_rows=routecode_rows,
        direct_rows=direct_rows,
        group_columns=["new_model_id", "examples_per_label"],
        metric="recovered_gap_vs_oracle",
    )
    value = matched[0] if matched is not None else float(routecode_best["recovered_gap_vs_oracle"] - direct_best["recovered_gap_vs_oracle"])
    primary_metric = (
        "mean_matched_routecode_minus_direct_recovered_gap"
        if matched is not None
        else "best_routecode_minus_best_direct_recovered_gap"
    )
    status = "diagnostic_alive" if value > 0 else "not_supported"
    return _claim_row(
        "new_model_calibration",
        status=status,
        primary_metric=primary_metric,
        primary_value=value,
        threshold="diagnostic if > 0; paper-level support requires stronger cost accounting and broader repeats",
        evidence=matched[1]
        if matched is not None
        else (
            _evidence(routecode_best, out_path, _source_file(routecode_best, "new_model_or_probe"))
            + "; direct_baseline="
            + _evidence(direct_best, out_path, _source_file(direct_best, "new_model_or_probe"))
        ),
        interpretation=(
            "Calibration remains alive as a bounded diagnostic."
            if status == "diagnostic_alive"
            else "Current calibration rows do not beat direct retraining."
        ),
    )


def _benchmark_diagnosis(tables: dict[str, pd.DataFrame], out_path: Path) -> dict[str, object]:
    split_rank = tables["split_rank"]
    recovered = tables["recovered_gap"]
    values = []
    evidence_parts = []
    if not split_rank.empty and "rank_correlation_vs_random" in split_rank.columns:
        min_idx = pd.to_numeric(split_rank["rank_correlation_vs_random"], errors="coerce").idxmin()
        if pd.notna(min_idx):
            min_row = split_rank.loc[min_idx]
            value = float(min_row["rank_correlation_vs_random"])
            values.append(("min_split_rank_correlation", value))
            evidence_parts.append(
                f"table_split_rank_correlation.csv: scenario={min_row.get('scenario', '')}, rank_correlation={value:.4f}"
            )
    dataset_row = _best_row(_method_filter(recovered, include_any=["dataset_label_lookup"]), "recovered_gap_vs_oracle")
    if dataset_row is not None:
        dataset_value = float(dataset_row["recovered_gap_vs_oracle"])
        values.append(("dataset_label_recovered_gap_vs_oracle", dataset_value))
        evidence_parts.append(_evidence(dataset_row, out_path, "table_recovered_gap.csv"))
    if not values:
        return _missing_row(
            "benchmark_diagnosis",
            "Missing split-sensitivity or dataset-label evidence.",
            ["table_split_rank_correlation.csv", "table_recovered_gap.csv"],
        )
    split_value = next((value for metric, value in values if metric == "min_split_rank_correlation"), np.nan)
    dataset_value = next((value for metric, value in values if metric == "dataset_label_recovered_gap_vs_oracle"), np.nan)
    status = (
        "diagnostic_supported"
        if (pd.notna(split_value) and split_value < 0.5) or (pd.notna(dataset_value) and dataset_value >= 0.25)
        else "not_supported"
    )
    primary_metric = "min_split_rank_correlation" if pd.notna(split_value) else "dataset_label_recovered_gap_vs_oracle"
    primary_value = split_value if pd.notna(split_value) else dataset_value
    return _claim_row(
        "benchmark_diagnosis",
        status=status,
        primary_metric=primary_metric,
        primary_value=float(primary_value),
        threshold="diagnostic if rank correlation < 0.50 or dataset-label recovered gap >= 0.25",
        evidence="; ".join(evidence_parts),
        interpretation=(
            "Benchmark diagnosis is supported as a diagnostic thread."
            if status == "diagnostic_supported"
            else "Current split/compressibility evidence is not strong enough for benchmark diagnosis."
        ),
    )


def _adaptive_refinement(tables: dict[str, pd.DataFrame], out_path: Path) -> dict[str, object]:
    adaptive = tables["adaptive"]
    if not adaptive.empty:
        row = _best_row(adaptive, "recovered_gap_vs_oracle") or _best_row(adaptive, "mean_utility")
        if row is None:
            return _missing_row(
                "adaptive_refinement",
                "Adaptive-refinement output exists but lacks utility/recovery columns.",
                ["table_adaptive_refinement.csv"],
            )
        value = float(row.get("recovered_gap_vs_oracle", row.get("mean_utility", np.nan)))
        status = "diagnostic_alive" if pd.notna(value) and value > 0 else "not_supported"
        return _claim_row(
            "adaptive_refinement",
            status=status,
            primary_metric="adaptive_refinement_recovered_gap_vs_oracle",
            primary_value=value,
            threshold="diagnostic if utility/recovery improves over fallback",
            evidence=_evidence(row, out_path, "table_adaptive_refinement.csv"),
            interpretation="Adaptive refinement has a metric row." if status != "not_supported" else "Adaptive refinement did not improve in current rows.",
        )
    residual = tables["residual_risk"]
    if residual.empty:
        return _missing_row(
            "adaptive_refinement",
            "Missing residual-risk gate and adaptive-refinement utility outputs.",
            ["table_residual_risk.csv", "table_adaptive_refinement.csv"],
        )
    top10 = residual[pd.to_numeric(residual.get("top_fraction", pd.Series(dtype=float)), errors="coerce").round(2) == 0.10]
    row = _best_row(top10 if not top10.empty else residual, "regret_mass_fraction")
    if row is None:
        return _missing_row(
            "adaptive_refinement",
            "Residual-risk table lacks regret-mass evidence.",
            ["table_residual_risk.csv"],
        )
    value = float(row["regret_mass_fraction"])
    status = "deferred" if value >= 0.3 else "not_supported"
    return _claim_row(
        "adaptive_refinement",
        status=status,
        primary_metric="top10_regret_mass_fraction",
        primary_value=value,
        threshold="defer unless residual-risk gate is strong; current heuristic expects top-10% regret mass >= 0.30",
        evidence=_evidence(row, out_path, "table_residual_risk.csv"),
        interpretation=(
            "Residual risk is concentrated enough to justify a future refinement experiment."
            if status == "deferred"
            else "Do not implement or claim adaptive refinement from the current residual-risk gate."
        ),
    )


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["__source_file"] = path.name
    return frame


def _concat_existing(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if not frame.empty]
    return pd.concat(non_empty, ignore_index=True, sort=False) if non_empty else pd.DataFrame()


def _method_filter(table: pd.DataFrame, include_any: list[str]) -> pd.DataFrame:
    if table.empty or "method" not in table.columns:
        return pd.DataFrame()
    pattern = "|".join(include_any)
    return table[table["method"].astype(str).str.contains(pattern, case=False, na=False)].copy()


def _numeric_filter(table: pd.DataFrame, column: str, max_value: float | None = None) -> pd.DataFrame:
    if table.empty or column not in table.columns:
        return table
    values = pd.to_numeric(table[column], errors="coerce")
    keep = values.notna()
    if max_value is not None:
        keep &= values <= max_value
    return table[keep].copy()


def _best_row(table: pd.DataFrame, metric: str) -> pd.Series | None:
    if table.empty or metric not in table.columns:
        return None
    values = pd.to_numeric(table[metric], errors="coerce")
    if values.dropna().empty:
        return None
    return table.loc[values.idxmax()]


def _matched_best_difference(
    *,
    routecode_rows: pd.DataFrame,
    direct_rows: pd.DataFrame,
    group_columns: list[str],
    metric: str,
) -> tuple[float, str] | None:
    available = [column for column in group_columns if column in routecode_rows.columns and column in direct_rows.columns]
    if not available:
        return None
    diffs: list[float] = []
    evidence: list[str] = []
    for key, route_group in routecode_rows.groupby(available, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        mask = pd.Series(True, index=direct_rows.index)
        for column, value in zip(available, key_tuple):
            mask &= direct_rows[column].eq(value)
        direct_group = direct_rows[mask]
        route_best = _best_row(route_group, metric)
        direct_best = _best_row(direct_group, metric)
        if route_best is None or direct_best is None:
            continue
        diff = float(route_best[metric] - direct_best[metric])
        diffs.append(diff)
        key_text = ", ".join(f"{column}={value}" for column, value in zip(available, key_tuple))
        evidence.append(
            f"{key_text}: routecode={float(route_best[metric]):.4f}, direct={float(direct_best[metric]):.4f}, diff={diff:.4f}"
        )
    if not diffs:
        return None
    return float(np.mean(diffs)), "; ".join(evidence[:6])


def _has_supported_recovery_ci(row: pd.Series) -> bool:
    # Most current tables only carry utility CIs, not recovered-gap CIs. Keep the
    # offensive claim gated if recovered-gap CI columns are added later.
    for candidate in ["recovered_gap_ci_low", "recovered_gap_vs_oracle_ci_low"]:
        if candidate in row and pd.notna(row[candidate]):
            return float(row[candidate]) >= 0.80
    return False


def _source_file(row: pd.Series, fallback: str) -> str:
    return str(row.get("__source_file", fallback))


def _evidence(row: pd.Series, out_path: Path, source_file: str) -> str:
    parts = [source_file]
    for column in [
        "method",
        "K",
        "alpha",
        "new_model_id",
        "examples_per_label",
        "transfer_scenario",
        "recovered_gap_vs_oracle",
        "rank_correlation_vs_random",
        "regret_mass_fraction",
    ]:
        if column in row and pd.notna(row[column]):
            value = row[column]
            if isinstance(value, float):
                value = f"{value:.4f}"
            parts.append(f"{column}={value}")
    return ", ".join(parts)


def _missing_row(claim_id: str, interpretation: str, expected: list[str]) -> dict[str, object]:
    return _claim_row(
        claim_id,
        status="missing_evidence",
        primary_metric="missing_evidence",
        primary_value=np.nan,
        threshold="requires current result artifact",
        evidence="expected: " + ", ".join(expected),
        interpretation=interpretation,
    )


def _claim_row(
    claim_id: str,
    *,
    status: str,
    primary_metric: str,
    primary_value: float,
    threshold: str,
    evidence: str,
    interpretation: str,
) -> dict[str, object]:
    claim = next(row["claim"] for row in CLAIM_ROWS if row["claim_id"] == claim_id)
    return {
        "claim_id": claim_id,
        "claim": claim,
        "status": status,
        "primary_metric": primary_metric,
        "primary_value": primary_value,
        "threshold": threshold,
        "evidence": evidence,
        "interpretation": interpretation,
    }
