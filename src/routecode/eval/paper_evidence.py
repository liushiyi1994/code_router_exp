from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd


SUMMARY_COLUMNS = [
    "section",
    "item",
    "status",
    "key_value",
    "evidence",
    "interpretation",
]


def build_paper_evidence_summary(
    global_claims: pd.DataFrame,
    readiness_tables: Mapping[str, pd.DataFrame],
    *,
    readiness_paths: Mapping[str, str | Path] | None = None,
) -> pd.DataFrame:
    """Build a compact, paper-facing summary from current claim gates.

    This is intentionally conservative. It does not convert diagnostic evidence
    into paper-level claims; it records what the current artifacts permit.
    """

    paths = {key: Path(value) for key, value in (readiness_paths or {}).items()}
    rows: list[dict[str, object]] = []
    rows.append(_paper_direction_row(global_claims))
    for _, claim in global_claims.iterrows():
        rows.append(_claim_row(claim))
    rows.extend(_external_readiness_rows(readiness_tables, paths))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _paper_direction_row(global_claims: pd.DataFrame) -> dict[str, object]:
    claims = global_claims.set_index("claim_id") if "claim_id" in global_claims.columns else pd.DataFrame()
    inferred_status = _claim_status(claims, "small_inferred_labels")
    low_rate_status = _claim_status(claims, "low_rate_oracle_codes")
    calibration_status = _claim_status(claims, "new_model_calibration")
    transfer_status = _claim_status(claims, "model_pool_transfer")
    if inferred_status == "not_supported" and low_rate_status in {"diagnostic_supported", "supported"}:
        status = "information_frontier_diagnostic"
        interpretation = (
            "Recommended framing: information-frontier and benchmark-diagnostic paper. "
            "Do not claim that few inferred bits are enough; current evidence supports low-rate oracle structure, "
            "modest deployable inferred-label recovery, and diagnostic calibration/transfer threads."
        )
    elif inferred_status in {"supported", "diagnostic_supported"}:
        status = "few_bits_claim_candidate"
        interpretation = (
            "Recommended framing: small inferred labels may be claim-critical, but verify bootstrap lower bounds, "
            "external baselines, and robustness before paper-level wording."
        )
    else:
        status = "diagnostic_only"
        interpretation = (
            "Recommended framing: keep claims diagnostic until inferred-label, calibration, transfer, and baseline "
            "coverage strengthen."
        )
    key_parts = []
    if inferred_status:
        key_parts.append(f"small_inferred_labels={inferred_status}")
    if low_rate_status:
        key_parts.append(f"low_rate_oracle_codes={low_rate_status}")
    if calibration_status:
        key_parts.append(f"new_model_calibration={calibration_status}")
    if transfer_status:
        key_parts.append(f"model_pool_transfer={transfer_status}")
    return {
        "section": "paper_direction",
        "item": "recommended_framing",
        "status": status,
        "key_value": "; ".join(key_parts),
        "evidence": "results/table_claim_status_global.csv",
        "interpretation": interpretation,
    }


def _claim_status(claims: pd.DataFrame, claim_id: str) -> str:
    if claims.empty or claim_id not in claims.index:
        return ""
    return str(claims.loc[claim_id].get("global_status", ""))


def _claim_row(claim: pd.Series) -> dict[str, object]:
    best = _format_float(claim.get("best_primary_value"))
    worst = _format_float(claim.get("worst_primary_value"))
    key_value = "; ".join(part for part in [f"best={best}" if best else "", f"worst={worst}" if worst else ""] if part)
    return {
        "section": "claim",
        "item": str(claim.get("claim_id", "")),
        "status": str(claim.get("global_status", "")),
        "key_value": key_value,
        "evidence": str(claim.get("evidence_summary", "")),
        "interpretation": str(claim.get("interpretation", "")),
    }


def _external_readiness_rows(
    readiness_tables: Mapping[str, pd.DataFrame],
    readiness_paths: Mapping[str, Path],
) -> list[dict[str, object]]:
    if not readiness_tables:
        return [
            {
                "section": "external_baselines",
                "item": "readiness_overview",
                "status": "missing_evidence",
                "key_value": "",
                "evidence": "",
                "interpretation": "No external command readiness tables were supplied.",
            }
        ]
    total_rows = 0
    runnable_rows = 0
    exact_rows = 0
    blocked: dict[str, str] = {}
    metric_rows: list[str] = []
    evidence_paths = []
    for run_id, table in readiness_tables.items():
        total_rows += len(table)
        runnable = _bool_series(table, "runnable_now")
        exact = _bool_series(table, "exact_upstream_command")
        routecode_metric = _bool_series(table, "routecode_metric_compatible")
        runnable_rows += int(runnable.sum())
        exact_rows += int((runnable & exact).sum())
        if run_id in readiness_paths:
            evidence_paths.append(str(readiness_paths[run_id]))
        for _, row in table.iterrows():
            check_id = str(row.get("check_id", ""))
            if _as_bool(row.get("routecode_metric_compatible", False)) and _as_bool(row.get("runnable_now", False)):
                metric_rows.append(check_id)
            if not _as_bool(row.get("runnable_now", False)) and not _as_bool(row.get("routecode_metric_compatible", False)):
                blocked.setdefault(check_id, str(row.get("blocking_reasons", "")))
    rows = [
        {
            "section": "external_baselines",
            "item": "readiness_overview",
            "status": "partial" if blocked else "available",
            "key_value": f"{total_rows} rows; {runnable_rows} runnable; {exact_rows} exact",
            "evidence": "; ".join(evidence_paths),
            "interpretation": (
                "RouteCode-compatible metric rows available: "
                + (", ".join(sorted(set(metric_rows))) if metric_rows else "none")
                + (". Blocked rows remain." if blocked else ". No blocked command-path rows in supplied tables.")
            ),
        }
    ]
    for check_id, reasons in sorted(blocked.items()):
        rows.append(
            {
                "section": "external_baselines",
                "item": check_id,
                "status": "blocked",
                "key_value": "",
                "evidence": "; ".join(evidence_paths),
                "interpretation": reasons,
            }
        )
    return rows


def _bool_series(table: pd.DataFrame, column: str) -> pd.Series:
    if column not in table.columns:
        return pd.Series(False, index=table.index)
    return table[column].map(_as_bool)


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def _format_float(value: object) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return ""
    return f"{float(number):.4f}"
