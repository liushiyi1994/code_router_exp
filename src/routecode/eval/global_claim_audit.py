from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from routecode.eval.claim_audit import audit_claims


NONMISSING_STATUSES = {"supported", "diagnostic_supported", "diagnostic_alive", "deferred", "not_supported"}


def audit_global_claims(result_dirs: Iterable[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_run_frames: list[pd.DataFrame] = []
    for result_dir in result_dirs:
        result_path = Path(result_dir)
        if (result_path / "table_claim_status.csv").exists():
            table = pd.read_csv(result_path / "table_claim_status.csv")
        else:
            table = audit_claims(result_path)
        table.insert(0, "result_id", result_path.name)
        table.insert(1, "result_dir", str(result_path))
        per_run_frames.append(table)
    per_run = pd.concat(per_run_frames, ignore_index=True, sort=False) if per_run_frames else pd.DataFrame()
    return per_run, aggregate_claim_tables(per_run)


def aggregate_claim_tables(per_run: pd.DataFrame) -> pd.DataFrame:
    if per_run.empty:
        return pd.DataFrame(
            columns=[
                "claim_id",
                "claim",
                "global_status",
                "run_count",
                "status_counts",
                "best_primary_value",
                "worst_primary_value",
                "best_result_id",
                "evidence_summary",
                "interpretation",
            ]
        )
    rows = []
    for claim_id, group in per_run.groupby("claim_id", sort=False):
        values = pd.to_numeric(group.get("primary_value", pd.Series(dtype=float)), errors="coerce")
        best_idx = values.idxmax() if not values.dropna().empty else group.index[0]
        best_row = group.loc[best_idx]
        nonmissing = group[group["status"].isin(NONMISSING_STATUSES)].copy()
        rows.append(
            {
                "claim_id": claim_id,
                "claim": str(best_row.get("claim", "")),
                "global_status": _global_status(nonmissing if not nonmissing.empty else group),
                "run_count": int(group["result_id"].nunique()),
                "status_counts": _status_counts(group["status"]),
                "best_primary_value": float(values.max()) if not values.dropna().empty else np.nan,
                "worst_primary_value": float(values.min()) if not values.dropna().empty else np.nan,
                "best_result_id": str(best_row.get("result_id", "")),
                "evidence_summary": _evidence_summary(group),
                "interpretation": _global_interpretation(claim_id, nonmissing if not nonmissing.empty else group),
            }
        )
    return pd.DataFrame(rows)


def _global_status(group: pd.DataFrame) -> str:
    statuses = [str(status) for status in group["status"].dropna().tolist()]
    if not statuses:
        return "missing_evidence"
    unique = set(statuses)
    if unique == {"missing_evidence"}:
        return "missing_evidence"
    nonmissing = unique - {"missing_evidence"}
    if not nonmissing:
        return "missing_evidence"
    if "supported" in nonmissing and nonmissing <= {"supported", "diagnostic_supported"}:
        return "supported" if "supported" in nonmissing else "diagnostic_supported"
    if "not_supported" in nonmissing and len(nonmissing) > 1:
        return "mixed_evidence"
    if nonmissing == {"not_supported"}:
        return "not_supported"
    if "diagnostic_supported" in nonmissing:
        return "diagnostic_supported"
    if "diagnostic_alive" in nonmissing:
        return "diagnostic_alive"
    if "deferred" in nonmissing:
        return "deferred"
    return sorted(nonmissing)[0]


def _status_counts(statuses: pd.Series) -> str:
    counts = statuses.astype(str).value_counts().sort_index()
    return "; ".join(f"{status}={count}" for status, count in counts.items())


def _evidence_summary(group: pd.DataFrame) -> str:
    parts = []
    for _, row in group.head(8).iterrows():
        value = row.get("primary_value", np.nan)
        value_text = "" if pd.isna(value) else f"{float(value):.4f}"
        parts.append(
            f"{row.get('result_id', '')}: {row.get('status', '')}"
            + (f" ({row.get('primary_metric', '')}={value_text})" if value_text else "")
        )
    if len(group) > 8:
        parts.append(f"... {len(group) - 8} more")
    return "; ".join(parts)


def _global_interpretation(claim_id: str, group: pd.DataFrame) -> str:
    status = _global_status(group)
    if claim_id == "small_inferred_labels" and status == "not_supported":
        return "Do not claim that small inferred route labels recover most routing performance across current runs."
    if status == "mixed_evidence":
        return "Evidence is mixed across runs; keep this claim diagnostic and identify the conditions that change it."
    if status in {"diagnostic_alive", "diagnostic_supported"}:
        return "Use diagnostic framing; broader coverage is still required for a paper-level claim."
    if status == "not_supported":
        return "Current cross-run evidence does not support this claim."
    if status == "missing_evidence":
        return "Required result artifacts are missing across the supplied runs."
    return "Current cross-run evidence meets the configured claim gate."
