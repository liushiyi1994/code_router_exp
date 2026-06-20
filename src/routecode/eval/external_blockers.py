from __future__ import annotations

from collections.abc import Mapping
import math

import pandas as pd


def summarize_external_blockers(readiness_tables: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize blocked external-command rows across one or more runs."""

    grouped: dict[str, dict[str, object]] = {}
    for run_name, table in readiness_tables.items():
        if table.empty or "check_id" not in table.columns:
            continue
        for _, row in table.iterrows():
            if str(row.get("status", "")).lower() != "blocked":
                continue
            check_id = str(row["check_id"])
            reasons = _split_reasons(row.get("blocking_reasons", ""))
            entry = grouped.setdefault(
                check_id,
                {
                    "check_id": check_id,
                    "blocked_runs": set(),
                    "blocking_reasons": set(),
                    "missing_modules": set(),
                    "missing_checkpoints": set(),
                    "missing_assets": set(),
                    "service_requirements": set(),
                    "other_blockers": set(),
                },
            )
            entry["blocked_runs"].add(str(run_name))
            entry["blocking_reasons"].update(reasons)
            parsed = _parse_reasons(reasons)
            for key, values in parsed.items():
                entry[key].update(values)

    rows = []
    for check_id, entry in sorted(grouped.items()):
        modules = sorted(entry["missing_modules"])
        checkpoints = sorted(entry["missing_checkpoints"])
        assets = sorted(entry["missing_assets"])
        service = sorted(entry["service_requirements"])
        other = sorted(entry["other_blockers"])
        can_progress_without_download = not checkpoints and not service
        rows.append(
            {
                "check_id": check_id,
                "blocked_runs": ",".join(sorted(entry["blocked_runs"])),
                "blocked_run_count": len(entry["blocked_runs"]),
                "blocking_reasons": ";".join(sorted(entry["blocking_reasons"])),
                "missing_modules": ",".join(modules),
                "missing_checkpoints": ",".join(checkpoints),
                "missing_assets": ",".join(assets),
                "service_requirements": ",".join(service),
                "other_blockers": ",".join(other),
                "can_progress_without_download": can_progress_without_download,
                "next_action": _next_action(modules, checkpoints, assets, service, other),
            }
        )
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)


_OUTPUT_COLUMNS = [
    "check_id",
    "blocked_runs",
    "blocked_run_count",
    "blocking_reasons",
    "missing_modules",
    "missing_checkpoints",
    "missing_assets",
    "service_requirements",
    "other_blockers",
    "can_progress_without_download",
    "next_action",
]


def _parse_reasons(reasons: list[str]) -> dict[str, set[str]]:
    parsed = {
        "missing_modules": set(),
        "missing_checkpoints": set(),
        "missing_assets": set(),
        "service_requirements": set(),
        "other_blockers": set(),
    }
    for reason in reasons:
        if reason.startswith("missing_python_modules:"):
            parsed["missing_modules"].update(_split_csv(reason.split(":", 1)[1]))
        elif reason.startswith("embedding_config_requires_env:"):
            parsed["service_requirements"].update(_split_csv(reason.split(":", 1)[1]))
        elif reason.startswith("embedding_service_env_missing:"):
            parsed["service_requirements"].update(_split_csv(reason.split(":", 1)[1]))
        elif reason == "requires_embedding_service":
            parsed["service_requirements"].add(reason)
        elif "checkpoint" in reason:
            parsed["missing_checkpoints"].add(reason)
        elif reason.startswith("missing_"):
            parsed["missing_assets"].add(reason)
        elif reason:
            parsed["other_blockers"].add(reason)
    return parsed


def _split_reasons(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    return [item.strip() for item in str(value).split(";") if item.strip() and item.strip().lower() != "nan"]


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _next_action(
    modules: list[str],
    checkpoints: list[str],
    assets: list[str],
    service: list[str],
    other: list[str],
) -> str:
    actions = []
    if checkpoints:
        actions.append("Provision local checkpoints: " + ",".join(checkpoints) + ".")
    if modules:
        noun = "module" if len(modules) == 1 else "modules"
        actions.append(f"Install Python {noun}: " + ",".join(modules) + ".")
    if assets:
        noun = "asset" if len(assets) == 1 else "assets"
        actions.append(f"Prepare missing local {noun}: " + ",".join(assets) + ".")
    if service:
        actions.append("Use cached/local embeddings or set service requirements: " + ",".join(service) + ".")
    if other:
        actions.append("Inspect blocker reasons: " + ",".join(other) + ".")
    return " ".join(actions)
