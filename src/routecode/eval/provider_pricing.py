from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


def provider_price_coverage(outcomes: pd.DataFrame, schedule: Mapping[str, Any]) -> pd.DataFrame:
    """Report which outcome models have provider price entries."""

    prices = _price_entries(schedule)
    notes = _unmapped_notes(schedule)
    provider = str(schedule.get("provider", ""))
    schedule_name = str(schedule.get("name", "provider_schedule"))
    checked_date = str(schedule.get("source_checked_date", ""))
    rows: list[dict[str, Any]] = []
    for model_id, group in outcomes.groupby("model_id", sort=True):
        model = str(model_id)
        entry = prices.get(model)
        note = notes.get(model)
        mapped = entry is not None
        rows.append(
            {
                "schedule": schedule_name,
                "provider": _entry_provider(entry, provider) if mapped else _entry_provider(note, provider),
                "source_checked_date": checked_date,
                "model_id": model,
                "mapped": mapped,
                "provider_model_id": str(entry.get("provider_model_id", "")) if mapped else _optional_entry_str(note, "provider_model_id"),
                "input_price_per_million_tokens": float(entry.get("input", 0.0)) if mapped else 0.0,
                "output_price_per_million_tokens": float(entry.get("output", 0.0)) if mapped else 0.0,
                "source_url": str(entry.get("source_url", "")) if mapped else _optional_entry_str(note, "source_url"),
                "row_count": int(len(group)),
                "query_count": int(group["query_id"].nunique()),
                "coverage_note": "mapped" if mapped else _optional_entry_str(note, "coverage_note", "model_not_in_provider_schedule"),
            }
        )
    return pd.DataFrame(rows)


def apply_provider_price_schedule(outcomes: pd.DataFrame, schedule: Mapping[str, Any]) -> pd.DataFrame:
    """Return outcomes with cost fields recomputed from provider token prices.

    Prices are expected in USD per million input/output tokens. By default,
    models not present in the schedule are dropped so downstream routers compare
    only models with known provider prices.
    """

    prices = _price_entries(schedule)
    policy = str(schedule.get("unmapped_policy", "drop"))
    if policy not in {"drop", "keep_benchmark"}:
        raise ValueError("provider pricing unmapped_policy must be 'drop' or 'keep_benchmark'")
    priced = outcomes.copy()
    priced["provider_price_mapped"] = priced["model_id"].astype(str).isin(prices)
    if policy == "drop":
        priced = priced[priced["provider_price_mapped"]].copy()
    if priced.empty:
        return priced

    provider = str(schedule.get("provider", ""))
    schedule_name = str(schedule.get("name", "provider_schedule"))
    checked_date = str(schedule.get("source_checked_date", ""))
    for index, row in priced.iterrows():
        model_id = str(row["model_id"])
        entry = prices.get(model_id)
        if entry is None:
            priced.at[index, "provider"] = ""
            priced.at[index, "price_schedule"] = schedule_name
            priced.at[index, "provider_model_id"] = ""
            priced.at[index, "price_source_url"] = ""
            priced.at[index, "price_source_checked_date"] = checked_date
            continue
        input_price = float(entry["input"])
        output_price = float(entry["output"])
        input_cost = float(row.get("tokens_input", 0.0)) * input_price / 1_000_000.0
        output_cost = float(row.get("tokens_output", 0.0)) * output_price / 1_000_000.0
        priced.at[index, "cost_input"] = input_cost
        priced.at[index, "cost_output"] = output_cost
        priced.at[index, "cost_total"] = input_cost + output_cost
        priced.at[index, "provider"] = _entry_provider(entry, provider)
        priced.at[index, "price_schedule"] = schedule_name
        priced.at[index, "provider_model_id"] = str(entry.get("provider_model_id", model_id))
        priced.at[index, "input_price_per_million_tokens"] = input_price
        priced.at[index, "output_price_per_million_tokens"] = output_price
        priced.at[index, "price_source_url"] = str(entry.get("source_url", ""))
        priced.at[index, "price_source_checked_date"] = checked_date
    return priced


def mapped_model_count(outcomes: pd.DataFrame, schedule: Mapping[str, Any]) -> int:
    coverage = provider_price_coverage(outcomes, schedule)
    return int(coverage["mapped"].sum()) if not coverage.empty else 0


def _entry_provider(entry: Mapping[str, Any] | None, fallback: str) -> str:
    if entry is None:
        return fallback
    return str(entry.get("provider", fallback))


def _optional_entry_str(entry: Mapping[str, Any] | None, key: str, default: str = "") -> str:
    if entry is None:
        return default
    return str(entry.get(key, default))


def _price_entries(schedule: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = schedule.get("prices_per_million_tokens") or {}
    if not isinstance(entries, Mapping):
        raise ValueError("provider pricing schedule must define prices_per_million_tokens as a mapping")
    normalized: dict[str, Mapping[str, Any]] = {}
    for model_id, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"provider pricing entry for {model_id} must be a mapping")
        if "input" not in entry or "output" not in entry:
            raise ValueError(f"provider pricing entry for {model_id} must define input and output prices")
        normalized[str(model_id)] = entry
    return normalized


def _unmapped_notes(schedule: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = schedule.get("unmapped_price_notes") or {}
    if not isinstance(entries, Mapping):
        raise ValueError("provider pricing schedule must define unmapped_price_notes as a mapping")
    normalized: dict[str, Mapping[str, Any]] = {}
    for model_id, entry in entries.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"provider pricing unmapped note for {model_id} must be a mapping")
        normalized[str(model_id)] = entry
    return normalized
