from __future__ import annotations

import pandas as pd
import pytest

from routecode.eval.provider_pricing import (
    apply_provider_price_schedule,
    provider_price_coverage,
)


def _outcomes() -> pd.DataFrame:
    rows = []
    for query_id, input_tokens, output_tokens in [("q0", 1000, 2000), ("q1", 500, 100)]:
        for model_id in ["cheap", "strong", "unmapped"]:
            rows.append(
                {
                    "query_id": query_id,
                    "query_text": f"query {query_id}",
                    "dataset": "demo",
                    "domain": "demo",
                    "split": "train",
                    "model_id": model_id,
                    "quality": 0.8 if model_id == "strong" else 0.6,
                    "cost_input": 0.0,
                    "cost_output": 0.0,
                    "cost_total": 0.0,
                    "latency": 0.0,
                    "tokens_input": input_tokens,
                    "tokens_output": output_tokens,
                    "judge": "test",
                    "metadata_json": "{}",
                }
            )
    return pd.DataFrame(rows)


def _schedule() -> dict:
    return {
        "name": "demo_provider",
        "provider": "ExampleProvider",
        "source_checked_date": "2026-06-15",
        "unmapped_policy": "drop",
        "prices_per_million_tokens": {
            "cheap": {
                "provider_model_id": "provider/cheap",
                "input": 0.10,
                "output": 0.20,
                "source_url": "https://example.com/cheap",
            },
            "strong": {
                "provider_model_id": "provider/strong",
                "input": 1.00,
                "output": 2.00,
                "source_url": "https://example.com/strong",
            },
        },
    }


def test_apply_provider_price_schedule_reprices_token_costs_and_drops_unmapped_models():
    priced = apply_provider_price_schedule(_outcomes(), _schedule())

    assert set(priced["model_id"]) == {"cheap", "strong"}
    cheap_q0 = priced[(priced["query_id"] == "q0") & (priced["model_id"] == "cheap")].iloc[0]
    assert cheap_q0["cost_input"] == pytest.approx(1000 * 0.10 / 1_000_000)
    assert cheap_q0["cost_output"] == pytest.approx(2000 * 0.20 / 1_000_000)
    assert cheap_q0["cost_total"] == pytest.approx((1000 * 0.10 + 2000 * 0.20) / 1_000_000)
    assert cheap_q0["provider"] == "ExampleProvider"
    assert cheap_q0["provider_model_id"] == "provider/cheap"
    assert cheap_q0["price_source_url"] == "https://example.com/cheap"


def test_apply_provider_price_schedule_uses_entry_provider_when_present():
    schedule = _schedule()
    schedule["prices_per_million_tokens"]["strong"]["provider"] = "OtherProvider"

    priced = apply_provider_price_schedule(_outcomes(), schedule)
    strong_q0 = priced[(priced["query_id"] == "q0") & (priced["model_id"] == "strong")].iloc[0]
    cheap_q0 = priced[(priced["query_id"] == "q0") & (priced["model_id"] == "cheap")].iloc[0]

    assert strong_q0["provider"] == "OtherProvider"
    assert cheap_q0["provider"] == "ExampleProvider"


def test_provider_price_coverage_reports_mapped_and_unmapped_models():
    coverage = provider_price_coverage(_outcomes(), _schedule()).set_index("model_id")

    assert bool(coverage.loc["cheap", "mapped"])
    assert bool(coverage.loc["strong", "mapped"])
    assert not bool(coverage.loc["unmapped", "mapped"])
    assert coverage.loc["cheap", "row_count"] == 2
    assert coverage.loc["cheap", "input_price_per_million_tokens"] == 0.10
    assert coverage.loc["strong", "output_price_per_million_tokens"] == 2.00
    assert coverage.loc["unmapped", "coverage_note"] == "model_not_in_provider_schedule"


def test_provider_price_coverage_attaches_unmapped_source_notes():
    schedule = _schedule()
    schedule["unmapped_price_notes"] = {
        "unmapped": {
            "provider": "FlatRateHost",
            "provider_model_id": "host/unmapped",
            "source_url": "https://example.com/unmapped-flat-rate",
            "coverage_note": "flat_rate_only_no_per_token_price",
        }
    }

    coverage = provider_price_coverage(_outcomes(), schedule).set_index("model_id")

    assert not bool(coverage.loc["unmapped", "mapped"])
    assert coverage.loc["unmapped", "provider"] == "FlatRateHost"
    assert coverage.loc["unmapped", "provider_model_id"] == "host/unmapped"
    assert coverage.loc["unmapped", "source_url"] == "https://example.com/unmapped-flat-rate"
    assert coverage.loc["unmapped", "coverage_note"] == "flat_rate_only_no_per_token_price"


def test_provider_price_coverage_uses_entry_provider_when_present():
    schedule = _schedule()
    schedule["prices_per_million_tokens"]["strong"]["provider"] = "OtherProvider"

    coverage = provider_price_coverage(_outcomes(), schedule).set_index("model_id")

    assert coverage.loc["strong", "provider"] == "OtherProvider"
    assert coverage.loc["cheap", "provider"] == "ExampleProvider"
