from __future__ import annotations

from typing import Any, Callable

import pandas as pd


EXPLAINABILITY_FIELDS = [
    "best_model",
    "second_best_model",
    "model_margin",
    "top_domains",
    "top_datasets",
    "representative_queries",
    "high_regret_failure_cases",
    "model_utility_vector",
    "human_readable_explanation",
]


def summarize_code_card_interpretability(
    codebook_name: str,
    codebook: Any,
    cards: list[dict[str, Any]],
) -> pd.DataFrame:
    """Compare label-only observability with generated code-card observability."""

    n_labels = int(getattr(codebook, "effective_labels", len(cards)))
    label_only = {
        "codebook": str(codebook_name),
        "condition": "label_only",
        "n_labels": n_labels,
        "available_explainability_fields": _label_only_field_count(codebook),
        "best_model_coverage": _label_only_best_model_coverage(codebook, n_labels),
        "second_best_model_coverage": 0.0,
        "model_margin_coverage": 0.0,
        "domain_summary_coverage": 0.0,
        "dataset_summary_coverage": 0.0,
        "representative_query_coverage": 0.0,
        "failure_case_coverage": 0.0,
        "utility_vector_coverage": 0.0,
        "human_explanation_coverage": 0.0,
        "avg_representative_queries": 0.0,
        "avg_failure_cases": 0.0,
    }
    with_cards = {
        "codebook": str(codebook_name),
        "condition": "with_code_cards",
        "n_labels": n_labels,
        "available_explainability_fields": _available_card_field_count(cards),
        "best_model_coverage": _coverage(cards, lambda card: _nonempty_scalar(card.get("best_model"))),
        "second_best_model_coverage": _coverage(cards, lambda card: _nonempty_scalar(card.get("second_best_model"))),
        "model_margin_coverage": _coverage(cards, lambda card: card.get("model_margin") is not None),
        "domain_summary_coverage": _coverage(cards, lambda card: _nonempty_sequence(card.get("top_domains"))),
        "dataset_summary_coverage": _coverage(cards, lambda card: _nonempty_sequence(card.get("top_datasets"))),
        "representative_query_coverage": _coverage(
            cards,
            lambda card: _nonempty_sequence(card.get("representative_queries")),
        ),
        "failure_case_coverage": _coverage(
            cards,
            lambda card: _nonempty_sequence(card.get("high_regret_failure_cases")),
        ),
        "utility_vector_coverage": _coverage(cards, lambda card: _nonempty_mapping(card.get("model_utility_vector"))),
        "human_explanation_coverage": _coverage(
            cards,
            lambda card: _nonempty_scalar(card.get("human_readable_explanation")),
        ),
        "avg_representative_queries": _average_count(cards, "representative_queries"),
        "avg_failure_cases": _average_count(cards, "high_regret_failure_cases"),
    }
    return pd.DataFrame([label_only, with_cards])


def _label_only_field_count(codebook: Any) -> int:
    return 1 if getattr(codebook, "label_to_model", None) else 0


def _label_only_best_model_coverage(codebook: Any, n_labels: int) -> float:
    if n_labels <= 0:
        return 0.0
    label_to_model = getattr(codebook, "label_to_model", {}) or {}
    return min(len(label_to_model), n_labels) / n_labels


def _available_card_field_count(cards: list[dict[str, Any]]) -> int:
    return sum(1 for field in EXPLAINABILITY_FIELDS if _field_coverage(cards, field) > 0.0)


def _field_coverage(cards: list[dict[str, Any]], field: str) -> float:
    return _coverage(cards, lambda card: _field_available(card.get(field)))


def _coverage(cards: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> float:
    if not cards:
        return 0.0
    return sum(1 for card in cards if predicate(card)) / len(cards)


def _average_count(cards: list[dict[str, Any]], field: str) -> float:
    if not cards:
        return 0.0
    return float(sum(len(card.get(field) or []) for card in cards) / len(cards))


def _field_available(value: Any) -> bool:
    if isinstance(value, dict):
        return _nonempty_mapping(value)
    if isinstance(value, list):
        return _nonempty_sequence(value)
    return _nonempty_scalar(value)


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _nonempty_sequence(value: Any) -> bool:
    return isinstance(value, list) and bool(value)


def _nonempty_scalar(value: Any) -> bool:
    return value is not None and str(value) != ""
