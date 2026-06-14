from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from routecode.codes.routecode import RouteCodeCodebook


def build_code_cards(
    codebook: RouteCodeCodebook,
    query_info: pd.DataFrame,
    utility: pd.DataFrame,
    max_examples: int = 4,
) -> list[dict[str, Any]]:
    if codebook.train_labels_ is None or codebook.label_utility_ is None:
        raise RuntimeError("Codebook must be fit before building code cards")

    cards = []
    oracle_utility = utility.max(axis=1)
    for label in range(codebook.effective_labels):
        query_ids = codebook.train_labels_.index[codebook.train_labels_ == label]
        label_info = query_info.loc[query_ids]
        avg_utility = codebook.label_utility_.loc[label].sort_values(ascending=False)
        best_model = str(avg_utility.index[0])
        second_model = str(avg_utility.index[1]) if len(avg_utility) > 1 else best_model
        margin = float(avg_utility.iloc[0] - avg_utility.iloc[1]) if len(avg_utility) > 1 else 0.0
        selected_utility = utility.loc[query_ids, best_model]
        regrets = (oracle_utility.loc[query_ids] - selected_utility).sort_values(ascending=False)

        top_domains = _top_counts(label_info, "domain")
        top_datasets = _top_counts(label_info, "dataset")
        examples = _query_examples(label_info.head(max_examples))
        failures = _query_examples(label_info.loc[regrets.head(max_examples).index])
        short_name = f"{top_domains[0]['name'] if top_domains else 'mixed'}__{best_model}"
        cards.append(
            {
                "label_id": int(label),
                "short_name": short_name,
                "size": int(len(query_ids)),
                "best_model": best_model,
                "second_best_model": second_model,
                "model_margin": margin,
                "top_domains": top_domains,
                "top_datasets": top_datasets,
                "representative_queries": examples,
                "high_regret_failure_cases": failures,
                "model_utility_vector": {str(model): float(value) for model, value in avg_utility.items()},
                "human_readable_explanation": _explanation(short_name, best_model, top_domains, top_datasets),
            }
        )
    return cards


def write_code_cards(
    path: str,
    codebook: RouteCodeCodebook,
    query_info: pd.DataFrame,
    utility: pd.DataFrame,
    max_examples: int = 4,
) -> None:
    cards = build_code_cards(codebook, query_info, utility, max_examples=max_examples)
    lines = [
        "# RouteCode Code Cards",
        "",
        "These cards summarize route labels learned from train-set utility profiles. "
        "They are synthetic-pilot diagnostics, not paper claims.",
        "",
    ]

    for card in cards:
        lines.extend(
            [
                f"## Route label {card['label_id']}: `{card['short_name']}`",
                "",
                f"- Size: {card['size']} train queries",
                f"- Best model: `{card['best_model']}`",
                f"- Second-best model: `{card['second_best_model']}`",
                f"- Mean utility margin: {card['model_margin']:.4f}",
                f"- Dominant domains: {_format_counts(card['top_domains'])}",
                f"- Dominant datasets: {_format_counts(card['top_datasets'])}",
                f"- Model utility vector: {_format_vector(card['model_utility_vector'])}",
                f"- Human-readable explanation: {card['human_readable_explanation']}",
                "- Representative queries:",
                *[f"  - {example['query_text']}" for example in card["representative_queries"]],
                "- Highest-regret train examples under this label:",
                *[f"  - {example['query_text']}" for example in card["high_regret_failure_cases"]],
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def write_code_cards_json(
    path: str | Path,
    codebook: RouteCodeCodebook,
    query_info: pd.DataFrame,
    utility: pd.DataFrame,
    max_examples: int = 4,
) -> None:
    payload = {
        "schema_version": "routecode.code_cards.v1",
        "cards": build_code_cards(codebook, query_info, utility, max_examples=max_examples),
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _top_counts(query_info: pd.DataFrame, column: str, n: int = 3) -> list[dict[str, int | str]]:
    if column not in query_info.columns or query_info.empty:
        return []
    counts = query_info[column].astype(str).value_counts().head(n)
    return [{"name": str(index), "count": int(value)} for index, value in counts.items()]


def _query_examples(query_info: pd.DataFrame) -> list[dict[str, str]]:
    examples = []
    for query_id, row in query_info.iterrows():
        examples.append(
            {
                "query_id": str(query_id),
                "query_text": str(row.get("query_text", "")),
                "dataset": str(row.get("dataset", "")),
                "domain": str(row.get("domain", "")),
            }
        )
    return examples


def _format_counts(counts: list[dict[str, int | str]]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{item['name']} ({item['count']})" for item in counts)


def _format_vector(values: dict[str, float]) -> str:
    return ", ".join(f"{model}={value:.3f}" for model, value in values.items())


def _explanation(
    short_name: str,
    best_model: str,
    top_domains: list[dict[str, int | str]],
    top_datasets: list[dict[str, int | str]],
) -> str:
    domain = str(top_domains[0]["name"]) if top_domains else "mixed domains"
    dataset = str(top_datasets[0]["name"]) if top_datasets else "mixed datasets"
    return (
        f"`{short_name}` groups queries whose train-set utility profile favors `{best_model}`. "
        f"It is most associated with domain `{domain}` and dataset `{dataset}` in this run."
    )
