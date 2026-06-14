from __future__ import annotations

import pandas as pd


REQUIRED_OUTCOME_COLUMNS = [
    "query_id",
    "query_text",
    "dataset",
    "model_id",
    "quality",
    "cost_total",
    "judge",
]

OPTIONAL_OUTCOME_COLUMNS = [
    "domain",
    "cost_input",
    "cost_output",
    "latency",
    "tokens_input",
    "tokens_output",
    "metadata_json",
]


def validate_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_OUTCOME_COLUMNS if column not in outcomes.columns]
    if missing:
        raise ValueError(f"Missing required outcome columns: {missing}")

    if outcomes[["query_id", "model_id"]].isna().any().any():
        raise ValueError("query_id and model_id cannot contain nulls")

    duplicated = outcomes.duplicated(["query_id", "model_id"])
    if duplicated.any():
        examples = outcomes.loc[duplicated, ["query_id", "model_id"]].head().to_dict("records")
        raise ValueError(f"Duplicate query/model outcomes found: {examples}")

    numeric_columns = ["quality", "cost_total"]
    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(outcomes[column]):
            raise ValueError(f"{column} must be numeric")
        if outcomes[column].isna().any():
            raise ValueError(f"{column} cannot contain nulls")

    if ((outcomes["quality"] < 0) | (outcomes["quality"] > 1)).any():
        raise ValueError("quality must be bounded in [0, 1]")
    if (outcomes["cost_total"] < 0).any():
        raise ValueError("cost_total must be non-negative")

    model_counts = outcomes.groupby("query_id")["model_id"].nunique()
    if model_counts.nunique() != 1:
        raise ValueError("Each query must have outcomes for the same number of models")

    return outcomes.copy()
