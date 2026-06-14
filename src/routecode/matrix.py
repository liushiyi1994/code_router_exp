from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from routecode.data.schema import validate_outcomes


@dataclass(frozen=True)
class Matrices:
    quality: pd.DataFrame
    cost: pd.DataFrame
    utility: pd.DataFrame
    query_info: pd.DataFrame
    model_ids: list[str]


def build_matrices(outcomes: pd.DataFrame, lambda_cost: float) -> Matrices:
    valid = validate_outcomes(outcomes)
    query_order = valid["query_id"].drop_duplicates().tolist()
    model_order = valid["model_id"].drop_duplicates().tolist()

    quality = (
        valid.pivot(index="query_id", columns="model_id", values="quality")
        .reindex(index=query_order, columns=model_order)
        .astype(float)
    )
    cost = (
        valid.pivot(index="query_id", columns="model_id", values="cost_total")
        .reindex(index=query_order, columns=model_order)
        .astype(float)
    )
    utility = quality - float(lambda_cost) * cost

    query_columns = [
        column
        for column in valid.columns
        if column
        not in {
            "model_id",
            "quality",
            "cost_input",
            "cost_output",
            "cost_total",
            "latency",
            "tokens_input",
            "tokens_output",
            "judge",
            "metadata_json",
        }
    ]
    query_info = (
        valid[query_columns]
        .drop_duplicates("query_id")
        .set_index("query_id")
        .reindex(query_order)
    )
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=model_order)
