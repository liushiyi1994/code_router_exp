from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from routecode.eval.evaluate import evaluate_selection
from routecode.matrix import Matrices
from routecode.metrics import selected_values
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


@dataclass(frozen=True)
class RouteCodeSplitMasks:
    train_row_indices: list[int]
    val_row_indices: list[int]
    test_row_indices: list[int]
    test_query_ids: list[str]
    test_query_positions: list[int]


def build_routecode_split_masks(router_data: pd.DataFrame, *, num_llms: int) -> RouteCodeSplitMasks:
    """Build GraphRouter edge-mask row indices from preserved RouteCode splits."""

    required = {"query_id", "routecode_split", "llm"}
    missing = sorted(required - set(router_data.columns))
    if missing:
        raise ValueError(f"GraphRouter router data missing required columns: {missing}")
    if num_llms <= 0:
        raise ValueError("num_llms must be positive")

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    test_query_ids: list[str] = []
    test_query_positions: list[int] = []
    supported = {"train", "val", "test"}
    query_position = 0
    for query_id, group in router_data.groupby("query_id", sort=False):
        if len(group) != num_llms:
            raise ValueError(
                f"GraphRouter query {query_id} has {len(group)} rows; expected exactly {num_llms}"
            )
        splits = sorted(set(group["routecode_split"].astype(str)))
        if len(splits) != 1:
            raise ValueError(f"GraphRouter query {query_id} has multiple routecode_split values: {splits}")
        split = splits[0]
        if split not in supported:
            raise ValueError(f"GraphRouter query {query_id} has unsupported routecode_split: {split}")
        row_indices = [int(index) for index in group.index]
        if split == "train":
            train.extend(row_indices)
        elif split == "val":
            val.extend(row_indices)
        else:
            test.extend(row_indices)
            test_query_ids.append(str(query_id))
            test_query_positions.append(query_position)
        query_position += 1
    if not train:
        raise ValueError("GraphRouter split-aligned training mask is empty")
    if not test:
        raise ValueError("GraphRouter split-aligned test mask is empty")
    return RouteCodeSplitMasks(
        train_row_indices=train,
        val_row_indices=val,
        test_row_indices=test,
        test_query_ids=test_query_ids,
        test_query_positions=test_query_positions,
    )


def evaluate_graphrouter_selected_models(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    selected_models: pd.Series,
    *,
    prediction_source: str,
    seed: int = 0,
    n_bootstrap: int = 300,
    ci: float = 0.95,
    knn_k: int = 15,
) -> dict[str, Any]:
    """Score GraphRouter selected models on the RouteCode test utility matrix."""

    selected_models = selected_models.astype(str).reindex(test.utility.index)
    if selected_models.isna().any():
        missing = selected_models[selected_models.isna()].index.astype(str).tolist()
        raise ValueError(f"GraphRouter predictions missing test query ids: {missing[:5]}")
    unknown = sorted(set(selected_models.astype(str)) - set(map(str, test.utility.columns)))
    if unknown:
        raise ValueError(f"GraphRouter selected unknown model ids: {unknown[:5]}")

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    internal_knn = KNNRouter(knn_k).fit(train.query_info, train.utility, embeddings).predict(
        test.query_info,
        embeddings,
    )
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, internal_knn).mean()))
    oracle_mean = float(test.utility.max(axis=1).mean())

    row = evaluate_selection(
        method="graphrouter_split_aligned_gnn",
        selected_models=selected_models,
        matrices=test,
        baseline_mean=baseline_mean,
        learned_reference_mean=learned_reference_mean,
        oracle_mean=oracle_mean,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        labels=selected_models,
    )
    row.update(
        {
            "baseline_family": "graphrouter_upstream_model_code_split_aligned",
            "split_aligned_with_routecode": True,
            "routecode_metric_compatible": True,
            "upstream_model_code_used": True,
            "exact_upstream_command": False,
            "external_api_calls": False,
            "official_upstream_result": False,
            "prediction_source": str(prediction_source),
            "prediction_count": int(len(selected_models)),
            "selected_models": ",".join(sorted(set(selected_models.astype(str)))),
            "paper_reference": "GraphRouter",
            "repo_reference": "https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/GraphRouter",
            "implementation_note": (
                "RouteCode split-aligned adapter around the upstream GraphRouter GNN/model code. "
                "The unmodified upstream command does not consume arbitrary RouteCode train/test masks "
                "or emit RouteCode utility metrics, so this row is not an exact upstream command result."
            ),
        }
    )
    return row
