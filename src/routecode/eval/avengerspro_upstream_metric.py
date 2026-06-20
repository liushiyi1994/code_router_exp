from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from routecode.eval.evaluate import evaluate_selection
from routecode.matrix import Matrices
from routecode.metrics import selected_values
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


def avengerspro_payload_has_routing_details(path: str | Path) -> bool:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    details = payload.get("results", {}).get("routing_details")
    return isinstance(details, list) and bool(details)


def selected_models_from_routing_details(
    routing_details: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> pd.Series:
    """Convert upstream Avengers-Pro routing_details to a selected-model Series.

    The upstream details do not echo RouteCode query ids, so alignment is by the
    test JSONL order emitted from RouteCode assets.
    """

    if len(routing_details) != len(test_records):
        raise ValueError(
            f"Avengers-Pro prediction count mismatch: expected {len(test_records)}, got {len(routing_details)}"
        )
    query_ids: list[str] = []
    selected_models: list[str] = []
    for index, (detail, record) in enumerate(zip(routing_details, test_records, strict=True)):
        query_id = str(record.get("query_id", ""))
        if not query_id:
            raise ValueError(f"Avengers-Pro test record {index} is missing query_id")
        models = detail.get("selected_models")
        if not isinstance(models, list) or not models:
            raise ValueError(f"Avengers-Pro routing detail {index} has no selected_models")
        query_ids.append(query_id)
        selected_models.append(str(models[0]))
    return pd.Series(selected_models, index=pd.Index(query_ids, name="query_id"), name="selected_model")


def evaluate_avengerspro_routing_details(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    *,
    routing_details: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    prediction_source: str,
    seed: int = 0,
    n_bootstrap: int = 300,
    ci: float = 0.95,
    knn_k: int = 15,
) -> dict[str, Any]:
    selected = selected_models_from_routing_details(routing_details, test_records).reindex(test.utility.index)
    if selected.isna().any():
        missing = selected[selected.isna()].index.astype(str).tolist()
        raise ValueError(f"Avengers-Pro predictions missing test query ids: {missing[:5]}")
    unknown = sorted(set(selected.astype(str)) - set(map(str, test.utility.columns)))
    if unknown:
        raise ValueError(f"Avengers-Pro selected unknown model ids: {unknown[:5]}")

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    internal_knn = KNNRouter(knn_k).fit(train.query_info, train.utility, embeddings).predict(
        test.query_info,
        embeddings,
    )
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, internal_knn).mean()))
    oracle_mean = float(test.utility.max(axis=1).mean())
    row = evaluate_selection(
        method="avengerspro_upstream_simple_cluster_postprocessed",
        selected_models=selected,
        matrices=test,
        baseline_mean=baseline_mean,
        learned_reference_mean=learned_reference_mean,
        oracle_mean=oracle_mean,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        labels=selected,
    )
    row.update(
        {
            "baseline_family": "avengerspro_upstream_model_code_postprocessed",
            "split_aligned_with_routecode": True,
            "routecode_metric_compatible": True,
            "upstream_model_code_used": True,
            "exact_upstream_command": False,
            "external_api_calls": False,
            "official_upstream_result": False,
            "prediction_source": str(prediction_source),
            "prediction_count": int(len(selected)),
            "selected_models": ",".join(sorted(set(selected.astype(str)))),
            "paper_reference": "Avengers-Pro",
            "repo_reference": "https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/AvengersPro",
            "implementation_note": (
                "RouteCode utility postprocessing over routing_details captured from the upstream "
                "Avengers-Pro SimpleClusterRouter class. The exact CLI JSON omits routing_details, "
                "so this is upstream model-code evidence rather than an exact command output."
            ),
        }
    )
    return row
