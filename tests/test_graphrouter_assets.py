from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from routecode.eval.graphrouter_assets import build_graphrouter_assets, write_graphrouter_assets
from routecode.matrix import Matrices


def _matrices(query_ids: list[str], split: str) -> Matrices:
    models = ["m0", "m1", "m2"]
    quality = pd.DataFrame(
        [[0.8, 0.6, 0.4], [0.3, 0.7, 0.5]][: len(query_ids)],
        index=pd.Index(query_ids, name="query_id"),
        columns=models,
    )
    cost = pd.DataFrame(
        [[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]][: len(query_ids)],
        index=pd.Index(query_ids, name="query_id"),
        columns=models,
    )
    query_info = pd.DataFrame(
        [
            {
                "query_id": query_id,
                "query_text": f"Question for {query_id}",
                "dataset": "math" if idx == 0 else "code",
                "domain": "reasoning",
                "task_family": "family",
                "split": split,
            }
            for idx, query_id in enumerate(query_ids)
        ]
    ).set_index("query_id")
    return Matrices(
        quality=quality,
        cost=cost,
        utility=quality - 0.1 * cost,
        query_info=query_info,
        model_ids=models,
    )


def test_build_graphrouter_assets_preserves_routecode_splits_and_graphrouter_schema(tmp_path):
    embeddings = pd.DataFrame(
        [[1.0, 0.0, 0.5, 0.1], [0.2, 1.0, 0.3, 0.4], [0.6, 0.5, 0.4, 0.3]],
        index=pd.Index(["q0", "q1", "q2"], name="query_id"),
        columns=["e0", "e1", "e2", "e3"],
    )

    assets = build_graphrouter_assets(
        {"train": _matrices(["q0", "q1"], "train"), "test": _matrices(["q2"], "test")},
        embeddings,
        seed=17,
    )

    router_data = assets.router_data
    assert len(router_data) == 9
    assert {
        "task_id",
        "task_description",
        "task_description_embedding",
        "query",
        "query_embedding",
        "ground_truth",
        "metric",
        "llm",
        "effect",
        "cost",
        "cost_usd",
        "query_id",
        "routecode_split",
    }.issubset(router_data.columns)
    assert router_data.groupby("query_id")["llm"].apply(list).to_dict() == {
        "q0": ["m0", "m1", "m2"],
        "q1": ["m0", "m1", "m2"],
        "q2": ["m0", "m1", "m2"],
    }
    assert router_data.groupby("query_id")["routecode_split"].first().to_dict() == {
        "q0": "train",
        "q1": "train",
        "q2": "test",
    }
    assert router_data["cost"].between(0.0, 1.0).all()
    assert json.loads(router_data.iloc[0]["query_embedding"])[0] == [1.0, 0.0, 0.5, 0.1]
    assert isinstance(json.loads(router_data.iloc[0]["task_description_embedding"])[0][0], float)
    assert assets.llm_description_embeddings.shape == (3, 4)
    assert list(assets.llm_descriptions) == ["m0", "m1", "m2"]

    written = write_graphrouter_assets(assets, tmp_path / "graphrouter_assets")

    assert written.router_data_path.exists()
    assert written.llm_description_path.exists()
    assert written.llm_embedding_path.exists()
    assert written.config_path.exists()
    config_text = written.config_path.read_text(encoding="utf-8")
    assert str(written.router_data_path) in config_text
    assert str(written.llm_embedding_path) in config_text
    assert pd.read_csv(written.router_data_path)["routecode_split"].tolist() == router_data["routecode_split"].tolist()
    assert json.loads(written.llm_description_path.read_text(encoding="utf-8"))["m0"]["model"] == "m0"


def test_build_graphrouter_assets_rejects_missing_query_embedding():
    embeddings = pd.DataFrame([[1.0, 0.0]], index=pd.Index(["q0"], name="query_id"))

    try:
        build_graphrouter_assets({"train": _matrices(["q0", "q1"], "train")}, embeddings)
    except ValueError as exc:
        assert "Missing embedding rows for GraphRouter assets" in str(exc)
    else:
        raise AssertionError("missing embeddings should fail")
