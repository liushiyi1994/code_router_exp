from __future__ import annotations

import sys
import types

import pandas as pd

from routecode.eval.transformer_embedding_router import (
    evaluate_transformer_embedding_router,
    extract_local_transformer_embeddings,
)
from routecode.matrix import Matrices


def _matrices(query_ids: list[str], utility_rows: list[list[float]]) -> Matrices:
    model_ids = ["m0", "m1"]
    utility = pd.DataFrame(utility_rows, index=pd.Index(query_ids, name="query_id"), columns=model_ids)
    quality = utility.copy()
    cost = pd.DataFrame(0.0, index=utility.index, columns=model_ids)
    query_info = pd.DataFrame(
        {
            "query_text": [f"text {query_id}" for query_id in query_ids],
            "dataset": ["synthetic"] * len(query_ids),
            "domain": ["d0"] * len(query_ids),
        },
        index=utility.index,
    )
    return Matrices(quality=quality, cost=cost, utility=utility, query_info=query_info, model_ids=model_ids)


def test_transformer_embedding_router_writes_skipped_rows_without_cached_encoder():
    readiness = pd.DataFrame(
        [
            {
                "model_id": "answerdotai/ModernBERT-base",
                "cache_status": "missing_local_cache",
                "runnable_as_encoder_baseline": False,
                "reason": "missing_local_cache",
                "local_path": "",
            }
        ]
    )

    table = evaluate_transformer_embedding_router(
        train=_matrices(["q0", "q1"], [[1.0, 0.0], [0.0, 1.0]]),
        test=_matrices(["q2"], [[1.0, 0.0]]),
        readiness_table=readiness,
        embedding_provider=None,
        direct_methods=["knn"],
        n_bootstrap=5,
    )

    row = table.iloc[0]
    assert row["status"] == "skipped"
    assert row["reason"] == "no_cached_encoder_candidate"
    assert row["readiness_reason"] == "missing_local_cache"
    assert pd.isna(row["mean_utility"])


def test_transformer_embedding_router_evaluates_injected_embeddings():
    readiness = pd.DataFrame(
        [
            {
                "model_id": "answerdotai/ModernBERT-base",
                "cache_status": "cached",
                "runnable_as_encoder_baseline": True,
                "reason": "cached_encoder_candidate",
                "local_path": "/tmp/modernbert",
            }
        ]
    )

    def provider(_row: pd.Series, query_info: pd.DataFrame) -> pd.DataFrame:
        values = {
            "q0": [1.0, 0.0],
            "q1": [0.0, 1.0],
            "q2": [0.9, 0.1],
            "q3": [0.1, 0.9],
        }
        return pd.DataFrame.from_dict(values, orient="index").loc[query_info.index]

    table = evaluate_transformer_embedding_router(
        train=_matrices(["q0", "q1"], [[1.0, 0.0], [0.0, 1.0]]),
        test=_matrices(["q2", "q3"], [[1.0, 0.0], [0.0, 1.0]]),
        readiness_table=readiness,
        embedding_provider=provider,
        direct_methods=["knn"],
        n_bootstrap=5,
        n_neighbors=1,
    )

    row = table.iloc[0]
    assert row["status"] == "executed"
    assert row["method"] == "transformer_embedding_direct_router_knn"
    assert row["direct_router_method"] == "knn"
    assert row["model_id"] == "answerdotai/ModernBERT-base"
    assert row["mean_utility"] == 1.0
    assert row["embedding_source"] == "local_transformer"


def test_transformer_embedding_router_keeps_skipped_rows_with_runnable_encoder():
    readiness = pd.DataFrame(
        [
            {
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "cache_status": "cached",
                "runnable_as_encoder_baseline": True,
                "reason": "cached_encoder_candidate",
                "local_path": "/tmp/minilm",
            },
            {
                "model_id": "answerdotai/ModernBERT-base",
                "cache_status": "missing_local_cache",
                "runnable_as_encoder_baseline": False,
                "reason": "missing_local_cache",
                "local_path": "",
            },
        ]
    )

    def provider(_row: pd.Series, query_info: pd.DataFrame) -> pd.DataFrame:
        values = {
            "q0": [1.0, 0.0],
            "q1": [0.0, 1.0],
            "q2": [0.9, 0.1],
            "q3": [0.1, 0.9],
        }
        return pd.DataFrame.from_dict(values, orient="index").loc[query_info.index]

    table = evaluate_transformer_embedding_router(
        train=_matrices(["q0", "q1"], [[1.0, 0.0], [0.0, 1.0]]),
        test=_matrices(["q2", "q3"], [[1.0, 0.0], [0.0, 1.0]]),
        readiness_table=readiness,
        embedding_provider=provider,
        direct_methods=["knn"],
        n_bootstrap=5,
        n_neighbors=1,
    )

    executed = table[table["status"] == "executed"]
    skipped = table[table["status"] == "skipped"]
    assert executed["model_id"].tolist() == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert skipped["model_id"].tolist() == ["answerdotai/ModernBERT-base"]
    assert skipped.iloc[0]["reason"] == "no_cached_encoder_candidate"
    assert skipped.iloc[0]["readiness_reason"] == "missing_local_cache"


def test_extract_local_transformer_embeddings_sets_tokenizer_regex_fix(monkeypatch):
    tokenizer_calls: list[dict] = []
    model_calls: list[dict] = []

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            tokenizer_calls.append(kwargs)
            return object()

    class FakeModel:
        @staticmethod
        def from_pretrained(*_args, **kwargs):
            model_calls.append(kwargs)
            return FakeModel()

        def to(self, _device):
            return self

        def eval(self):
            return self

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *_args):
            return False

    fake_transformers = types.SimpleNamespace(AutoTokenizer=FakeTokenizer, AutoModel=FakeModel)
    fake_torch = types.SimpleNamespace(cuda=FakeCuda(), no_grad=lambda: FakeNoGrad())
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    query_info = pd.DataFrame({"query_text": []}, index=pd.Index([], name="query_id"))
    embeddings = extract_local_transformer_embeddings(local_path="/tmp/model", query_info=query_info, device="cpu")

    assert embeddings.empty
    assert tokenizer_calls[0]["local_files_only"] is True
    assert tokenizer_calls[0]["trust_remote_code"] is False
    assert tokenizer_calls[0]["fix_mistral_regex"] is True
    assert model_calls[0]["local_files_only"] is True
    assert model_calls[0]["trust_remote_code"] is False
