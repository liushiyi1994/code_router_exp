from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from routecode.eval.evaluate import evaluate_selection
from routecode.eval.new_model_calibration import fit_predict_budgeted_direct_router
from routecode.matrix import Matrices
from routecode.metrics import selected_values
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


EmbeddingProvider = Callable[[pd.Series, pd.DataFrame], pd.DataFrame]


def evaluate_transformer_embedding_router(
    *,
    train: Matrices,
    test: Matrices,
    readiness_table: pd.DataFrame,
    embedding_provider: EmbeddingProvider | None,
    direct_methods: Sequence[str],
    random_state: int = 0,
    n_bootstrap: int = 100,
    ci: float = 0.95,
    max_iter: int = 200,
    n_neighbors: int = 15,
    logistic_solver: str = "lbfgs",
    svm_backend: str = "linear_svc",
    tol: float = 1e-4,
) -> pd.DataFrame:
    """Evaluate direct routers on local transformer embeddings when available.

    The caller supplies a readiness table produced from local cache metadata and
    an embedding provider. If no runnable checkpoint is present, this still
    returns an explicit skipped artifact row instead of silently doing nothing.
    """

    if readiness_table.empty:
        return pd.DataFrame([_skipped_row(None, "no_readiness_rows")])

    runnable_mask = readiness_table["runnable_as_encoder_baseline"].map(_as_bool)
    runnable = readiness_table[runnable_mask].copy()
    non_runnable = readiness_table[~runnable_mask].copy()
    if runnable.empty:
        return pd.DataFrame([_skipped_row(row, "no_cached_encoder_candidate") for _, row in readiness_table.iterrows()])

    all_query_info = _combined_query_info(train, test)
    rows: list[dict[str, Any]] = []
    baseline_selected = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, baseline_selected).mean())
    oracle_selected = OracleRouter().predict(test.utility)
    oracle_mean = float(selected_values(test.utility, oracle_selected).mean())
    train_labels = train.utility.idxmax(axis=1).astype(str).rename("selected_model")

    for model_index, (_, readiness_row) in enumerate(runnable.iterrows()):
        if embedding_provider is None:
            rows.append(_failed_row(readiness_row, "missing_embedding_provider"))
            continue
        try:
            embeddings = embedding_provider(readiness_row, all_query_info)
            _validate_embeddings(embeddings, train.utility.index, test.utility.index)
        except Exception as exc:  # pragma: no cover - exercised by real cache environments.
            rows.append(_failed_row(readiness_row, f"embedding_extraction_failed:{type(exc).__name__}:{exc}"))
            continue

        train_embeddings = embeddings.loc[train.utility.index]
        test_embeddings = embeddings.loc[test.utility.index]
        for method_index, method in enumerate(direct_methods):
            method_name = str(method)
            try:
                selected = fit_predict_budgeted_direct_router(
                    method=method_name,
                    train_labels=train_labels,
                    train_embeddings=train_embeddings,
                    test_embeddings=test_embeddings,
                    random_state=int(random_state) + 100 * model_index + method_index,
                    max_iter=max_iter,
                    n_neighbors=n_neighbors,
                    logistic_solver=logistic_solver,
                    svm_backend=svm_backend,
                    tol=tol,
                )
            except Exception as exc:  # pragma: no cover - defensive experiment artifact path.
                failed = _failed_row(readiness_row, f"direct_router_failed:{method_name}:{type(exc).__name__}:{exc}")
                failed["direct_router_method"] = method_name
                failed["method"] = f"transformer_embedding_direct_router_{method_name}"
                rows.append(failed)
                continue
            row = evaluate_selection(
                method=f"transformer_embedding_direct_router_{method_name}",
                selected_models=selected,
                matrices=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=oracle_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=int(random_state) + 1000 * model_index + method_index,
            )
            row.update(_metadata(readiness_row))
            row.update(
                {
                    "status": "executed",
                    "reason": "",
                    "direct_router_method": method_name,
                    "embedding_source": "local_transformer",
                    "embedding_dim": int(train_embeddings.shape[1]),
                    "reference_policy": "oracle_reference_for_transformer_ablation",
                }
            )
            rows.append(row)

    rows.extend(_skipped_row(row, "no_cached_encoder_candidate") for _, row in non_runnable.iterrows())
    return pd.DataFrame(rows)


def extract_local_transformer_embeddings(
    *,
    local_path: str,
    query_info: pd.DataFrame,
    text_column: str = "query_text",
    batch_size: int = 16,
    max_length: int = 256,
    device: str = "auto",
) -> pd.DataFrame:
    """Extract mean-pooled transformer embeddings from a cached local model.

    This intentionally uses `local_files_only=True` and `trust_remote_code=False`
    so experiment scripts do not download models or execute repository code.
    """

    if text_column not in query_info.columns:
        raise ValueError(f"query_info is missing text column: {text_column}")
    from transformers import AutoModel, AutoTokenizer
    import torch

    selected_device = _select_device(device, torch)
    tokenizer = AutoTokenizer.from_pretrained(
        local_path,
        local_files_only=True,
        trust_remote_code=False,
        fix_mistral_regex=True,
    )
    model = AutoModel.from_pretrained(local_path, local_files_only=True, trust_remote_code=False)
    model.to(selected_device)
    model.eval()

    texts = query_info[text_column].fillna("").astype(str).tolist()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = texts[start : start + int(batch_size)]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(selected_device) for key, value in encoded.items()}
            output = model(**encoded)
            hidden = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(pooled.detach().cpu().numpy())
    values = np.vstack(chunks) if chunks else np.empty((0, 0), dtype=float)
    return pd.DataFrame(
        values,
        index=query_info.index,
        columns=[f"transformer_{idx}" for idx in range(values.shape[1])],
    )


def _combined_query_info(train: Matrices, test: Matrices) -> pd.DataFrame:
    combined = pd.concat([train.query_info, test.query_info], axis=0)
    return combined.loc[~combined.index.duplicated(keep="first")]


def _validate_embeddings(embeddings: pd.DataFrame, train_index: pd.Index, test_index: pd.Index) -> None:
    required = pd.Index(train_index.tolist() + test_index.tolist())
    missing = required.difference(embeddings.index)
    if len(missing) > 0:
        preview = ", ".join(str(item) for item in missing[:5])
        raise ValueError(f"Transformer embeddings missing query rows: {preview}")
    if embeddings.shape[1] == 0:
        raise ValueError("Transformer embeddings have zero columns")


def _select_device(device: str, torch_module) -> str:
    requested = str(device).lower()
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return requested


def _skipped_row(readiness_row: pd.Series | None, reason: str) -> dict[str, Any]:
    row = _empty_metric_row()
    if readiness_row is not None:
        row.update(_metadata(readiness_row))
    row.update(
        {
            "method": "transformer_embedding_direct_router",
            "status": "skipped",
            "reason": reason,
            "direct_router_method": "",
            "embedding_source": "local_transformer",
            "reference_policy": "not_evaluated",
        }
    )
    return row


def _failed_row(readiness_row: pd.Series, reason: str) -> dict[str, Any]:
    row = _empty_metric_row()
    row.update(_metadata(readiness_row))
    row.update(
        {
            "method": "transformer_embedding_direct_router",
            "status": "failed",
            "reason": reason,
            "direct_router_method": "",
            "embedding_source": "local_transformer",
            "reference_policy": "not_evaluated",
        }
    )
    return row


def _empty_metric_row() -> dict[str, Any]:
    return {
        "method": "",
        "mean_utility": np.nan,
        "oracle_regret": np.nan,
        "mean_quality": np.nan,
        "normalized_cost": np.nan,
        "K": "",
        "utility_ci_low": np.nan,
        "utility_ci_high": np.nan,
        "recovered_gap_vs_learned": np.nan,
        "recovered_gap_vs_oracle": np.nan,
        "selected_model_entropy": np.nan,
        "rate_log2K": np.nan,
        "empirical_H_Z": "",
        "status": "",
        "reason": "",
        "direct_router_method": "",
        "embedding_source": "",
        "embedding_dim": np.nan,
        "reference_policy": "",
    }


def _metadata(readiness_row: pd.Series) -> dict[str, Any]:
    return {
        "probe_scope": "transformer_embedding_direct_router",
        "model_id": str(readiness_row.get("model_id", "")),
        "cache_status": str(readiness_row.get("cache_status", "")),
        "readiness_reason": str(readiness_row.get("reason", "")),
        "architecture": str(readiness_row.get("architecture", "")),
        "model_type": str(readiness_row.get("model_type", "")),
        "hidden_size": readiness_row.get("hidden_size", ""),
        "size_gb": readiness_row.get("size_gb", np.nan),
        "local_path": str(readiness_row.get("local_path", "")),
    }


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)
