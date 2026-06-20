from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from routecode.matrix import Matrices


@dataclass(frozen=True)
class ExternalBaselineAssets:
    frugalgpt_train_records: list[dict[str, Any]]
    frugalgpt_test_records: list[dict[str, Any]]
    embedllm_train: pd.DataFrame
    embedllm_test: pd.DataFrame
    embedllm_question_embeddings: np.ndarray
    best_route_train_records: list[dict[str, Any]]
    best_route_validation_records: list[dict[str, Any]]
    best_route_test_records: list[dict[str, Any]]
    routerdc_train_records: list[dict[str, Any]]
    routerdc_test_records: list[dict[str, Any]]
    routerdc_final_eval_records: list[dict[str, Any]]
    modelsat_train_records: list[dict[str, Any]]
    modelsat_validation_records: list[dict[str, Any]]
    modelsat_ood_records: list[dict[str, Any]]
    modelsat_model_descriptions: dict[str, dict[str, Any]]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WrittenExternalBaselineAssets:
    asset_dir: Path
    metadata_path: Path
    frugalgpt_train_path: Path
    frugalgpt_test_path: Path
    embedllm_train_path: Path
    embedllm_test_path: Path
    embedllm_smoke_train_path: Path
    embedllm_smoke_test_path: Path
    embedllm_question_embeddings_path: Path
    embedllm_mf_question_embeddings_path: Path
    best_route_train_path: Path
    best_route_validation_path: Path
    best_route_test_path: Path
    routerdc_train_path: Path
    routerdc_test_path: Path
    routerdc_final_eval_path: Path
    modelsat_train_path: Path
    modelsat_validation_path: Path
    modelsat_ood_path: Path
    modelsat_model_description_path: Path


def build_external_baseline_assets(
    matrices_by_split: Mapping[str, Matrices],
    embeddings: pd.DataFrame,
    *,
    correctness_threshold: float = 0.5,
) -> ExternalBaselineAssets:
    ordered_splits = _ordered_splits(matrices_by_split)
    _validate_embeddings(matrices_by_split, embeddings, ordered_splits)
    model_ids = [str(model) for model in matrices_by_split[ordered_splits[0]].model_ids]
    for split in ordered_splits:
        if [str(model) for model in matrices_by_split[split].model_ids] != model_ids:
            raise ValueError("External baseline assets require the same model order in every split")

    prompt_index = _prompt_index(matrices_by_split, ordered_splits)
    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    validation_split = "val" if "val" in matrices_by_split else "test"

    frugalgpt = {
        split: [_frugalgpt_record(matrices_by_split[split], query_id, split) for query_id in matrices_by_split[split].quality.index]
        for split in ordered_splits
    }
    embedllm = {
        split: _embedllm_frame(
            matrices_by_split[split],
            split,
            prompt_index,
            model_index,
            correctness_threshold,
        )
        for split in ordered_splits
    }
    best_route = {
        split: [
            _best_route_record(matrices_by_split[split], query_id, split)
            for query_id in matrices_by_split[split].quality.index
        ]
        for split in ordered_splits
    }
    routerdc = {
        split: [
            _routerdc_record(matrices_by_split[split], query_id, split, prompt_index, correctness_threshold)
            for query_id in matrices_by_split[split].quality.index
        ]
        for split in ordered_splits
    }
    modelsat = {
        split: [
            _modelsat_record(matrices_by_split[split], query_id, split, correctness_threshold)
            for query_id in matrices_by_split[split].quality.index
        ]
        for split in ordered_splits
    }

    question_embeddings = np.asarray(
        [embeddings.loc[query_id].to_numpy(dtype=np.float32) for query_id, _ in sorted(prompt_index.items(), key=lambda item: item[1])],
        dtype=np.float32,
    )
    metadata = {
        "split_aligned_with_routecode": True,
        "routecode_metric_compatible": False,
        "official_upstream_result": False,
        "asset_families": ["frugalgpt", "embedllm", "best_route", "routerdc", "modelsat"],
        "routecode_splits": ordered_splits,
        "validation_split": validation_split,
        "query_count": int(sum(len(matrices_by_split[split].quality.index) for split in ordered_splits)),
        "model_count": int(len(model_ids)),
        "embedding_dim": int(embeddings.shape[1]),
        "correctness_threshold": float(correctness_threshold),
        "compatibility_note": (
            "Split-aligned input assets for upstream baseline command paths. "
            "These are not trained models or metric rows."
        ),
    }
    return ExternalBaselineAssets(
        frugalgpt_train_records=frugalgpt.get("train", []),
        frugalgpt_test_records=frugalgpt.get("test", []),
        embedllm_train=embedllm.get("train", pd.DataFrame()),
        embedllm_test=embedllm.get("test", pd.DataFrame()),
        embedllm_question_embeddings=question_embeddings,
        best_route_train_records=best_route.get("train", []),
        best_route_validation_records=best_route.get(validation_split, []),
        best_route_test_records=best_route.get("test", []),
        routerdc_train_records=routerdc.get("train", []),
        routerdc_test_records=routerdc.get("test", []),
        routerdc_final_eval_records=routerdc.get("test", []),
        modelsat_train_records=modelsat.get("train", []),
        modelsat_validation_records=modelsat.get(validation_split, []),
        modelsat_ood_records=modelsat.get("test", []),
        modelsat_model_descriptions=_model_descriptions(matrices_by_split, ordered_splits, model_ids),
        metadata=metadata,
    )


def write_external_baseline_assets(
    assets: ExternalBaselineAssets,
    out_dir: str | Path,
) -> WrittenExternalBaselineAssets:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    frugal_dir = root / "frugalgpt_split_aligned"
    embed_dir = root / "embedllm_assets"
    best_route_dir = root / "best_route_assets"
    routerdc_dir = root / "routerdc_assets"
    modelsat_dir = root / "modelsat_assets/seed42"
    for path in [frugal_dir, embed_dir, best_route_dir, routerdc_dir, modelsat_dir]:
        path.mkdir(parents=True, exist_ok=True)

    frugal_train = frugal_dir / "train.jsonl"
    frugal_test = frugal_dir / "test.jsonl"
    embed_train = embed_dir / "train.csv"
    embed_test = embed_dir / "test.csv"
    embed_smoke_train = embed_dir / "smoke_train.csv"
    embed_smoke_test = embed_dir / "smoke_test.csv"
    embed_embeddings = embed_dir / "question_embeddings.pth"
    embed_mf_embeddings = embed_dir / "question_embeddings_3584.pth"
    best_route_train = best_route_dir / "train.jsonl"
    best_route_validation = best_route_dir / "validation.jsonl"
    best_route_test = best_route_dir / "test.jsonl"
    routerdc_train = routerdc_dir / "train.json"
    routerdc_test = routerdc_dir / "test.json"
    routerdc_final = routerdc_dir / "final_eval.json"
    modelsat_train = modelsat_dir / "train.json"
    modelsat_validation = modelsat_dir / "test.json"
    modelsat_ood = modelsat_dir / "ood.json"
    modelsat_description = modelsat_dir / "model_description.json"
    metadata_path = root / "external_baseline_assets_metadata.json"

    _write_jsonl(frugal_train, assets.frugalgpt_train_records)
    _write_jsonl(frugal_test, assets.frugalgpt_test_records)
    assets.embedllm_train.to_csv(embed_train, index=False)
    assets.embedllm_test.to_csv(embed_test, index=False)
    _embedllm_smoke_frame(assets.embedllm_train, prompt_limit=24).to_csv(embed_smoke_train, index=False)
    _embedllm_smoke_frame(assets.embedllm_test, prompt_limit=12).to_csv(embed_smoke_test, index=False)
    _write_torch_tensor(embed_embeddings, assets.embedllm_question_embeddings)
    _write_torch_tensor(embed_mf_embeddings, _embedllm_mf_question_embeddings(assets.embedllm_question_embeddings))
    _write_jsonl(best_route_train, assets.best_route_train_records)
    _write_jsonl(best_route_validation, assets.best_route_validation_records)
    _write_jsonl(best_route_test, assets.best_route_test_records)
    _write_json(routerdc_train, assets.routerdc_train_records)
    _write_json(routerdc_test, assets.routerdc_test_records)
    _write_json(routerdc_final, assets.routerdc_final_eval_records)
    _write_json(modelsat_train, assets.modelsat_train_records)
    _write_json(modelsat_validation, assets.modelsat_validation_records)
    _write_json(modelsat_ood, assets.modelsat_ood_records)
    _write_json(modelsat_description, assets.modelsat_model_descriptions)
    _write_json(metadata_path, assets.metadata)

    return WrittenExternalBaselineAssets(
        asset_dir=root,
        metadata_path=metadata_path,
        frugalgpt_train_path=frugal_train,
        frugalgpt_test_path=frugal_test,
        embedllm_train_path=embed_train,
        embedllm_test_path=embed_test,
        embedllm_smoke_train_path=embed_smoke_train,
        embedllm_smoke_test_path=embed_smoke_test,
        embedllm_question_embeddings_path=embed_embeddings,
        embedllm_mf_question_embeddings_path=embed_mf_embeddings,
        best_route_train_path=best_route_train,
        best_route_validation_path=best_route_validation,
        best_route_test_path=best_route_test,
        routerdc_train_path=routerdc_train,
        routerdc_test_path=routerdc_test,
        routerdc_final_eval_path=routerdc_final,
        modelsat_train_path=modelsat_train,
        modelsat_validation_path=modelsat_validation,
        modelsat_ood_path=modelsat_ood,
        modelsat_model_description_path=modelsat_description,
    )


def _embedllm_smoke_frame(frame: pd.DataFrame, *, prompt_limit: int) -> pd.DataFrame:
    if frame.empty or "prompt_id" not in frame.columns:
        return frame.copy()
    prompt_ids = frame["prompt_id"].drop_duplicates().head(prompt_limit)
    return frame[frame["prompt_id"].isin(prompt_ids)].copy()


def _embedllm_mf_question_embeddings(question_embeddings: np.ndarray, *, target_dim: int = 3584) -> np.ndarray:
    source = np.asarray(question_embeddings, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("EmbedLLM MF question embeddings must be a 2D matrix")
    if source.shape[1] > target_dim:
        raise ValueError(
            f"EmbedLLM MF compatibility embeddings cannot pad source dim {source.shape[1]} to {target_dim}"
        )
    padded = np.zeros((source.shape[0], target_dim), dtype=np.float32)
    padded[:, : source.shape[1]] = source
    return padded


def summarize_external_baseline_assets(assets: ExternalBaselineAssets) -> pd.DataFrame:
    rows = [
        _summary_row("frugalgpt", len(assets.frugalgpt_train_records), 0, len(assets.frugalgpt_test_records)),
        _summary_row("embedllm", len(assets.embedllm_train), 0, len(assets.embedllm_test)),
        _summary_row(
            "best_route",
            len(assets.best_route_train_records),
            len(assets.best_route_validation_records),
            len(assets.best_route_test_records),
        ),
        _summary_row(
            "routerdc",
            len(assets.routerdc_train_records),
            0,
            len(assets.routerdc_test_records),
            final_eval_records=len(assets.routerdc_final_eval_records),
        ),
        _summary_row(
            "modelsat",
            len(assets.modelsat_train_records),
            len(assets.modelsat_validation_records),
            len(assets.modelsat_ood_records),
            model_description_count=len(assets.modelsat_model_descriptions),
        ),
    ]
    return pd.DataFrame(rows)


def _summary_row(
    asset_family: str,
    train_records: int,
    validation_records: int,
    test_records: int,
    *,
    final_eval_records: int = 0,
    model_description_count: int = 0,
) -> dict[str, Any]:
    return {
        "asset_family": asset_family,
        "train_records": int(train_records),
        "validation_records": int(validation_records),
        "test_records": int(test_records),
        "final_eval_records": int(final_eval_records),
        "model_description_count": int(model_description_count),
        "split_aligned_with_routecode": True,
        "routecode_metric_compatible": False,
        "official_upstream_result": False,
        "implementation_note": "Input assets only; not a trained upstream baseline or metric row.",
    }


def _ordered_splits(matrices_by_split: Mapping[str, Matrices]) -> list[str]:
    ordered = [split for split in ["train", "val", "test"] if split in matrices_by_split]
    ordered.extend(split for split in matrices_by_split if split not in ordered)
    if "train" not in matrices_by_split or "test" not in matrices_by_split:
        raise ValueError("External baseline assets require train and test matrices")
    return ordered


def _validate_embeddings(
    matrices_by_split: Mapping[str, Matrices],
    embeddings: pd.DataFrame,
    ordered_splits: list[str],
) -> None:
    missing = []
    for split in ordered_splits:
        for query_id in matrices_by_split[split].query_info.index:
            if query_id not in embeddings.index:
                missing.append(str(query_id))
    if missing:
        raise ValueError(f"Missing embedding rows for external baseline assets: {missing[:5]}")


def _prompt_index(matrices_by_split: Mapping[str, Matrices], ordered_splits: list[str]) -> dict[str, int]:
    query_ids: list[str] = []
    for split in ordered_splits:
        query_ids.extend(str(query_id) for query_id in matrices_by_split[split].query_info.index)
    return {query_id: idx for idx, query_id in enumerate(dict.fromkeys(query_ids))}


def _frugalgpt_record(matrices: Matrices, query_id: str, split: str) -> dict[str, Any]:
    query_info = matrices.query_info.loc[query_id].to_dict()
    return {
        "dataset": str(query_info.get("dataset", "")),
        "index": str(query_id),
        "query": _prompt_from_query_info(query_info),
        "records": {model: float(matrices.quality.at[query_id, model]) for model in matrices.model_ids},
        "usages": {
            model: {
                "cost": float(matrices.cost.at[query_id, model]),
                "utility": float(matrices.utility.at[query_id, model]),
            }
            for model in matrices.model_ids
        },
        "routecode_split": str(split),
    }


def _embedllm_frame(
    matrices: Matrices,
    split: str,
    prompt_index: dict[str, int],
    model_index: dict[str, int],
    correctness_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for query_id in matrices.quality.index:
        query_info = matrices.query_info.loc[query_id].to_dict()
        for model in matrices.model_ids:
            rows.append(
                {
                    "model_id": int(model_index[str(model)]),
                    "model_name": str(model),
                    "prompt_id": int(prompt_index[str(query_id)]),
                    "prompt": _prompt_from_query_info(query_info),
                    "label": int(float(matrices.quality.at[query_id, model]) >= correctness_threshold),
                    "quality": float(matrices.quality.at[query_id, model]),
                    "cost": float(matrices.cost.at[query_id, model]),
                    "utility": float(matrices.utility.at[query_id, model]),
                    "query_id": str(query_id),
                    "routecode_split": str(split),
                }
            )
    return pd.DataFrame(rows)


def _best_route_record(matrices: Matrices, query_id: str, split: str) -> dict[str, Any]:
    query_info = matrices.query_info.loc[query_id].to_dict()
    prompt = _prompt_from_query_info(query_info)
    candidates = []
    for model in matrices.model_ids:
        quality = float(matrices.quality.at[query_id, model])
        cost = float(matrices.cost.at[query_id, model])
        candidates.append(
            {
                "model": str(model),
                "text": "",
                "scores": {
                    "quality": quality,
                    "utility": float(matrices.utility.at[query_id, model]),
                },
                "token_num_prompt": 1,
                "token_num_responses": [1],
                "cost": [cost] * 10,
                "decoding_method": "routecode",
            }
        )
    return {
        "id": str(query_id),
        "instruction": prompt,
        "input": "",
        "output": "",
        "candidates": candidates,
        "dataset": str(query_info.get("dataset", "")),
        "domain": str(query_info.get("domain", "")),
        "routecode_split": str(split),
    }


def _routerdc_record(
    matrices: Matrices,
    query_id: str,
    split: str,
    prompt_index: dict[str, int],
    correctness_threshold: float,
) -> dict[str, Any]:
    query_info = matrices.query_info.loc[query_id].to_dict()
    return {
        "id": str(query_id),
        "question": _prompt_from_query_info(query_info),
        "scores": {model: float(matrices.quality.at[query_id, model]) for model in matrices.model_ids},
        "correct": {
            model: bool(float(matrices.quality.at[query_id, model]) >= correctness_threshold)
            for model in matrices.model_ids
        },
        "cluster_id": int(prompt_index[str(query_id)] % max(1, min(16, len(prompt_index)))),
        "dataset": str(query_info.get("dataset", "")),
        "routecode_split": str(split),
    }


def _modelsat_record(
    matrices: Matrices,
    query_id: str,
    split: str,
    correctness_threshold: float,
) -> dict[str, Any]:
    query_info = matrices.query_info.loc[query_id].to_dict()
    return {
        "id": str(query_id),
        "query": _prompt_from_query_info(query_info),
        "scores": {model: float(matrices.quality.at[query_id, model]) for model in matrices.model_ids},
        "is_correct_sc": {
            model: bool(float(matrices.quality.at[query_id, model]) >= correctness_threshold)
            for model in matrices.model_ids
        },
        "task": str(query_info.get("domain", query_info.get("dataset", ""))),
        "dataset": str(query_info.get("dataset", "")),
        "routecode_split": str(split),
    }


def _model_descriptions(
    matrices_by_split: Mapping[str, Matrices],
    ordered_splits: list[str],
    model_ids: list[str],
) -> dict[str, dict[str, Any]]:
    quality_parts = [matrices_by_split[split].quality[model_ids] for split in ordered_splits]
    cost_parts = [matrices_by_split[split].cost[model_ids] for split in ordered_splits]
    quality = pd.concat(quality_parts)
    cost = pd.concat(cost_parts)
    return {
        model: {
            "model": model,
            "description": (
                f"RouteCode benchmark model {model}; mean quality "
                f"{float(quality[model].mean()):.4f}, mean cost {float(cost[model].mean()):.4f}."
            ),
            "mean_quality": float(quality[model].mean()),
            "mean_cost": float(cost[model].mean()),
        }
        for model in model_ids
    }


def _prompt_from_query_info(query_info: dict[str, Any]) -> str:
    for key in ["query_text", "prompt", "question", "text"]:
        value = query_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(query_info.get("query_id", "")).strip()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_torch_tensor(path: Path, values: np.ndarray) -> None:
    import torch

    torch.save(torch.tensor(values, dtype=torch.float32), path)
