from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from routecode.eval.evaluate import evaluate_selection
from routecode.matrix import Matrices
from routecode.metrics import selected_values
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


@dataclass
class _SklearnRouterState:
    query_embedding_list: list[Any]
    model_name_list: list[str]
    cfg: dict[str, Any]
    knn_model: Any | None = None
    svm_model: Any | None = None


def evaluate_llmrouter_library_adapters(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    *,
    output_dir: Path,
    llmrouter_root: Path,
    seed: int = 0,
    n_bootstrap: int = 300,
    ci: float = 0.95,
    knn_k: int = 15,
    svm_kernel: str = "rbf",
) -> pd.DataFrame:
    """Train selected local LLMRouter library trainers on RouteCode splits.

    This intentionally avoids LLMRouter route methods because those paths embed
    text with Longformer and may call model APIs. The metric rows use the
    upstream trainer classes, then load the saved sklearn artifacts and predict
    from RouteCode's deterministic local embeddings.
    """

    _validate_embeddings(train, test, embeddings)
    trainers = _load_llmrouter_trainers(llmrouter_root)
    artifact_dir = output_dir / "llmrouter_library_adapters"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_llmrouter_cli_assets(
        train,
        test,
        embeddings,
        artifact_dir=artifact_dir,
        knn_k=knn_k,
        svm_kernel=svm_kernel,
    )

    train_embeddings = embeddings.loc[train.utility.index].to_numpy(dtype=float)
    train_labels = train.utility.idxmax(axis=1).astype(str).tolist()
    test_embeddings = embeddings.loc[test.utility.index].to_numpy(dtype=float)

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    internal_knn = KNNRouter(knn_k).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, internal_knn).mean()))
    oracle_mean = float(test.utility.max(axis=1).mean())

    rows: list[dict[str, Any]] = []
    rows.append(
        _train_predict_row(
            method="llmrouter_library_knn",
            trainer_cls=trainers["knn"],
            router_state=_SklearnRouterState(
                query_embedding_list=[row for row in train_embeddings],
                model_name_list=train_labels,
                cfg={
                    "model_path": {
                        "ini_model_path": "",
                        "save_model_path": str((artifact_dir / "knn_model.pkl").resolve()),
                    }
                },
                knn_model=KNeighborsClassifier(
                    n_neighbors=max(1, min(int(knn_k), len(train_embeddings))),
                    metric="euclidean",
                ),
            ),
            model_path=artifact_dir / "knn_model.pkl",
            test_embeddings=test_embeddings,
            test_index=test.utility.index,
            test=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
        )
    )
    if len(set(train_labels)) >= 2:
        rows.append(
            _train_predict_row(
                method="llmrouter_library_svm",
                trainer_cls=trainers["svm"],
                router_state=_SklearnRouterState(
                    query_embedding_list=[row for row in train_embeddings],
                    model_name_list=train_labels,
                    cfg={
                        "model_path": {
                            "ini_model_path": "",
                            "save_model_path": str((artifact_dir / "svm_model.pkl").resolve()),
                        }
                    },
                    svm_model=SVC(kernel=svm_kernel, gamma="scale", random_state=seed),
                ),
                model_path=artifact_dir / "svm_model.pkl",
                test_embeddings=test_embeddings,
                test_index=test.utility.index,
                test=test,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 1,
            )
        )
    return pd.DataFrame(rows)


def evaluate_llmrouter_cli_predictions(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    *,
    predictions: dict[str, Path],
    seed: int = 0,
    n_bootstrap: int = 300,
    ci: float = 0.95,
    knn_k: int = 15,
) -> pd.DataFrame:
    """Score exact LLMRouter route-only CLI outputs on RouteCode utility.

    LLMRouter's upstream inference CLI emits routing decisions, not RouteCode
    utility metrics. This function keeps that boundary explicit: selections are
    read from exact CLI outputs, then scored against the RouteCode test matrix.
    Outputs are matched to the test split by row order because the upstream CLI
    preserves input order but does not echo arbitrary metadata fields.
    """

    _validate_embeddings(train, test, embeddings)
    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    internal_knn = KNNRouter(knn_k).fit(train.query_info, train.utility, embeddings).predict(test.query_info, embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, internal_knn).mean()))
    oracle_mean = float(test.utility.max(axis=1).mean())

    rows: list[dict[str, Any]] = []
    for offset, (router_short, prediction_path) in enumerate(sorted(predictions.items())):
        route_rows = _read_llmrouter_cli_predictions(Path(prediction_path))
        if len(route_rows) != len(test.utility.index):
            raise ValueError(
                f"LLMRouter CLI prediction count mismatch for {router_short}: "
                f"expected {len(test.utility.index)}, got {len(route_rows)}"
            )
        failed = [row for row in route_rows if not row.get("success")]
        if failed:
            first = failed[0]
            raise ValueError(
                f"LLMRouter CLI output contains failed rows for {router_short}: "
                f"{first.get('error', 'unknown error')}"
            )
        selected = pd.Series(
            [str(row.get("model_name", "")) for row in route_rows],
            index=test.utility.index,
            name="selected_model",
        )
        missing = sorted(set(selected) - set(map(str, test.utility.columns)))
        if missing:
            raise ValueError(f"LLMRouter CLI selected unknown model ids for {router_short}: {missing[:5]}")
        eval_row = evaluate_selection(
            method=f"llmrouter_cli_{router_short}",
            selected_models=selected,
            matrices=test,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + offset,
            labels=selected,
        )
        eval_row.update(
            {
                "baseline_family": "llmrouter_exact_cli_postprocessed",
                "split_aligned_with_routecode": True,
                "routecode_metric_compatible": True,
                "upstream_training_class_used": False,
                "exact_upstream_command": True,
                "external_api_calls": False,
                "official_upstream_result": False,
                "prediction_source": str(prediction_path),
                "prediction_count": int(len(route_rows)),
                "selected_models": ",".join(sorted(set(selected.astype(str)))),
                "paper_reference": "LLMRouter",
                "repo_reference": "https://github.com/ulab-uiuc/LLMRouter",
                "implementation_note": (
                    "Exact LLMRouter route-only CLI predictions scored with RouteCode "
                    "test-split utility; LLMRouter CLI itself does not emit RouteCode metrics."
                ),
            }
        )
        rows.append(eval_row)
    return pd.DataFrame(rows)


def _train_predict_row(
    *,
    method: str,
    trainer_cls: Any,
    router_state: _SklearnRouterState,
    model_path: Path,
    test_embeddings: Any,
    test_index: pd.Index,
    test: Matrices,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
) -> dict[str, Any]:
    trainer = trainer_cls(router_state, device="cpu")
    trainer.train()
    model = _load_pickle_model(model_path)
    selected = pd.Series(model.predict(test_embeddings), index=test_index, name="selected_model").astype(str)
    row = evaluate_selection(
        method=method,
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
            "baseline_family": "llmrouter_library_adapter",
            "split_aligned_with_routecode": True,
            "routecode_metric_compatible": True,
            "upstream_training_class_used": True,
            "exact_upstream_command": False,
            "external_api_calls": False,
            "official_upstream_result": False,
            "model_artifact_path": str(model_path),
            "selected_models": ",".join(sorted(set(selected.astype(str)))),
            "paper_reference": "LLMRouter",
            "repo_reference": "https://github.com/ulab-uiuc/LLMRouter",
            "implementation_note": (
                "Uses local LLMRouter trainer class on RouteCode precomputed embeddings; "
                "prediction/evaluation avoid Longformer embedding and API route paths."
            ),
        }
    )
    return row


def _load_llmrouter_trainers(llmrouter_root: Path) -> dict[str, Any]:
    root = Path(llmrouter_root)
    if not (root / "llmrouter").exists():
        raise FileNotFoundError(f"LLMRouter checkout not found: {root}")
    root_str = str(root.resolve())
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        from llmrouter.models.knnrouter.trainer import KNNRouterTrainer
        from llmrouter.models.svmrouter.trainer import SVMRouterTrainer
    finally:
        if inserted:
            try:
                sys.path.remove(root_str)
            except ValueError:
                pass
    return {"knn": KNNRouterTrainer, "svm": SVMRouterTrainer}


def _load_pickle_model(path: Path) -> Any:
    import pickle

    with Path(path).open("rb") as file:
        return pickle.load(file)


def _write_llmrouter_cli_assets(
    train: Matrices,
    test: Matrices,
    embeddings: pd.DataFrame,
    *,
    artifact_dir: Path,
    knn_k: int,
    svm_kernel: str,
) -> None:
    query_ids = [str(query_id) for query_id in train.utility.index]
    embedding_id = {query_id: idx for idx, query_id in enumerate(query_ids)}
    train_embeddings = embeddings.loc[query_ids].to_numpy(dtype=float)

    query_rows = []
    routing_rows = []
    embedding_lookup: dict[str, torch.Tensor] = {}
    for query_id in query_ids:
        query_info = train.query_info.loc[query_id].to_dict()
        prompt = _prompt_from_query_info(query_info, query_id)
        embedding_lookup[prompt] = torch.tensor(embeddings.loc[query_id].to_numpy(dtype=float), dtype=torch.float32)
        task_name = str(query_info.get("dataset", query_info.get("domain", "routecode")))
        query_rows.append(
            {
                "id": query_id,
                "task_name": task_name,
                "query": prompt,
                "ground_truth": "",
                "metric": "routecode_utility",
            }
        )
        for model in train.model_ids:
            routing_rows.append(
                {
                    "task_name": task_name,
                    "query": prompt,
                    "ground_truth": "",
                    "gt": "",
                    "metric": "routecode_utility",
                    "model_name": str(model),
                    "performance": float(train.utility.at[query_id, model]),
                    "quality": float(train.quality.at[query_id, model]),
                    "cost": float(train.cost.at[query_id, model]),
                    "embedding_id": int(embedding_id[query_id]),
                    "query_id": query_id,
                }
            )

    smoke_rows = []
    test_rows = []
    for query_id in [str(query_id) for query_id in test.utility.index]:
        query_info = test.query_info.loc[query_id].to_dict()
        prompt = _prompt_from_query_info(query_info, query_id)
        embedding_lookup[prompt] = torch.tensor(embeddings.loc[query_id].to_numpy(dtype=float), dtype=torch.float32)
        test_rows.append({"query": prompt, "query_id": query_id})
        if len(smoke_rows) < 32:
            smoke_rows.append({"query": prompt, "query_id": query_id})

    _write_jsonl(artifact_dir / "query_train.jsonl", query_rows)
    _write_jsonl(artifact_dir / "routing_train.jsonl", routing_rows)
    _write_jsonl(artifact_dir / "query_inference_smoke.jsonl", smoke_rows)
    _write_jsonl(artifact_dir / "query_inference_test.jsonl", test_rows)
    torch.save(torch.tensor(train_embeddings, dtype=torch.float32), artifact_dir / "query_embeddings.pt")
    torch.save(embedding_lookup, artifact_dir / "query_embedding_lookup.pt")
    _write_json(
        artifact_dir / "llm_candidates.json",
        {
            str(model): {
                "model": str(model),
                "service": "RouteCode",
                "api_endpoint": "http://localhost:0/v1",
                "size": "unknown",
                "feature": "RouteCode benchmark candidate",
            }
            for model in train.model_ids
        },
    )
    _write_llmrouter_train_config(
        artifact_dir / "knnrouter_train.yaml",
        artifact_dir=artifact_dir,
        router="knn",
        save_model_path=artifact_dir / "knn_cli_model.pkl",
        hparam={
            "n_neighbors": max(1, min(int(knn_k), len(query_ids))),
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "n_jobs": -1,
        },
    )
    _write_llmrouter_train_config(
        artifact_dir / "svmrouter_train.yaml",
        artifact_dir=artifact_dir,
        router="svm",
        save_model_path=artifact_dir / "svm_cli_model.pkl",
        hparam={"kernel": str(svm_kernel), "gamma": "scale"},
    )


def _write_llmrouter_train_config(
    path: Path,
    *,
    artifact_dir: Path,
    router: str,
    save_model_path: Path,
    hparam: dict[str, Any],
) -> None:
    del router
    config = {
        "data_path": {
            "query_data_train": str((artifact_dir / "query_train.jsonl").resolve()),
            "query_embedding_data": str((artifact_dir / "query_embeddings.pt").resolve()),
            "query_embedding_lookup": str((artifact_dir / "query_embedding_lookup.pt").resolve()),
            "routing_data_train": str((artifact_dir / "routing_train.jsonl").resolve()),
            "llm_data": str((artifact_dir / "llm_candidates.json").resolve()),
        },
        "model_path": {
            "ini_model_path": "",
            "save_model_path": str(save_model_path.resolve()),
            "load_model_path": str(save_model_path.resolve()),
        },
        "hparam": hparam,
        "metric": {"weights": {"performance": 1, "cost": 0, "llm_judge": 0}},
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _prompt_from_query_info(query_info: dict[str, Any], query_id: str) -> str:
    for key in ["query_text", "prompt", "question", "text"]:
        value = query_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return query_id


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_llmrouter_cli_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"LLMRouter CLI prediction file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"LLMRouter CLI prediction file is empty: {path}")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    if not isinstance(payload, list):
        raise ValueError(f"LLMRouter CLI prediction file must contain a JSON list: {path}")
    return payload


def _validate_embeddings(train: Matrices, test: Matrices, embeddings: pd.DataFrame) -> None:
    required = set(train.utility.index.astype(str)) | set(test.utility.index.astype(str))
    available = set(embeddings.index.astype(str))
    missing = sorted(required - available)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f" and {len(missing) - 5} more"
        raise ValueError(f"Missing embedding rows for LLMRouter library adapters: {preview}{suffix}")
