from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.external_baseline_assets import build_external_baseline_assets, write_external_baseline_assets
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


RUN_DIRNAME = "embedllm_knn_split_aligned"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    baseline_config = config.get("external_baselines", {})
    seed = int(config.get("run", {}).get("random_seed", 0))

    assets = build_external_baseline_assets({"train": train, "test": test}, prepared.embeddings)
    written = write_external_baseline_assets(assets, out_dir)
    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)

    train_csv = pd.read_csv(written.embedllm_train_path)
    test_csv = pd.read_csv(written.embedllm_test_path)
    backend = str(baseline_config.get("embedllm_knn_embedding_backend", "sentence_transformers"))
    embedding_info, train_embeddings, test_embeddings = _embed_prompt_tables(
        train_csv,
        test_csv,
        prepared.embeddings,
        baseline_config,
        backend,
    )

    neighbors = [int(value) for value in baseline_config.get("embedllm_knn_neighbors", [131])]
    table, raw_predictions = _evaluate_neighbors(
        train_csv=train_csv,
        test_csv=test_csv,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
        neighbors=neighbors,
        train_matrices=train,
        test_matrices=test,
        routecode_embeddings=prepared.embeddings,
        config=config,
        seed=seed,
        embedding_info=embedding_info,
    )
    table["split_aligned_with_routecode"] = True
    table["routecode_metric_compatible"] = True
    table["official_upstream_checkpoint"] = False
    table["exact_upstream_command"] = False
    table["baseline_family"] = "embedllm_knn_local_metric_adapter"
    table["implementation_note"] = (
        "Local RouteCode metric adapter using the same per-model kNN correctness idea as the "
        "LLMRouterBench EmbedLLM KNN script, then selecting the model with highest predicted "
        "correctness probability. This is not an upstream published checkpoint."
    )

    table.to_csv(out_dir / "table_embedllm_knn_split_aligned.csv", index=False)
    _write_json(run_dir / "raw_predictions.json", raw_predictions)
    _write_json(run_dir / "embedding_info.json", embedding_info)
    write_memo(out_dir, config_path, run_dir, table, embedding_info)
    append_readme(out_dir, config_path, run_dir, table)
    print(f"Wrote split-aligned EmbedLLM KNN outputs to {run_dir}")


def _embed_prompt_tables(
    train_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    routecode_embeddings: pd.DataFrame,
    baseline_config: dict[str, Any],
    backend: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if backend == "routecode_embeddings":
        train_embeddings = _routecode_prompt_embeddings(train_csv, routecode_embeddings)
        test_embeddings = _routecode_prompt_embeddings(test_csv, routecode_embeddings)
        return (
            {
                "embedding_backend": backend,
                "embedding_model": "routecode_prepared_embeddings",
                "embedding_dim": int(train_embeddings.shape[1]),
                "local_files_only": True,
            },
            train_embeddings,
            test_embeddings,
        )
    if backend != "sentence_transformers":
        raise ValueError(f"Unknown EmbedLLM KNN embedding backend: {backend}")

    from sentence_transformers import SentenceTransformer

    model_id = str(baseline_config.get("embedllm_knn_sentence_transformer", "all-mpnet-base-v2"))
    cache_folder = baseline_config.get("embedllm_knn_cache_folder")
    local_files_only = bool(baseline_config.get("embedllm_knn_local_files_only", True))
    batch_size = int(baseline_config.get("embedllm_knn_batch_size", 32))
    device = baseline_config.get("embedllm_knn_device")
    embedder = SentenceTransformer(
        model_id,
        cache_folder=str(Path(cache_folder).expanduser()) if cache_folder else None,
        local_files_only=local_files_only,
        device=str(device) if device else None,
    )
    train_embeddings = _sentence_transformer_prompt_embeddings(train_csv, embedder, batch_size=batch_size)
    test_embeddings = _sentence_transformer_prompt_embeddings(test_csv, embedder, batch_size=batch_size)
    return (
        {
            "embedding_backend": backend,
            "embedding_model": model_id,
            "embedding_dim": int(train_embeddings.shape[1]),
            "local_files_only": local_files_only,
        },
        train_embeddings,
        test_embeddings,
    )


def _routecode_prompt_embeddings(csv: pd.DataFrame, routecode_embeddings: pd.DataFrame) -> pd.DataFrame:
    prompts = _unique_prompts(csv)
    rows = []
    for row in prompts.itertuples(index=False):
        rows.append(routecode_embeddings.loc[str(row.query_id)].to_numpy(dtype=float))
    return pd.DataFrame(rows, index=prompts["prompt_id"].astype(int), dtype=float)


def _sentence_transformer_prompt_embeddings(csv: pd.DataFrame, embedder: Any, *, batch_size: int) -> pd.DataFrame:
    prompts = _unique_prompts(csv)
    vectors = embedder.encode(
        prompts["prompt"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return pd.DataFrame(np.asarray(vectors, dtype=float), index=prompts["prompt_id"].astype(int))


def _unique_prompts(csv: pd.DataFrame) -> pd.DataFrame:
    prompts = (
        csv[["prompt_id", "query_id", "prompt"]]
        .drop_duplicates("prompt_id")
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    if prompts["prompt_id"].duplicated().any():
        raise ValueError("EmbedLLM KNN input contains duplicate prompt ids after deduplication")
    return prompts


def _evaluate_neighbors(
    *,
    train_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    train_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
    neighbors: list[int],
    train_matrices,
    test_matrices,
    routecode_embeddings: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    embedding_info: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_single = BestSingleRouter().fit(train_matrices.query_info, train_matrices.utility).predict(test_matrices.query_info)
    baseline_mean = float(selected_values(test_matrices.utility, best_single).mean())
    oracle_mean = float(test_matrices.utility.max(axis=1).mean())
    knn = KNNRouter(int(config.get("routers", {}).get("knn_k", 15))).fit(
        train_matrices.query_info,
        train_matrices.utility,
        routecode_embeddings,
    ).predict(test_matrices.query_info, routecode_embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test_matrices.utility, knn).mean()))
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))

    model_names = _model_order(train_csv)
    train_mean_utility = train_csv.groupby("model_name")["utility"].mean().to_dict()
    train_label_rates = train_csv.groupby("model_name")["label"].mean().to_dict()
    rows = []
    raw: dict[str, Any] = {
        "embedding_info": embedding_info,
        "neighbors": neighbors,
        "model_names": model_names,
        "predictions": {},
    }
    for index, requested_k in enumerate(neighbors):
        effective_k = max(1, min(int(requested_k), int(train_embeddings.shape[0])))
        model_probabilities: dict[str, pd.Series] = {}
        model_predictions: dict[str, pd.Series] = {}
        model_accuracies: dict[str, float] = {}
        for model_name in model_names:
            train_rows = _model_rows(train_csv, model_name)
            test_rows = _model_rows(test_csv, model_name)
            classifier = KNeighborsClassifier(n_neighbors=effective_k)
            classifier.fit(train_embeddings.loc[train_rows["prompt_id"].astype(int)].to_numpy(), train_rows["label"].astype(int))
            test_x = test_embeddings.loc[test_rows["prompt_id"].astype(int)].to_numpy()
            pred = pd.Series(classifier.predict(test_x), index=test_rows["query_id"].astype(str), name=model_name)
            proba = _positive_probability(classifier, test_x)
            probability = pd.Series(proba, index=test_rows["query_id"].astype(str), name=model_name)
            model_predictions[model_name] = pred
            model_probabilities[model_name] = probability
            model_accuracies[model_name] = float((pred.to_numpy() == test_rows["label"].astype(int).to_numpy()).mean())

        probability_frame = pd.DataFrame(model_probabilities).reindex(test_matrices.utility.index)
        selected = _select_models(probability_frame, train_mean_utility, train_label_rates)
        eval_row = evaluate_selection(
            method=f"embedllm_knn_split_aligned_k{requested_k:g}",
            selected_models=selected,
            matrices=test_matrices,
            baseline_mean=baseline_mean,
            learned_reference_mean=learned_reference_mean,
            oracle_mean=oracle_mean,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed + 200 + index,
            k=len(model_names),
            labels=selected,
        )
        selected_probs = [
            float(probability_frame.at[query_id, selected_model]) for query_id, selected_model in selected.items()
        ]
        eval_row.update(
            {
                "requested_neighbors": int(requested_k),
                "effective_neighbors": int(effective_k),
                "embedding_backend": embedding_info["embedding_backend"],
                "embedding_model": embedding_info["embedding_model"],
                "mean_correctness_accuracy": float(np.mean(list(model_accuracies.values()))),
                "mean_selected_correctness_probability": float(np.mean(selected_probs)),
            }
        )
        rows.append(eval_row)
        raw["predictions"][str(requested_k)] = {
            "effective_neighbors": int(effective_k),
            "model_correctness_accuracy": model_accuracies,
            "selected_models": selected.astype(str).to_dict(),
            "selected_correctness_probability": {
                str(query_id): float(probability_frame.at[query_id, model]) for query_id, model in selected.items()
            },
        }
    return pd.DataFrame(rows), raw


def _model_order(csv: pd.DataFrame) -> list[str]:
    ordered = csv[["model_id", "model_name"]].drop_duplicates().sort_values("model_id")
    return ordered["model_name"].astype(str).tolist()


def _model_rows(csv: pd.DataFrame, model_name: str) -> pd.DataFrame:
    return csv[csv["model_name"].astype(str) == str(model_name)].sort_values("prompt_id").reset_index(drop=True)


def _positive_probability(classifier: KNeighborsClassifier, x: np.ndarray) -> np.ndarray:
    probabilities = classifier.predict_proba(x)
    classes = list(classifier.classes_)
    if 1 in classes:
        return np.asarray(probabilities[:, classes.index(1)], dtype=float)
    return np.zeros(x.shape[0], dtype=float)


def _select_models(
    probability_frame: pd.DataFrame,
    train_mean_utility: dict[str, float],
    train_label_rates: dict[str, float],
) -> pd.Series:
    selected = {}
    utility_tiebreak = pd.Series({model: float(train_mean_utility.get(model, 0.0)) for model in probability_frame.columns})
    label_tiebreak = pd.Series({model: float(train_label_rates.get(model, 0.0)) for model in probability_frame.columns})
    for query_id, probabilities in probability_frame.iterrows():
        ranking = pd.DataFrame(
            {
                "probability": probabilities.astype(float),
                "train_label_rate": label_tiebreak,
                "train_mean_utility": utility_tiebreak,
                "model_name": probability_frame.columns.astype(str),
            }
        ).sort_values(
            ["probability", "train_label_rate", "train_mean_utility", "model_name"],
            ascending=[False, False, False, True],
        )
        selected[str(query_id)] = str(ranking.index[0])
    return pd.Series(selected, name="selected_model")


def append_readme(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## EmbedLLM KNN Split-Aligned Evaluation"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/29_embedllm_knn_split_aligned.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        f"- `{RUN_DIRNAME}/raw_predictions.json`: per-neighbor selected-model and probability evidence.",
        f"- `{RUN_DIRNAME}/embedding_info.json`: embedding backend details.",
        "- `table_embedllm_knn_split_aligned.csv`: RouteCode utility metrics for the local metric adapter.",
        "- `phase_e_embedllm_knn_split_aligned_memo.md`: Phase E memo and caveats.",
        "",
        f"Artifact directory: `{run_dir}`.",
        "",
        "This is a local metric adapter around the same per-model kNN correctness idea used by EmbedLLM KNN; it is not an upstream published checkpoint.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame, embedding_info: dict[str, Any]) -> None:
    lines = [
        "# Phase E EmbedLLM KNN Split-Aligned Memo",
        "",
        f"Command: `python experiments/29_embedllm_knn_split_aligned.py --config {config_path}`",
        "",
        f"Artifact directory: `{run_dir}`.",
        "",
        "This run evaluates a RouteCode-compatible local metric adapter for EmbedLLM KNN.",
        "",
        "It uses the same per-model kNN correctness idea as the LLMRouterBench EmbedLLM KNN script, then turns the per-model correctness probabilities into a selected model by choosing the largest probability with deterministic train-set tie breaks.",
        "",
        "It is split-aligned with RouteCode and does not make external API calls. It is not an upstream published EmbedLLM checkpoint or the full EmbedLLM MF method.",
        "",
        "## Embeddings",
        "",
        f"- Backend: `{embedding_info['embedding_backend']}`",
        f"- Model: `{embedding_info['embedding_model']}`",
        f"- Dimension: `{embedding_info['embedding_dim']}`",
        "",
        "## Evaluation Summary",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "phase_e_embedllm_knn_split_aligned_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{float(value):.4f}")
    headers = [str(column) for column in display.columns]
    rows = display.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
