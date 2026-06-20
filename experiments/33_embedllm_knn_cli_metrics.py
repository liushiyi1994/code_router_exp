from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from routecode.config import load_config, output_dir
from routecode.eval.external_baseline_assets import build_external_baseline_assets, write_external_baseline_assets
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


RUN_DIRNAME = "embedllm_knn_cli_metrics"
EMBEDLLM_ROOT = ROOT / "data/raw/external/LLMRouterBench/baselines/EmbedLLM"
EMBEDLLM_KNN_SCRIPT = EMBEDLLM_ROOT / "algorithm/knn.py"


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
    np.random.seed(seed)

    assets = build_external_baseline_assets({"train": train, "test": test}, prepared.embeddings)
    written = write_external_baseline_assets(assets, out_dir)
    asset_dir = written.asset_dir / "embedllm_assets"
    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)

    train_csv = pd.read_csv(written.embedllm_train_path)
    test_csv = pd.read_csv(written.embedllm_test_path)
    backend = str(
        baseline_config.get(
            "embedllm_knn_cli_embedding_backend",
            baseline_config.get("embedllm_knn_embedding_backend", "sentence_transformers"),
        )
    )
    embedding_info, train_embeddings, test_embeddings = _embed_prompt_tables(
        train_csv,
        test_csv,
        prepared.embeddings,
        baseline_config,
        backend,
    )
    tensor_paths = _write_knn_tensors(asset_dir, train_csv, test_csv, train_embeddings, test_embeddings)
    neighbors = [int(value) for value in baseline_config.get("embedllm_knn_neighbors", [131])]
    rows = []
    for requested_k in neighbors:
        effective_k = max(1, min(int(requested_k), int(tensor_paths["train_y_shape"][1])))
        log_path = run_dir / f"embedllm_knn_k{requested_k:g}_stdout.log"
        command = _command(tensor_paths, effective_k)
        completed = subprocess.run(
            command,
            cwd=EMBEDLLM_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        log_text = completed.stdout + completed.stderr
        log_path.write_text(log_text, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"EmbedLLM KNN command failed for k={requested_k}; see {log_path}")
        mean_accuracy = _parse_mean_accuracy(log_text, effective_k)
        rows.append(
            {
                "method": f"embedllm_knn_cli_k{requested_k:g}",
                "requested_neighbors": int(requested_k),
                "effective_neighbors": int(effective_k),
                "mean_correctness_accuracy": float(mean_accuracy),
                "train_prompts": int(tensor_paths["train_y_shape"][1]),
                "test_prompts": int(tensor_paths["test_y_shape"][1]),
                "model_count": int(tensor_paths["train_y_shape"][0]),
                "embedding_backend": embedding_info["embedding_backend"],
                "embedding_model": embedding_info["embedding_model"],
                "embedding_dim": int(embedding_info["embedding_dim"]),
                "split_aligned_with_routecode": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "official_upstream_checkpoint": False,
                "baseline_family": "embedllm_knn_exact_cli_correctness",
                "execution_evidence": str(log_path),
                "implementation_note": (
                    "Exact upstream EmbedLLM KNN command on RouteCode split-aligned tensor assets. "
                    "The upstream command reports correctness-prediction accuracy, not routing utility."
                ),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_embedllm_knn_cli_metrics.csv", index=False)
    _write_json(
        run_dir / "run_config.json",
        {
            "config_path": config_path,
            "embedding_info": embedding_info,
            "tensor_paths": {key: str(value) for key, value in tensor_paths.items() if isinstance(value, Path)},
            "neighbors": neighbors,
        },
    )
    write_memo(out_dir, config_path, table, tensor_paths)
    append_readme(out_dir, config_path, table)
    print(f"Wrote EmbedLLM KNN CLI metric outputs to {out_dir}")


def _embed_prompt_tables(
    train_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    routecode_embeddings: pd.DataFrame,
    baseline_config: dict[str, Any],
    backend: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if backend == "routecode_embeddings":
        return _routecode_embedding_tables(train_csv, test_csv, routecode_embeddings)
    if backend != "sentence_transformers":
        raise ValueError(f"Unknown EmbedLLM KNN CLI embedding backend: {backend}")

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
    train_embeddings = _sentence_transformer_embeddings(train_csv, embedder, batch_size=batch_size)
    test_embeddings = _sentence_transformer_embeddings(test_csv, embedder, batch_size=batch_size)
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


def _routecode_embedding_tables(
    train_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    routecode_embeddings: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_embeddings = _routecode_prompt_embeddings(train_csv, routecode_embeddings)
    test_embeddings = _routecode_prompt_embeddings(test_csv, routecode_embeddings)
    return (
        {
            "embedding_backend": "routecode_embeddings",
            "embedding_model": "routecode_prepared_embeddings",
            "embedding_dim": int(train_embeddings.shape[1]),
            "local_files_only": True,
        },
        train_embeddings,
        test_embeddings,
    )


def _routecode_prompt_embeddings(csv: pd.DataFrame, routecode_embeddings: pd.DataFrame) -> pd.DataFrame:
    prompts = _unique_prompts(csv)
    vectors = [routecode_embeddings.loc[str(row.query_id)].to_numpy(dtype=np.float32) for row in prompts.itertuples()]
    return pd.DataFrame(np.asarray(vectors, dtype=np.float32), index=prompts["prompt_id"].astype(int))


def _sentence_transformer_embeddings(csv: pd.DataFrame, embedder: Any, *, batch_size: int) -> pd.DataFrame:
    prompts = _unique_prompts(csv)
    vectors = embedder.encode(
        prompts["prompt"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return pd.DataFrame(np.asarray(vectors, dtype=np.float32), index=prompts["prompt_id"].astype(int))


def _unique_prompts(csv: pd.DataFrame) -> pd.DataFrame:
    return (
        csv[["prompt_id", "query_id", "prompt"]]
        .drop_duplicates("prompt_id")
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )


def _write_knn_tensors(
    asset_dir: Path,
    train_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    train_embeddings: pd.DataFrame,
    test_embeddings: pd.DataFrame,
) -> dict[str, Any]:
    train_x, train_y = _tensor_pair(train_csv, train_embeddings)
    test_x, test_y = _tensor_pair(test_csv, test_embeddings)
    paths = {
        "train_x": asset_dir / "knn_train_x.pth",
        "train_y": asset_dir / "knn_train_y.pth",
        "test_x": asset_dir / "knn_test_x.pth",
        "test_y": asset_dir / "knn_test_y.pth",
        "train_x_shape": tuple(train_x.shape),
        "train_y_shape": tuple(train_y.shape),
        "test_x_shape": tuple(test_x.shape),
        "test_y_shape": tuple(test_y.shape),
    }
    torch.save(train_x, paths["train_x"])
    torch.save(train_y, paths["train_y"])
    torch.save(test_x, paths["test_x"])
    torch.save(test_y, paths["test_y"])
    return paths


def _tensor_pair(csv: pd.DataFrame, embeddings: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
    labels = (
        csv.groupby(["model_id", "prompt_id"])["label"]
        .max()
        .reset_index()
        .pivot(index="model_id", columns="prompt_id", values="label")
        .fillna(0)
        .astype(int)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    prompt_ids = labels.columns.astype(int).tolist()
    question_embeddings = embeddings.loc[prompt_ids].to_numpy(dtype=np.float32)
    x = np.stack([question_embeddings] * labels.shape[0], axis=0)
    y = labels.to_numpy(dtype=np.int64)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def _command(tensor_paths: dict[str, Any], neighbors: int) -> list[str]:
    return [
        sys.executable,
        "algorithm/knn.py",
        "--input-format",
        "tensor",
        "--train-x-path",
        str(tensor_paths["train_x"].resolve()),
        "--train-y-path",
        str(tensor_paths["train_y"].resolve()),
        "--test-x-path",
        str(tensor_paths["test_x"].resolve()),
        "--test-y-path",
        str(tensor_paths["test_y"].resolve()),
        "--num-neighbors",
        str(neighbors),
    ]


def _parse_mean_accuracy(log_text: str, neighbors: int) -> float:
    pattern = rf"Mean Test Accuracy for {int(neighbors)} neighbors:\s*([0-9.]+)"
    match = re.search(pattern, log_text)
    if not match:
        raise ValueError(f"Could not parse EmbedLLM KNN mean accuracy for k={neighbors}")
    return float(match.group(1))


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, tensor_paths: dict[str, Any]) -> None:
    lines = [
        "# Phase E EmbedLLM KNN CLI Metrics Memo",
        "",
        f"Command: `python experiments/33_embedllm_knn_cli_metrics.py --config {config_path}`",
        "",
        "This runs the exact upstream EmbedLLM KNN command on RouteCode split-aligned tensor assets. The upstream metric is correctness-prediction accuracy, not RouteCode routing utility; RouteCode utility remains tracked in `table_embedllm_knn_split_aligned.csv`.",
        "",
        "Tensor assets:",
        "",
        f"- `{tensor_paths['train_x']}`",
        f"- `{tensor_paths['train_y']}`",
        f"- `{tensor_paths['test_x']}`",
        f"- `{tensor_paths['test_y']}`",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "phase_e_embedllm_knn_cli_metrics_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## EmbedLLM KNN CLI Metrics"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/33_embedllm_knn_cli_metrics.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_embedllm_knn_cli_metrics.csv`: exact upstream EmbedLLM KNN correctness metrics on split-aligned tensor inputs.",
        "- `phase_e_embedllm_knn_cli_metrics_memo.md`: compatibility notes and tensor asset paths.",
        f"- `{RUN_DIRNAME}/embedllm_knn_k*_stdout.log`: exact command logs.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_correctness_accuracy",
                    "train_prompts",
                    "test_prompts",
                    "exact_upstream_command",
                ]
            ]
            if not table.empty
            else table
        ),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    columns = list(table.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
