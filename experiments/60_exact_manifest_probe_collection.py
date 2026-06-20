from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir as config_output_dir
from routecode.local_eval.generation_runner import OpenAICompatibleLocalClient, TransformersLocalClient
from routecode.local_eval.probe_runner import DryRunProbeClient, LocalProbeTask, run_aligned_probe_matrix
from routecode.local_eval.task_manifest import tasks_from_manifest
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()
    run(config_path=args.config, output_dir=args.output_dir or None)


def run(*, config_path: str, output_dir: str | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    probe_config = config.get("phase2_exact_manifest_probe", {})
    out_dir = Path(output_dir) if output_dir else config_output_dir(config)
    manifest_path = Path(str(probe_config.get("task_manifest_path", "results/phase2/local_exact_task_manifest.csv")))
    max_queries = int(probe_config.get("max_queries", 200))
    tasks = _probe_tasks_from_manifest(manifest_path, max_queries=max_queries)
    model_ids = [str(model_id) for model_id in probe_config.get("model_ids", ["dry_probe"])]
    generation_params = dict(probe_config.get("generation_params", {"temperature": 0.0, "max_tokens": 64}))
    dry_run = bool(probe_config.get("dry_run", True))
    backend = "dry_run" if dry_run else str(probe_config.get("backend", "openai")).lower()
    model_revision = str(probe_config.get("model_revision", "dry-run" if dry_run else "local-server"))
    client = _build_client(probe_config, dry_run=dry_run, backend=backend, model_ids=model_ids)
    if not dry_run and backend != "transformers":
        model_ids = _resolve_openai_model_ids(model_ids, client)
    features, raw_logs, errors = run_aligned_probe_matrix(
        tasks=tasks,
        model_ids=model_ids,
        client=client,
        generation_params=generation_params,
        model_revision=model_revision,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(
        out_dir=out_dir,
        features=features,
        raw_logs=raw_logs,
        errors=errors,
        metadata={
            "config_path": config_path,
            "task_manifest_path": str(manifest_path),
            "created_at": datetime.now(UTC).isoformat(),
            "dry_run": dry_run,
            "backend": backend,
            "model_ids": model_ids,
            "model_revision": model_revision,
            "generation_params": generation_params,
            "task_count": len(tasks),
            "feature_rows": len(features),
            "error_rows": len(errors),
        },
    )
    write_memo(out_dir, config_path, manifest_path, features, paths, dry_run, backend)
    append_readme(out_dir, config_path, manifest_path, features, paths, dry_run, backend)
    print(f"Wrote exact manifest probe features to {paths['features']}")
    return features


def write_outputs(
    *,
    out_dir: Path,
    features: pd.DataFrame,
    raw_logs: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, str]:
    paths = {
        "features": str(out_dir / "exact_manifest_probe_features.parquet"),
        "raw_outputs": str(out_dir / "exact_manifest_probe_raw_outputs.jsonl"),
        "errors": str(out_dir / "exact_manifest_probe_errors.jsonl"),
        "metadata": str(out_dir / "exact_manifest_probe_run_metadata.json"),
    }
    features.to_parquet(paths["features"], index=False)
    _write_jsonl(Path(paths["raw_outputs"]), raw_logs)
    _write_jsonl(Path(paths["errors"]), errors)
    Path(paths["metadata"]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def write_memo(
    out_dir: Path,
    config_path: str,
    manifest_path: Path,
    features: pd.DataFrame,
    paths: dict[str, str],
    dry_run: bool,
    backend: str,
) -> None:
    lines = [
        "# Phase 2 Exact Manifest Probe Collection",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/60_exact_manifest_probe_collection.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        _evidence_sentence(dry_run, backend),
        "",
        f"Task manifest: `{manifest_path}`.",
        "",
        "Outputs:",
        "",
        "- `exact_manifest_probe_features.parquet`: schema-compatible probe observations.",
        "- `exact_manifest_probe_raw_outputs.jsonl`: raw prompts and model outputs.",
        "- `exact_manifest_probe_errors.jsonl`: generation errors, if any.",
        "- `exact_manifest_probe_run_metadata.json`: config and run metadata.",
        "- `m11_exact_manifest_probe_collection_memo.md`: this memo.",
        "",
        "Summary:",
        "",
        _markdown_table(_summary(features)),
        "",
        "Files:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    (out_dir / "m11_exact_manifest_probe_collection_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    config_path: str,
    manifest_path: Path,
    features: pd.DataFrame,
    paths: dict[str, str],
    dry_run: bool,
    backend: str,
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Exact Manifest Probe Collection"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/60_exact_manifest_probe_collection.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        _evidence_sentence(dry_run, backend),
        "",
        f"Task manifest: `{manifest_path}`.",
        "",
        "Outputs:",
        "",
        "- `exact_manifest_probe_features.parquet`",
        "- `exact_manifest_probe_raw_outputs.jsonl`",
        "- `exact_manifest_probe_errors.jsonl`",
        "- `exact_manifest_probe_run_metadata.json`",
        "- `m11_exact_manifest_probe_collection_memo.md`",
        "",
        _markdown_table(_summary(features)),
        "",
        "Files:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _probe_tasks_from_manifest(manifest_path: Path, *, max_queries: int) -> list[LocalProbeTask]:
    manifest = pd.read_csv(manifest_path)
    eval_tasks = tasks_from_manifest(manifest)
    if max_queries > 0:
        eval_tasks = eval_tasks[:max_queries]
    return [
        LocalProbeTask(
            query_id=task.query_id,
            query_text=task.query_text,
            dataset=task.dataset,
            domain=task.domain,
        )
        for task in eval_tasks
    ]


def _build_client(probe_config: dict[str, Any], *, dry_run: bool, backend: str, model_ids: list[str]):
    if dry_run:
        return DryRunProbeClient()
    if backend == "transformers":
        return TransformersLocalClient(
            model_id_or_path=str(probe_config.get("model_id_or_path", model_ids[0])),
            torch_dtype=str(probe_config.get("torch_dtype", "auto")),
            device_map=str(probe_config.get("device_map", "auto")),
            local_files_only=bool(probe_config.get("local_files_only", True)),
            trust_remote_code=bool(probe_config.get("trust_remote_code", True)),
        )
    return OpenAICompatibleLocalClient(
        base_url=str(probe_config.get("base_url", "http://127.0.0.1:8000/v1")),
        api_key=str(probe_config.get("api_key", "local-routecode")),
        timeout_sec=float(probe_config.get("timeout_sec", 120.0)),
    )


def _resolve_openai_model_ids(model_ids: list[str], client) -> list[str]:
    if "__first_listed__" not in model_ids:
        return model_ids
    listed = list(client.list_models())
    if not listed:
        raise ValueError("model_ids includes __first_listed__, but the local server returned no models")
    return [listed[0] if model_id == "__first_listed__" else model_id for model_id in model_ids]


def _summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    return (
        features.groupby(["probe_type", "probe_model_id"], as_index=False)
        .agg(
            rows=("query_id", "count"),
            unique_queries=("query_id", "nunique"),
            mean_self_confidence=("self_confidence", "mean"),
            mean_entropy_proxy=("entropy_proxy", "mean"),
            mean_probe_cost_proxy=("probe_cost_proxy", "mean"),
            errors=("error_type", lambda values: int(values.fillna("").astype(str).ne("").sum())),
        )
        .sort_values(["probe_type", "probe_model_id"])
    )


def _evidence_sentence(dry_run: bool, backend: str) -> str:
    if dry_run:
        return (
            "This run uses a deterministic dry-run probe client over the exact-task manifest. "
            "It validates manifest-backed probe logging and M4 plumbing; it is not true local-model probe evidence."
        )
    if backend == "transformers":
        return (
            "This run uses a local Hugging Face Transformers backend over the exact-task manifest. "
            "It uses local model weights directly and makes no GPT/Claude/Gemini API calls."
        )
    return (
        "This run uses an OpenAI-compatible local serving endpoint over the exact-task manifest. "
        "It is true local probe evidence only if the metadata records the local model IDs and endpoint used."
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
