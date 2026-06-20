from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.local_eval.generation_runner import (
    DryRunLocalClient,
    OpenAICompatibleLocalClient,
    TransformersLocalClient,
    run_generation_matrix,
)
from routecode.local_eval.task_manifest import tasks_from_manifest
from routecode.local_eval.tasks import load_smoke_tasks
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> pd.DataFrame:
    config = load_config(config_path)
    out_dir = output_dir(config)
    local_config = config.get("phase2_local_eval", {})
    dry_run = bool(local_config.get("dry_run", True))
    backend = str(local_config.get("backend", "dry_run" if dry_run else "openai")).lower()
    model_ids = [str(model_id) for model_id in local_config.get("model_ids", ["dry_run_model"])]
    model_revision = str(local_config.get("model_revision", "dry-run" if dry_run else "unknown"))
    generation_params = dict(local_config.get("generation_params", {"temperature": 0.0, "max_tokens": 128}))
    datasets = [str(dataset) for dataset in local_config.get("datasets", ["gsm8k_smoke", "mmlu_smoke"])]
    max_queries = int(local_config.get("max_queries", 20))
    task_manifest_path = str(local_config.get("task_manifest_path", "") or "")
    tasks = _load_tasks(
        task_manifest_path=task_manifest_path,
        datasets=datasets,
        max_queries=max_queries,
    )
    if not tasks:
        raise ValueError("No Phase 2 local-eval tasks selected")

    endpoint_specs = _endpoint_specs(local_config)
    if endpoint_specs:
        backend = "openai_multi_endpoint"
        frame, raw_logs, errors, model_ids, model_revisions, endpoint_metadata = _run_openai_endpoint_specs(
            tasks=tasks,
            endpoint_specs=endpoint_specs,
            default_generation_params=generation_params,
            default_api_key=str(local_config.get("api_key", "local-routecode")),
            default_timeout_sec=float(local_config.get("timeout_sec", 120.0)),
        )
        model_revision = "; ".join(model_revisions)
    else:
        endpoint_metadata = []
        if backend == "dry_run" or dry_run:
            client = DryRunLocalClient()
            backend = "dry_run"
        elif backend == "transformers":
            client = TransformersLocalClient(
                model_id_or_path=str(local_config.get("model_id_or_path", model_ids[0])),
                torch_dtype=str(local_config.get("torch_dtype", "auto")),
                device_map=str(local_config.get("device_map", "auto")),
                local_files_only=bool(local_config.get("local_files_only", True)),
                trust_remote_code=bool(local_config.get("trust_remote_code", True)),
            )
        else:
            client = OpenAICompatibleLocalClient(
                base_url=str(local_config.get("base_url", "http://localhost:8000/v1")),
                api_key=str(local_config.get("api_key", "local-routecode")),
                timeout_sec=float(local_config.get("timeout_sec", 120.0)),
            )
            model_ids = _resolve_openai_model_ids(model_ids, client)

        frame, raw_logs, errors = run_generation_matrix(
            tasks=tasks,
            model_ids=model_ids,
            client=client,
            generation_params=generation_params,
            model_revision=model_revision,
        )
    outcomes_path = out_dir / "local_model_outcomes.parquet"
    raw_path = out_dir / "local_model_raw_outputs.jsonl"
    errors_path = out_dir / "local_model_errors.jsonl"
    metadata_path = out_dir / "local_model_run_metadata.json"
    frame.to_parquet(outcomes_path, index=False)
    _write_jsonl(raw_path, raw_logs)
    _write_jsonl(errors_path, errors)
    metadata = _metadata(
        config_path=config_path,
        out_dir=out_dir,
        outcomes_path=outcomes_path,
        dry_run=dry_run,
        backend=backend,
        model_ids=model_ids,
        model_revision=model_revision,
        generation_params=generation_params,
        endpoint_metadata=endpoint_metadata,
        task_manifest_path=task_manifest_path,
        task_count=len(tasks),
        row_count=len(frame),
        error_count=len(errors),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_memo(out_dir, config_path, frame, metadata)
    append_readme(out_dir, config_path, frame, metadata)
    print(f"Wrote Phase 2 local model outcomes to {outcomes_path}")
    return frame


def _load_tasks(*, task_manifest_path: str, datasets: list[str], max_queries: int) -> list:
    if task_manifest_path:
        manifest = pd.read_csv(task_manifest_path)
        tasks = tasks_from_manifest(manifest)
        if max_queries > 0:
            return tasks[:max_queries]
        return tasks
    return load_smoke_tasks(datasets=datasets, max_queries=max_queries)


def _resolve_openai_model_ids(model_ids: list[str], client) -> list[str]:
    if "__first_listed__" not in model_ids:
        return model_ids
    listed = list(client.list_models())
    if not listed:
        raise ValueError("model_ids includes __first_listed__, but the local server returned no models")
    return [listed[0] if model_id == "__first_listed__" else model_id for model_id in model_ids]


def _endpoint_specs(local_config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = local_config.get("openai_endpoints", [])
    if specs is None:
        return []
    if not isinstance(specs, list):
        raise ValueError("phase2_local_eval.openai_endpoints must be a list")
    return [dict(spec) for spec in specs]


def _run_openai_endpoint_specs(
    *,
    tasks: list,
    endpoint_specs: list[dict[str, Any]],
    default_generation_params: dict[str, Any],
    default_api_key: str,
    default_timeout_sec: float,
) -> tuple[pd.DataFrame, list[dict], list[dict], list[str], list[str], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    raw_logs: list[dict] = []
    errors: list[dict] = []
    resolved_model_ids: list[str] = []
    model_revisions: list[str] = []
    endpoint_metadata: list[dict[str, Any]] = []
    for index, spec in enumerate(endpoint_specs):
        base_url = str(spec.get("base_url", "")).strip()
        if not base_url:
            raise ValueError(f"openai_endpoints[{index}] is missing base_url")
        client = OpenAICompatibleLocalClient(
            base_url=base_url,
            api_key=str(spec.get("api_key", default_api_key)),
            timeout_sec=float(spec.get("timeout_sec", default_timeout_sec)),
        )
        model_ids = [str(model_id) for model_id in spec.get("model_ids", ["__first_listed__"])]
        model_ids = _resolve_openai_model_ids(model_ids, client)
        revision = str(spec.get("model_revision", spec.get("name", base_url)))
        params = dict(default_generation_params)
        params.update(dict(spec.get("generation_params", {})))
        frame, endpoint_raw_logs, endpoint_errors = run_generation_matrix(
            tasks=tasks,
            model_ids=model_ids,
            client=client,
            generation_params=params,
            model_revision=revision,
        )
        frames.append(frame)
        raw_logs.extend(endpoint_raw_logs)
        errors.extend(endpoint_errors)
        resolved_model_ids.extend(model_ids)
        model_revisions.append(revision)
        endpoint_metadata.append(
            {
                "name": str(spec.get("name", f"endpoint_{index}")),
                "base_url": base_url,
                "model_ids": model_ids,
                "model_revision": revision,
                "generation_params": params,
            }
        )
    if not frames:
        raise ValueError("No OpenAI-compatible endpoint specs configured")
    return (
        pd.concat(frames, ignore_index=True),
        raw_logs,
        errors,
        resolved_model_ids,
        model_revisions,
        endpoint_metadata,
    )


def write_memo(out_dir: Path, config_path: str, frame: pd.DataFrame, metadata: dict) -> None:
    lines = [
        "# Phase 2 True Local Model Generation Matrix",
        "",
        f"Command: `python experiments/51_true_model_generation_matrix.py --config {config_path}`",
        "",
        _mode_sentence(metadata),
        "",
        "Outputs:",
        "",
        "- `local_model_outcomes.parquet`: exact-scored local outcome rows.",
        "- `local_model_raw_outputs.jsonl`: prompt/output logs for every attempted generation.",
        "- `local_model_errors.jsonl`: error rows, if any.",
        "- `local_model_run_metadata.json`: command, git SHA, checksum, model IDs, generation parameters.",
        "",
        "Summary:",
        "",
        _markdown_table(_summary(frame)),
        "",
    ]
    (out_dir / "m2_local_model_generation_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, frame: pd.DataFrame, metadata: dict) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Local Model Outcomes"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/51_true_model_generation_matrix.py --config {config_path}",
        "```",
        "",
        _mode_sentence(metadata),
        "",
        "Outputs:",
        "",
        "- `local_model_outcomes.parquet`",
        "- `local_model_raw_outputs.jsonl`",
        "- `local_model_errors.jsonl`",
        "- `local_model_run_metadata.json`",
        "- `m2_local_model_generation_memo.md`",
        "",
        _markdown_table(_summary(frame)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _metadata(
    *,
    config_path: str,
    out_dir: Path,
    outcomes_path: Path,
    dry_run: bool,
    backend: str,
    model_ids: list[str],
    model_revision: str,
    generation_params: dict,
    endpoint_metadata: list[dict[str, Any]],
    task_manifest_path: str,
    task_count: int,
    row_count: int,
    error_count: int,
) -> dict:
    return {
        "command": f"python experiments/51_true_model_generation_matrix.py --config {config_path}",
        "config_path": str(config_path),
        "output_dir": str(out_dir),
        "dry_run": dry_run,
        "backend": backend,
        "model_ids": model_ids,
        "model_revision": model_revision,
        "generation_params": generation_params,
        "openai_endpoints": endpoint_metadata or [],
        "task_manifest_path": task_manifest_path,
        "task_count": int(task_count),
        "row_count": int(row_count),
        "error_count": int(error_count),
        "git_commit": _git_commit(),
        "outcomes_sha256": _sha256(outcomes_path),
    }


def _mode_sentence(metadata: dict) -> str:
    if metadata.get("dry_run"):
        return (
            "Mode: `dry_run`. This validates local-eval logging, parsing, scoring, and parquet output; "
            "it is not true model-performance evidence."
        )
    if metadata.get("backend") == "transformers":
        return (
            "Mode: local Hugging Face Transformers backend. This uses local model weights directly and makes no GPT/Claude/Gemini API calls."
        )
    if metadata.get("backend") == "openai_multi_endpoint":
        return (
            "Mode: multiple local OpenAI-compatible servers. This is intended for 2--4 vLLM endpoints, "
            "typically one base model per server, and makes no GPT/Claude/Gemini API calls."
        )
    return (
        "Mode: local OpenAI-compatible server. This expects a local vLLM/llama.cpp/SGLang endpoint and makes no GPT/Claude/Gemini API calls."
    )


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.groupby(["dataset", "model_id"], as_index=False)
        .agg(
            rows=("query_id", "count"),
            mean_quality=("quality", "mean"),
            mean_latency_sec=("latency_sec", "mean"),
            mean_tokens_output=("tokens_output", "mean"),
            errors=("error_type", lambda values: int((values.astype(str) != "").sum())),
        )
        .sort_values(["dataset", "model_id"])
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


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
