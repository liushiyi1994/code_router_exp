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

from routecode.config import load_config
from routecode.local_eval.generation_runner import OpenAICompatibleLocalClient
from routecode.local_eval.probe_runner import DryRunProbeClient, LocalProbeTask, run_aligned_probe_matrix
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


DEFAULT_OUTPUT_DIR = "results/phase2/aligned_local_probes"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--state-targets", default="")
    args = parser.parse_args()
    run(
        config_path=args.config,
        output_dir=args.output_dir,
        state_targets_path=args.state_targets or None,
    )


def run(
    *,
    config_path: str,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    state_targets_path: str | None = None,
) -> pd.DataFrame:
    config = load_config(config_path)
    probe_config = config.get("phase2_aligned_local_probe", {})
    prepared = prepare_from_config(config)
    tasks = _select_tasks(
        query_info=prepared.matrices["test"].query_info,
        state_targets_path=state_targets_path,
        max_queries=int(probe_config.get("max_queries", 50)),
    )
    model_ids = [str(model_id) for model_id in probe_config.get("model_ids", ["dry_probe"])]
    generation_params = dict(probe_config.get("generation_params", {"temperature": 0.0, "max_tokens": 64}))
    dry_run = bool(probe_config.get("dry_run", True))
    model_revision = str(probe_config.get("model_revision", "dry-run" if dry_run else "local-server"))
    client = _build_client(probe_config, dry_run=dry_run)
    features, raw_logs, errors = run_aligned_probe_matrix(
        tasks=tasks,
        model_ids=model_ids,
        client=client,
        generation_params=generation_params,
        model_revision=model_revision,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(
        out_dir=out_dir,
        features=features,
        raw_logs=raw_logs,
        errors=errors,
        metadata={
            "config_path": config_path,
            "state_targets_path": state_targets_path or "",
            "created_at": datetime.now(UTC).isoformat(),
            "dry_run": dry_run,
            "model_ids": model_ids,
            "model_revision": model_revision,
            "generation_params": generation_params,
            "task_count": len(tasks),
            "feature_rows": len(features),
            "error_rows": len(errors),
        },
    )
    write_memo(out_dir, config_path, state_targets_path, features, paths, dry_run)
    append_readme(out_dir, config_path, state_targets_path, features, paths, dry_run)
    print(f"Wrote aligned local probe features to {paths['features']}")
    print(f"Wrote aligned local probe raw logs to {paths['raw_outputs']}")
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
        "features": str(out_dir / "aligned_local_probe_features.parquet"),
        "raw_outputs": str(out_dir / "aligned_local_probe_raw_outputs.jsonl"),
        "errors": str(out_dir / "aligned_local_probe_errors.jsonl"),
        "metadata": str(out_dir / "aligned_local_probe_run_metadata.json"),
    }
    features.to_parquet(paths["features"], index=False)
    _write_jsonl(Path(paths["raw_outputs"]), raw_logs)
    _write_jsonl(Path(paths["errors"]), errors)
    Path(paths["metadata"]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def write_memo(
    out_dir: Path,
    config_path: str,
    state_targets_path: str | None,
    features: pd.DataFrame,
    paths: dict[str, str],
    dry_run: bool,
) -> None:
    lines = [
        "# Phase 2 Aligned Local Probe Collection",
        "",
        "Command:",
        "",
        "```bash",
        _command(config_path, out_dir, state_targets_path),
        "```",
        "",
        _evidence_sentence(dry_run),
        "",
        "Outputs:",
        "",
        "- `aligned_local_probe_features.parquet`: schema-compatible probe observations.",
        "- `aligned_local_probe_raw_outputs.jsonl`: raw prompts and model outputs.",
        "- `aligned_local_probe_errors.jsonl`: generation errors, if any.",
        "- `aligned_local_probe_run_metadata.json`: config and run metadata.",
        "- `m8_aligned_local_probe_collection_memo.md`: this memo.",
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
    (out_dir / "m8_aligned_local_probe_collection_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    config_path: str,
    state_targets_path: str | None,
    features: pd.DataFrame,
    paths: dict[str, str],
    dry_run: bool,
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Aligned Local Probes"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        _command(config_path, out_dir, state_targets_path),
        "```",
        "",
        _evidence_sentence(dry_run),
        "",
        "Outputs:",
        "",
        "- `aligned_local_probe_features.parquet`",
        "- `aligned_local_probe_raw_outputs.jsonl`",
        "- `aligned_local_probe_errors.jsonl`",
        "- `aligned_local_probe_run_metadata.json`",
        "- `m8_aligned_local_probe_collection_memo.md`",
        "",
        _markdown_table(_summary(features)),
        "",
        "Files:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _select_tasks(
    *,
    query_info: pd.DataFrame,
    state_targets_path: str | None,
    max_queries: int,
) -> list[LocalProbeTask]:
    frame = query_info.reset_index().rename(columns={query_info.index.name or "index": "query_id"})
    if state_targets_path:
        targets = pd.read_csv(state_targets_path)
        test_ids = targets[targets.get("split", "test").astype(str).eq("test")]["query_id"].astype(str).tolist()
        frame = frame[frame["query_id"].astype(str).isin(test_ids)]
    if max_queries > 0:
        frame = frame.head(max_queries)
    tasks: list[LocalProbeTask] = []
    for _, row in frame.iterrows():
        tasks.append(
            LocalProbeTask(
                query_id=str(row["query_id"]),
                query_text=str(row.get("query_text", "")),
                dataset=str(row.get("dataset", "unknown")),
                domain=str(row.get("domain", "unknown")),
            )
        )
    return tasks


def _build_client(probe_config: dict[str, Any], *, dry_run: bool):
    if dry_run:
        return DryRunProbeClient()
    return OpenAICompatibleLocalClient(
        base_url=str(probe_config.get("base_url", "http://127.0.0.1:8000/v1")),
        api_key=str(probe_config.get("api_key", "local-routecode")),
        timeout_sec=float(probe_config.get("timeout_sec", 120.0)),
    )


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
            errors=("error_type", lambda values: int((values.astype(str) != "").sum())),
        )
        .sort_values(["probe_type", "probe_model_id"])
    )


def _evidence_sentence(dry_run: bool) -> str:
    if dry_run:
        return (
            "This run uses a deterministic dry-run probe client. It validates aligned query selection, "
            "logging, schema compatibility, and downstream plumbing; it is not true local-model probe evidence."
        )
    return (
        "This run uses an OpenAI-compatible local serving endpoint. It is aligned local probe evidence "
        "only if the metadata records the local model IDs and endpoint used."
    )


def _command(config_path: str, out_dir: Path, state_targets_path: str | None) -> str:
    parts = [
        "python experiments/57_aligned_local_probe_collection.py",
        f"--config {config_path}",
        f"--output-dir {out_dir}",
    ]
    if state_targets_path:
        parts.append(f"--state-targets {state_targets_path}")
    return " ".join(parts)


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
