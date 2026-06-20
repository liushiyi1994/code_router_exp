from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.external_baselines import build_avengerspro_records
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


ASSET_DIRNAME = "avengerspro_split_aligned"
RUN_DIRNAME = "avengerspro_cli_metrics"
LLMROUTERBENCH_ROOT = ROOT / "data/raw/external/LLMRouterBench"


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

    asset_dir = out_dir / ASSET_DIRNAME
    run_dir = out_dir / RUN_DIRNAME
    asset_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    assets = build_avengerspro_records({"train": train, "test": test})
    train_path = asset_dir / "train.jsonl"
    test_path = asset_dir / "test.jsonl"
    baseline_scores_path = asset_dir / "baseline_scores.json"
    _write_jsonl(train_path, assets.train_records)
    _write_jsonl(test_path, assets.test_records)
    _write_json(baseline_scores_path, assets.baseline_scores)

    cache_path = run_dir / "full_embedding_cache.jsonl"
    _write_jsonl(cache_path, _embedding_cache_rows(assets.train_records + assets.test_records, prepared.embeddings))

    clusters = _cluster_values(config, baseline_config)
    top_k = int(baseline_config.get("avengerspro_top_k", 1))
    beta = float(baseline_config.get("avengerspro_beta", 9.0))
    performance_weight = float(baseline_config.get("avengerspro_performance_weight", 0.7))
    cost_sensitivity = float(baseline_config.get("avengerspro_cost_sensitivity", 0.3))

    rows = []
    for n_clusters in clusters:
        effective_clusters = max(1, min(int(n_clusters), len(assets.train_records)))
        output_path = run_dir / "simple_cluster_full_results.json"
        stdout_path = run_dir / "avengerspro_simple_cluster_stdout.log"
        config_path_full = run_dir / "simple_cluster_config.full.json"
        command_config = {
            "train_data_path": str(train_path.resolve()),
            "test_data_path": str(test_path.resolve()),
            "baseline_scores_path": str(baseline_scores_path.resolve()),
            "embedding_cache_path": str(cache_path.resolve()),
            "embedding_api_key": "",
            "embedding_base_url": "",
            "embedding_model": "routecode-cache",
            "n_clusters": effective_clusters,
            "max_router": 1,
            "top_k": max(1, min(top_k, effective_clusters)),
            "beta": beta,
            "seed": seed,
            "max_workers": 1,
            "cluster_batch_size": int(baseline_config.get("avengerspro_cluster_batch_size", 1000)),
            "performance_weight": performance_weight,
            "cost_sensitivity": cost_sensitivity,
        }
        _write_json(config_path_full, command_config)
        command = _command(config_path_full, output_path)
        _run_upstream_command(command, LLMROUTERBENCH_ROOT, stdout_path)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        parsed = _parse_results(payload)
        rows.append(
            {
                "method": f"avengerspro_cli_simple_cluster_k{int(n_clusters)}",
                "requested_clusters": int(n_clusters),
                "effective_clusters": int(effective_clusters),
                "top_k": int(command_config["top_k"]),
                "beta": beta,
                "dataset_level_accuracy": parsed["dataset_level_accuracy"],
                "sample_level_accuracy": parsed["sample_level_accuracy"],
                "correct_routes": parsed["correct_routes"],
                "total_queries": parsed["total_queries"],
                "total_cost": parsed["total_cost"],
                "avg_cost_per_query": parsed["avg_cost_per_query"],
                "model_selection_stats_json": json.dumps(parsed["model_selection_stats"], sort_keys=True),
                "train_queries": int(len(train.query_info)),
                "test_queries": int(len(test.query_info)),
                "split_aligned_with_routecode": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "official_upstream_checkpoint": False,
                "baseline_family": "avengerspro_exact_cli_simple_cluster_accuracy",
                "execution_evidence": str(stdout_path),
                "result_json": str(output_path),
                "implementation_note": (
                    "Exact upstream Avengers-Pro simple-cluster command on RouteCode split-aligned assets "
                    "with a local embedding cache. The upstream command reports accuracy/cost, not RouteCode utility."
                ),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_avengerspro_cli_metrics.csv", index=False)
    _write_json(
        run_dir / "run_config.json",
        {
            "config_path": config_path,
            "clusters": clusters,
            "train_path": str(train_path),
            "test_path": str(test_path),
            "baseline_scores_path": str(baseline_scores_path),
            "embedding_cache_path": str(cache_path),
        },
    )
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote Avengers-Pro CLI metric outputs to {out_dir}")


def _cluster_values(config: dict[str, Any], baseline_config: dict[str, Any]) -> list[int]:
    configured = baseline_config.get("avengerspro_clusters")
    if configured is None:
        configured = [int(config.get("routers", {}).get("embedding_clusters", 16))]
    if isinstance(configured, int):
        values = [configured]
    else:
        values = [int(value) for value in configured]
    return sorted({max(1, value) for value in values})


def _command(config_path: Path, output_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "baselines.AvengersPro.simple_cluster_router",
        "--config",
        str(config_path.resolve()),
        "--output",
        str(output_path.resolve()),
    ]


def _run_upstream_command(command: list[str], cwd: Path, stdout_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    completed = subprocess.run(command, cwd=cwd, env=env, check=False, capture_output=True, text=True)
    stdout_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Avengers-Pro simple-cluster command failed; see {stdout_path}")
    return int(completed.returncode)


def _parse_results(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {})
    cost = results.get("cost_analysis", {})
    return {
        "dataset_level_accuracy": float(results.get("accuracy", 0.0)) / 100.0,
        "sample_level_accuracy": float(results.get("all_sample_avg", 0.0)) / 100.0,
        "correct_routes": float(results.get("correct_routes", 0.0)),
        "total_queries": int(results.get("total_queries", 0)),
        "total_cost": float(cost.get("total_cost", 0.0)),
        "avg_cost_per_query": float(cost.get("avg_cost_per_query", 0.0)),
        "model_selection_stats": dict(results.get("model_selection_stats", {})),
    }


def _embedding_cache_rows(records: list[dict[str, Any]], embeddings: pd.DataFrame) -> list[dict[str, Any]]:
    rows_by_query: dict[str, dict[str, Any]] = {}
    for record in records:
        query = str(record["query"])
        if query in rows_by_query:
            continue
        query_id = str(record["query_id"])
        if query_id not in embeddings.index:
            raise ValueError(f"Missing embedding for Avengers-Pro query_id={query_id}")
        rows_by_query[query] = {
            "query_id": query_id,
            "query": query,
            "embedding": embeddings.loc[query_id].to_numpy(dtype=float).tolist(),
        }
    return list(rows_by_query.values())


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    lines = [
        "# Phase E Avengers-Pro CLI Metrics Memo",
        "",
        f"Command: `python experiments/37_avengerspro_cli_metrics.py --config {config_path}`",
        "",
        "This runs the exact upstream Avengers-Pro simple-cluster command on RouteCode split-aligned assets using a local embedding cache. The upstream metric is routing accuracy/cost, not RouteCode routing utility.",
        "",
        "Outputs:",
        "",
        "- `table_avengerspro_cli_metrics.csv`",
        f"- `{RUN_DIRNAME}/simple_cluster_full_results.json`",
        f"- `{RUN_DIRNAME}/avengerspro_simple_cluster_stdout.log`",
        f"- `{RUN_DIRNAME}/simple_cluster_config.full.json`",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "phase_e_avengerspro_cli_metrics_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## Avengers-Pro CLI Metrics"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/37_avengerspro_cli_metrics.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_avengerspro_cli_metrics.csv`: exact upstream Avengers-Pro simple-cluster accuracy/cost metrics on split-aligned assets.",
        "- `phase_e_avengerspro_cli_metrics_memo.md`: compatibility notes and command evidence.",
        f"- `{RUN_DIRNAME}/simple_cluster_full_results.json`: exact command JSON output.",
        f"- `{RUN_DIRNAME}/avengerspro_simple_cluster_stdout.log`: exact command log.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "dataset_level_accuracy",
                    "sample_level_accuracy",
                    "total_queries",
                    "total_cost",
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


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
