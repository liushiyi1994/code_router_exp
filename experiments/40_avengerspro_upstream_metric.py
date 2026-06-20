from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.avengerspro_upstream_metric import evaluate_avengerspro_routing_details
from routecode.eval.external_baselines import build_avengerspro_records
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


ASSET_DIRNAME = "avengerspro_split_aligned"
RUN_DIRNAME = "avengerspro_upstream_metric"
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
    bootstrap = config.get("bootstrap", {})

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

    cache_path = run_dir / "embedding_cache.jsonl"
    _write_jsonl(cache_path, _embedding_cache_rows(assets.train_records + assets.test_records, prepared.embeddings))
    command_config = _command_config(
        config=config,
        baseline_config=baseline_config,
        train_path=train_path,
        test_path=test_path,
        baseline_scores_path=baseline_scores_path,
        cache_path=cache_path,
        seed=seed,
        train_count=len(assets.train_records),
    )
    _write_json(run_dir / "simple_cluster_config.local.json", command_config)

    upstream_results = _run_upstream_router(command_config)
    routing_details = upstream_results.get("routing_details")
    if not isinstance(routing_details, list) or not routing_details:
        raise RuntimeError("Upstream Avengers-Pro router did not return routing_details")
    raw_path = run_dir / "raw_routing_details.json"
    _write_json(raw_path, routing_details)
    _write_json(
        run_dir / "run_config.json",
        {
            "config_path": config_path,
            "command_config": command_config,
            "prediction_count": int(len(routing_details)),
            "upstream_summary": _summary_payload(upstream_results),
            "split_aligned_with_routecode": True,
            "routecode_metric_compatible": True,
            "exact_upstream_command": False,
        },
    )

    row = evaluate_avengerspro_routing_details(
        train,
        test,
        prepared.embeddings,
        routing_details=routing_details,
        test_records=assets.test_records,
        prediction_source=str(raw_path),
        seed=seed,
        n_bootstrap=int(bootstrap.get("n_bootstrap", 300)),
        ci=float(bootstrap.get("ci", 0.95)),
        knn_k=int(config.get("routers", {}).get("knn_k", 15)),
    )
    row.update(
        {
            "requested_clusters": int(command_config["n_clusters"]),
            "effective_clusters": int(command_config["n_clusters"]),
            "top_k": int(command_config["top_k"]),
            "beta": float(command_config["beta"]),
            "upstream_accuracy": float(upstream_results.get("accuracy", 0.0)) / 100.0,
            "upstream_total_cost": float(upstream_results.get("cost_analysis", {}).get("total_cost", 0.0)),
            "config_path": str(run_dir / "simple_cluster_config.local.json"),
        }
    )
    table = pd.DataFrame([row])
    table.to_csv(out_dir / "table_avengerspro_upstream_metric.csv", index=False)
    write_memo(out_dir, config_path, table, raw_path)
    append_readme(out_dir, config_path, table, raw_path)
    print(f"Wrote Avengers-Pro upstream-code metric outputs to {out_dir}")


def _command_config(
    *,
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    train_path: Path,
    test_path: Path,
    baseline_scores_path: Path,
    cache_path: Path,
    seed: int,
    train_count: int,
) -> dict[str, Any]:
    clusters = _cluster_values(config, baseline_config)
    requested_clusters = int(clusters[0])
    effective_clusters = max(1, min(requested_clusters, train_count))
    top_k = int(baseline_config.get("avengerspro_top_k", 1))
    return {
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
        "beta": float(baseline_config.get("avengerspro_beta", 9.0)),
        "seed": int(seed),
        "max_workers": 1,
        "cluster_batch_size": int(baseline_config.get("avengerspro_cluster_batch_size", 1000)),
        "performance_weight": float(baseline_config.get("avengerspro_performance_weight", 0.7)),
        "cost_sensitivity": float(baseline_config.get("avengerspro_cost_sensitivity", 0.3)),
    }


def _cluster_values(config: dict[str, Any], baseline_config: dict[str, Any]) -> list[int]:
    configured = baseline_config.get("avengerspro_clusters")
    if configured is None:
        configured = [int(config.get("routers", {}).get("embedding_clusters", 16))]
    if isinstance(configured, int):
        values = [configured]
    else:
        values = [int(value) for value in configured]
    return sorted({max(1, value) for value in values})


def _run_upstream_router(command_config: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(LLMROUTERBENCH_ROOT))
    from baselines.AvengersPro.config import SimpleClusterConfig
    from baselines.AvengersPro.simple_cluster_router import SimpleClusterRouter

    previous_key = os.environ.get("EMBEDDING_API_KEY")
    os.environ["EMBEDDING_API_KEY"] = ""
    try:
        router = SimpleClusterRouter(SimpleClusterConfig(**command_config))
        return router.run_routing()
    finally:
        if previous_key is None:
            os.environ.pop("EMBEDDING_API_KEY", None)
        else:
            os.environ["EMBEDDING_API_KEY"] = previous_key


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


def _summary_payload(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": float(results.get("accuracy", 0.0)),
        "correct_routes": float(results.get("correct_routes", 0.0)),
        "total_queries": int(results.get("total_queries", 0)),
        "all_sample_avg": float(results.get("all_sample_avg", 0.0)),
        "model_selection_stats": dict(results.get("model_selection_stats", {})),
        "cost_analysis": dict(results.get("cost_analysis", {})),
    }


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, raw_path: Path) -> None:
    lines = [
        "# Phase E Avengers-Pro Upstream Metric Memo",
        "",
        f"Command: `python experiments/40_avengerspro_upstream_metric.py --config {config_path}`",
        "",
        "This run calls the upstream Avengers-Pro `SimpleClusterRouter` class on RouteCode split-aligned assets "
        "with a local embedding cache and saves `routing_details` before scoring selected models with RouteCode "
        "test-split utility. It is not an exact upstream command because the exact CLI JSON omits routing details.",
        "",
        "Outputs:",
        "",
        "- `table_avengerspro_upstream_metric.csv`",
        f"- `{raw_path}`",
        f"- `{RUN_DIRNAME}/simple_cluster_config.local.json`",
        f"- `{RUN_DIRNAME}/run_config.json`",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
                    "upstream_accuracy",
                    "routecode_metric_compatible",
                ]
            ]
        ),
        "",
    ]
    (out_dir / "phase_e_avengerspro_upstream_metric_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame, raw_path: Path) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## Avengers-Pro Upstream Metric"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/40_avengerspro_upstream_metric.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_avengerspro_upstream_metric.csv`: RouteCode utility metrics over captured upstream Avengers-Pro routing details.",
        "- `phase_e_avengerspro_upstream_metric_memo.md`: compatibility and leakage notes.",
        f"- `{raw_path}`: captured upstream `routing_details` for RouteCode test queries.",
        "",
        "This is not an exact upstream command row; the exact CLI JSON omits `routing_details`.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
                    "upstream_accuracy",
                ]
            ]
        ),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
                values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
