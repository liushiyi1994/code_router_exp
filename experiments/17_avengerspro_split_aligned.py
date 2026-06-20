from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.external_baselines import (
    AvengersProClusterRouter,
    build_avengerspro_records,
)
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.knn import KNNRouter
from routecode.routers.single_best import BestSingleRouter


RUN_DIRNAME = "avengerspro_split_aligned"
AVENGERSPRO_REPO = "https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/AvengersPro"
GRAPHROUTER_REPO = "https://github.com/ynulihao/LLMRouterBench/tree/main/baselines/GraphRouter"


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
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))

    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)
    assets = build_avengerspro_records({"train": train, "test": test})
    _write_jsonl(run_dir / "train.jsonl", assets.train_records)
    _write_jsonl(run_dir / "test.jsonl", assets.test_records)
    _write_json(run_dir / "baseline_scores.json", assets.baseline_scores)
    clusters = _cluster_values(config, baseline_config)
    top_k = int(baseline_config.get("avengerspro_top_k", 1))
    beta = float(baseline_config.get("avengerspro_beta", 9.0))
    performance_weight = float(baseline_config.get("avengerspro_performance_weight", 0.7))
    cost_sensitivity = float(baseline_config.get("avengerspro_cost_sensitivity", 0.3))
    min_quality_threshold = float(baseline_config.get("avengerspro_min_quality_threshold", 0.0))
    smoke_assets = _write_upstream_smoke_assets(
        run_dir=run_dir,
        train_records=assets.train_records,
        test_records=assets.test_records,
        embeddings=prepared.embeddings,
        clusters=clusters,
        top_k=top_k,
        beta=beta,
        seed=seed,
        performance_weight=performance_weight,
        cost_sensitivity=cost_sensitivity,
    )

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    oracle_mean = float(test.utility.max(axis=1).mean())
    knn = KNNRouter(int(config.get("routers", {}).get("knn_k", 15))).fit(
        train.query_info,
        train.utility,
        prepared.embeddings,
    ).predict(test.query_info, prepared.embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, knn).mean()))

    rows: list[dict[str, Any]] = []
    raw_results: dict[str, Any] = {}

    for offset, n_clusters in enumerate(clusters):
        simple = AvengersProClusterRouter(
            n_clusters=n_clusters,
            top_k=top_k,
            beta=beta,
            mode="simple",
            random_state=seed,
        ).fit(train.query_info, train.quality, train.cost, prepared.embeddings)
        rows.append(
            _evaluate_router(
                method=f"avengerspro_simple_cluster_k{n_clusters}",
                router=simple,
                train=train,
                test=test,
                embeddings=prepared.embeddings,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + offset,
                n_clusters=n_clusters,
                mode="simple",
                top_k=top_k,
                beta=beta,
            )
        )
        raw_results[rows[-1]["method"]] = _raw_router_result(simple, test, prepared.embeddings)

        balanced = AvengersProClusterRouter(
            n_clusters=n_clusters,
            top_k=top_k,
            beta=beta,
            mode="balance",
            performance_weight=performance_weight,
            cost_sensitivity=cost_sensitivity,
            min_quality_threshold=min_quality_threshold,
            random_state=seed,
        ).fit(train.query_info, train.quality, train.cost, prepared.embeddings)
        rows.append(
            _evaluate_router(
                method=(
                    f"avengerspro_balance_cluster_k{n_clusters}"
                    f"_w{performance_weight:g}_c{cost_sensitivity:g}"
                ),
                router=balanced,
                train=train,
                test=test,
                embeddings=prepared.embeddings,
                baseline_mean=baseline_mean,
                learned_reference_mean=learned_reference_mean,
                oracle_mean=oracle_mean,
                n_bootstrap=n_bootstrap,
                ci=ci,
                seed=seed + 100 + offset,
                n_clusters=n_clusters,
                mode="balance",
                top_k=top_k,
                beta=beta,
                performance_weight=performance_weight,
                cost_sensitivity=cost_sensitivity,
                min_quality_threshold=min_quality_threshold,
            )
        )
        raw_results[rows[-1]["method"]] = _raw_router_result(balanced, test, prepared.embeddings)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_avengerspro_split_aligned.csv", index=False)
    _write_json(run_dir / "raw_results.json", raw_results)
    _write_json(
        run_dir / "metadata.json",
        {
            "config": config_path,
            "train_queries": int(len(train.query_info)),
            "test_queries": int(len(test.query_info)),
            "clusters": clusters,
            "top_k": top_k,
            "beta": beta,
            "performance_weight": performance_weight,
            "cost_sensitivity": cost_sensitivity,
            "source_repo": AVENGERSPRO_REPO,
            "official_command_path": False,
            "no_api_calls": True,
            "split_aligned_with_routecode": True,
            "upstream_smoke_assets": smoke_assets,
        },
    )
    write_memo(out_dir, config_path, run_dir, table)
    append_readme(out_dir, config_path, run_dir, table)
    print(f"Wrote split-aligned Avengers-Pro outputs to {run_dir}")


def _cluster_values(config: dict[str, Any], baseline_config: dict[str, Any]) -> list[int]:
    configured = baseline_config.get("avengerspro_clusters")
    if configured is None:
        configured = [int(config.get("routers", {}).get("embedding_clusters", 16))]
    if isinstance(configured, int):
        values = [configured]
    else:
        values = [int(value) for value in configured]
    return sorted({max(1, value) for value in values})


def _write_upstream_smoke_assets(
    *,
    run_dir: Path,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    embeddings: pd.DataFrame,
    clusters: list[int],
    top_k: int,
    beta: float,
    seed: int,
    performance_weight: float,
    cost_sensitivity: float,
) -> dict[str, Any]:
    max_clusters = max(clusters) if clusters else 1
    smoke_train_count = min(len(train_records), max(8, max_clusters * 2))
    smoke_test_count = min(len(test_records), 32)
    smoke_train = list(train_records[:smoke_train_count])
    smoke_test = list(test_records[:smoke_test_count])
    if not smoke_train:
        raise ValueError("Avengers-Pro smoke train asset requires at least one train record")
    if not smoke_test:
        raise ValueError("Avengers-Pro smoke test asset requires at least one test record")

    smoke_train_path = run_dir / "smoke_train.jsonl"
    smoke_test_path = run_dir / "smoke_test.jsonl"
    cache_path = run_dir / "embedding_cache.jsonl"
    config_path = run_dir / "simple_cluster_config.local.json"
    output_path = run_dir / "simple_cluster_smoke_results.json"
    _write_jsonl(smoke_train_path, smoke_train)
    _write_jsonl(smoke_test_path, smoke_test)
    cache_rows = _embedding_cache_rows(smoke_train + smoke_test, embeddings)
    _write_jsonl(cache_path, cache_rows)

    smoke_clusters = max(1, min(max_clusters, len(smoke_train)))
    smoke_top_k = max(1, min(top_k, smoke_clusters))
    _write_json(
        config_path,
        {
            "train_data_path": str(smoke_train_path.resolve()),
            "test_data_path": str(smoke_test_path.resolve()),
            "baseline_scores_path": str((run_dir / "baseline_scores.json").resolve()),
            "embedding_cache_path": str(cache_path.resolve()),
            "embedding_api_key": "",
            "embedding_base_url": "",
            "embedding_model": "routecode-cache",
            "n_clusters": smoke_clusters,
            "max_router": 1,
            "top_k": smoke_top_k,
            "beta": beta,
            "seed": seed,
            "max_workers": 1,
            "cluster_batch_size": 64,
            "performance_weight": performance_weight,
            "cost_sensitivity": cost_sensitivity,
        },
    )
    return {
        "smoke_train_path": str(smoke_train_path),
        "smoke_test_path": str(smoke_test_path),
        "embedding_cache_path": str(cache_path),
        "config_path": str(config_path),
        "expected_output_path": str(output_path),
        "smoke_train_queries": len(smoke_train),
        "smoke_test_queries": len(smoke_test),
        "smoke_clusters": smoke_clusters,
    }


def _embedding_cache_rows(records: list[dict[str, Any]], embeddings: pd.DataFrame) -> list[dict[str, Any]]:
    rows_by_query: dict[str, dict[str, Any]] = {}
    for record in records:
        query = str(record["query"])
        if query in rows_by_query:
            continue
        query_id = str(record["query_id"])
        if query_id not in embeddings.index:
            raise ValueError(f"Missing embedding for Avengers-Pro smoke query_id={query_id}")
        rows_by_query[query] = {
            "query_id": query_id,
            "query": query,
            "embedding": embeddings.loc[query_id].to_numpy(dtype=float).tolist(),
        }
    return list(rows_by_query.values())


def _evaluate_router(
    *,
    method: str,
    router: AvengersProClusterRouter,
    train,
    test,
    embeddings: pd.DataFrame,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    n_clusters: int,
    mode: str,
    top_k: int,
    beta: float,
    performance_weight: float | str = "",
    cost_sensitivity: float | str = "",
    min_quality_threshold: float | str = "",
) -> dict[str, Any]:
    del train
    selected = router.predict(test.query_info, embeddings)
    labels = router.predict_labels(embeddings.loc[test.query_info.index])
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
        k=n_clusters,
        labels=labels,
    )
    row.update(
        {
            "baseline_family": "official_algorithm_local_embedding",
            "source_repo": AVENGERSPRO_REPO,
            "paper_reference": "Avengers-Pro / LLMRouterBench",
            "mode": mode,
            "n_clusters": int(n_clusters),
            "effective_clusters": int(router.effective_clusters),
            "top_k": int(top_k),
            "beta": float(beta),
            "performance_weight": performance_weight,
            "cost_sensitivity": cost_sensitivity,
            "min_quality_threshold": min_quality_threshold,
            "split_aligned_with_routecode": True,
            "routecode_metric_compatible": True,
            "no_api_calls": True,
            "official_command_path": False,
            "official_upstream_checkpoint": False,
            "implementation_note": (
                "Split-aligned local implementation of the Avengers-Pro cluster-routing contract "
                "using RouteCode deterministic embeddings; not an official upstream command-path run."
            ),
        }
    )
    return row


def _raw_router_result(router: AvengersProClusterRouter, test, embeddings: pd.DataFrame) -> dict[str, Any]:
    selected = router.predict(test.query_info, embeddings)
    labels = router.predict_labels(embeddings.loc[test.query_info.index])
    return {
        "selected_model_counts": selected.value_counts().sort_index().to_dict(),
        "cluster_counts": labels.value_counts().sort_index().to_dict(),
        "cluster_rankings": router.cluster_rankings,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame) -> None:
    summary = table.sort_values("mean_utility", ascending=False)[
        [
            "method",
            "mean_utility",
            "oracle_regret",
            "recovered_gap_vs_oracle",
            "selected_model_entropy",
        ]
    ]
    lines = [
        "# Phase E Avengers-Pro Split-Aligned Memo",
        "",
        f"Command: `python experiments/17_avengerspro_split_aligned.py --config {config_path}`",
        "",
        f"Run assets: `{run_dir}`.",
        "",
        "This run uses a local implementation of the Avengers-Pro cluster-routing contract: K-means over query embeddings, train-only per-cluster model rankings, and nearest-cluster routing. It uses RouteCode deterministic embeddings and the RouteCode train/test split, so it makes no embedding API calls.",
        "",
        "This is not an official upstream command-path run, not an upstream checkpoint, and not evidence for paper-level Avengers-Pro performance. It is a split-aligned compatibility baseline for the RouteCode pilot.",
        "",
        _markdown_table(summary),
        "",
        "## Adapter Notes",
        "",
        "- Avengers-Pro source inspected: `data/raw/external/LLMRouterBench/baselines/AvengersPro`.",
        "- Official Avengers-Pro scripts require an embedding service configuration by default; RouteCode also writes a bounded cache-backed smoke config for local no-API upstream command checks.",
        "- GraphRouter remains blocked for local metric rows because the current environment lacks PyG packages and its data construction path expects generated graph inputs plus embedding configuration.",
        "",
        "## References Used",
        "",
        f"- Avengers-Pro source in LLMRouterBench: {AVENGERSPRO_REPO}",
        f"- GraphRouter source in LLMRouterBench: {GRAPHROUTER_REPO}",
        "",
    ]
    (out_dir / "phase_e_avengerspro_split_aligned_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, run_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## Avengers-Pro Split-Aligned Evaluation"
    summary = table.sort_values("mean_utility", ascending=False)[
        ["method", "mean_utility", "recovered_gap_vs_oracle"]
    ]
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/17_avengerspro_split_aligned.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_avengerspro_split_aligned.csv`: RouteCode utility rows from a local implementation of the Avengers-Pro cluster-routing contract.",
        f"- `{run_dir.name}/train.jsonl`, `{run_dir.name}/test.jsonl`, and `{run_dir.name}/baseline_scores.json`: split-aligned Avengers-Pro-format assets.",
        f"- `{run_dir.name}/smoke_train.jsonl`, `{run_dir.name}/smoke_test.jsonl`, `{run_dir.name}/embedding_cache.jsonl`, and `{run_dir.name}/simple_cluster_config.local.json`: bounded no-API assets for the upstream Avengers-Pro command path.",
        "- `phase_e_avengerspro_split_aligned_memo.md`: caveats and adapter notes.",
        "",
        _markdown_table(summary),
        "",
    ]
    existing = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
