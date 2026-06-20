from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.transformer_backbones import DEFAULT_REQUESTED_BACKBONES, inspect_transformer_backbone_cache
from routecode.eval.transformer_embedding_router import (
    evaluate_transformer_embedding_router,
    extract_local_transformer_embeddings,
)
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


DEFAULT_CACHE_DIR = Path("~/.cache/huggingface/hub")


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
    seed = int(config.get("run", {}).get("random_seed", 0))
    bootstrap = config.get("bootstrap", {})
    backbone_config = config.get("transformer_backbones", {})
    router_config = config.get("transformer_embedding_router", {})

    cache_dir = Path(backbone_config.get("cache_dir", DEFAULT_CACHE_DIR)).expanduser()
    requested = list(backbone_config.get("requested_model_ids", DEFAULT_REQUESTED_BACKBONES))
    max_runnable_gb = float(backbone_config.get("max_runnable_gb", 2.0))
    direct_methods = [str(method) for method in router_config.get("direct_router_methods", ["logistic", "svm", "knn"])]
    batch_size = int(router_config.get("batch_size", 16))
    max_length = int(router_config.get("max_length", 256))
    device = str(router_config.get("device", "auto"))

    readiness = inspect_transformer_backbone_cache(
        cache_dir,
        requested_model_ids=requested,
        max_runnable_gb=max_runnable_gb,
    )
    requested_set = {str(model_id) for model_id in requested}
    router_readiness = readiness[readiness["model_id"].astype(str).isin(requested_set)].copy()

    def provider(row: pd.Series, query_info: pd.DataFrame) -> pd.DataFrame:
        return extract_local_transformer_embeddings(
            local_path=str(row["local_path"]),
            query_info=query_info,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )

    table = evaluate_transformer_embedding_router(
        train=train,
        test=test,
        readiness_table=router_readiness,
        embedding_provider=provider,
        direct_methods=direct_methods,
        random_state=seed,
        n_bootstrap=int(bootstrap.get("n_bootstrap", 100)),
        ci=float(bootstrap.get("ci", 0.95)),
        max_iter=int(router_config.get("max_iter", 200)),
        n_neighbors=int(router_config.get("knn_k", config.get("routers", {}).get("knn_k", 15))),
        logistic_solver=str(router_config.get("logistic_solver", "lbfgs")),
        svm_backend=str(router_config.get("svm_backend", "linear_svc")),
        tol=float(router_config.get("tol", 1e-4)),
    )
    table.to_csv(out_dir / "table_transformer_embedding_router.csv", index=False)
    write_memo(out_dir, config_path, cache_dir, requested, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote transformer embedding router outputs to {out_dir}")


def write_memo(
    out_dir: Path,
    config_path: str,
    cache_dir: Path,
    requested: list[str],
    table: pd.DataFrame,
) -> None:
    executed = table[table["status"].eq("executed")] if "status" in table else pd.DataFrame()
    skipped = table[table["status"].eq("skipped")] if "status" in table else pd.DataFrame()
    failed = table[table["status"].eq("failed")] if "status" in table else pd.DataFrame()
    lines = [
        "# Phase F/G Transformer Embedding Router Memo",
        "",
        f"Command: `python experiments/28_transformer_embedding_router.py --config {config_path}`",
        "",
        "This script is the local-files-only execution path for pretrained encoder direct-router rows. It reads local cache metadata, loads cached checkpoints only with `local_files_only=True`, and performs no downloads or external API calls.",
        "",
        f"Cache directory: `{cache_dir}`.",
        f"Requested text backbones: `{', '.join(requested)}`.",
        "",
        "## Status",
        "",
    ]
    if executed.empty:
        lines.append("No transformer direct-router metric row was executed because no requested lightweight encoder checkpoint was cached/runnable.")
    else:
        lines.append(f"Executed transformer direct-router metric rows: `{len(executed)}`.")
    if not skipped.empty:
        lines.append(f"Skipped rows: `{len(skipped)}`.")
    if not failed.empty:
        lines.append(f"Failed rows: `{len(failed)}`. See `reason` in the CSV for dependency or model-load details.")
    lines.extend(
        [
            "",
            "## Summary",
            "",
            _markdown_table(_summary_table(table)),
            "",
            "## Compatibility",
            "",
            "- Executed rows are split-aligned direct-router metrics over query IDs.",
            "- Skipped rows are not metric evidence; they preserve the current blocker in a machine-readable artifact.",
            _compatibility_status_line(executed, skipped),
            "",
        ]
    )
    (out_dir / "phase_f_g_transformer_embedding_router_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _compatibility_status_line(executed: pd.DataFrame, skipped: pd.DataFrame) -> str:
    if executed.empty:
        return "- This validates the local-files-only execution path but does not yet provide transformer encoder routing metric evidence."
    executed_models = ", ".join(sorted(executed["model_id"].astype(str).unique()))
    if skipped.empty:
        return f"- This provides split-aligned transformer encoder routing metric evidence for the requested cached backbones: `{executed_models}`."
    skipped_models = ", ".join(sorted(skipped["model_id"].astype(str).unique()))
    return (
        "- This provides split-aligned transformer encoder routing metric evidence "
        f"for `{executed_models}`; skipped requested backbones remain incomplete until cached/evaluated: `{skipped_models}`."
    )


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# Transformer Embedding Router\n"
    marker = "## Transformer Embedding Router"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/28_transformer_embedding_router.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_transformer_embedding_router.csv`: local-files-only pretrained encoder direct-router rows, or explicit skipped/failed rows if checkpoints/dependencies are absent.",
        "- `phase_f_g_transformer_embedding_router_memo.md`: status memo for the transformer predictor-type ablation.",
        "",
        _markdown_table(_summary_table(table)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _summary_table(table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "status",
        "method",
        "model_id",
        "direct_router_method",
        "mean_utility",
        "recovered_gap_vs_oracle",
        "reason",
        "readiness_reason",
    ]
    existing = [column for column in columns if column in table.columns]
    if not existing:
        return table
    return table[existing].sort_values(["status", "model_id", "method"])


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
