from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.llmrouter_library_adapters import evaluate_llmrouter_cli_predictions
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


ASSET_DIRNAME = "llmrouter_library_adapters"


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
    asset_dir = out_dir / ASSET_DIRNAME
    predictions = _prediction_files(asset_dir)
    if not predictions:
        raise FileNotFoundError(f"No full LLMRouter CLI prediction files found under {asset_dir}")
    bootstrap = config.get("bootstrap", {})
    seed = int(config.get("run", {}).get("random_seed", 0))
    table = evaluate_llmrouter_cli_predictions(
        train,
        test,
        prepared.embeddings,
        predictions=predictions,
        seed=seed,
        n_bootstrap=int(bootstrap.get("n_bootstrap", 300)),
        ci=float(bootstrap.get("ci", 0.95)),
        knn_k=int(config.get("routers", {}).get("knn_k", 15)),
    )
    table.to_csv(out_dir / "table_llmrouter_cli_metrics.csv", index=False)
    write_memo(out_dir, config_path, table)
    append_readme(out_dir, config_path, table)
    print(f"Wrote LLMRouter CLI metric outputs to {out_dir}")


def _prediction_files(asset_dir: Path) -> dict[str, Path]:
    candidates = {
        "knn": asset_dir / "llmrouter_knn_full_predictions.json",
        "svm": asset_dir / "llmrouter_svm_full_predictions.json",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    lines = [
        "# Phase E LLMRouter CLI Metrics Memo",
        "",
        f"Command: `python experiments/31_llmrouter_cli_metrics.py --config {config_path}`",
        "",
        "Exact LLMRouter route-only CLI predictions are scored here with RouteCode test-split utility. The upstream CLI emits model selections, not RouteCode utility metrics, so this table is a RouteCode post-processing metric over exact upstream command outputs.",
        "",
        "Outputs:",
        "",
        "- `table_llmrouter_cli_metrics.csv`",
        "- `llmrouter_library_adapters/llmrouter_knn_full_predictions.json` and/or `llmrouter_library_adapters/llmrouter_svm_full_predictions.json`",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
                    "prediction_source",
                    "exact_upstream_command",
                ]
            ]
            if not table.empty
            else table
        ),
        "",
    ]
    (out_dir / "phase_e_llmrouter_cli_metrics_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## LLMRouter CLI Metrics"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/31_llmrouter_cli_metrics.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_llmrouter_cli_metrics.csv`: RouteCode utility metrics over exact LLMRouter route-only CLI outputs.",
        "- `phase_e_llmrouter_cli_metrics_memo.md`: compatibility notes for these post-processed exact-command rows.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "mean_utility",
                    "recovered_gap_vs_oracle",
                    "prediction_count",
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


if __name__ == "__main__":
    main()
