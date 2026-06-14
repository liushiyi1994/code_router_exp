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
from routecode.eval.external_baselines import (
    build_routellm_pairwise_records,
    choose_strong_weak_pair,
)
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


PAIRWISE_DIRNAME = "routellm_pairwise"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


def run(config_path: str) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    baseline_config = config.get("external_baselines", {})
    pair = choose_strong_weak_pair(
        prepared.matrices["train"].utility,
        strong_model=baseline_config.get("strong_model"),
        weak_model=baseline_config.get("weak_model"),
    )
    records = build_routellm_pairwise_records(
        {
            "train": prepared.matrices["train"],
            "test": prepared.matrices["test"],
        },
        pair,
    )

    pairwise_dir = out_dir / PAIRWISE_DIRNAME
    pairwise_dir.mkdir(parents=True, exist_ok=True)
    _write_json(pairwise_dir / "pairwise_train.json", records["train"])
    _write_json(pairwise_dir / "pairwise_test.json", records["test"])

    table = _summary_table(records, pair)
    table.to_csv(out_dir / "table_routellm_pairwise_alignment.csv", index=False)

    metadata = _metadata(config_path, config, pair, records)
    _write_json(pairwise_dir / "metadata.json", metadata)
    write_memo(out_dir, config_path, pairwise_dir, table, metadata)
    append_readme(out_dir, config_path, pairwise_dir, table)
    print(f"Wrote RouteLLM pairwise alignment substrate to {pairwise_dir}")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata(
    config_path: str,
    config: dict[str, Any],
    pair,
    records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    train_ids = {row["query_id"] for row in records["train"]}
    test_ids = {row["query_id"] for row in records["test"]}
    return {
        "config_path": config_path,
        "data_source": config.get("data", {}).get("source", "synthetic"),
        "strong_model": pair.strong_model,
        "weak_model": pair.weak_model,
        "model_a": pair.strong_model,
        "model_b": pair.weak_model,
        "split_aligned_with_routecode": True,
        "official_routellm_result": False,
        "routecode_metric_compatible": False,
        "query_id_overlap_train_test": len(train_ids & test_ids),
        "record_counts": {split: len(rows) for split, rows in records.items()},
        "schema": {
            "winner": "model_a if strong utility is higher, model_b if weak utility is higher, else tie",
            "model_a_utility": "RouteCode utility for the configured strong model",
            "model_b_utility": "RouteCode utility for the configured weak model",
        },
        "compatibility_note": (
            "Split-aligned RouteLLM-style pairwise data substrate. This is not an "
            "official RouteLLM MF/BERT fitted result."
        ),
    }


def _summary_table(records: dict[str, list[dict[str, Any]]], pair) -> pd.DataFrame:
    rows = [_summary_row(split, split_records, pair) for split, split_records in records.items()]
    rows.append(_summary_row("overall", [row for split_records in records.values() for row in split_records], pair))
    return pd.DataFrame(rows)


def _summary_row(split: str, records: list[dict[str, Any]], pair) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    count = int(len(frame))
    model_a_wins = int((frame["winner"] == "model_a").sum()) if count else 0
    model_b_wins = int((frame["winner"] == "model_b").sum()) if count else 0
    ties = int((frame["winner"] == "tie").sum()) if count else 0
    decisive = model_a_wins + model_b_wins
    return {
        "split": split,
        "record_count": count,
        "decisive_count": decisive,
        "tie_count": ties,
        "model_a_win_count": model_a_wins,
        "model_b_win_count": model_b_wins,
        "model_a_win_rate": model_a_wins / count if count else 0.0,
        "model_b_win_rate": model_b_wins / count if count else 0.0,
        "tie_rate": ties / count if count else 0.0,
        "mean_utility_margin_model_a_minus_b": float(frame["utility_margin_model_a_minus_b"].mean()) if count else 0.0,
        "strong_model": pair.strong_model,
        "weak_model": pair.weak_model,
        "model_a": pair.strong_model,
        "model_b": pair.weak_model,
        "split_aligned_with_routecode": True,
        "official_routellm_result": False,
        "routecode_metric_compatible": False,
        "implementation_note": (
            "Pairwise data substrate for later official RouteLLM evaluation; "
            "not an official RouteLLM MF/BERT result."
        ),
    }


def append_readme(out_dir: Path, config_path: str, pairwise_dir: Path, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## RouteLLM Pairwise Alignment Substrate"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/14_routellm_pairwise_alignment.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        f"- `{PAIRWISE_DIRNAME}/pairwise_train.json`: RouteCode train-split strong/weak pairwise records.",
        f"- `{PAIRWISE_DIRNAME}/pairwise_test.json`: RouteCode test-split strong/weak pairwise records.",
        f"- `{PAIRWISE_DIRNAME}/metadata.json`: split-alignment and compatibility metadata.",
        "- `table_routellm_pairwise_alignment.csv`: winner distribution and split-alignment summary.",
        "- `phase_e_routellm_pairwise_alignment_memo.md`: Phase E memo explaining remaining official RouteLLM work.",
        "",
        f"Artifact directory: `{pairwise_dir}`.",
        "",
        "This is not an official RouteLLM MF/BERT result; it is a split-aligned substrate for a future official run.",
        "",
        _markdown_table(table),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(
    out_dir: Path,
    config_path: str,
    pairwise_dir: Path,
    table: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Phase E RouteLLM Pairwise Alignment Memo",
        "",
        f"Command: `python experiments/14_routellm_pairwise_alignment.py --config {config_path}`",
        "",
        f"Artifact directory: `{pairwise_dir}`.",
        "",
        f"Binary pair: strong/model_a `{metadata['strong_model']}`, weak/model_b `{metadata['weak_model']}`.",
        "",
        "This memo records a RouteCode split-aligned RouteLLM-style pairwise data substrate. It preserves the RouteCode query-level train/test split and writes the strong/weak utility winner needed by RouteLLM-style binary routers.",
        "",
        "The official RouteLLM evaluation remains incomplete: no RouteLLM MF/BERT model is trained or evaluated by this script, and no external embedding API or checkpoint download is used.",
        "",
        "## Split Alignment",
        "",
        f"- `split_aligned_with_routecode`: `{metadata['split_aligned_with_routecode']}`",
        f"- Train/test query overlap: `{metadata['query_id_overlap_train_test']}`",
        f"- `official_routellm_result`: `{metadata['official_routellm_result']}`",
        "",
        "## Pairwise Summary",
        "",
        _markdown_table(table),
        "",
        "## References Used",
        "",
        "- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; https://github.com/lm-sys/routellm",
        "- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; https://github.com/ynulihao/LLMRouterBench",
        "",
        "## Remaining External-Baseline Gap",
        "",
        "- Run official RouteLLM-MF/BERT on this split-aligned pairwise substrate after local embedding/checkpoint dependencies are pinned.",
        "- Report the exact command, pair, split, thresholds, and metric compatibility before ranking against RouteCode.",
        "- GraphRouter and Avengers/Avengers-Pro remain separate external-baseline adapter tasks.",
        "",
    ]
    (out_dir / "phase_e_routellm_pairwise_alignment_memo.md").write_text("\n".join(lines), encoding="utf-8")


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
