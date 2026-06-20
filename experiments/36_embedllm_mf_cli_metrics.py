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

from routecode.config import load_config, output_dir
from routecode.eval.external_baseline_assets import build_external_baseline_assets, write_external_baseline_assets
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


RUN_DIRNAME = "embedllm_mf_cli_metrics"
EMBEDLLM_ROOT = ROOT / "data/raw/external/LLMRouterBench/baselines/EmbedLLM"


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
    run_dir = out_dir / RUN_DIRNAME
    run_dir.mkdir(parents=True, exist_ok=True)

    train_csv = pd.read_csv(written.embedllm_train_path)
    test_csv = pd.read_csv(written.embedllm_test_path)
    settings = _settings(baseline_config)
    log_path = run_dir / "embedllm_mf_stdout.log"
    model_embedding_path = run_dir / "model_embeddings.pth"
    model_path = run_dir / "saved_model.pth"
    command = _command(
        train_path=written.embedllm_train_path,
        test_path=written.embedllm_test_path,
        question_embedding_path=written.embedllm_mf_question_embeddings_path,
        model_embedding_path=model_embedding_path,
        model_path=model_path,
        settings=settings,
        model_count=int(test_csv["model_id"].nunique()),
    )
    _run_upstream_command(command, EMBEDLLM_ROOT, log_path)
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = _parse_router_metrics(log_text)

    table = pd.DataFrame(
        [
            {
                "method": "embedllm_mf_cli_full_split",
                "num_epochs": int(settings["num_epochs"]),
                "embedding_dim": int(settings["embedding_dim"]),
                "batch_size": int(settings["batch_size"]),
                "learning_rate": float(settings["learning_rate"]),
                "alpha": float(settings["alpha"]),
                "best_dataset_level_accuracy": float(parsed["best_dataset_level_accuracy"]),
                "best_epoch": int(parsed["best_epoch"]),
                "final_dataset_level_accuracy": float(parsed["final_dataset_level_accuracy"]),
                "final_sample_level_accuracy": float(parsed["final_sample_level_accuracy"]),
                "train_prompts": int(train_csv["prompt_id"].nunique()),
                "test_prompts": int(test_csv["prompt_id"].nunique()),
                "model_count": int(test_csv["model_id"].nunique()),
                "split_aligned_with_routecode": True,
                "routecode_metric_compatible": False,
                "exact_upstream_command": True,
                "official_upstream_checkpoint": False,
                "baseline_family": "embedllm_mf_exact_cli_router_accuracy",
                "execution_evidence": str(log_path),
                "implementation_note": (
                    "Exact upstream EmbedLLM MF command on RouteCode split-aligned CSV assets. "
                    "The upstream command reports router correctness accuracy, not RouteCode routing utility."
                ),
            }
        ]
    )
    table.to_csv(out_dir / "table_embedllm_mf_cli_metrics.csv", index=False)
    _write_json(
        run_dir / "run_config.json",
        {
            "config_path": config_path,
            "command": command,
            "settings": settings,
            "train_path": str(written.embedllm_train_path),
            "test_path": str(written.embedllm_test_path),
            "question_embedding_path": str(written.embedllm_mf_question_embeddings_path),
            "model_embedding_path": str(model_embedding_path),
            "model_path": str(model_path),
        },
    )
    write_memo(out_dir, config_path, table, command)
    append_readme(out_dir, config_path, table)
    print(f"Wrote EmbedLLM MF CLI metric outputs to {out_dir}")


def _settings(baseline_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_epochs": int(baseline_config.get("embedllm_mf_num_epochs", 1)),
        "embedding_dim": int(baseline_config.get("embedllm_mf_embedding_dim", 16)),
        "batch_size": int(baseline_config.get("embedllm_mf_batch_size", 32768)),
        "learning_rate": float(baseline_config.get("embedllm_mf_learning_rate", 1e-4)),
        "alpha": float(baseline_config.get("embedllm_mf_alpha", 0.05)),
    }


def _command(
    *,
    train_path: Path,
    test_path: Path,
    question_embedding_path: Path,
    model_embedding_path: Path,
    model_path: Path,
    settings: dict[str, Any],
    model_count: int,
) -> list[str]:
    return [
        sys.executable,
        "algorithm/mf.py",
        "--train-data-path",
        str(train_path.resolve()),
        "--test-data-path",
        str(test_path.resolve()),
        "--question-embedding-path",
        str(question_embedding_path.resolve()),
        "--embedding-save-path",
        str(model_embedding_path.resolve()),
        "--model-save-path",
        str(model_path.resolve()),
        "--eval-mode",
        "router",
        "--model-num",
        str(int(model_count)),
        "--num-epochs",
        str(int(settings["num_epochs"])),
        "--batch-size",
        str(int(settings["batch_size"])),
        "--embedding-dim",
        str(int(settings["embedding_dim"])),
        "--learning-rate",
        str(float(settings["learning_rate"])),
        "--alpha",
        str(float(settings["alpha"])),
        "--wandb-run-name",
        "routecode-mf-full-split",
    ]


def _run_upstream_command(command: list[str], cwd: Path, stdout_path: Path) -> int:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)
    stdout_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"EmbedLLM MF command failed; see {stdout_path}")
    return int(completed.returncode)


def _parse_router_metrics(log_text: str) -> dict[str, float | int]:
    final_text = _final_test_block(log_text)
    dataset_matches = re.findall(r"Dataset-Level Average Accuracy:\s*([0-9]*\.?[0-9]+)", final_text)
    sample_matches = re.findall(r"Sample-Level Average Accuracy:\s*([0-9]*\.?[0-9]+)", final_text)
    best_match = re.search(
        r"Best Dataset-Level Accuracy:\s*([0-9]*\.?[0-9]+)\s+at Epoch\s+(\d+)",
        log_text,
    )
    if not dataset_matches:
        raise ValueError("Could not parse EmbedLLM MF final dataset-level accuracy")
    if not sample_matches:
        raise ValueError("Could not parse EmbedLLM MF final sample-level accuracy")
    if not best_match:
        raise ValueError("Could not parse EmbedLLM MF best dataset-level accuracy")
    return {
        "best_dataset_level_accuracy": float(best_match.group(1)),
        "best_epoch": int(best_match.group(2)),
        "final_dataset_level_accuracy": float(dataset_matches[-1]),
        "final_sample_level_accuracy": float(sample_matches[-1]),
    }


def _final_test_block(log_text: str) -> str:
    match = re.search(
        r"FINAL TEST SET RESULTS(?P<block>.*?)(?:FINAL TRAINING SET RESULTS|Final model saved|Model saved|\Z)",
        log_text,
        flags=re.DOTALL,
    )
    if match:
        return match.group("block")
    return log_text


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, command: list[str]) -> None:
    lines = [
        "# Phase E EmbedLLM MF CLI Metrics Memo",
        "",
        f"Command: `python experiments/36_embedllm_mf_cli_metrics.py --config {config_path}`",
        "",
        "This runs the exact upstream EmbedLLM MF command on RouteCode split-aligned CSV assets. The upstream metric is router correctness accuracy, not RouteCode routing utility.",
        "",
        "Upstream command:",
        "",
        "```bash",
        " ".join(command),
        "```",
        "",
        "Outputs:",
        "",
        "- `table_embedllm_mf_cli_metrics.csv`",
        f"- `{RUN_DIRNAME}/embedllm_mf_stdout.log`",
        f"- `{RUN_DIRNAME}/run_config.json`",
        "",
        _markdown_table(table),
        "",
    ]
    (out_dir / "phase_e_embedllm_mf_cli_metrics_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    marker = "## EmbedLLM MF CLI Metrics"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/36_embedllm_mf_cli_metrics.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_embedllm_mf_cli_metrics.csv`: exact upstream EmbedLLM MF router-accuracy metrics on split-aligned CSV inputs.",
        "- `phase_e_embedllm_mf_cli_metrics_memo.md`: compatibility notes and command evidence.",
        f"- `{RUN_DIRNAME}/embedllm_mf_stdout.log`: exact command log.",
        "",
        _markdown_table(
            table[
                [
                    "method",
                    "best_dataset_level_accuracy",
                    "final_sample_level_accuracy",
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
