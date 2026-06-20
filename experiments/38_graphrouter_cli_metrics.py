from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

from routecode.config import load_config, output_dir
from routecode.reporting import upsert_markdown_section


GRAPHROUTER_SOURCE = ROOT / "data/raw/external/LLMRouterBench/baselines/GraphRouter"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    run(args.config, epochs=args.epochs)


def run(config_path: str, *, epochs: int = 1) -> None:
    config = load_config(config_path)
    out_dir = output_dir(config)
    asset_dir = out_dir / "graphrouter_assets"
    run_dir = out_dir / "graphrouter_cli_metrics"
    run_dir.mkdir(parents=True, exist_ok=True)

    base_config = _load_yaml(asset_dir / "config.local.yaml")
    smoke_config_path = run_dir / "config.smoke.yaml"
    model_path = run_dir / "model_path/best_model.pth"
    stdout_path = run_dir / "graphrouter_stdout.log"
    smoke_config = _smoke_config(base_config, asset_dir=asset_dir, run_dir=run_dir, model_path=model_path, epochs=epochs)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_config_path.write_text(yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8")

    command = ["python", "run_exp.py", "--config_file", str(smoke_config_path.resolve())]
    return_code = _run_upstream_command(command, GRAPHROUTER_SOURCE, stdout_path, _wandb_offline_env(run_dir))
    metrics = _parse_graphrouter_metrics(stdout_path.read_text(encoding="utf-8", errors="replace"))
    table = pd.DataFrame(
        [
            {
                "method": "graphrouter_cli_smoke",
                "epochs": int(epochs),
                "return_code": int(return_code),
                "dataset_level_accuracy": metrics.get("dataset_level_accuracy"),
                "sample_level_accuracy": metrics.get("sample_level_accuracy"),
                "total_cost": metrics.get("total_cost"),
                "cost_source": metrics.get("cost_source", ""),
                "model_path": str(model_path),
                "stdout_path": str(stdout_path),
                "config_path": str(smoke_config_path),
                "exact_upstream_command": True,
                "no_api_compatible": True,
                "routecode_metric_compatible": False,
            }
        ]
    )
    table.to_csv(out_dir / "table_graphrouter_cli_metrics.csv", index=False)
    _write_memo(out_dir, config_path, table)
    _append_readme(out_dir, config_path, table)
    print(f"Wrote GraphRouter CLI smoke metrics to {out_dir}")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GraphRouter asset config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"GraphRouter asset config must be a mapping: {path}")
    return payload


def _smoke_config(
    config: dict[str, Any],
    *,
    asset_dir: Path,
    run_dir: Path,
    model_path: Path,
    epochs: int,
) -> dict[str, Any]:
    result = dict(config)
    result.update(
        {
            "saved_router_data_path": str((asset_dir / "router_data.csv").resolve()),
            "llm_description_path": str((asset_dir / "LLM_Descriptions.json").resolve()),
            "llm_embedding_path": str((asset_dir / "llm_description_embedding.pkl").resolve()),
            "model_path": str(model_path.resolve()),
            "train_epoch": int(epochs),
            "wandb_key": "",
        }
    )
    result.setdefault("split_ratio", [0.7, 0.0, 0.3])
    result.setdefault("batch_size", 32)
    result.setdefault("embedding_dim", 8)
    result.setdefault("edge_dim", 3)
    result["output_dir"] = str(run_dir.resolve())
    return result


def _wandb_offline_env(run_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "WANDB_MODE": "offline",
            "WANDB_SILENT": "true",
            "WANDB_DIR": str(run_dir.resolve()),
        }
    )
    return env


def _run_upstream_command(command: list[str], cwd: Path, stdout_path: Path, env: dict[str, str]) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"GraphRouter command failed with exit code {completed.returncode}; see {stdout_path}")
    return int(completed.returncode)


def _parse_graphrouter_metrics(text: str) -> dict[str, float | str]:
    section = _preferred_metrics_section(text)
    return {
        "dataset_level_accuracy": _parse_float(section, r"Dataset-Level Average Accuracy:\s*([0-9.]+)"),
        "sample_level_accuracy": _parse_float(section, r"Sample-Level Average Accuracy:\s*([0-9.]+)"),
        "total_cost": _parse_float(section, r"Total Cost:\s*([0-9.]+)"),
        "cost_source": _parse_text(section, r"Cost Source:\s*([A-Za-z0-9_.-]+)"),
    }


def _preferred_metrics_section(text: str) -> str:
    marker = "BEST TEST CHECKPOINT METRICS"
    index = text.find(marker)
    if index >= 0:
        return text[index:]
    marker = "LAST EPOCH METRICS"
    index = text.find(marker)
    return text[index:] if index >= 0 else text


def _parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def _parse_text(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return str(match.group(1)) if match else ""


def _write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    row = table.iloc[0]
    lines = [
        "# Phase E GraphRouter CLI Metrics Memo",
        "",
        f"Command: `python experiments/38_graphrouter_cli_metrics.py --config {config_path}`",
        "",
        "This run executes the upstream GraphRouter `run_exp.py` command path with generated RouteCode assets, "
        "offline wandb mode, and no external API calls. It reports upstream accuracy/cost metrics, not RouteCode utility.",
        "",
        "## Outputs",
        "",
        f"- Metrics table: `{out_dir / 'table_graphrouter_cli_metrics.csv'}`",
        f"- Smoke config: `{row['config_path']}`",
        f"- Stdout log: `{row['stdout_path']}`",
        f"- Model checkpoint: `{row['model_path']}`",
        "",
        "## Summary",
        "",
        f"- Dataset-level accuracy: `{row['dataset_level_accuracy']:.4f}`",
        f"- Sample-level accuracy: `{row['sample_level_accuracy']:.4f}`",
        f"- Total cost: `{row['total_cost']:.4f}`" if pd.notna(row["total_cost"]) else "- Total cost: ``",
    ]
    (out_dir / "phase_e_graphrouter_cli_metrics_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Results\n"
    marker = "## GraphRouter CLI Metrics"
    section = [
        marker,
        "",
        "Command:",
        "",
        f"`python experiments/38_graphrouter_cli_metrics.py --config {config_path}`",
        "",
        "Outputs:",
        "",
        "- `table_graphrouter_cli_metrics.csv`: exact upstream GraphRouter smoke-command metrics.",
        "- `phase_e_graphrouter_cli_metrics_memo.md`: command and interpretation notes.",
        "- `graphrouter_cli_metrics/config.smoke.yaml`: absolute-path offline config for the upstream runner.",
        "- `graphrouter_cli_metrics/graphrouter_stdout.log`: captured upstream command output.",
        "",
        _markdown_table(table),
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, section), encoding="utf-8")


def _markdown_table(table: pd.DataFrame) -> str:
    columns = list(table.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in table.iterrows():
        rows.append("| " + " | ".join(_markdown_cell(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _markdown_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value).replace("\n", " ").replace("|", "\\|")


if __name__ == "__main__":
    main()
