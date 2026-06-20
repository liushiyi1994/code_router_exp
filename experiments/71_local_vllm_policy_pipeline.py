from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir as config_output_dir
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_local_vllm_two_model_all200_nothink.yaml")
    parser.add_argument("--phase2-dir", default="results/phase2")
    parser.add_argument("--state-targets", default="results/phase2/aligned_offline/aligned_state_targets.csv")
    parser.add_argument("--query-features", default="results/phase2/aligned_offline/aligned_query_features.csv")
    parser.add_argument(
        "--probe-features",
        default="results/phase2/exact_manifest_probes_vllm_qwen3_4b_all200/exact_manifest_probe_features.parquet",
    )
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--skip-readiness", action="store_true")
    args = parser.parse_args()
    run(
        config_path=args.config,
        phase2_dir=args.phase2_dir,
        state_targets_path=args.state_targets,
        query_features_path=args.query_features,
        probe_features_path=args.probe_features,
        lambda_cost=args.lambda_cost,
        skip_readiness=args.skip_readiness,
    )


def run(
    *,
    config_path: str,
    phase2_dir: str,
    state_targets_path: str,
    query_features_path: str,
    probe_features_path: str,
    lambda_cost: float = 0.0,
    skip_readiness: bool = False,
) -> dict[str, str]:
    config = load_config(config_path)
    generation_dir = config_output_dir(config)
    phase2 = Path(phase2_dir)
    run_name = str(config.get("run", {}).get("name", generation_dir.name))
    readiness_dir = phase2 / f"local_server_readiness_{run_name}"
    matrix_dir = phase2 / f"local_policy_matrices_{run_name}"
    inputs_dir = phase2 / f"true_probe_policy_inputs_{run_name}"
    policy_dir = phase2 / f"true_probe_policy_{run_name}"
    paths: dict[str, str] = {}

    if not skip_readiness:
        readiness = _script("58_local_server_readiness").run(
            config_path=config_path,
            output_dir=str(readiness_dir),
        )
        paths["readiness_table"] = str(readiness_dir / "table_local_server_readiness.csv")
        blocked = readiness[readiness["status"].astype(str).eq("blocked")]
        if not blocked.empty:
            write_pipeline_memo(
                phase2=phase2,
                config_path=config_path,
                run_name=run_name,
                paths=paths,
                status="blocked_readiness",
                notes="At least one local vLLM endpoint is blocked; generation was not started.",
            )
            raise SystemExit("Blocked: at least one local vLLM endpoint failed readiness")

    _script("51_true_model_generation_matrix").run(config_path)
    local_outcomes = generation_dir / "local_model_outcomes.parquet"
    paths["local_outcomes"] = str(local_outcomes)

    matrix_paths = _script("70_local_outcomes_policy_matrices").run(
        local_outcomes_path=str(local_outcomes),
        state_targets_path=state_targets_path,
        output_dir=str(matrix_dir),
        lambda_cost=lambda_cost,
    )
    paths.update({f"matrices_{key}": value for key, value in matrix_paths.items()})

    input_paths = _script("64_true_probe_policy_inputs").run(
        probe_features_path=probe_features_path,
        state_targets_path=state_targets_path,
        query_features_path=query_features_path,
        state_model_utility_path=matrix_paths["state_model_utility"],
        query_model_utility_path=matrix_paths["query_model_utility"],
        output_dir=str(inputs_dir),
    )
    paths.update({f"inputs_{key}": value for key, value in input_paths.items()})

    _script("54_proberoute_policy").run(
        output_dir=str(policy_dir),
        before_beliefs_path=input_paths["before_beliefs"],
        after_beliefs_path=input_paths["after_beliefs"],
        state_model_utility_path=input_paths["state_model_utility"],
        query_model_utility_path=input_paths["query_model_utility"],
        probe_cost_path=input_paths["probe_cost"],
        predicted_gain_path=input_paths["predicted_gain"],
    )
    paths["policy_table"] = str(policy_dir / "table_proberoute_policy.csv")
    paths["policy_figure"] = str(policy_dir / "fig_gap_closed_vs_probe_cost.pdf")

    audit_paths = _script("69_phase2_completion_audit").run(root=ROOT, output_dir=phase2)
    paths.update({f"audit_{key}": value for key, value in audit_paths.items()})
    write_pipeline_memo(
        phase2=phase2,
        config_path=config_path,
        run_name=run_name,
        paths=paths,
        status="completed",
        notes="Local vLLM generation, local policy matrices, true-probe policy inputs, M5 policy evaluation, and audit refresh completed.",
    )
    return paths


def write_pipeline_memo(
    *,
    phase2: Path,
    config_path: str,
    run_name: str,
    paths: dict[str, str],
    status: str,
    notes: str,
) -> None:
    phase2.mkdir(parents=True, exist_ok=True)
    memo_path = phase2 / f"{run_name}_local_vllm_policy_pipeline_memo.md"
    lines = [
        "# Local vLLM Policy Pipeline",
        "",
        "Command:",
        "",
        "```bash",
        f"PYTHONPATH=src python experiments/71_local_vllm_policy_pipeline.py --config {config_path}",
        "```",
        "",
        f"Status: `{status}`.",
        "",
        notes,
        "",
        "Outputs:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    memo_path.write_text("\n".join(lines), encoding="utf-8")
    report_path = phase2 / "PHASE2_EVIDENCE_REPORT.md"
    existing = report_path.read_text(encoding="utf-8") if report_path.exists() else "# Phase 2 Evidence Report\n"
    marker = "## Local vLLM Policy Pipeline"
    report_lines = [
        marker,
        "",
        f"The local vLLM policy pipeline command is available for `{run_name}`:",
        "",
        "```bash",
        f"PYTHONPATH=src python experiments/71_local_vllm_policy_pipeline.py --config {config_path}",
        "```",
        "",
        f"Latest pipeline memo: `{memo_path}`.",
        "",
    ]
    report_path.write_text(upsert_markdown_section(existing, marker, report_lines), encoding="utf-8")


def _script(name: str):
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
