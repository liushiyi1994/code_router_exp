from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.local_eval.policy_matrices import LocalPolicyMatrices, build_local_policy_matrices
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-outcomes", required=True)
    parser.add_argument("--state-targets", required=True)
    parser.add_argument("--output-dir", default="results/phase2/local_policy_matrices")
    parser.add_argument("--lambda-cost", type=float, default=0.0)
    parser.add_argument("--policy-split", default="test")
    parser.add_argument("--train-split", default="train")
    args = parser.parse_args()
    run(
        local_outcomes_path=args.local_outcomes,
        state_targets_path=args.state_targets,
        output_dir=args.output_dir,
        lambda_cost=args.lambda_cost,
        policy_split=args.policy_split,
        train_split=args.train_split,
    )


def run(
    *,
    local_outcomes_path: str,
    state_targets_path: str,
    output_dir: str,
    lambda_cost: float = 0.0,
    policy_split: str = "test",
    train_split: str = "train",
) -> dict[str, str]:
    outcomes = pd.read_parquet(local_outcomes_path)
    state_targets = pd.read_csv(state_targets_path)
    matrices = build_local_policy_matrices(
        local_outcomes=outcomes,
        state_targets=state_targets,
        lambda_cost=lambda_cost,
        policy_split=policy_split,
        train_split=train_split,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(out_dir, matrices)
    write_memo(out_dir, local_outcomes_path, state_targets_path, matrices, paths)
    append_readme(out_dir, local_outcomes_path, state_targets_path, matrices, paths)
    print(f"Wrote local policy matrices to {out_dir}")
    return paths


def write_outputs(out_dir: Path, matrices: LocalPolicyMatrices) -> dict[str, str]:
    paths = {
        "query_model_utility": str(out_dir / "local_query_model_utility.csv"),
        "query_model_quality": str(out_dir / "local_query_model_quality.csv"),
        "query_model_cost": str(out_dir / "local_query_model_cost.csv"),
        "state_model_utility": str(out_dir / "local_state_model_utility.csv"),
        "state_model_quality": str(out_dir / "local_state_model_quality.csv"),
        "state_model_cost": str(out_dir / "local_state_model_cost.csv"),
        "metadata": str(out_dir / "local_policy_matrix_metadata.json"),
    }
    matrices.query_model_utility.to_csv(paths["query_model_utility"], index=False)
    matrices.query_model_quality.to_csv(paths["query_model_quality"], index=False)
    matrices.query_model_cost.to_csv(paths["query_model_cost"], index=False)
    matrices.state_model_utility.to_csv(paths["state_model_utility"], index=False)
    matrices.state_model_quality.to_csv(paths["state_model_quality"], index=False)
    matrices.state_model_cost.to_csv(paths["state_model_cost"], index=False)
    Path(paths["metadata"]).write_text(
        json.dumps(matrices.metadata.iloc[0].to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def write_memo(
    out_dir: Path,
    local_outcomes_path: str,
    state_targets_path: str,
    matrices: LocalPolicyMatrices,
    paths: dict[str, str],
) -> None:
    metadata = matrices.metadata.iloc[0].to_dict()
    lines = [
        "# Local Outcomes Policy Matrices",
        "",
        "This step converts exact-scored local model outcomes into ProbeRoute++ policy matrices. It reuses learned route-state targets and does not introduce human route labels.",
        "",
        "Inputs:",
        "",
        f"- Local outcomes: `{local_outcomes_path}`",
        f"- State targets: `{state_targets_path}`",
        "",
        "Utility:",
        "",
        f"`utility = quality - lambda_cost * cost_proxy`, with `lambda_cost = {float(metadata['lambda_cost']):.6g}`.",
        "",
        "Summary:",
        "",
        _markdown_table(matrices.metadata),
        "",
        "Outputs:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    (out_dir / "m15_local_policy_matrices_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(
    out_dir: Path,
    local_outcomes_path: str,
    state_targets_path: str,
    matrices: LocalPolicyMatrices,
    paths: dict[str, str],
) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Local Outcomes Policy Matrices"
    metadata = matrices.metadata.iloc[0].to_dict()
    lines = [
        marker,
        "",
        "Converts exact-scored local model outcomes to `query_model_utility` and `state_model_utility` matrices for ProbeRoute++ policy evaluation.",
        "",
        f"- Local outcomes: `{local_outcomes_path}`",
        f"- State targets: `{state_targets_path}`",
        f"- Utility formula: `quality - {float(metadata['lambda_cost']):.6g} * cost_proxy`.",
        "",
        _markdown_table(matrices.metadata),
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


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
