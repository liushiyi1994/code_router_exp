from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config
from routecode.pipeline import prepare_from_config
from routecode.probes.routecode_policy_inputs import RouteCodePolicyInputs, build_routecode_policy_inputs
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llmrouterbench_pilot.yaml")
    parser.add_argument(
        "--query-model-utility",
        default="results/phase2/true_probe_policy_inputs_vllm_qwen3_4b_all200/true_probe_query_model_utility.csv",
    )
    parser.add_argument("--output-dir", default="results/phase2/routecode_target_rate_policy_inputs_vllm_all200")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0)
    args = parser.parse_args()
    paths = run(
        config_path=args.config,
        query_model_utility_path=args.query_model_utility,
        output_dir=args.output_dir,
        k=args.k,
        alpha=args.alpha,
    )
    print(f"Wrote target-rate RouteCode policy inputs to {paths['metadata']}")


def run(
    *,
    config_path: str,
    query_model_utility_path: str,
    output_dir: str,
    k: int = 32,
    alpha: float = 0.0,
) -> dict[str, str]:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    query_model_utility = _read_matrix(query_model_utility_path)
    d2_config = config.get("predictability_constrained", {})
    route_config = config.get("routecode", {})
    bundle = build_routecode_policy_inputs(
        train=prepared.matrices["train"],
        embeddings=prepared.embeddings,
        query_model_utility=query_model_utility,
        k=k,
        alpha=alpha,
        beta=float(d2_config.get("beta", 0.0)),
        random_state=int(config.get("run", {}).get("random_seed", 0)),
        max_iter=int(d2_config.get("max_iter", route_config.get("max_iter", 25))),
        refinement_iter=int(d2_config.get("refinement_iter", 10)),
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(out_dir, bundle)
    write_memo(out_dir, config_path, query_model_utility_path, paths, bundle)
    append_readme(out_dir, paths, bundle)
    return paths


def write_outputs(out_dir: Path, bundle: RouteCodePolicyInputs) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "before_beliefs": str(out_dir / "true_probe_before_beliefs.csv"),
        "after_beliefs": str(out_dir / "true_probe_after_beliefs.csv"),
        "state_model_utility": str(out_dir / "true_probe_state_model_utility.csv"),
        "query_model_utility": str(out_dir / "true_probe_query_model_utility.csv"),
        "probe_cost": str(out_dir / "true_probe_cost.csv"),
        "predicted_gain": str(out_dir / "true_probe_predicted_gain.csv"),
        "metadata": str(out_dir / "routecode_target_rate_policy_input_metadata.json"),
    }
    bundle.before_beliefs.reset_index().to_csv(paths["before_beliefs"], index=False)
    bundle.after_beliefs.reset_index().to_csv(paths["after_beliefs"], index=False)
    bundle.state_model_utility.reset_index().to_csv(paths["state_model_utility"], index=False)
    bundle.query_model_utility.reset_index().rename(
        columns={bundle.query_model_utility.index.name or "index": "query_id"}
    ).to_csv(paths["query_model_utility"], index=False)
    bundle.probe_cost.reset_index().rename(columns={bundle.probe_cost.index.name or "index": "query_id"}).to_csv(
        paths["probe_cost"],
        index=False,
    )
    bundle.predicted_gain.reset_index().rename(
        columns={bundle.predicted_gain.index.name or "index": "query_id"}
    ).to_csv(paths["predicted_gain"], index=False)
    Path(paths["metadata"]).write_text(json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def write_memo(
    out_dir: Path,
    config_path: str,
    query_model_utility_path: str,
    paths: dict[str, str],
    bundle: RouteCodePolicyInputs,
) -> None:
    lines = [
        "# Target-Rate RouteCode Policy Inputs",
        "",
        "Command:",
        "",
        "```bash",
        (
            "PYTHONPATH=src python experiments/73_routecode_target_rate_policy_inputs.py "
            f"--config {config_path} "
            f"--query-model-utility {query_model_utility_path} "
            f"--output-dir {out_dir} "
            f"--k {bundle.metadata['k']} --alpha {bundle.metadata['alpha']}"
        ),
        "```",
        "",
        "These inputs fit the RouteCode codebook on train only and predict one-hot route-state beliefs from query embeddings. They do not use held-out utility to assign labels.",
        "",
        "Summary:",
        "",
        f"- K: `{bundle.metadata['k']}`.",
        f"- Effective labels: `{bundle.metadata['effective_labels']}`.",
        f"- Train rows: `{bundle.metadata['train_rows']}`.",
        f"- Policy rows: `{bundle.metadata['policy_rows']}`.",
        "",
        "Outputs:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    (out_dir / "m_routecode_target_rate_policy_inputs.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, paths: dict[str, str], bundle: RouteCodePolicyInputs) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Target-Rate RouteCode Policy Inputs"
    lines = [
        marker,
        "",
        "Train-fit RouteCode one-hot beliefs for the standard ProbeRoute++ policy evaluator.",
        "",
        f"- K: `{bundle.metadata['k']}`.",
        f"- Effective labels: `{bundle.metadata['effective_labels']}`.",
        f"- Policy rows: `{bundle.metadata['policy_rows']}`.",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _read_matrix(path: str | Path) -> pd.DataFrame:
    matrix_path = Path(path)
    frame = pd.read_parquet(matrix_path) if matrix_path.suffix == ".parquet" else pd.read_csv(matrix_path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    return frame


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
