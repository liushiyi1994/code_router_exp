from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config
from routecode.pipeline import prepare_from_config
from routecode.probes.aligned_inputs import AlignedOfflineInputs, build_aligned_offline_inputs
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/phase2/aligned_offline")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--probe-cost-proxy", type=float, default=0.0001)
    args = parser.parse_args()
    run(
        config_path=args.config,
        output_dir=args.output_dir,
        n_neighbors=args.n_neighbors,
        probe_cost_proxy=args.probe_cost_proxy,
    )


def run(
    *,
    config_path: str,
    output_dir: str,
    n_neighbors: int = 15,
    probe_cost_proxy: float = 0.0001,
) -> dict[str, str]:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    d2_config = config.get("predictability_constrained", {})
    route_config = config.get("routecode", {})
    seed = int(config.get("run", {}).get("random_seed", 0))
    bundle = build_aligned_offline_inputs(
        train=train,
        test=test,
        embeddings=prepared.embeddings,
        k=int(d2_config.get("k", route_config.get("selected_k_for_cards", 16))),
        alpha=float(d2_config.get("selected_alpha", 3.0)),
        beta=float(d2_config.get("beta", 0.0)),
        random_state=seed,
        max_iter=int(d2_config.get("max_iter", route_config.get("max_iter", 25))),
        refinement_iter=int(d2_config.get("refinement_iter", 10)),
        n_neighbors=n_neighbors,
        probe_cost_proxy=probe_cost_proxy,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_bundle(out_dir, bundle)
    write_memo(out_dir, config_path, paths, bundle)
    append_phase2_readme(_phase2_readme_dir(out_dir), config_path, paths)
    print(f"Wrote aligned offline Phase 2 inputs to {out_dir}")
    return paths


def write_bundle(out_dir: Path, bundle: AlignedOfflineInputs) -> dict[str, str]:
    paths = {
        "probe_features": str(out_dir / "aligned_probe_features.parquet"),
        "state_targets": str(out_dir / "aligned_state_targets.csv"),
        "query_features": str(out_dir / "aligned_query_features.csv"),
        "before_beliefs": str(out_dir / "aligned_before_beliefs.csv"),
        "after_beliefs": str(out_dir / "aligned_after_beliefs.csv"),
        "state_model_utility": str(out_dir / "aligned_state_model_utility.csv"),
        "query_model_utility": str(out_dir / "aligned_query_model_utility.csv"),
        "probe_cost": str(out_dir / "aligned_probe_cost.csv"),
        "predicted_gain": str(out_dir / "aligned_predicted_gain.csv"),
    }
    bundle.probe_features.to_parquet(paths["probe_features"], index=False)
    bundle.state_targets.to_csv(paths["state_targets"], index=False)
    bundle.query_features.to_csv(paths["query_features"], index=False)
    bundle.before_beliefs.reset_index().rename(columns={bundle.before_beliefs.index.name or "index": "query_id"}).to_csv(
        paths["before_beliefs"],
        index=False,
    )
    bundle.after_beliefs.reset_index().rename(columns={bundle.after_beliefs.index.name or "index": "query_id"}).to_csv(
        paths["after_beliefs"],
        index=False,
    )
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
    return paths


def write_memo(out_dir: Path, config_path: str, paths: dict[str, str], bundle: AlignedOfflineInputs) -> None:
    lines = [
        "# Aligned Offline Probe Inputs",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/56_aligned_offline_probe_inputs.py --config {config_path} --output-dir {out_dir}",
        "```",
        "",
        "These files are aligned benchmark-derived scaffolding for M4/M5. They use train-only kNN uncertainty as an offline probe feature; they are not true local model probe evidence.",
        "",
        "Summary:",
        "",
        f"- Probe rows: `{len(bundle.probe_features)}`.",
        f"- State-target rows: `{len(bundle.state_targets)}`.",
        f"- Policy test queries: `{len(bundle.before_beliefs)}`.",
        "",
        "Files:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    (out_dir / "m7_aligned_offline_probe_inputs_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_phase2_readme(phase2_dir: Path, config_path: str, paths: dict[str, str]) -> None:
    phase2_dir.mkdir(parents=True, exist_ok=True)
    readme_path = phase2_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Aligned Offline Inputs"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/56_aligned_offline_probe_inputs.py --config {config_path} --output-dir {Path(paths['probe_features']).parent}",
        "```",
        "",
        "These artifacts make M4/M5 executable on aligned benchmark-derived route states. They are offline scaffolding, not true local probe evidence.",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _phase2_readme_dir(out_dir: Path) -> Path:
    if out_dir.name == "aligned_offline" and out_dir.parent.name == "phase2":
        return out_dir.parent
    return out_dir


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
