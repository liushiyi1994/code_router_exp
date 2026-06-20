from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.probes.probe_features import build_probe_features_from_outcomes
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outcomes", default="results/phase2/local_model_outcomes.parquet")
    parser.add_argument("--output-dir", default="results/phase2")
    args = parser.parse_args()
    run(outcomes_path=args.outcomes, output_dir=args.output_dir)


def run(*, outcomes_path: str, output_dir: str) -> pd.DataFrame:
    outcomes = pd.read_parquet(outcomes_path)
    features = build_probe_features_from_outcomes(outcomes)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_path = out_dir / "probe_features.parquet"
    features.to_parquet(features_path, index=False)
    write_memo(out_dir, outcomes_path, features)
    append_readme(out_dir, outcomes_path, features)
    print(f"Wrote Phase 2 probe features to {features_path}")
    return features


def write_memo(out_dir: Path, outcomes_path: str, features: pd.DataFrame) -> None:
    lines = [
        "# Phase 2 Probe Feature Collection",
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/52_probe_collection.py --outcomes {outcomes_path} --output-dir {out_dir}",
        "```",
        "",
        "This M3 step collects generic cheap-probe observations from local model outputs. "
        "It is not evidence that probes close the observability gap.",
        "",
        "Outputs:",
        "",
        "- `probe_features.parquet`: one probe-observation row per local outcome row.",
        "- `m3_probe_collection_memo.md`: this memo.",
        "",
        "Summary:",
        "",
        _markdown_table(_summary(features)),
        "",
        "Notes:",
        "",
        "- `local_answer_probe` rows reuse the M2 local outcome generations as cheap probe observations.",
        "- `logprob_mean` and `entropy_proxy` are null unless the serving backend exposes those values.",
        "- `knn_label_entropy` and `knn_winner_entropy` are null unless aligned train-only state embeddings were supplied.",
        "- These features must be used to update beliefs over latent route states, not to map directly from probe output to final model.",
        "",
    ]
    (out_dir / "m3_probe_collection_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, outcomes_path: str, features: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Probe Features"
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/52_probe_collection.py --outcomes {outcomes_path} --output-dir {out_dir}",
        "```",
        "",
        "This writes `probe_features.parquet` from local cheap-probe outputs without external API calls. "
        "These rows validate the probe-feature schema and logging path; probe usefulness is evaluated separately.",
        "",
        "Outputs:",
        "",
        "- `probe_features.parquet`",
        "- `m3_probe_collection_memo.md`",
        "",
        _markdown_table(_summary(features)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    return (
        features.groupby(["probe_type", "probe_model_id"], as_index=False)
        .agg(
            rows=("query_id", "count"),
            unique_queries=("query_id", "nunique"),
            mean_agreement=("agreement_score", "mean"),
            mean_probe_cost_proxy=("probe_cost_proxy", "mean"),
            errors=("error_type", lambda values: int((values.astype(str) != "").sum())),
        )
        .sort_values(["probe_type", "probe_model_id"])
    )


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
