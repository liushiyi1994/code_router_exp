from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.eval.residuals import (
    label_residual_summary,
    residual_concentration_table,
    residual_query_table,
    residual_risk_coverage_table,
)
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.plots import save_residual_concentration, save_risk_coverage
from routecode.reporting import upsert_markdown_section
from routecode.predictors.classifiers import RouteCodeLabelClassifier
from routecode.routers.knn import KNNRouter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    seed = int(config.get("run", {}).get("random_seed", 0))
    route_config = config.get("routecode", {})
    router_config = config.get("routers", {})
    route_k = int(route_config.get("selected_k_for_cards", 16))

    codebook = RouteCodeCodebook(
        route_k,
        random_state=seed,
        max_iter=int(route_config.get("max_iter", 25)),
    ).fit(train.query_info, train.utility, embeddings)
    classifier = RouteCodeLabelClassifier(random_state=seed).fit(codebook, embeddings)
    test_embeddings = embeddings.loc[test.utility.index]
    labels = classifier.predict_labels(test_embeddings)
    confidence = classifier.predict_confidence(test_embeddings)
    selected = classifier.predict(test.query_info, embeddings)

    residuals = residual_query_table(test.utility, selected, labels, test_embeddings)
    residuals["route_label_confidence"] = confidence
    residuals["low_route_label_confidence"] = 1.0 - confidence
    knn_selected = KNNRouter(int(router_config.get("knn_k", 15))).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    residuals["knn_selected_model"] = knn_selected
    residuals["knn_disagreement"] = (knn_selected != selected).astype(int)
    residuals["centroid_distance_risk"] = residuals["distance_to_label_centroid"]
    residuals["oracle_margin_risk_diagnostic"] = residuals["oracle_margin"]
    residuals["knn_disagreement_plus_distance"] = residuals["knn_disagreement"] + residuals[
        "distance_to_label_centroid"
    ].rank(pct=True)
    residuals = residuals.join(test.query_info[["query_text", "dataset", "domain"]], how="left")

    concentration = residual_concentration_table(
        test.utility.max(axis=1),
        selected_values(test.utility, selected),
        fractions=[0.05, 0.10, 0.20],
    )
    risk = residual_risk_coverage_table(
        residuals,
        score_columns=[
            "low_route_label_confidence",
            "centroid_distance_risk",
            "knn_disagreement",
            "knn_disagreement_plus_distance",
            "oracle_margin_risk_diagnostic",
        ],
        top_fractions=[0.05, 0.10, 0.20],
    )
    concentration.to_csv(out_dir / "table_residual_concentration.csv", index=False)
    risk.to_csv(out_dir / "table_residual_risk.csv", index=False)
    residuals.sort_values("regret", ascending=False).to_csv(out_dir / "table_residual_queries.csv")
    label_residual_summary(residuals).to_csv(out_dir / "table_residual_by_label.csv", index=False)
    save_residual_concentration(concentration, out_dir / "fig_residual_concentration.pdf")
    save_risk_coverage(risk, out_dir / "fig_risk_coverage.pdf")
    write_gate_memo(out_dir, args.config, concentration, risk)
    append_readme(out_dir, args.config, concentration, risk)
    print(f"Wrote residual concentration outputs to {out_dir}")


def write_gate_memo(out_dir: Path, config_path: str, concentration: pd.DataFrame, risk: pd.DataFrame) -> None:
    best_deployable = (
        risk[~risk["score"].astype(str).str.contains("diagnostic")]
        .sort_values(["top_fraction", "regret_mass_fraction"], ascending=[True, False])
        .groupby("top_fraction", as_index=False)
        .head(1)
    )
    lines = [
        "# Phase D5 Adaptive-Refinement Gate Memo",
        "",
        f"Command: `python experiments/03_residual_concentration.py --config {config_path}`",
        "",
        "This memo checks whether residual failures are concentrated and predictable enough to justify implementing adaptive refinement. It is a gate, not an adaptive-refinement result.",
        "",
        "## Residual Concentration",
        "",
        _markdown_table(concentration),
        "",
        "## Best Deployable Risk Signals",
        "",
        _markdown_table(
            best_deployable[
                [
                    "score",
                    "top_fraction",
                    "n_flagged",
                    "regret_mass_fraction",
                    "positive_regret_recall",
                    "auc_regret_positive",
                ]
            ]
        ),
        "",
        "## Current Decision",
        "",
    ]
    top_5 = best_deployable[best_deployable["top_fraction"] == 0.05]
    top_10 = best_deployable[best_deployable["top_fraction"] == 0.10]
    if top_5.empty or top_10.empty:
        lines.append("- No deployable risk signal was available, so adaptive refinement is not justified yet.")
    else:
        top_5_mass = float(top_5.iloc[0]["regret_mass_fraction"])
        top_10_mass = float(top_10.iloc[0]["regret_mass_fraction"])
        lines.append(
            f"- Best deployable signals capture `{top_5_mass:.4f}` of regret in the top 5% and `{top_10_mass:.4f}` in the top 10% of flagged queries."
        )
        if top_5_mass >= 0.50 or top_10_mass >= 0.70:
            lines.append("- Adaptive refinement is worth a small follow-up experiment, but it is not yet a main claim.")
        else:
            lines.append("- Adaptive refinement should remain deferred; the current gate is not strong enough for a core claim.")
    lines.extend(
        [
            "- The oracle-margin diagnostic is included to show an upper-bound-style non-deployable signal; it should not be treated as a deployable trigger.",
            "",
        ]
    )
    (out_dir / "phase_d5_adaptive_refinement_gate_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, concentration: pd.DataFrame, risk: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Residual Concentration"
    risk_compact = risk[
        (risk["top_fraction"].isin([0.05, 0.10]))
        & (~risk["score"].astype(str).str.contains("diagnostic"))
    ][["score", "top_fraction", "regret_mass_fraction", "positive_regret_recall", "auc_regret_positive"]]
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/03_residual_concentration.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_residual_concentration.csv`: fraction of residual regret captured by top-regret queries.",
        "- `table_residual_risk.csv`: regret capture and AUC for deployable residual-risk signals.",
        "- `table_residual_queries.csv`: per-query regret, margin, label, confidence, centroid distance, and kNN disagreement.",
        "- `table_residual_by_label.csv`: per-label residual summary.",
        "- `fig_residual_concentration.pdf`: residual concentration curve.",
        "- `fig_risk_coverage.pdf`: regret mass captured by top-risk query fractions.",
        "- `phase_d5_adaptive_refinement_gate_memo.md`: gate memo for whether adaptive refinement is justified.",
        "",
        "Residual concentration:",
        "",
        _markdown_table(concentration),
        "",
        "Deployable risk coverage:",
        "",
        _markdown_table(risk_compact),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
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
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
