from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.codes.code_cards import build_code_cards
from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.codes.routecode import RouteCodeCodebook
from routecode.config import load_config, output_dir
from routecode.eval.code_card_interpretability import summarize_code_card_interpretability
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = output_dir(config)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    table = build_interpretability_table(config, train.query_info, train.utility, prepared.embeddings)
    table.to_csv(out_dir / "table_code_card_interpretability.csv", index=False)
    write_memo(out_dir, args.config, table)
    append_readme(out_dir, args.config, table)
    print(f"Wrote code-card interpretability outputs to {out_dir}")


def build_interpretability_table(
    config: dict,
    query_info: pd.DataFrame,
    utility: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> pd.DataFrame:
    seed = int(config.get("run", {}).get("random_seed", 0))
    route_config = config.get("routecode", {})
    d2_config = config.get("predictability_constrained", {})
    max_examples = int(config.get("code_cards", {}).get("max_examples", 4))

    flat_k = int(route_config.get("selected_k_for_cards", 16))
    flat = RouteCodeCodebook(
        flat_k,
        random_state=seed,
        max_iter=int(route_config.get("max_iter", 25)),
    ).fit(query_info, utility, embeddings)
    flat_cards = build_code_cards(flat, query_info, utility, max_examples=max_examples)
    flat_summary = summarize_code_card_interpretability("flat_routecode", flat, flat_cards)
    flat_summary["K"] = flat_k
    flat_summary["alpha"] = ""
    flat_summary["beta"] = ""

    d2_k = int(d2_config.get("k", flat_k))
    d2_alpha = float(d2_config.get("selected_alpha", 3.0))
    d2_beta = float(d2_config.get("beta", 0.0))
    d2 = PredictabilityConstrainedRouteCode(
        d2_k,
        alpha=d2_alpha,
        beta=d2_beta,
        random_state=seed,
        max_iter=int(d2_config.get("max_iter", route_config.get("max_iter", 25))),
        refinement_iter=int(d2_config.get("refinement_iter", 10)),
    ).fit(query_info, utility, embeddings)
    d2_cards = build_code_cards(d2, query_info, utility, max_examples=max_examples)
    d2_summary = summarize_code_card_interpretability("predictability_constrained_routecode", d2, d2_cards)
    d2_summary["K"] = d2_k
    d2_summary["alpha"] = d2_alpha
    d2_summary["beta"] = d2_beta

    table = pd.concat([flat_summary, d2_summary], ignore_index=True)
    columns = ["codebook", "K", "alpha", "beta", "condition"] + [
        column for column in table.columns if column not in {"codebook", "K", "alpha", "beta", "condition"}
    ]
    return table[columns]


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    with_cards = table[table["condition"] == "with_code_cards"].copy()
    label_only = table[table["condition"] == "label_only"].copy()
    lines = [
        "# Phase F Code-Card Interpretability Memo",
        "",
        f"Command: `python experiments/11_code_card_interpretability.py --config {config_path}`",
        "",
        "This is an observability ablation, not a routing-utility result. It compares what is inspectable from a route-label lookup table alone against what is inspectable after generating code cards from train-set labels and utility profiles.",
        "",
        "## Summary Table",
        "",
        _markdown_table(
            table[
                [
                    "codebook",
                    "condition",
                    "n_labels",
                    "available_explainability_fields",
                    "best_model_coverage",
                    "domain_summary_coverage",
                    "dataset_summary_coverage",
                    "representative_query_coverage",
                    "failure_case_coverage",
                    "utility_vector_coverage",
                    "human_explanation_coverage",
                ]
            ]
        ),
        "",
        "## Current Readout",
        "",
    ]
    if with_cards.empty or label_only.empty:
        lines.append("- Interpretability rows were not produced for both conditions.")
    else:
        min_human = float(with_cards["human_explanation_coverage"].min())
        min_representatives = float(with_cards["representative_query_coverage"].min())
        min_failures = float(with_cards["failure_case_coverage"].min())
        max_label_only_fields = int(label_only["available_explainability_fields"].max())
        min_card_fields = int(with_cards["available_explainability_fields"].min())
        lines.extend(
            [
                f"- Label-only rows expose at most `{max_label_only_fields}` explanatory field in this audit.",
                f"- Code-card rows expose at least `{min_card_fields}` explanatory fields across the tested codebooks.",
                f"- Minimum code-card coverage: human explanations `{min_human:.4f}`, representative queries `{min_representatives:.4f}`, and high-regret examples `{min_failures:.4f}`.",
            ]
        )
    lines.extend(
        [
            "- The result supports treating code cards as an explainability and diagnosis layer. It does not show that code cards improve routing utility, because model selection is unchanged by this audit.",
            "",
        ]
    )
    (out_dir / "phase_f_code_card_interpretability_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## Code-Card Interpretability"
    compact = table[
        [
            "codebook",
            "condition",
            "available_explainability_fields",
            "representative_query_coverage",
            "failure_case_coverage",
            "human_explanation_coverage",
        ]
    ]
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/11_code_card_interpretability.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_code_card_interpretability.csv`: label-only versus code-card observability coverage for flat and D2 RouteCode.",
        "- `phase_f_code_card_interpretability_memo.md`: Phase F memo for the code-card interpretability ablation.",
        "",
        _markdown_table(compact),
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
        values = [_format_cell(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
