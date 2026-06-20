from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from routecode.config import load_config
from routecode.eval.transformer_backbones import DEFAULT_REQUESTED_BACKBONES, inspect_transformer_backbone_cache
from routecode.eval.transformer_embedding_router import extract_local_transformer_embeddings
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.states.observability import compute_observability_gap_for_result_dir
from routecode.states.strong_encoders import evaluate_strong_encoder_state_observability


DEFAULT_RESULT_DIRS = [
    Path("results/llmrouterbench_pilot"),
    Path("results/llmrouterbench_broad10"),
    Path("results/llmrouterbench_broad20"),
    Path("results/llmrouterbench_scale20"),
    Path("results/llmrouterbench_32model"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", action="append", type=Path, default=[])
    parser.add_argument("--config", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase2"))
    args = parser.parse_args()
    result_dirs = args.result_dir or [path for path in DEFAULT_RESULT_DIRS if path.exists()]
    run(result_dirs, args.output_dir, config_paths=args.config)


def run(
    result_dirs: list[str | Path],
    output_dir: str | Path,
    config_paths: list[str | Path] | None = None,
    strong_readiness_table: pd.DataFrame | None = None,
    strong_embedding_provider=None,
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    skipped: list[tuple[Path, str]] = []
    for result_dir in result_dirs:
        result_path = Path(result_dir)
        try:
            frames.append(compute_observability_gap_for_result_dir(result_path))
        except FileNotFoundError as exc:
            skipped.append((result_path, str(exc)))
    strong_frames = _strong_encoder_frames(
        config_paths or [],
        readiness_table_override=strong_readiness_table,
        embedding_provider_override=strong_embedding_provider,
    )
    frames.extend(strong_frames)

    table = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_table()
    if not table.empty:
        table = table.sort_values(
            ["result_id", "state_family", "alpha", "comparison"],
            na_position="last",
        ).reset_index(drop=True)
    table.to_csv(out_dir / "table_observability_strong_encoders.csv", index=False)
    _save_observability_gap_figure(table, out_dir / "fig_observability_gap.pdf")
    write_memo(out_dir, result_dirs, config_paths or [], skipped, table)
    append_readme(out_dir, result_dirs, config_paths or [], table)
    print(f"Wrote Phase 2 M0 observability outputs to {out_dir}")
    return table


def write_memo(
    out_dir: Path,
    result_dirs: list[str | Path],
    config_paths: list[str | Path],
    skipped: list[tuple[Path, str]],
    table: pd.DataFrame,
) -> None:
    strong_rows = _strong_rows(table)
    lines = [
        "# Phase 2 M0 Previous Findings Recap",
        "",
        "Command:",
        "",
        "```bash",
        _command(result_dirs, out_dir, config_paths),
        "```",
        "",
        "Purpose: restate the Phase 1 oracle-vs-deployable result as a partial-observability problem before adding probes or running local models.",
        "",
        _strong_encoder_status_sentence(strong_rows),
        "",
        "Inputs:",
        "",
        *[f"- `{Path(path)}`" for path in result_dirs],
        "",
    ]
    if config_paths:
        lines.extend(
            [
                "Strong encoder configs:",
                "",
                *[f"- `{Path(path)}`" for path in config_paths],
                "",
            ]
        )
    if skipped:
        lines.extend(
            [
                "Skipped inputs:",
                "",
                *[f"- `{path}`: {reason}" for path, reason in skipped],
                "",
            ]
        )
    lines.extend(
        [
            "## Main Observability Rows",
            "",
            _markdown_table(_display_rows(table)),
            "",
            "## Strong Encoder Rows",
            "",
            _markdown_table(_display_strong_rows(strong_rows)),
            "",
            "## Observations",
            "",
            *_observations(table),
            "",
            "## Outputs",
            "",
            "- `table_observability_strong_encoders.csv`: route-state oracle vs deployable query-to-state comparisons.",
            "- `fig_observability_gap.pdf`: visual summary of state and query-oracle gaps.",
            "- `m0_previous_findings_recap.md`: this recap.",
            "",
        ]
    )
    (out_dir / "m0_previous_findings_recap.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, result_dirs: list[str | Path], config_paths: list[str | Path], table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Phase 2 Observability Gap"
    strong_rows = _strong_rows(table)
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        _command(result_dirs, out_dir, config_paths),
        "```",
        "",
        "This is the M0 bridge from Phase 1 to Phase 2. It does not run probes, APIs, or local generative models. It quantifies the current observability gap using saved Phase 1 tables.",
        "",
        _strong_encoder_status_sentence(strong_rows),
        "",
        "Outputs:",
        "",
        "- `table_observability_strong_encoders.csv`",
        "- `fig_observability_gap.pdf`",
        "- `m0_previous_findings_recap.md`",
        "",
        _markdown_table(_display_rows(table)),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _strong_encoder_frames(
    config_paths: list[str | Path],
    readiness_table_override: pd.DataFrame | None = None,
    embedding_provider_override=None,
) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for config_path in config_paths:
        config = load_config(config_path)
        prepared = prepare_from_config(config)
        train = prepared.matrices["train"]
        test = prepared.matrices["test"]
        seed = int(config.get("run", {}).get("random_seed", 0))
        bootstrap = config.get("bootstrap", {})
        phase2_config = config.get("phase2_observability", {})
        backbone_config = config.get("transformer_backbones", {})
        router_config = config.get("transformer_embedding_router", {})
        requested = list(backbone_config.get("requested_model_ids", DEFAULT_REQUESTED_BACKBONES))
        cache_dir = Path(backbone_config.get("cache_dir", "~/.cache/huggingface/hub")).expanduser()
        readiness = (
            readiness_table_override.copy()
            if readiness_table_override is not None
            else inspect_transformer_backbone_cache(
                cache_dir,
                requested_model_ids=requested,
                max_runnable_gb=float(backbone_config.get("max_runnable_gb", 2.0)),
            )
        )
        requested_set = {str(model_id) for model_id in requested}
        if "model_id" in readiness.columns:
            readiness = readiness[readiness["model_id"].astype(str).isin(requested_set)].copy()

        def provider(row: pd.Series, query_info: pd.DataFrame) -> pd.DataFrame:
            if embedding_provider_override is not None:
                return embedding_provider_override(row, query_info)
            return extract_local_transformer_embeddings(
                local_path=str(row["local_path"]),
                query_info=query_info,
                batch_size=int(router_config.get("batch_size", phase2_config.get("batch_size", 16))),
                max_length=int(router_config.get("max_length", phase2_config.get("max_length", 256))),
                device=str(router_config.get("device", phase2_config.get("device", "auto"))),
            )

        table = evaluate_strong_encoder_state_observability(
            train=train,
            test=test,
            readiness_table=readiness,
            embedding_provider=provider,
            k=int(phase2_config.get("k", config.get("predictability_constrained", {}).get("k", 16))),
            alpha=float(phase2_config.get("alpha", config.get("predictability_constrained", {}).get("selected_alpha", 3.0))),
            beta=float(phase2_config.get("beta", config.get("predictability_constrained", {}).get("beta", 0.0))),
            state_families=[str(item) for item in phase2_config.get("state_families", ["flat_routecode", "d2"])],
            predictors=[str(item) for item in phase2_config.get("state_predictors", ["centroid", "knn", "logistic", "mlp"])],
            random_state=seed,
            n_bootstrap=int(bootstrap.get("n_bootstrap", 100)),
            ci=float(bootstrap.get("ci", 0.95)),
            n_neighbors=int(phase2_config.get("knn_k", router_config.get("knn_k", config.get("routers", {}).get("knn_k", 15)))),
            max_iter=int(phase2_config.get("max_iter", router_config.get("max_iter", 200))),
            refinement_iter=int(
                phase2_config.get("refinement_iter", config.get("predictability_constrained", {}).get("refinement_iter", 10))
            ),
        )
        table["result_id"] = str(config.get("run", {}).get("name") or Path(config.get("run", {}).get("output_dir", Path(config_path).stem)).name)
        table["config_path"] = str(config_path)
        frames.append(table)
    return frames


def _save_observability_gap_figure(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No observability rows", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return

    plot_table = table.dropna(subset=["query_oracle_gap", "state_observability_gap"]).copy()
    if plot_table.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "No metric observability rows", ha="center", va="center")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return
    plot_table["label"] = plot_table["result_id"].astype(str) + "\n" + plot_table["comparison"].astype(str)
    plot_table = plot_table.sort_values("query_oracle_gap", ascending=True).tail(16)
    y_pos = range(len(plot_table))

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(plot_table))))
    ax.barh(
        list(y_pos),
        plot_table["query_oracle_gap"],
        color="#E45756",
        alpha=0.35,
        label="query oracle - deployable",
    )
    ax.barh(
        list(y_pos),
        plot_table["state_observability_gap"],
        color="#4C78A8",
        alpha=0.85,
        label="state oracle - deployable",
    )
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(plot_table["label"])
    ax.set_xlabel("Mean utility gap")
    ax.set_title("Phase 2 M0 Observability Gap From Phase 1 Tables")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _observations(table: pd.DataFrame) -> list[str]:
    if table.empty:
        return ["- No complete Phase 1 result directories were available for M0."]
    rows = []
    flat = table[table["comparison"] == "flat_routecode_logistic_label_predictor"].copy()
    if not flat.empty:
        worst_flat = flat.sort_values("state_observability_gap", ascending=False).iloc[0]
        rows.append(
            "- Flat utility-oracle labels remain a diagnostic upper bound: "
            f"`{worst_flat['result_id']}` has state-observability gap "
            f"{worst_flat['state_observability_gap']:.4f} and full query-oracle gap "
            f"{worst_flat['query_oracle_gap']:.4f}."
        )
    deployable = table.sort_values("full_gap_closed_vs_query_oracle", ascending=False).iloc[0]
    rows.append(
        "- Best current deployable state assignment in this M0 table: "
        f"`{deployable['result_id']}` / `{deployable['comparison']}` with "
        f"{deployable['full_gap_closed_vs_query_oracle']:.4f} of the best-single-to-query-oracle gap recovered."
    )
    strong = _strong_rows(table)
    executed_strong = strong[strong["strong_encoder_status"].eq("executed")] if not strong.empty else pd.DataFrame()
    if not executed_strong.empty:
        best_strong = executed_strong.sort_values("full_gap_closed_vs_query_oracle", ascending=False).iloc[0]
        rows.append(
            "- Best executed strong-encoder state predictor: "
            f"`{best_strong['result_id']}` / `{best_strong['model_id']}` / `{best_strong['comparison']}` with "
            f"label accuracy `{float(best_strong['label_accuracy']):.4f}` and recovered query-oracle gap "
            f"`{float(best_strong['full_gap_closed_vs_query_oracle']):.4f}`."
        )
    positive_gaps = table[table["state_observability_gap"] > 0.01]
    if not positive_gaps.empty:
        rows.append(
            "- The Phase 1 evidence supports the Phase 2 premise: useful route states can exist while the query-only assignment remains partially observed."
        )
    if executed_strong.empty:
        rows.append(
            "- This is not evidence that probes or strong encoders close the gap; those rows must be added by later Phase 2 experiments."
        )
    else:
        rows.append(
            "- This is still not evidence that probes close the gap; probe rows and cost-adjusted policies must be added by later Phase 2 experiments."
        )
    return rows


def _display_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    columns = [
        "result_id",
        "comparison",
        "K",
        "alpha",
        "oracle_state_mean_utility",
        "deployable_state_mean_utility",
        "state_observability_gap",
        "state_observability_gap_ci_low",
        "state_observability_gap_ci_high",
        "query_oracle_gap",
        "query_oracle_gap_ci_low",
        "query_oracle_gap_ci_high",
        "full_gap_closed_vs_query_oracle",
        "strong_encoder_status",
    ]
    return table[columns].copy()


def _display_strong_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    columns = [
        "result_id",
        "model_id",
        "state_family",
        "state_predictor",
        "status",
        "label_accuracy",
        "deployable_state_mean_utility",
        "deployable_state_mean_utility_ci_low",
        "deployable_state_mean_utility_ci_high",
        "state_observability_gap",
        "state_observability_gap_ci_low",
        "state_observability_gap_ci_high",
        "full_gap_closed_vs_query_oracle",
        "routing_invariant",
        "reason",
    ]
    existing = [column for column in columns if column in table.columns]
    return table[existing].copy()


def _strong_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty or "evidence_source" not in table.columns:
        return pd.DataFrame()
    return table[table["evidence_source"].eq("phase2_strong_encoder_state_predictor")].copy()


def _strong_encoder_status_sentence(strong_rows: pd.DataFrame) -> str:
    if strong_rows.empty:
        return "Strong encoders have not been run by this script. The table name follows the Phase 2 deliverable contract, but M0 rows are derived from existing Phase 1 result tables and mark `strong_encoder_status=not_run_in_m0`."
    counts = strong_rows["strong_encoder_status"].astype(str).value_counts().sort_index()
    count_text = ", ".join(f"{status}={count}" for status, count in counts.items())
    return (
        "Strong encoder rows were produced for M1 using cached/local encoder paths only. "
        f"Status counts: `{count_text}`. These rows preserve the invariant `query -> state -> model`."
    )


def _empty_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "result_id",
            "comparison",
            "state_family",
            "oracle_state_method",
            "deployable_state_method",
            "K",
            "alpha",
            "encoder_family",
            "strong_encoder_status",
            "query_oracle_mean_utility",
            "best_single_mean_utility",
            "oracle_state_mean_utility",
            "deployable_state_mean_utility",
            "state_observability_gap",
            "query_oracle_gap",
            "state_oracle_gap_vs_best_single",
            "deployable_gap_vs_best_single",
            "state_gap_closed",
            "full_gap_closed_vs_query_oracle",
            "oracle_state_recovered_gap_vs_oracle",
            "deployable_recovered_gap_vs_oracle",
            "label_accuracy",
            "mean_confidence",
            "evidence_source",
            "interpretation",
            "status",
            "reason",
            "model_id",
            "state_predictor",
            "routing_invariant",
            "config_path",
        ]
    )


def _command(result_dirs: list[str | Path], out_dir: Path, config_paths: list[str | Path] | None = None) -> str:
    parts = ["python experiments/50_observability_gap_strong_encoders.py"]
    for result_dir in result_dirs:
        parts.extend(["--result-dir", str(Path(result_dir))])
    for config_path in config_paths or []:
        parts.extend(["--config", str(Path(config_path))])
    parts.extend(["--output-dir", str(out_dir)])
    return " ".join(parts)


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
