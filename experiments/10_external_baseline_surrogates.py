from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.config import load_config, output_dir
from routecode.eval.evaluate import evaluate_selection
from routecode.eval.external_baselines import choose_strong_weak_pair
from routecode.metrics import selected_values
from routecode.pipeline import prepare_from_config
from routecode.reporting import upsert_markdown_section
from routecode.routers.knn import KNNRouter
from routecode.routers.matrix_factorization import BinaryThresholdRouter, MatrixFactorizationRouter
from routecode.routers.oracle import OracleRouter
from routecode.routers.single_best import BestSingleRouter


ROUTELLM_REPO = "https://github.com/lm-sys/routellm"
LLMROUTER_REPO = "https://github.com/ulab-uiuc/LLMRouter"
LLMROUTERBENCH_REPO = "https://github.com/ynulihao/LLMRouterBench"


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
    bootstrap = config.get("bootstrap", {})
    n_bootstrap = int(bootstrap.get("n_bootstrap", 300))
    ci = float(bootstrap.get("ci", 0.95))
    baseline_config = config.get("external_baselines", {})
    thresholds = [float(value) for value in baseline_config.get("thresholds", [0.25, 0.5, 0.75])]
    pair = choose_strong_weak_pair(
        train.utility,
        strong_model=baseline_config.get("strong_model"),
        weak_model=baseline_config.get("weak_model"),
    )

    best_single = BestSingleRouter().fit(train.query_info, train.utility).predict(test.query_info)
    baseline_mean = float(selected_values(test.utility, best_single).mean())
    oracle_mean = float(test.utility.max(axis=1).mean())
    knn = KNNRouter(int(config.get("routers", {}).get("knn_k", 15))).fit(
        train.query_info,
        train.utility,
        embeddings,
    ).predict(test.query_info, embeddings)
    learned_reference_mean = max(baseline_mean, float(selected_values(test.utility, knn).mean()))

    rows: list[dict[str, Any]] = []
    rows.append(
        _row(
            "best_single",
            best_single,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed,
            baseline_family="reference",
        )
    )
    rows.append(
        _row(
            "kNN",
            knn,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 1,
            baseline_family="reference",
        )
    )
    oracle = OracleRouter().predict(test.utility)
    rows.append(
        _row(
            "query_oracle",
            oracle,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 2,
            baseline_family="reference",
        )
    )

    mf_router = MatrixFactorizationRouter(
        rank=int(baseline_config.get("mf_rank", 4)),
        alpha=float(baseline_config.get("mf_alpha", 1.0)),
        random_state=seed,
    ).fit(train.query_info, train.utility, embeddings)
    mf_selected = mf_router.predict(test.query_info, embeddings)
    rows.append(
        _row(
            "routellm_style_mf_utility_router",
            mf_selected,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 3,
            baseline_family="external_style_surrogate",
            implementation_note=(
                "Local low-rank utility-factor router inspired by RouteLLM/EmbedLLM MF; "
                "not an official RouteLLM checkpoint."
            ),
        )
    )

    strong_only = pd.Series(pair.strong_model, index=test.utility.index, name="selected_model")
    weak_only = pd.Series(pair.weak_model, index=test.utility.index, name="selected_model")
    rows.append(
        _row(
            "routellm_pair_strong_only",
            strong_only,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 4,
            baseline_family="binary_pair_reference",
            strong_model=pair.strong_model,
            weak_model=pair.weak_model,
            k=2,
            labels=strong_only,
        )
    )
    rows.append(
        _row(
            "routellm_pair_weak_only",
            weak_only,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 5,
            baseline_family="binary_pair_reference",
            strong_model=pair.strong_model,
            weak_model=pair.weak_model,
            k=2,
            labels=weak_only,
        )
    )
    for threshold_index, threshold in enumerate(thresholds):
        binary_router = BinaryThresholdRouter(
            pair.strong_model,
            pair.weak_model,
            threshold=threshold,
            random_state=seed,
        ).fit(train.query_info, train.utility, embeddings)
        selected = binary_router.predict(test.query_info, embeddings)
        row = _row(
            f"routellm_binary_logistic_surrogate_t{threshold:g}",
            selected,
            test,
            baseline_mean,
            learned_reference_mean,
            oracle_mean,
            n_bootstrap,
            ci,
            seed + 10 + threshold_index,
            baseline_family="external_style_surrogate",
            strong_model=pair.strong_model,
            weak_model=pair.weak_model,
            threshold=threshold,
            k=2,
            labels=selected,
            implementation_note=(
                "RouteLLM-style strong/weak threshold router using local embeddings and "
                "logistic strong-win prediction; not official RouteLLM MF/BERT."
            ),
        )
        row["strong_selection_rate"] = float((selected == pair.strong_model).mean())
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_external_baselines.csv", index=False)
    write_memo(out_dir, args.config, table, pair)
    append_readme(out_dir, args.config, table)
    print(f"Wrote external baseline surrogate outputs to {out_dir}")


def _row(
    method: str,
    selected_models: pd.Series,
    test,
    baseline_mean: float,
    learned_reference_mean: float,
    oracle_mean: float,
    n_bootstrap: int,
    ci: float,
    seed: int,
    baseline_family: str,
    strong_model: str = "",
    weak_model: str = "",
    threshold: float | str = "",
    k: int | None = None,
    labels: pd.Series | None = None,
    implementation_note: str = "",
) -> dict[str, Any]:
    row = evaluate_selection(
        method=method,
        selected_models=selected_models,
        matrices=test,
        baseline_mean=baseline_mean,
        learned_reference_mean=learned_reference_mean,
        oracle_mean=oracle_mean,
        n_bootstrap=n_bootstrap,
        ci=ci,
        seed=seed,
        k=k,
        labels=labels,
    )
    row.update(
        {
            "baseline_family": baseline_family,
            "strong_model": strong_model,
            "weak_model": weak_model,
            "threshold": threshold,
            "paper_reference": "RouteLLM / LLMRouter / LLMRouterBench baseline ecosystem",
            "repo_reference": f"{ROUTELLM_REPO}; {LLMROUTER_REPO}; {LLMROUTERBENCH_REPO}",
            "implementation_note": implementation_note,
        }
    )
    return row


def append_readme(out_dir: Path, config_path: str, table: pd.DataFrame) -> None:
    readme_path = out_dir / "README.md"
    if not readme_path.exists():
        return
    existing = readme_path.read_text(encoding="utf-8")
    marker = "## External Baseline Surrogates"
    summary = table.sort_values("mean_utility", ascending=False)[
        ["method", "baseline_family", "mean_utility", "recovered_gap_vs_oracle"]
    ]
    lines = [
        marker,
        "",
        "Command:",
        "",
        "```bash",
        f"python experiments/10_external_baseline_surrogates.py --config {config_path}",
        "```",
        "",
        "Outputs:",
        "",
        "- `table_external_baselines.csv`: local external-style baseline surrogate rows with explicit implementation notes.",
        "- `phase_e_external_baseline_memo.md`: Phase E checkpoint memo for these surrogate baselines.",
        "",
        _markdown_table(summary),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def write_memo(out_dir: Path, config_path: str, table: pd.DataFrame, pair) -> None:
    lines = [
        "# Phase E External Baseline Surrogate Memo",
        "",
        f"Command: `python experiments/10_external_baseline_surrogates.py --config {config_path}`",
        "",
        f"Binary pair: strong `{pair.strong_model}`, weak `{pair.weak_model}`.",
        "",
        "These rows are local surrogates inspired by RouteLLM/LLMRouter-style baselines. They are not official external-repo reproductions.",
        "",
        "Official RouteLLM/LLMRouterBench RouteLLM inspection found that the upstream adapter expects its own embedding/checkpoint pipeline; this run keeps the no-API local RouteCode split and deterministic embeddings.",
        "",
        _markdown_table(
            table.sort_values("mean_utility", ascending=False)[
                ["method", "baseline_family", "mean_utility", "oracle_regret", "recovered_gap_vs_oracle"]
            ]
        ),
        "",
        "## References Used",
        "",
        f"- RouteLLM paper/repo: https://arxiv.org/abs/2406.18665 ; {ROUTELLM_REPO}",
        f"- LLMRouter repo: {LLMROUTER_REPO}",
        f"- LLMRouterBench paper/repo: https://arxiv.org/abs/2601.07206 ; {LLMROUTERBENCH_REPO}",
        "",
        "## Remaining External-Baseline Gap",
        "",
        "- Run an official RouteLLM-MF/BERT baseline or an LLMRouterBench adapter output when its dependency and embedding pipeline can be pinned locally.",
        "- Run GraphRouter/Avengers-Pro only after their commands, data contracts, and leakage controls are pinned.",
        "",
    ]
    (out_dir / "phase_e_external_baseline_memo.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


if __name__ == "__main__":
    main()
