from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


FRONTIER_MODELS = {"gpt-5.5", "gemini-3.5-flash"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached-only probe-signal sandbox for broad100 routing.",
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_probe_signal_cached"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-features", type=int, default=12000)
    parser.add_argument("--k-values", default="1,3,5,10,20")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_broad_package_module()
    outputs = package.load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    main_eval, selections = package.build_main_eval(outputs, lambda_cost=args.lambda_cost)

    recall, confusion = model_win_recall_tables(package, outputs, selections[package.DEFAULT_METHOD])
    recall.to_csv(args.output_dir / "table_model_win_recall.csv", index=False)
    confusion.to_csv(args.output_dir / "table_oracle_selected_confusion.csv", index=False)

    k_values = [int(value) for value in str(args.k_values).split(",") if str(value).strip()]
    knn_table, selected_table = run_knn_experiments(
        package,
        outputs,
        k_values=k_values,
        lambda_cost=float(args.lambda_cost),
        max_features=int(args.max_features),
    )
    knn_table.to_csv(args.output_dir / "table_probe_signal_knn.csv", index=False)
    selected_table.to_csv(args.output_dir / "table_probe_signal_selected.csv", index=False)

    diagnostics = local_probe_feature_diagnostics(package, outputs)
    diagnostics.to_csv(args.output_dir / "table_local_probe_feature_diagnostics.csv", index=False)

    write_figure(args.output_dir, knn_table, main_eval)
    write_memo(args.output_dir / "PROBE_SIGNAL_CACHED_SANDBOX_MEMO.md", args.outputs, recall, knn_table, selected_table)
    print(f"Wrote cached probe-signal sandbox to {args.output_dir}")


def load_broad_package_module():
    path = Path("experiments/125_phase3_broad_target_method_package.py")
    spec = importlib.util.spec_from_file_location("broad_target_package", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def model_win_recall_tables(package, outputs: pd.DataFrame, selected: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = compare_selection_to_oracle(package, outputs, selected, split="test")
    comp["oracle_is_frontier"] = comp["oracle_model"].isin(FRONTIER_MODELS)
    comp["selected_is_frontier"] = comp["selected_model"].isin(FRONTIER_MODELS)
    comp["miss_type"] = np.select(
        [
            comp["oracle_model"].eq(comp["selected_model"]),
            comp["oracle_is_frontier"] & ~comp["selected_is_frontier"],
            ~comp["oracle_is_frontier"] & comp["selected_is_frontier"],
        ],
        ["hit", "miss_frontier_needed", "miss_unneeded_frontier"],
        default="miss_wrong_same_family",
    )

    rows: list[dict[str, Any]] = []
    for (benchmark, oracle_model), group in comp.groupby(["benchmark", "oracle_model"], dropna=False):
        hits = group["oracle_model"].eq(group["selected_model"])
        rows.append(
            {
                "benchmark": benchmark,
                "oracle_model": oracle_model,
                "n_oracle_wins": int(len(group)),
                "selected_recall": float(hits.mean()),
                "n_selected_correct": int(hits.sum()),
                "mean_quality_gap": float(group["quality_gap"].mean()),
                "mean_utility_gap": float(group["utility_gap"].mean()),
                "oracle_frontier_rate": float(group["oracle_is_frontier"].mean()),
                "selected_frontier_rate": float(group["selected_is_frontier"].mean()),
                "miss_type_counts_json": group["miss_type"].value_counts().sort_index().to_json(),
            }
        )
    recall = pd.DataFrame(rows).sort_values(["mean_utility_gap", "n_oracle_wins"], ascending=[False, False])

    confusion = (
        comp.groupby(["benchmark", "oracle_model", "selected_model"], dropna=False)
        .agg(
            n_queries=("selected_model", "size"),
            mean_quality_gap=("quality_gap", "mean"),
            mean_utility_gap=("utility_gap", "mean"),
        )
        .reset_index()
        .sort_values(["mean_utility_gap", "n_queries"], ascending=[False, False])
    )
    return recall, confusion


def compare_selection_to_oracle(package, outputs: pd.DataFrame, selected: pd.Series, *, split: str) -> pd.DataFrame:
    selected_rows = package.selected_to_rows(outputs, selected, split=split).set_index("query_id")
    target = outputs[outputs["split"].eq(split)].copy()
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()].set_index("query_id")
    query_info = target.drop_duplicates("query_id").set_index("query_id")
    comp = query_info[["benchmark", "domain", "metric", "query_text"]].join(
        selected_rows[["model_id", "quality_score", "utility", "normalized_remote_cost"]].rename(
            columns={
                "model_id": "selected_model",
                "quality_score": "selected_quality",
                "utility": "selected_utility",
                "normalized_remote_cost": "selected_normalized_cost",
            }
        )
    )
    comp = comp.join(
        cost_oracle[["model_id", "quality_score", "utility", "normalized_remote_cost"]].rename(
            columns={
                "model_id": "oracle_model",
                "quality_score": "oracle_quality",
                "utility": "oracle_utility",
                "normalized_remote_cost": "oracle_normalized_cost",
            }
        )
    )
    comp["quality_gap"] = comp["oracle_quality"] - comp["selected_quality"]
    comp["utility_gap"] = comp["oracle_utility"] - comp["selected_utility"]
    return comp.reset_index()


def run_knn_experiments(
    package,
    outputs: pd.DataFrame,
    *,
    k_values: list[int],
    lambda_cost: float,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    for feature_view in ["query_text", "query_text_with_benchmark", "query_text_with_local_answers"]:
        for k in k_values:
            for split in ["val", "test"]:
                selected = knn_utility_selection(
                    package,
                    outputs,
                    split=split,
                    feature_view=feature_view,
                    k=int(k),
                    max_features=max_features,
                )
                row = evaluate_selection(package, outputs, selected, split=split, lambda_cost=lambda_cost)
                row.update({"method": f"knn_utility_{feature_view}_k{k}", "feature_view": feature_view, "k": int(k)})
                rows.append(row)

    table = pd.DataFrame(rows)
    validation = table[table["split"].eq("val")].copy()
    validation = validation.sort_values(["mean_utility", "mean_quality"], ascending=[False, False])
    if validation.empty:
        return table, pd.DataFrame()
    best = validation.iloc[0]
    test_match = table[
        table["split"].eq("test")
        & table["feature_view"].eq(best["feature_view"])
        & table["k"].eq(int(best["k"]))
    ].copy()
    for _, row in pd.concat([best.to_frame().T, test_match], ignore_index=True).iterrows():
        out = row.to_dict()
        out["selection_rule"] = "validation_best_mean_utility"
        selected_rows.append(out)
    return table, pd.DataFrame(selected_rows)


def knn_utility_selection(
    package,
    outputs: pd.DataFrame,
    *,
    split: str,
    feature_view: str,
    k: int,
    max_features: int,
) -> pd.Series:
    query_info = outputs.drop_duplicates("query_id").set_index("query_id")
    train_queries = query_info[query_info["split"].eq("train")].copy()
    target_queries = query_info[query_info["split"].eq(split)].copy()
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    train_ids = train_queries.index.astype(str).tolist()
    target_ids = target_queries.index.astype(str).tolist()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
    train_text = [
        feature_text(str(query_id), train_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in train_ids
    ]
    target_text = [
        feature_text(str(query_id), target_queries.loc[query_id], by_query, local_models, feature_view)
        for query_id in target_ids
    ]
    train_x = vectorizer.fit_transform(train_text)
    target_x = vectorizer.transform(target_text)
    n_neighbors = min(max(int(k), 1), len(train_ids))
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    neighbors.fit(train_x)
    _, neighbor_idx = neighbors.kneighbors(target_x)

    utility = outputs.pivot_table(index="query_id", columns="model_id", values="utility", aggfunc="first")
    candidate_models = [
        model_id
        for model_id in sorted(outputs["model_id"].astype(str).unique())
        if model_id != package.TOOL_MODEL
    ]
    selected: dict[str, str] = {}
    for row_index, query_id in enumerate(target_ids):
        tool_choice = package.deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        nearest_ids = [train_ids[int(idx)] for idx in neighbor_idx[row_index]]
        neighbor_utility = utility.loc[nearest_ids, candidate_models].mean(axis=0).sort_values(ascending=False)
        selected[query_id] = str(neighbor_utility.index[0])
    return pd.Series(selected)


def feature_text(
    query_id: str,
    row: pd.Series,
    by_query: pd.DataFrame,
    local_models: list[str],
    feature_view: str,
) -> str:
    parts = [str(row.get("query_text", ""))]
    if feature_view in {"query_text_with_benchmark", "query_text_with_local_answers"}:
        parts.extend(
            [
                f"benchmark_{row.get('benchmark', '')}",
                f"domain_{row.get('domain', '')}",
                f"metric_{row.get('metric', '')}",
            ]
        )
    if feature_view == "query_text_with_local_answers":
        answers = []
        for model_id in local_models:
            answer = local_probe_answer(by_query, query_id, model_id)
            if answer:
                answers.append(answer)
                parts.append(f"{model_id}_answer_{answer}")
                parts.append(f"{model_id}_valid")
            else:
                parts.append(f"{model_id}_empty")
        counts = pd.Series(answers).value_counts() if answers else pd.Series(dtype=int)
        parts.append(f"local_valid_count_{len(answers)}")
        parts.append(f"local_unique_count_{int(len(counts))}")
        parts.append(f"local_majority_count_{int(counts.iloc[0]) if not counts.empty else 0}")
    return " ".join(parts)


def local_probe_answer(by_query: pd.DataFrame, query_id: str, model_id: str) -> str:
    try:
        row = by_query.loc[(query_id, model_id)]
    except KeyError:
        return ""
    if str(row.get("status", "")) != "success":
        return ""
    value = row.get("parsed_answer", "")
    if pd.isna(value):
        return ""
    answer = str(value).strip().lower()
    if not answer or answer in {"nan", "none", "null", "no_code"} or answer.startswith("failed"):
        return ""
    return answer[:120]


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    lambda_cost: float,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    return package.evaluation_row("candidate", selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)


def local_probe_feature_diagnostics(package, outputs: pd.DataFrame) -> pd.DataFrame:
    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = package.observable_local_models(outputs)
    rows: list[dict[str, Any]] = []
    for query_id, row in outputs.drop_duplicates("query_id").set_index("query_id").iterrows():
        query_id = str(query_id)
        answers = [local_probe_answer(by_query, query_id, model_id) for model_id in local_models]
        answers = [answer for answer in answers if answer]
        counts = pd.Series(answers).value_counts() if answers else pd.Series(dtype=int)
        local_rows = []
        for model_id in local_models:
            try:
                local_rows.append(by_query.loc[(query_id, model_id)])
            except KeyError:
                continue
        frame = pd.DataFrame(local_rows)
        rows.append(
            {
                "query_id": query_id,
                "split": row["split"],
                "benchmark": row["benchmark"],
                "metric": row["metric"],
                "local_valid_count": int(len(answers)),
                "local_unique_count": int(len(counts)),
                "local_majority_count": int(counts.iloc[0]) if not counts.empty else 0,
                "best_local_quality": float(frame["quality_score"].max()) if not frame.empty else np.nan,
                "best_local_utility": float(frame["utility"].max()) if not frame.empty else np.nan,
                "any_local_correct": bool(float(frame["quality_score"].max()) > 0.0) if not frame.empty else False,
            }
        )
    diagnostics = pd.DataFrame(rows)
    return (
        diagnostics.groupby(["split", "benchmark", "local_valid_count", "local_majority_count"], dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            any_local_correct_rate=("any_local_correct", "mean"),
            mean_best_local_quality=("best_local_quality", "mean"),
            mean_best_local_utility=("best_local_utility", "mean"),
        )
        .reset_index()
        .sort_values(["split", "benchmark", "local_valid_count", "local_majority_count"])
    )


def write_figure(out_dir: Path, table: pd.DataFrame, main_eval: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].copy()
    plot["label"] = plot["feature_view"] + " k=" + plot["k"].astype(str)
    plot = plot.sort_values("mean_utility", ascending=False).head(12).sort_values("mean_utility")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(plot["label"], plot["mean_utility"], color="#4c78a8")
    oracle = main_eval[main_eval["method"].eq("cost_aware_oracle")]
    current = main_eval[main_eval["method"].eq("observable_local_state_v5")]
    if not oracle.empty:
        ax.axvline(float(oracle.iloc[0]["mean_utility"]), color="#d62728", linestyle="--", label="oracle")
    if not current.empty:
        ax.axvline(float(current.iloc[0]["mean_utility"]), color="#2ca02c", linestyle=":", label="current")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Cached kNN Probe-Signal Routing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_probe_signal_knn_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, outputs_path: Path, recall: pd.DataFrame, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(1)
    lines = [
        "# Cached Probe-Signal Sandbox",
        "",
        f"Source outputs: `{outputs_path}`.",
        "",
        "This run makes no model or provider API calls. It evaluates the first probe-note ideas on cached broad100 outputs.",
        "",
        "## Ideas Tested",
        "",
        "- LLMRouterBench-style model-win recall and oracle-vs-selected confusion.",
        "- kNN utility routing from train-only neighbor utility vectors.",
        "- kNN feature views: query text, query text plus benchmark tags, and query text plus local probe answers.",
        "- Local probe vote/count diagnostics from cached local model answers.",
        "",
        "## Validation-Selected kNN",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out kNN Diagnostic",
        "",
        markdown_table(best_test),
        "",
        "## Largest Model-Win Recall Gaps",
        "",
        markdown_table(recall.head(20)),
        "",
        "## Interpretation",
        "",
        "- kNN rows are deployable only in the sense that neighbors are fit on train data; validation still selects the reported configuration.",
        "- Local probe-answer features use cached local model outputs as if those local probes were gathered before routing.",
        "- If query+local-answer kNN does not improve over query-only kNN, then surface probe answers are not enough; the next notes to try are logprob/confidence probes and prefill/activation probes.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
