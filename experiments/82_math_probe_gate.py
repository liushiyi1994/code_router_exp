from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from routecode.controlled.live_stage0 import normalize_answer, score_output


LOCAL_MODEL = "qwen3-4b-local"
PROBE_MODEL = "qwen3-0.6b-probe"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate held-out MATH500 query+probe routing gates.")
    parser.add_argument("--run-dir", default="results/controlled/math500_live_pilot_1024")
    parser.add_argument("--manifest", default="results/phase2/all200_exact_task_manifest/local_exact_task_manifest.csv")
    parser.add_argument("--dataset", default="math500")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-frontier-rate", type=float, default=0.40)
    return parser.parse_args()


def build_query_table(run_dir: Path, manifest_path: Path, dataset: str, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(run_dir / "model_outputs.parquet")
    outputs = outputs[outputs["status"].eq("success")].copy()
    rescored = [
        score_output(str(parsed), str(gold), str(metric))
        for parsed, gold, metric in zip(outputs["parsed_answer"], outputs["gold_answer"], outputs["metric"])
    ]
    outputs["parsed_answer"] = [parsed for parsed, _ in rescored]
    outputs["quality_score"] = [quality for _, quality in rescored]
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["dataset"].eq(dataset)][["query_id", "routecode_split", "source_split"]].copy()
    outputs = outputs.merge(manifest, on="query_id", how="left")
    missing = sorted(outputs.loc[outputs["routecode_split"].isna(), "query_id"].unique())
    if missing:
        raise ValueError(f"Missing manifest split for {len(missing)} query_ids, first={missing[:3]}")

    gpt_cost = outputs.loc[outputs["model_id"].eq(GPT_MODEL)].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()) if not gpt_cost.empty else float(outputs["cost_total_usd"].max()), 1e-12)
    outputs["normalized_remote_cost"] = outputs["cost_total_usd"] / cost_norm
    outputs["utility"] = outputs["quality_score"] - lambda_cost * outputs["normalized_remote_cost"]

    rows: list[dict[str, object]] = []
    by_query = outputs.groupby("query_id", sort=True)
    for query_id, group in by_query:
        by_model = group.drop_duplicates("model_id").set_index("model_id")
        required = {LOCAL_MODEL, PROBE_MODEL, GEMINI_MODEL, GPT_MODEL}
        if not required.issubset(set(by_model.index)):
            continue
        local_ans = normalize_answer(by_model.loc[LOCAL_MODEL, "parsed_answer"])
        probe_ans = normalize_answer(by_model.loc[PROBE_MODEL, "parsed_answer"])
        gemini_ans = normalize_answer(by_model.loc[GEMINI_MODEL, "parsed_answer"])
        gpt_ans = normalize_answer(by_model.loc[GPT_MODEL, "parsed_answer"])
        local_models = [LOCAL_MODEL, PROBE_MODEL]
        best_local_model = max(local_models, key=lambda model: float(by_model.loc[model, "utility"]))
        candidate_models = [LOCAL_MODEL, PROBE_MODEL, GEMINI_MODEL, GPT_MODEL]
        cost_oracle_model = max(candidate_models, key=lambda model: float(by_model.loc[model, "utility"]))
        quality_oracle_model = max(candidate_models, key=lambda model: float(by_model.loc[model, "quality_score"]))
        query_text = str(by_model.iloc[0]["query_text"])
        rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "routecode_split": str(by_model.iloc[0]["routecode_split"]),
                "best_local_model": best_local_model,
                "cost_oracle_model": cost_oracle_model,
                "quality_oracle_model": quality_oracle_model,
                "local_answer": local_ans,
                "probe_answer": probe_ans,
                "gemini_answer": gemini_ans,
                "gpt_answer": gpt_ans,
                "local_probe_agree": bool(local_ans and local_ans == probe_ans),
                "local_answer_len": len(local_ans),
                "probe_answer_len": len(probe_ans),
                "query_len": len(query_text),
                "latex_count": query_text.count("\\"),
                "number_count": len(re.findall(r"-?\d+(?:\.\d+)?", query_text)),
                "sqrt_count": query_text.count("\\sqrt"),
                "frac_count": query_text.count("\\frac"),
                "best_local_quality": float(by_model.loc[best_local_model, "quality_score"]),
                "gemini_quality": float(by_model.loc[GEMINI_MODEL, "quality_score"]),
                "gpt_quality": float(by_model.loc[GPT_MODEL, "quality_score"]),
                "best_local_utility": float(by_model.loc[best_local_model, "utility"]),
                "gemini_utility": float(by_model.loc[GEMINI_MODEL, "utility"]),
                "gpt_utility": float(by_model.loc[GPT_MODEL, "utility"]),
                "best_local_cost": float(by_model.loc[best_local_model, "cost_total_usd"]),
                "gemini_cost": float(by_model.loc[GEMINI_MODEL, "cost_total_usd"]),
                "gpt_cost": float(by_model.loc[GPT_MODEL, "cost_total_usd"]),
                "best_local_latency": float(by_model.loc[best_local_model, "latency_s"]),
                "gemini_latency": float(by_model.loc[GEMINI_MODEL, "latency_s"]),
                "gpt_latency": float(by_model.loc[GPT_MODEL, "latency_s"]),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No complete query rows available for math probe gate evaluation.")
    return table


def feature_text(row: pd.Series, *, include_probe: bool) -> str:
    tags = [
        f"query_len_bin_{min(int(row.query_len) // 100, 8)}",
        f"num_count_bin_{min(int(row.number_count), 12)}",
        f"latex_count_bin_{min(int(row.latex_count) // 4, 10)}",
        "has_sqrt" if int(row.sqrt_count) else "no_sqrt",
        "has_frac" if int(row.frac_count) else "no_frac",
    ]
    if include_probe:
        tags.extend(
            [
                "local_probe_agree" if bool(row.local_probe_agree) else "local_probe_disagree",
                f"local_answer_len_bin_{min(int(row.local_answer_len) // 4, 10)}",
                f"probe_answer_len_bin_{min(int(row.probe_answer_len) // 4, 10)}",
                f"local_answer_{row.local_answer}",
                f"probe_answer_{row.probe_answer}",
            ]
        )
    return " ".join([str(row.query_text), *tags])


def evaluate_selection(
    table: pd.DataFrame,
    selected_models: pd.Series,
    *,
    method: str,
    split: str,
    probe_call_rate: float,
) -> dict[str, object]:
    rows = table.copy()
    rows["selected_model"] = selected_models.loc[rows.index].values
    quality = []
    utility = []
    costs = []
    latencies = []
    frontier = []
    for row in rows.itertuples():
        model = str(row.selected_model)
        if model in {LOCAL_MODEL, PROBE_MODEL}:
            prefix = "best_local" if model == str(row.best_local_model) else "best_local"
            quality.append(float(row.best_local_quality))
            utility.append(float(row.best_local_utility))
            costs.append(float(row.best_local_cost))
            latencies.append(float(row.best_local_latency))
            frontier.append(False)
        elif model == GEMINI_MODEL:
            quality.append(float(row.gemini_quality))
            utility.append(float(row.gemini_utility))
            costs.append(float(row.gemini_cost))
            latencies.append(float(row.gemini_latency))
            frontier.append(True)
        elif model == GPT_MODEL:
            quality.append(float(row.gpt_quality))
            utility.append(float(row.gpt_utility))
            costs.append(float(row.gpt_cost))
            latencies.append(float(row.gpt_latency))
            frontier.append(True)
        else:
            raise ValueError(f"Unknown selected model: {model}")
    oracle_quality = table[["best_local_quality", "gemini_quality", "gpt_quality"]].max(axis=1)
    oracle_utility = table[["best_local_utility", "gemini_utility", "gpt_utility"]].max(axis=1)
    mean_quality = float(np.mean(quality))
    mean_utility = float(np.mean(utility))
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(rows)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "oracle_mean_quality": float(oracle_quality.mean()),
        "oracle_mean_utility": float(oracle_utility.mean()),
        "quality_gap_to_oracle": float(oracle_quality.mean() - mean_quality),
        "utility_gap_to_oracle": float(oracle_utility.mean() - mean_utility),
        "oracle_utility_ratio": float(mean_utility / oracle_utility.mean()) if abs(float(oracle_utility.mean())) > 1e-12 else np.nan,
        "remote_cost_total_usd": float(np.sum(costs)),
        "frontier_call_rate": float(np.mean(frontier)),
        "probe_call_rate": float(probe_call_rate),
        "mean_latency_s": float(np.mean(latencies)),
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
    }


def static_selection(table: pd.DataFrame, model: str) -> pd.Series:
    if model == "best_local":
        return table["best_local_model"].copy()
    return pd.Series(model, index=table.index)


def agreement_selection(table: pd.DataFrame) -> pd.Series:
    selected = pd.Series(GEMINI_MODEL, index=table.index)
    selected.loc[table["local_probe_agree"].astype(bool)] = table.loc[table["local_probe_agree"].astype(bool), "best_local_model"]
    return selected


def fit_binary_gate(
    train: pd.DataFrame,
    *,
    include_probe: bool,
    target_fn: Callable[[pd.DataFrame], pd.Series],
) -> tuple[object, Callable[[pd.DataFrame], np.ndarray]]:
    x_train = train.apply(lambda row: feature_text(row, include_probe=include_probe), axis=1)
    y_train = target_fn(train).astype(int)
    if y_train.nunique() < 2:
        constant = int(y_train.iloc[0])
        return None, lambda frame: np.full(len(frame), constant, dtype=float)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=3000),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(x_train, y_train)

    def predict_proba(frame: pd.DataFrame) -> np.ndarray:
        x = frame.apply(lambda row: feature_text(row, include_probe=include_probe), axis=1)
        return model.predict_proba(x)[:, 1]

    return model, predict_proba


def threshold_selection(table: pd.DataFrame, proba_local: np.ndarray, threshold: float) -> pd.Series:
    selected = pd.Series(GEMINI_MODEL, index=table.index)
    use_local = proba_local >= threshold
    selected.loc[use_local] = table.loc[use_local, "best_local_model"]
    return selected


def choose_threshold(
    val: pd.DataFrame,
    proba_local: np.ndarray,
    *,
    max_frontier_rate: float,
    method: str,
    probe_call_rate: float,
) -> float:
    candidates = np.linspace(0.05, 0.95, 19)
    scored = []
    for threshold in candidates:
        selected = threshold_selection(val, proba_local, float(threshold))
        row = evaluate_selection(val, selected, method=method, split="val", probe_call_rate=probe_call_rate)
        row["threshold"] = float(threshold)
        scored.append(row)
    table = pd.DataFrame(scored)
    feasible = table[table["frontier_call_rate"].le(max_frontier_rate)]
    pool = feasible if not feasible.empty else table
    best = pool.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]
    return float(best["threshold"])


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    query_table = build_query_table(run_dir, Path(args.manifest), args.dataset, args.lambda_cost)
    split_order = {"train": 0, "val": 1, "test": 2}
    query_table = query_table.sort_values(
        by=["routecode_split", "query_id"], key=lambda col: col.map(split_order).fillna(9) if col.name == "routecode_split" else col
    ).reset_index(drop=True)
    train = query_table[query_table["routecode_split"].eq("train")].copy()
    val = query_table[query_table["routecode_split"].eq("val")].copy()
    test = query_table[query_table["routecode_split"].eq("test")].copy()
    if train.empty or val.empty or test.empty:
        raise ValueError(f"Need non-empty train/val/test splits; got {query_table['routecode_split'].value_counts().to_dict()}")

    target_local_beats_gemini = lambda frame: frame["best_local_utility"].ge(frame["gemini_utility"])
    _, query_only_proba = fit_binary_gate(train, include_probe=False, target_fn=target_local_beats_gemini)
    _, query_probe_proba = fit_binary_gate(train, include_probe=True, target_fn=target_local_beats_gemini)

    query_only_threshold = choose_threshold(
        val,
        query_only_proba(val),
        max_frontier_rate=args.max_frontier_rate,
        method="query_text_local_gate_then_gemini",
        probe_call_rate=0.0,
    )
    query_probe_threshold = choose_threshold(
        val,
        query_probe_proba(val),
        max_frontier_rate=args.max_frontier_rate,
        method="query_probe_local_gate_then_gemini",
        probe_call_rate=1.0,
    )

    rows: list[dict[str, object]] = []
    for split_name, split_table in [("val", val), ("test", test)]:
        rows.append(
            evaluate_selection(
                split_table,
                static_selection(split_table, "best_local"),
                method="best_local",
                split=split_name,
                probe_call_rate=0.0,
            )
        )
        for model in [GEMINI_MODEL, GPT_MODEL]:
            rows.append(
                evaluate_selection(
                    split_table,
                    static_selection(split_table, model),
                    method=f"all_{model}",
                    split=split_name,
                    probe_call_rate=0.0,
                )
            )
        rows.append(
            evaluate_selection(
                split_table,
                agreement_selection(split_table),
                method="probe_answer_agreement_then_gemini",
                split=split_name,
                probe_call_rate=1.0,
            )
        )
        rows.append(
            evaluate_selection(
                split_table,
                threshold_selection(split_table, query_only_proba(split_table), query_only_threshold),
                method="query_text_local_gate_then_gemini",
                split=split_name,
                probe_call_rate=0.0,
            )
            | {"threshold": query_only_threshold}
        )
        rows.append(
            evaluate_selection(
                split_table,
                threshold_selection(split_table, query_probe_proba(split_table), query_probe_threshold),
                method="query_probe_local_gate_then_gemini",
                split=split_name,
                probe_call_rate=1.0,
            )
            | {"threshold": query_probe_threshold}
        )
        rows.append(
            evaluate_selection(
                split_table,
                split_table["cost_oracle_model"],
                method="cost_aware_oracle",
                split=split_name,
                probe_call_rate=0.0,
            )
        )

    result = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    out_path = run_dir / "table_math_probe_gate.csv"
    result.to_csv(out_path, index=False)
    memo_path = run_dir / "MATH500_PROBE_GATE_MEMO.md"
    test_rows = result[result["split"].eq("test")].copy()
    best_deployable = test_rows[
        test_rows["method"].isin(
            [
                "probe_answer_agreement_then_gemini",
                "query_text_local_gate_then_gemini",
                "query_probe_local_gate_then_gemini",
            ]
        )
    ].sort_values(["mean_utility", "mean_quality"], ascending=False)
    best_line = best_deployable.iloc[0] if not best_deployable.empty else None
    lines = [
        "# MATH500 Probe Gate Memo",
        "",
        f"Run directory: `{run_dir}`.",
        f"Manifest: `{args.manifest}`.",
        f"Rows by split: `{query_table['routecode_split'].value_counts().to_dict()}`.",
        "",
        "The gates are selected without test leakage: classifiers fit on `train`, thresholds are selected on `val`, and the held-out `test` rows are reported once.",
        "",
        f"Query-only threshold selected on val: `{query_only_threshold:.2f}`.",
        f"Query+probe threshold selected on val: `{query_probe_threshold:.2f}`.",
        "",
        "## Held-Out Test Summary",
        "",
        markdown_table(
            test_rows[
                [
                    "method",
                    "n_queries",
                    "mean_quality",
                    "mean_utility",
                    "quality_gap_to_oracle",
                    "oracle_utility_ratio",
                    "frontier_call_rate",
                    "probe_call_rate",
                    "remote_cost_total_usd",
                ]
            ]
        ),
    ]
    if best_line is not None:
        lines.extend(
            [
                "",
                "## Best Deployable Gate On Test",
                "",
                (
                    f"`{best_line.method}` reaches quality `{best_line.mean_quality:.4f}`, "
                    f"quality gap `{best_line.quality_gap_to_oracle:.4f}`, utility ratio "
                    f"`{best_line.oracle_utility_ratio:.4f}`, frontier rate `{best_line.frontier_call_rate:.4f}`, "
                    f"and probe rate `{best_line.probe_call_rate:.4f}`."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- `{out_path}`",
        ]
    )
    memo_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
