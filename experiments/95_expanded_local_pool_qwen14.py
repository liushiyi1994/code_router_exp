from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from routecode.controlled.live_stage0 import normalize_answer, score_output


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
QWEN14 = "qwen3-14b-awq-local"
GEMINI = "gemini-3.5-flash"
GPT = "gpt-5.5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the expanded local pool after Qwen14-AWQ collection.")
    parser.add_argument("--query-table", default="results/controlled/gemini_metadata_gate/query_table_with_gemini_metadata.csv")
    parser.add_argument(
        "--qwen14-runs",
        nargs="+",
        default=[
            "results/controlled/math500_qwen14_awq_live_pilot_1024",
            "results/controlled/mixed_qwen14_awq_live_pilot_1024",
        ],
    )
    parser.add_argument("--output-dir", default="results/controlled/expanded_local_pool_qwen14")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--target-quality-gap", type=float, default=0.03)
    return parser.parse_args()


def load_qwen14_outputs(run_dirs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[frame["model_id"].eq(QWEN14) & frame["status"].eq("success")].copy()
        rescored = [
            score_output(str(answer), str(gold), str(metric))
            for answer, gold, metric in zip(frame["parsed_answer"], frame["gold_answer"], frame["metric"])
        ]
        frame[f"{QWEN14}_answer"] = [answer for answer, _ in rescored]
        frame[f"{QWEN14}_quality"] = [quality for _, quality in rescored]
        frame[f"{QWEN14}_latency"] = frame["latency_s"].astype(float)
        frames.append(frame[["query_id", f"{QWEN14}_answer", f"{QWEN14}_quality", f"{QWEN14}_latency"]])
    if not frames:
        raise ValueError("No Qwen14 outputs found.")
    return pd.concat(frames, ignore_index=True).drop_duplicates("query_id", keep="last")


def merge_table(query_table: pd.DataFrame, qwen14: pd.DataFrame) -> pd.DataFrame:
    table = query_table.merge(qwen14, on="query_id", how="left")
    missing = table[f"{QWEN14}_quality"].isna().sum()
    if missing:
        raise ValueError(f"Missing Qwen14 outputs for {missing} query rows.")
    for model_id in LOCAL_MODELS:
        table[f"{model_id}_answer"] = table[f"{model_id}_answer"].fillna("").astype(str)
        table[f"{model_id}_answer_norm"] = table[f"{model_id}_answer"].map(normalize_answer)
        table[f"{model_id}_answer_len"] = table[f"{model_id}_answer_norm"].str.len()
        table[f"{model_id}_utility_direct"] = table[f"{model_id}_quality"].astype(float)
    table["local_oracle_quality"] = table[[f"{model_id}_quality" for model_id in LOCAL_MODELS]].max(axis=1)
    table["expanded_quality_oracle"] = table[
        [f"{model_id}_quality" for model_id in LOCAL_MODELS] + [f"{GEMINI}_quality", f"{GPT}_quality"]
    ].max(axis=1)
    add_agreement_features(table)
    add_local_ensemble(table)
    return table


def add_agreement_features(table: pd.DataFrame) -> None:
    for idx, model_a in enumerate(LOCAL_MODELS):
        for model_b in LOCAL_MODELS[idx + 1 :]:
            column = f"agree__{model_a}__{model_b}"
            table[column] = table[f"{model_a}_answer_norm"].eq(table[f"{model_b}_answer_norm"]) & table[
                f"{model_a}_answer_norm"
            ].ne("")
    gemini_answer = table[f"{GEMINI}_answer"].fillna("").map(normalize_answer)
    for model_id in LOCAL_MODELS:
        table[f"{model_id}_gemini_agree"] = table[f"{model_id}_answer_norm"].eq(gemini_answer) & table[
            f"{model_id}_answer_norm"
        ].ne("")
    max_votes: list[int] = []
    for _, row in table.iterrows():
        counts: dict[str, int] = {}
        for model_id in LOCAL_MODELS:
            answer = str(row[f"{model_id}_answer_norm"])
            if answer and answer != "nan":
                counts[answer] = counts.get(answer, 0) + 1
        max_votes.append(max(counts.values()) if counts else 0)
    table["local_max_vote"] = max_votes


def add_local_ensemble(table: pd.DataFrame) -> None:
    answers: list[str] = []
    sources: list[str] = []
    votes: list[int] = []
    for _, row in table.iterrows():
        counts: dict[str, int] = {}
        for model_id in LOCAL_MODELS:
            answer = str(row[f"{model_id}_answer_norm"])
            if answer and answer != "nan":
                counts[answer] = counts.get(answer, 0) + 1
        if not counts:
            answers.append("")
            sources.append("none")
            votes.append(0)
            continue
        max_vote = max(counts.values())
        candidates = {answer for answer, count in counts.items() if count == max_vote}
        source = "unknown"
        chosen = next(iter(candidates))
        for model_id in [QWEN14, "qwen3-8b-local", "qwen3-4b-local", "qwen3-0.6b-probe"]:
            answer = str(row[f"{model_id}_answer_norm"])
            if answer in candidates:
                chosen = answer
                source = model_id
                break
        answers.append(chosen)
        sources.append(source)
        votes.append(max_vote)
    table["local_ensemble_answer"] = answers
    table["local_ensemble_source"] = sources
    table["local_ensemble_votes"] = votes
    rescored = [
        score_output(str(answer), str(gold), str(metric))
        for answer, gold, metric in zip(table["local_ensemble_answer"], table["gold_answer"], table["metric"])
    ]
    table["local_ensemble_quality"] = [quality for _, quality in rescored]


def split_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        row: dict[str, object] = {"split": split, "n_queries": len(frame)}
        for model_id in LOCAL_MODELS:
            row[f"{model_id}_quality"] = float(frame[f"{model_id}_quality"].mean())
        row["local_ensemble_quality"] = float(frame["local_ensemble_quality"].mean())
        row["local_oracle_quality"] = float(frame["local_oracle_quality"].mean())
        row["gemini_quality"] = float(frame[f"{GEMINI}_quality"].mean())
        row["gpt_quality"] = float(frame[f"{GPT}_quality"].mean())
        row["expanded_quality_oracle"] = float(frame["expanded_quality_oracle"].mean())
        row["base_quality_oracle"] = float(
            frame[["qwen3-8b-local_quality", f"{GEMINI}_quality", f"{GPT}_quality"]].max(axis=1).mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def best_quality_under_frontier(frame: pd.DataFrame, max_frontier: int) -> float:
    dp = {0: 0.0}
    for _, row in frame.iterrows():
        local_quality = float(row["local_oracle_quality"])
        frontier_quality = max(float(row[f"{GEMINI}_quality"]), float(row[f"{GPT}_quality"]))
        next_dp: dict[int, float] = {}
        for used_frontier, quality in dp.items():
            next_dp[used_frontier] = max(next_dp.get(used_frontier, -1.0), quality + local_quality)
            if used_frontier + 1 <= max_frontier:
                next_dp[used_frontier + 1] = max(
                    next_dp.get(used_frontier + 1, -1.0), quality + frontier_quality
                )
        dp = next_dp
    return max(dp.values()) / len(frame)


def frontier_bounds(table: pd.DataFrame, split: str, target_quality_gap: float) -> pd.DataFrame:
    frame = table[table["split"].eq(split)].copy()
    oracle_quality = float(frame["expanded_quality_oracle"].mean())
    target_quality = oracle_quality - target_quality_gap
    rows = []
    for rate in [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]:
        max_frontier = min(len(frame), int(np.floor(rate * len(frame))))
        rows.append(
            {
                "split": split,
                "frontier_rate_cap": rate,
                "frontier_call_cap": max_frontier,
                "max_quality": best_quality_under_frontier(frame, max_frontier),
            }
        )
    min_calls = None
    for calls in range(len(frame) + 1):
        if best_quality_under_frontier(frame, calls) >= target_quality - 1e-12:
            min_calls = calls
            break
    rows.append(
        {
            "split": split,
            "frontier_rate_cap": "min_for_target",
            "frontier_call_cap": min_calls,
            "max_quality": best_quality_under_frontier(frame, int(min_calls)) if min_calls is not None else np.nan,
        }
    )
    return pd.DataFrame(rows)


def evaluate_actions(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    lambda_cost: float,
    cost_norm: float,
    base_cost: str = "none",
) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    local_calls: list[bool] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        observation_cost = 0.0
        if base_cost == "gemini_full":
            observation_cost += float(row[f"{GEMINI}_cost"]) + float(row.get("verifier_cost", 0.0) or 0.0) + float(
                row.get("self_cost", 0.0) or 0.0
            )
        if action == "local_ensemble":
            quality = float(row["local_ensemble_quality"])
            cost = observation_cost
            gpt = False
            gemini = base_cost == "gemini_full"
            local = True
        elif action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            cost = observation_cost
            gpt = False
            gemini = base_cost == "gemini_full"
            local = True
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            cost = observation_cost if base_cost == "gemini_full" else float(row[f"{GEMINI}_cost"])
            gpt = False
            gemini = True
            local = False
        elif action == GPT:
            quality = float(row[f"{GPT}_quality"])
            cost = observation_cost + float(row[f"{GPT}_cost"])
            gpt = True
            gemini = base_cost == "gemini_full"
            local = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
        local_calls.append(local)
    expanded_cost_oracle = []
    for _, row in frame.iterrows():
        expanded_cost_oracle.append(
            max(
                [float(row[f"{model_id}_quality"]) for model_id in LOCAL_MODELS]
                + [float(row[f"{GEMINI}_utility_selected_cost"]), float(row[f"{GPT}_utility_selected_cost"])]
            )
        )
    mean_quality = float(np.mean(qualities))
    mean_utility = float(mean_quality - lambda_cost * (np.mean(costs) / cost_norm))
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_expanded_oracle": float(frame["expanded_quality_oracle"].mean() - mean_quality),
        "utility_ratio_to_expanded_cost_oracle": float(mean_utility / np.mean(expanded_cost_oracle)),
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / frame[f"{GPT}_cost"].sum()),
        "frontier_call_rate": float(np.mean([gpt or gemini for gpt, gemini in zip(gpt_calls, gemini_calls)])),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "local_final_rate": float(np.mean(local_calls)),
        "action_counts": json.dumps({str(key): int(value) for key, value in actions.value_counts().to_dict().items()}),
    }


def simple_gate_rows(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    cost_norm = max(float(table[f"{GPT}_cost"].mean()), 1e-12)
    rows: list[dict[str, object]] = []
    for split, frame in table.groupby("split", sort=False):
        policies = {
            "all_local_ensemble": pd.Series("local_ensemble", index=frame.index),
            "local_vote2_else_gemini": pd.Series("gemini-3.5-flash", index=frame.index),
            "local_vote2_else_gpt": pd.Series("gpt-5.5", index=frame.index),
        }
        policies["local_vote2_else_gemini"].loc[frame["local_ensemble_votes"].ge(2)] = "local_ensemble"
        policies["local_vote2_else_gpt"].loc[frame["local_ensemble_votes"].ge(2)] = "local_ensemble"
        for method, actions in policies.items():
            rows.append(evaluate_actions(frame, actions, method=method, lambda_cost=lambda_cost, cost_norm=cost_norm))
    return pd.DataFrame(rows)


def classifier_gate_rows(table: pd.DataFrame, lambda_cost: float) -> pd.DataFrame:
    cost_norm = max(float(table[f"{GPT}_cost"].mean()), 1e-12)
    rows: list[dict[str, object]] = []
    utility_columns = [f"{model_id}_utility_direct" for model_id in LOCAL_MODELS] + [
        f"{GEMINI}_utility_selected_cost",
        f"{GPT}_utility_selected_cost",
    ]
    target = (
        table[utility_columns]
        .idxmax(axis=1)
        .str.replace("_utility_direct", "", regex=False)
        .str.replace("_utility_selected_cost", "", regex=False)
    )
    cat_columns = ["dataset"] + [column for column in table.columns if column.startswith("agree__")]
    num_columns = [
        "query_len",
        "number_count",
        "latex_count",
        "frac_count",
        "sqrt_count",
        "local_max_vote",
    ] + [f"{model_id}_answer_len" for model_id in LOCAL_MODELS]
    for column in cat_columns:
        table[column] = table[column].fillna(False).astype(str)
    for column in num_columns:
        table[column] = pd.to_numeric(table[column], errors="coerce").fillna(0.0)
    train = table[table["split"].eq("train")].copy()
    if train.empty:
        return pd.DataFrame(rows)
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_columns),
            ("num", StandardScaler(), num_columns),
        ]
    )
    classifiers = {
        "expanded_local_logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "expanded_local_rf": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=3, class_weight="balanced", random_state=42
        ),
        "expanded_local_extra_trees": ExtraTreesClassifier(
            n_estimators=800, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
        "expanded_local_gb": GradientBoostingClassifier(random_state=42),
    }
    for method, estimator in classifiers.items():
        clf = make_pipeline(preprocessor, estimator)
        clf.fit(train[cat_columns + num_columns], target.loc[train.index])
        for split, frame in table.groupby("split", sort=False):
            if split == "train":
                continue
            actions = pd.Series(clf.predict(frame[cat_columns + num_columns]), index=frame.index)
            rows.append(evaluate_actions(frame, actions, method=method, lambda_cost=lambda_cost, cost_norm=cost_norm))
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False):
        values = [f"{value:.4f}" if isinstance(value, float) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(args.query_table)
    qwen14 = load_qwen14_outputs([Path(path) for path in args.qwen14_runs])
    table = merge_table(base, qwen14)
    query_path = output_dir / "query_table_expanded_local_pool.csv"
    table.to_csv(query_path, index=False)
    summary = split_summary(table)
    summary_path = output_dir / "table_expanded_local_pool_summary.csv"
    summary.to_csv(summary_path, index=False)
    bounds = frontier_bounds(table, "test", args.target_quality_gap)
    bounds_path = output_dir / "table_expanded_frontier_bounds.csv"
    bounds.to_csv(bounds_path, index=False)
    gates = pd.concat(
        [simple_gate_rows(table, args.lambda_cost), classifier_gate_rows(table, args.lambda_cost)],
        ignore_index=True,
    )
    gates_path = output_dir / "table_expanded_local_pool_gates.csv"
    gates.to_csv(gates_path, index=False)
    memo_path = output_dir / "EXPANDED_LOCAL_POOL_QWEN14_MEMO.md"
    memo = [
        "# Expanded Local Pool Qwen14 Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        f"Qwen14 run dirs: `{', '.join(args.qwen14_runs)}`.",
        "This analysis includes Qwen3-0.6B, Qwen3-4B, Qwen3-8B, Qwen3-14B-AWQ, Gemini 3.5 Flash, and GPT-5.5.",
        "",
        "## Split Summary",
        "",
        markdown_table(summary),
        "",
        "## Held-Out Frontier Feasibility With Expanded Local Oracle",
        "",
        markdown_table(bounds),
        "",
        "## Deployable Gate Attempts",
        "",
        markdown_table(gates[gates["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)),
        "",
        "## Interpretation",
        "",
        "Adding Qwen14-AWQ and the small locals makes the outcome-oracle frontier-call target feasible again on the held-out exact-math split, but the tested deployable local-ensemble and classifier gates still miss the quality target. The remaining bottleneck is predicting which local model or frontier solver will win, not the existence of useful local coverage.",
        "",
        "## Files",
        "",
        f"- `{query_path}`",
        f"- `{summary_path}`",
        f"- `{bounds_path}`",
        f"- `{gates_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
