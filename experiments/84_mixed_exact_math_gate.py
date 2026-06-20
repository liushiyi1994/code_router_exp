from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from routecode.controlled.live_stage0 import normalize_answer, score_output


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
REQUIRED_MODELS = (LOCAL_MODEL, GEMINI_MODEL, GPT_MODEL)


def has_normalized_answer(answer: object) -> bool:
    text = normalize_answer(answer)
    return bool(text and text.lower() not in {"nan", "none", "null"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate mixed exact-math routing gates from cached live outputs.")
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=[
            "results/controlled/math500_qwen8_live_pilot_1024",
            "results/controlled/livemathbench_live_pilot_1024",
        ],
    )
    parser.add_argument(
        "--manifest",
        default="results/controlled/mixed_exact_math_manifest/local_exact_task_manifest.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/mixed_exact_math_gate")
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--target-quality-gap", type=float, default=0.03)
    return parser.parse_args()


def load_rescored_outputs(run_dirs: list[Path], manifest_path: Path, lambda_cost: float) -> tuple[pd.DataFrame, float]:
    frames: list[pd.DataFrame] = []
    for order, run_dir in enumerate(run_dirs):
        path = run_dir / "model_outputs.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame = frame[frame["status"].eq("success")].copy()
        if frame.empty:
            continue
        rescored = [
            score_output(str(parsed), str(gold), str(metric))
            for parsed, gold, metric in zip(frame["parsed_answer"], frame["gold_answer"], frame["metric"])
        ]
        frame["parsed_answer"] = [parsed for parsed, _ in rescored]
        frame["quality_score"] = [quality for _, quality in rescored]
        frame["_run_order"] = order
        frame["_run_dir"] = str(run_dir)
        frames.append(frame)
    if not frames:
        raise ValueError("No successful model_outputs.parquet rows found.")

    outputs = pd.concat(frames, ignore_index=True)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.rename(columns={"dataset": "manifest_dataset"})
    keep_cols = ["query_id", "manifest_dataset", "routecode_split", "source_split", "task_type"]
    outputs = outputs.merge(manifest[keep_cols], on="query_id", how="left")
    outputs["dataset"] = outputs["manifest_dataset"].fillna(outputs.get("benchmark", ""))
    outputs["split"] = outputs["routecode_split"].fillna(outputs.get("source_split", ""))
    outputs = outputs.dropna(subset=["split", "dataset"])
    outputs = outputs.sort_values(["_run_order", "query_id", "model_id"])
    outputs = outputs.drop_duplicates(["query_id", "model_id"], keep="first")

    gpt_cost = outputs.loc[outputs["model_id"].eq(GPT_MODEL)].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()), 1e-12)
    outputs["normalized_remote_cost"] = outputs["cost_total_usd"].astype(float) / cost_norm
    outputs["utility_selected_cost"] = outputs["quality_score"].astype(float) - float(lambda_cost) * outputs[
        "normalized_remote_cost"
    ]
    return outputs, cost_norm


def build_query_table(outputs: pd.DataFrame, cost_norm: float, lambda_cost: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_id, group in outputs.groupby("query_id", sort=True):
        by_model = group.drop_duplicates("model_id").set_index("model_id")
        if not set(REQUIRED_MODELS).issubset(set(by_model.index)):
            continue
        query_text = str(by_model.iloc[0]["query_text"])
        row: dict[str, object] = {
            "query_id": query_id,
            "query_text": query_text,
            "dataset": str(by_model.iloc[0]["dataset"]),
            "split": str(by_model.iloc[0]["split"]),
            "domain": str(by_model.iloc[0].get("domain", "")),
            "query_len": len(query_text),
            "number_count": len(re.findall(r"-?\d+(?:\.\d+)?", query_text)),
            "latex_count": query_text.count("\\"),
            "frac_count": query_text.count("\\frac"),
            "sqrt_count": query_text.count("\\sqrt"),
        }
        for model_id in REQUIRED_MODELS:
            parsed = normalize_answer(by_model.loc[model_id, "parsed_answer"])
            row[f"{model_id}_answer"] = parsed
            row[f"{model_id}_answer_len"] = len(parsed)
            row[f"{model_id}_quality"] = float(by_model.loc[model_id, "quality_score"])
            row[f"{model_id}_cost"] = float(by_model.loc[model_id, "cost_total_usd"])
            row[f"{model_id}_latency"] = float(by_model.loc[model_id, "latency_s"])
            row[f"{model_id}_utility_selected_cost"] = float(by_model.loc[model_id, "utility_selected_cost"])
        row["qwen8_gemini_agree"] = bool(row[f"{LOCAL_MODEL}_answer"] and row[f"{LOCAL_MODEL}_answer"] == row[f"{GEMINI_MODEL}_answer"])
        row["gemini_gpt_agree"] = bool(row[f"{GEMINI_MODEL}_answer"] and row[f"{GEMINI_MODEL}_answer"] == row[f"{GPT_MODEL}_answer"])
        row["gemini_then_gpt_cost"] = float(row[f"{GEMINI_MODEL}_cost"]) + float(row[f"{GPT_MODEL}_cost"])
        row["gemini_then_gpt_utility"] = float(row[f"{GPT_MODEL}_quality"]) - lambda_cost * (
            float(row["gemini_then_gpt_cost"]) / cost_norm
        )
        row["gpt_answer_available"] = has_normalized_answer(row[f"{GPT_MODEL}_answer"])
        row["gemini_then_gpt_guarded_quality"] = (
            float(row[f"{GPT_MODEL}_quality"]) if bool(row["gpt_answer_available"]) else float(row[f"{GEMINI_MODEL}_quality"])
        )
        row["gemini_then_gpt_guarded_utility"] = float(row["gemini_then_gpt_guarded_quality"]) - lambda_cost * (
            float(row["gemini_then_gpt_cost"]) / cost_norm
        )
        candidates = list(REQUIRED_MODELS)
        row["quality_oracle_model"] = max(candidates, key=lambda model: float(row[f"{model}_quality"]))
        row["cost_oracle_model_selected_cost"] = max(
            candidates, key=lambda model: float(row[f"{model}_utility_selected_cost"])
        )
        # Sequential oracle pays Gemini cost first, then GPT cost only on rows where GPT is worth rescuing.
        gemini_quality = float(row[f"{GEMINI_MODEL}_quality"])
        gemini_utility = float(row[f"{GEMINI_MODEL}_utility_selected_cost"])
        gpt_rescue_utility = float(row["gemini_then_gpt_guarded_utility"])
        local_utility = float(row[f"{LOCAL_MODEL}_utility_selected_cost"])
        if local_utility >= gemini_utility and local_utility >= gpt_rescue_utility:
            row["sequential_cost_oracle_action"] = "local"
            row["sequential_cost_oracle_quality"] = float(row[f"{LOCAL_MODEL}_quality"])
            row["sequential_cost_oracle_utility"] = local_utility
        elif gpt_rescue_utility > gemini_utility:
            row["sequential_cost_oracle_action"] = "gemini_then_gpt_guarded"
            row["sequential_cost_oracle_quality"] = float(row["gemini_then_gpt_guarded_quality"])
            row["sequential_cost_oracle_utility"] = gpt_rescue_utility
        else:
            row["sequential_cost_oracle_action"] = "gemini"
            row["sequential_cost_oracle_quality"] = gemini_quality
            row["sequential_cost_oracle_utility"] = gemini_utility
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No complete query rows for required model set.")
    return table


def feature_text(row: pd.Series, mode: str) -> str:
    tags = [
        f"dataset_{row.dataset}",
        f"query_len_bin_{min(int(row.query_len) // 100, 10)}",
        f"number_count_bin_{min(int(row.number_count), 16)}",
        f"latex_count_bin_{min(int(row.latex_count) // 4, 12)}",
        "has_frac" if int(row.frac_count) else "no_frac",
        "has_sqrt" if int(row.sqrt_count) else "no_sqrt",
    ]
    if mode in {"qwen8", "gemini"}:
        local_answer = str(row[f"{LOCAL_MODEL}_answer"])
        tags.extend(
            [
                f"qwen8_answer_{local_answer}",
                f"qwen8_answer_len_bin_{min(len(local_answer) // 4, 16)}",
            ]
        )
    if mode == "gemini":
        gemini_answer = str(row[f"{GEMINI_MODEL}_answer"])
        tags.extend(
            [
                f"gemini_answer_{gemini_answer}",
                f"gemini_answer_len_bin_{min(len(gemini_answer) // 4, 16)}",
                "qwen8_gemini_agree" if bool(row.qwen8_gemini_agree) else "qwen8_gemini_disagree",
            ]
        )
    return " ".join([str(row.query_text), *tags])


def fit_probability_model(train: pd.DataFrame, target: pd.Series, mode: str) -> Callable[[pd.DataFrame], np.ndarray]:
    y = target.astype(int)
    if y.nunique() < 2:
        constant = float(y.iloc[0]) if len(y) else 0.0
        return lambda frame: np.full(len(frame), constant, dtype=float)
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
    )
    model.fit(train.apply(lambda row: feature_text(row, mode), axis=1), y)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.array([], dtype=float)
        return model.predict_proba(frame.apply(lambda row: feature_text(row, mode), axis=1))[:, 1]

    return predict


def evaluate_actions(
    table: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    split: str,
    lambda_cost: float,
    cost_norm: float,
) -> dict[str, object]:
    qualities: list[float] = []
    utilities: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    local_final: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        if action == "local":
            quality = float(row[f"{LOCAL_MODEL}_quality"])
            cost = 0.0
            latency = float(row[f"{LOCAL_MODEL}_latency"])
            gemini = False
            gpt = False
            local = True
        elif action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"])
            latency = max(float(row[f"{LOCAL_MODEL}_latency"]), float(row[f"{GEMINI_MODEL}_latency"]))
            gemini = True
            gpt = False
            local = False
        elif action == "gpt":
            quality = float(row[f"{GPT_MODEL}_quality"])
            cost = float(row[f"{GPT_MODEL}_cost"])
            latency = float(row[f"{GPT_MODEL}_latency"])
            gemini = False
            gpt = True
            local = False
        elif action == "gemini_then_gpt":
            quality = float(row[f"{GPT_MODEL}_quality"])
            cost = float(row[f"{GEMINI_MODEL}_cost"]) + float(row[f"{GPT_MODEL}_cost"])
            latency = max(float(row[f"{LOCAL_MODEL}_latency"]), float(row[f"{GEMINI_MODEL}_latency"])) + float(
                row[f"{GPT_MODEL}_latency"]
            )
            gemini = True
            gpt = True
            local = False
        elif action == "gemini_then_gpt_guarded":
            quality = (
                float(row[f"{GPT_MODEL}_quality"])
                if bool(row["gpt_answer_available"])
                else float(row[f"{GEMINI_MODEL}_quality"])
            )
            cost = float(row[f"{GEMINI_MODEL}_cost"]) + float(row[f"{GPT_MODEL}_cost"])
            latency = max(float(row[f"{LOCAL_MODEL}_latency"]), float(row[f"{GEMINI_MODEL}_latency"])) + float(
                row[f"{GPT_MODEL}_latency"]
            )
            gemini = True
            gpt = True
            local = False
        else:
            raise ValueError(f"Unknown action: {action}")
        qualities.append(quality)
        costs.append(cost)
        utilities.append(quality - lambda_cost * (cost / cost_norm))
        latencies.append(latency)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
        local_final.append(local)

    oracle_quality = table[[f"{model}_quality" for model in REQUIRED_MODELS]].max(axis=1)
    oracle_utility = table[[f"{model}_utility_selected_cost" for model in REQUIRED_MODELS]].max(axis=1)
    sequential_oracle_utility = table["sequential_cost_oracle_utility"].astype(float)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(np.mean(utilities))
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_oracle_mean_quality": float(oracle_quality.mean()),
        "cost_oracle_mean_utility": float(oracle_utility.mean()),
        "sequential_oracle_mean_utility": float(sequential_oracle_utility.mean()),
        "quality_gap_to_oracle": float(oracle_quality.mean() - mean_quality),
        "utility_gap_to_cost_oracle": float(oracle_utility.mean() - mean_utility),
        "utility_ratio_to_cost_oracle": float(mean_utility / oracle_utility.mean())
        if abs(float(oracle_utility.mean())) > 1e-12
        else np.nan,
        "utility_ratio_to_sequential_oracle": float(mean_utility / sequential_oracle_utility.mean())
        if abs(float(sequential_oracle_utility.mean())) > 1e-12
        else np.nan,
        "remote_cost_total_usd": float(np.sum(costs)),
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum())
        if float(table[f"{GPT_MODEL}_cost"].astype(float).sum()) > 0
        else np.nan,
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gemini_calls, gpt_calls)])),
        "gemini_call_rate": float(np.mean(gemini_calls)),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "local_final_rate": float(np.mean(local_final)),
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
        "action_counts": json.dumps(actions.value_counts().to_dict(), sort_keys=True),
    }


def constant_actions(table: pd.DataFrame, action: str) -> pd.Series:
    return pd.Series(action, index=table.index)


def oracle_actions(table: pd.DataFrame, column: str) -> pd.Series:
    if column == "quality":
        out = []
        for _, row in table.iterrows():
            best = max(REQUIRED_MODELS, key=lambda model: float(row[f"{model}_quality"]))
            out.append("local" if best == LOCAL_MODEL else "gemini" if best == GEMINI_MODEL else "gpt")
        return pd.Series(out, index=table.index)
    if column == "selected_cost":
        out = []
        for _, row in table.iterrows():
            best = max(REQUIRED_MODELS, key=lambda model: float(row[f"{model}_utility_selected_cost"]))
            out.append("local" if best == LOCAL_MODEL else "gemini" if best == GEMINI_MODEL else "gpt")
        return pd.Series(out, index=table.index)
    if column == "sequential":
        return table["sequential_cost_oracle_action"].astype(str).copy()
    raise ValueError(column)


def threshold_actions(table: pd.DataFrame, proba: np.ndarray, threshold: float, *, action_if_true: str) -> pd.Series:
    actions = pd.Series("gemini", index=table.index)
    actions.iloc[np.where(proba >= threshold)[0]] = action_if_true
    return actions


def agreement_else_gpt_guarded_actions(table: pd.DataFrame) -> pd.Series:
    actions = pd.Series("gemini_then_gpt_guarded", index=table.index)
    actions.loc[table["qwen8_gemini_agree"].astype(bool)] = "gemini"
    return actions


def choose_threshold(
    val: pd.DataFrame,
    proba: np.ndarray,
    *,
    action_if_true: str,
    method: str,
    lambda_cost: float,
    cost_norm: float,
    target_quality_gap: float,
) -> float:
    candidates = np.linspace(0.0, 1.0, 51)
    rows = []
    for threshold in candidates:
        actions = threshold_actions(val, proba, float(threshold), action_if_true=action_if_true)
        row = evaluate_actions(
            val,
            actions,
            method=method,
            split="val",
            lambda_cost=lambda_cost,
            cost_norm=cost_norm,
        )
        row["threshold"] = float(threshold)
        rows.append(row)
    scores = pd.DataFrame(rows)
    feasible = scores[scores["quality_gap_to_oracle"].le(target_quality_gap)]
    pool = feasible if not feasible.empty else scores
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs, cost_norm = load_rescored_outputs([Path(p) for p in args.run_dirs], Path(args.manifest), args.lambda_cost)
    query_table = build_query_table(outputs, cost_norm, args.lambda_cost)
    query_table.to_csv(output_dir / "query_table.csv", index=False)

    train = query_table[query_table["split"].eq("train")].copy()
    val = query_table[query_table["split"].eq("val")].copy()
    if train.empty or val.empty:
        raise ValueError("Need non-empty train and val splits for gate threshold selection.")

    target_gpt_rescue = train["gemini_then_gpt_utility"].astype(float) > train[f"{GEMINI_MODEL}_utility_selected_cost"].astype(float)
    target_guarded_rescue = train["gemini_then_gpt_guarded_utility"].astype(float) > train[
        f"{GEMINI_MODEL}_utility_selected_cost"
    ].astype(float)
    query_gate = fit_probability_model(train, target_gpt_rescue, "qwen8")
    gemini_gate = fit_probability_model(train, target_gpt_rescue, "gemini")
    gemini_guarded_gate = fit_probability_model(train, target_guarded_rescue, "gemini")

    val_query_proba = query_gate(val)
    val_gemini_proba = gemini_gate(val)
    val_gemini_guarded_proba = gemini_guarded_gate(val)
    query_threshold = choose_threshold(
        val,
        val_query_proba,
        action_if_true="gpt",
        method="query_qwen8_to_gpt_rescue_gate",
        lambda_cost=args.lambda_cost,
        cost_norm=cost_norm,
        target_quality_gap=args.target_quality_gap,
    )
    gemini_threshold = choose_threshold(
        val,
        val_gemini_proba,
        action_if_true="gemini_then_gpt",
        method="gemini_answer_to_gpt_rescue_gate",
        lambda_cost=args.lambda_cost,
        cost_norm=cost_norm,
        target_quality_gap=args.target_quality_gap,
    )
    gemini_guarded_threshold = choose_threshold(
        val,
        val_gemini_guarded_proba,
        action_if_true="gemini_then_gpt_guarded",
        method="gemini_answer_to_gpt_guarded_rescue_gate",
        lambda_cost=args.lambda_cost,
        cost_norm=cost_norm,
        target_quality_gap=args.target_quality_gap,
    )

    rows: list[dict[str, object]] = []
    for split, split_frame in query_table.groupby("split", sort=False):
        split_frame = split_frame.copy()
        policies = {
            "all_qwen3-8b-local": constant_actions(split_frame, "local"),
            "all_gemini-3.5-flash": constant_actions(split_frame, "gemini"),
            "all_gpt-5.5": constant_actions(split_frame, "gpt"),
            "all_gemini_then_gpt_guarded": constant_actions(split_frame, "gemini_then_gpt_guarded"),
            "qwen8_gemini_agreement_else_gpt_guarded": agreement_else_gpt_guarded_actions(split_frame),
            "quality_oracle": oracle_actions(split_frame, "quality"),
            "cost_aware_oracle_selected_cost": oracle_actions(split_frame, "selected_cost"),
            "sequential_cost_aware_oracle": oracle_actions(split_frame, "sequential"),
            "query_qwen8_to_gpt_rescue_gate": threshold_actions(
                split_frame, query_gate(split_frame), query_threshold, action_if_true="gpt"
            ),
            "gemini_answer_to_gpt_rescue_gate": threshold_actions(
                split_frame, gemini_gate(split_frame), gemini_threshold, action_if_true="gemini_then_gpt"
            ),
            "gemini_answer_to_gpt_guarded_rescue_gate": threshold_actions(
                split_frame,
                gemini_guarded_gate(split_frame),
                gemini_guarded_threshold,
                action_if_true="gemini_then_gpt_guarded",
            ),
        }
        for method, actions in policies.items():
            row = evaluate_actions(
                split_frame,
                actions,
                method=method,
                split=str(split),
                lambda_cost=args.lambda_cost,
                cost_norm=cost_norm,
            )
            row["threshold"] = (
                query_threshold
                if method == "query_qwen8_to_gpt_rescue_gate"
                else gemini_threshold
                if method == "gemini_answer_to_gpt_rescue_gate"
                else gemini_guarded_threshold
                if method == "gemini_answer_to_gpt_guarded_rescue_gate"
                else np.nan
            )
            rows.append(row)
        for dataset, dataset_frame in split_frame.groupby("dataset", sort=True):
            for method in [
                "all_gemini-3.5-flash",
                "all_gpt-5.5",
                "query_qwen8_to_gpt_rescue_gate",
                "gemini_answer_to_gpt_rescue_gate",
                "gemini_answer_to_gpt_guarded_rescue_gate",
                "qwen8_gemini_agreement_else_gpt_guarded",
                "cost_aware_oracle_selected_cost",
                "sequential_cost_aware_oracle",
            ]:
                row = evaluate_actions(
                    dataset_frame,
                    policies[method].loc[dataset_frame.index],
                    method=method,
                    split=f"{split}:{dataset}",
                    lambda_cost=args.lambda_cost,
                    cost_norm=cost_norm,
                )
                row["threshold"] = (
                    query_threshold
                    if method == "query_qwen8_to_gpt_rescue_gate"
                    else gemini_threshold
                    if method == "gemini_answer_to_gpt_rescue_gate"
                    else gemini_guarded_threshold
                    if method == "gemini_answer_to_gpt_guarded_rescue_gate"
                    else np.nan
                )
                rows.append(row)

    results = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    table_path = output_dir / "table_mixed_exact_math_gate.csv"
    results.to_csv(table_path, index=False)

    test_rows = results[results["split"].eq("test")].copy()
    deployable = test_rows[
        test_rows["method"].isin(
            [
                "query_qwen8_to_gpt_rescue_gate",
                "gemini_answer_to_gpt_rescue_gate",
                "gemini_answer_to_gpt_guarded_rescue_gate",
                "qwen8_gemini_agreement_else_gpt_guarded",
            ]
        )
    ]
    best_deployable = deployable.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]
    memo = [
        "# Mixed Exact-Math Gate Memo",
        "",
        f"Run dirs: `{', '.join(args.run_dirs)}`.",
        f"Manifest: `{args.manifest}`.",
        f"Rows with complete `{LOCAL_MODEL}`, `{GEMINI_MODEL}`, and `{GPT_MODEL}` outputs: `{len(query_table)}`.",
        f"Cost normalization: mean all-GPT cost per query = `${cost_norm:.6f}`.",
        "",
        "All rows are rescored from cached parsed/gold answers with the current exact-answer scorer before evaluation.",
        "The Gemini-answer gate is deployable as a sequential policy: run local Qwen8 and Gemini, then call GPT only when the learned gate fires. Its utility charges both Gemini and GPT on rescued rows.",
        "",
        "## Held-Out Test Results",
        "",
        markdown_table(
            test_rows[
                [
                    "method",
                    "n_queries",
                    "mean_quality",
                    "mean_utility",
                    "quality_gap_to_oracle",
                    "utility_ratio_to_cost_oracle",
                    "utility_ratio_to_sequential_oracle",
                    "normalized_remote_cost_vs_all_gpt",
                    "frontier_call_rate",
                    "gpt_call_rate",
                    "remote_cost_total_usd",
                    "p95_latency_s",
                    "action_counts",
                ]
            ]
        ),
        "",
        "## Best Deployable Gate",
        "",
        (
            f"`{best_deployable.method}` quality `{best_deployable.mean_quality:.4f}` leaves "
            f"`{best_deployable.quality_gap_to_oracle:.4f}` absolute quality gap to the quality oracle. "
            f"Utility ratio to selected-cost oracle is `{best_deployable.utility_ratio_to_cost_oracle:.4f}`; "
            f"normalized remote cost versus all-GPT is `{best_deployable.normalized_remote_cost_vs_all_gpt:.4f}`; "
            f"GPT call rate is `{best_deployable.gpt_call_rate:.4f}`."
        ),
        "",
        "## Files",
        "",
        f"- `{table_path}`",
        f"- `{output_dir / 'query_table.csv'}`",
    ]
    memo_path = output_dir / "MIXED_EXACT_MATH_GATE_MEMO.md"
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {table_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
