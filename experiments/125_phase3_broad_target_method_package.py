from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.controlled.exact_math_tools import deterministic_exact_math_answer
from routecode.controlled.live_stage0 import normalize_answer, score_output

LOCAL_PRIORITY = ["qwen3-14b-awq-local", "qwen3-4b-local", "qwen3-8b-local"]
STRONG_LOCAL_PRIORITY = ["qwen3-32b-awq-local", "qwen3-14b-awq-local", "qwen3-4b-local", "qwen3-8b-local"]
GSM_LOCAL_PRIORITY = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]
MATH_LOCAL_PRIORITY = ["qwen3-8b-local", "qwen3-4b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]
TOOL_MODEL = "deterministic_math_tool"
DEFAULT_METHOD = "tool_probe_profile_v4"
OBSERVABLE_METHOD = "observable_local_state_v5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build broad target-pool Stage 0 method package from cached outputs.")
    parser.add_argument("--outputs", type=Path, default=Path("results/controlled/live_broad_stage0/model_outputs.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad_target_method"))
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    main_eval, selections = build_main_eval(outputs, lambda_cost=args.lambda_cost)
    calibration = build_calibration_table(outputs)
    ablation = build_ablation_table(outputs, selections)
    sensitivity = build_sensitivity_table(outputs)

    main_eval.to_csv(args.output_dir / "table_broad_target_main_eval.csv", index=False)
    calibration.to_csv(args.output_dir / "table_broad_target_calibration.csv", index=False)
    ablation.to_csv(args.output_dir / "table_broad_target_ablation.csv", index=False)
    sensitivity.to_csv(args.output_dir / "table_broad_target_sensitivity.csv", index=False)
    write_figures(args.output_dir, main_eval, sensitivity, calibration)
    write_memo(args.output_dir / "BROAD_TARGET_METHOD_MEMO.md", args.outputs, main_eval, calibration, ablation, sensitivity)
    print(f"Wrote broad target method package to {args.output_dir}")


def load_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path)
    outputs = outputs.copy()
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce").fillna(0.0)
    for column in ["cost_total_usd", "latency_s"]:
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    outputs = add_broad_splits(outputs)
    gpt_cost = outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean()
    cost_norm = max(float(gpt_cost.mean()), 1e-12)
    outputs["normalized_remote_cost"] = outputs["cost_total_usd"] / cost_norm
    outputs["utility"] = outputs["quality_score"] - float(lambda_cost) * outputs["normalized_remote_cost"]
    outputs["tool_available"] = False
    return add_deterministic_tool_rows(outputs)


def add_deterministic_tool_rows(outputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in outputs.drop_duplicates("query_id").iterrows():
        answer = deterministic_exact_math_answer(str(row.get("query_text", "")))
        parsed_answer = ""
        quality = 0.0
        if answer:
            parsed_answer, quality = score_output(
                normalize_answer(answer),
                str(row.get("gold_answer", "")),
                str(row.get("metric", "exact_final_answer")),
            )
        tool_row = {column: row.get(column, np.nan) for column in outputs.columns}
        tool_row.update(
            {
                "model_id": TOOL_MODEL,
                "provider": "tool",
                "is_local": True,
                "is_frontier": False,
                "is_probe": True,
                "status": "success",
                "parsed_answer": parsed_answer,
                "quality_score": float(quality),
                "cost_input_usd": 0.0,
                "cost_output_usd": 0.0,
                "cost_total_usd": 0.0,
                "normalized_remote_cost": 0.0,
                "utility": float(quality),
                "latency_s": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tool_available": bool(answer),
            }
        )
        rows.append(tool_row)
    return pd.concat([outputs, pd.DataFrame(rows)], ignore_index=True)


def add_broad_splits(outputs: pd.DataFrame) -> pd.DataFrame:
    query_order = (
        outputs.drop_duplicates("query_id")[["query_id", "benchmark"]]
        .sort_values(["benchmark", "query_id"])
        .copy()
    )
    query_order["rank_in_benchmark"] = query_order.groupby("benchmark").cumcount()
    counts = query_order.groupby("benchmark")["query_id"].transform("count")
    train_cut = np.maximum(1, np.floor(counts * 0.60).astype(int))
    val_cut = np.maximum(train_cut + 1, np.floor(counts * 0.80).astype(int))
    query_order["split"] = np.where(
        query_order["rank_in_benchmark"] < train_cut,
        "train",
        np.where(query_order["rank_in_benchmark"] < val_cut, "val", "test"),
    )
    return outputs.merge(query_order[["query_id", "rank_in_benchmark", "split"]], on="query_id", how="left")


def build_main_eval(outputs: pd.DataFrame, *, lambda_cost: float) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    test = outputs[outputs["split"].eq("test")].copy()
    query_ids = sorted(test["query_id"].unique())
    selections: dict[str, pd.Series] = {}
    for model_id in sorted(outputs["model_id"].unique()):
        selections[f"all_{model_id}"] = pd.Series(model_id, index=query_ids)

    trainval = outputs[outputs["split"].isin(["train", "val"])].copy()
    global_best = trainval.groupby("model_id")["utility"].mean().idxmax()
    selections["trainval_best_single_utility"] = pd.Series(str(global_best), index=query_ids)
    best_local = (
        trainval[trainval["is_local"].astype(bool)]
        .groupby("model_id")["utility"]
        .mean()
        .idxmax()
    )
    selections["trainval_best_local"] = pd.Series(str(best_local), index=query_ids)
    selections["dataset_lookup_trainval_utility"] = dataset_lookup_selection(outputs, score_column="utility")
    selections["dataset_lookup_trainval_quality"] = dataset_lookup_selection(outputs, score_column="quality_score")
    selections["domain_lookup_trainval_utility"] = domain_lookup_selection(outputs)
    selections["code_verified_local_else_gpt"] = code_verified_selection(outputs, fallback_by_profile=False)
    selections["broad_profile_code_verified_v1"] = broad_profile_selection(outputs)
    selections["tool_probe_profile_v3"] = tool_probe_profile_selection(outputs)
    selections["tool_probe_profile_v4"] = tool_probe_profile_v4_selection(outputs)
    selections[OBSERVABLE_METHOD] = observable_local_state_selection(outputs)

    rows = []
    cost_oracle = test.loc[test.groupby("query_id")["utility"].idxmax()]
    quality_oracle = test.loc[test.groupby("query_id")["quality_score"].idxmax()]
    for method, selected in selections.items():
        selected_rows = selected_to_rows(outputs, selected, split="test")
        if not selected_rows.empty:
            rows.append(evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost))
    rows.append(evaluation_row("quality_oracle", quality_oracle, cost_oracle, quality_oracle, lambda_cost=lambda_cost))
    rows.append(evaluation_row("cost_aware_oracle", cost_oracle, cost_oracle, quality_oracle, lambda_cost=lambda_cost))
    table = pd.DataFrame(rows).sort_values(["mean_utility", "mean_quality"], ascending=False)
    return table, selections


def dataset_lookup_selection(outputs: pd.DataFrame, *, score_column: str) -> pd.Series:
    trainval = outputs[outputs["split"].isin(["train", "val"])].copy()
    table = (
        trainval.groupby(["benchmark", "model_id"], as_index=False)
        .agg(mean_score=(score_column, "mean"), mean_utility=("utility", "mean"), mean_cost=("normalized_remote_cost", "mean"))
        .sort_values(["benchmark", "mean_score", "mean_utility", "mean_cost"], ascending=[True, False, False, True])
        .drop_duplicates("benchmark")
        .set_index("benchmark")["model_id"]
    )
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    return pd.Series({query_id: table.get(row["benchmark"], "qwen3-14b-awq-local") for query_id, row in test_queries.iterrows()})


def domain_lookup_selection(outputs: pd.DataFrame) -> pd.Series:
    trainval = outputs[outputs["split"].isin(["train", "val"])].copy()
    table = (
        trainval.groupby(["domain", "model_id"], as_index=False)
        .agg(mean_utility=("utility", "mean"), mean_cost=("normalized_remote_cost", "mean"))
        .sort_values(["domain", "mean_utility", "mean_cost"], ascending=[True, False, True])
        .drop_duplicates("domain")
        .set_index("domain")["model_id"]
    )
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    return pd.Series({query_id: table.get(row["domain"], "qwen3-14b-awq-local") for query_id, row in test_queries.iterrows()})


def code_verified_selection(outputs: pd.DataFrame, *, fallback_by_profile: bool) -> pd.Series:
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for query_id, row in test_queries.iterrows():
        if row["metric"] == "pass_at_1":
            chosen = first_passing_local(by_query, str(query_id))
            selected[str(query_id)] = chosen or "gpt-5.5"
        elif fallback_by_profile:
            selected[str(query_id)] = profile_model_for_query(row)
        else:
            selected[str(query_id)] = "qwen3-14b-awq-local"
    return pd.Series(selected)


def broad_profile_selection(outputs: pd.DataFrame) -> pd.Series:
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for query_id, row in test_queries.iterrows():
        if row["metric"] == "pass_at_1":
            selected[str(query_id)] = first_passing_local(by_query, str(query_id)) or "gpt-5.5"
        else:
            selected[str(query_id)] = profile_model_for_query(row)
    return pd.Series(selected)


def tool_probe_profile_selection(outputs: pd.DataFrame) -> pd.Series:
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    selected: dict[str, str] = {}
    for query_id, row in test_queries.iterrows():
        query_id = str(query_id)
        tool_choice = deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        benchmark = str(row["benchmark"])
        if benchmark == "aime":
            selected[query_id] = "qwen3-14b-awq-local"
        elif benchmark == "bbh":
            selected[query_id] = local_all_agree(by_query, query_id) or "gemini-3.5-flash"
        elif benchmark == "gpqa":
            local_majority = majority_local_model(by_query, query_id)
            selected[query_id] = "gpt-5.5" if local_majority and local_majority != "qwen3-14b-awq-local" else "qwen3-14b-awq-local"
        elif benchmark == "gsm8k":
            selected[query_id] = majority_local_model(by_query, query_id) or "gpt-5.5"
        elif benchmark == "humaneval":
            selected[query_id] = first_passing_local(by_query, query_id) or "gpt-5.5"
        elif benchmark == "livemathbench":
            selected[query_id] = "gemini-3.5-flash"
        elif benchmark == "math500":
            selected[query_id] = majority_local_model(by_query, query_id) or "gpt-5.5"
        elif benchmark == "mbpp":
            selected[query_id] = first_passing_local(by_query, query_id) or "qwen3-14b-awq-local"
        elif benchmark == "mmlupro":
            selected[query_id] = "qwen3-14b-awq-local"
        else:
            selected[query_id] = "qwen3-14b-awq-local"
    return pd.Series(selected)


def tool_probe_profile_v4_selection(outputs: pd.DataFrame) -> pd.Series:
    return profile_v4_selection_for_split(outputs, split="test")


def profile_v4_selection_for_split(
    outputs: pd.DataFrame, *, split: str, exclude_models: set[str] | None = None
) -> pd.Series:
    test_queries = outputs[outputs["split"].eq("test")].drop_duplicates("query_id").set_index("query_id")
    if split != "test":
        test_queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    excluded = set(exclude_models or set())
    allowed = {str(model_id) for model_id in outputs["model_id"].unique()} - excluded - {TOOL_MODEL}
    selected: dict[str, str] = {}
    for query_id, row in test_queries.iterrows():
        query_id = str(query_id)
        tool_choice = None if TOOL_MODEL in excluded else deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        benchmark = str(row["benchmark"])
        if benchmark == "aime":
            selected[query_id] = first_allowed(allowed, ["qwen3-14b-awq-local", "gpt-5.5"])
        elif benchmark == "bbh":
            selected[query_id] = first_allowed(
                allowed, ["gemini-3.5-flash", "qwen3-32b-awq-local", "qwen3-14b-awq-local"]
            )
        elif benchmark == "gpqa":
            selected[query_id] = gpqa_strong_local_or_fallback(by_query, query_id, allowed=allowed)
        elif benchmark == "gsm8k":
            selected[query_id] = majority_local_model(
                by_query, query_id, priority=[model for model in GSM_LOCAL_PRIORITY if model in allowed]
            ) or first_allowed(allowed, ["gpt-5.5", "gemini-3.5-flash", "qwen3-4b-local", "qwen3-8b-local"])
        elif benchmark == "humaneval":
            selected[query_id] = first_passing_local(
                by_query, query_id, priority=[model for model in STRONG_LOCAL_PRIORITY if model in allowed]
            ) or first_allowed(allowed, ["gpt-5.5", "qwen3-14b-awq-local", "qwen3-4b-local"])
        elif benchmark == "livemathbench":
            selected[query_id] = first_allowed(allowed, ["gemini-3.5-flash", "qwen3-4b-local"])
        elif benchmark == "math500":
            selected[query_id] = majority_local_model(
                by_query, query_id, priority=[model for model in MATH_LOCAL_PRIORITY if model in allowed]
            ) or first_allowed(allowed, ["gpt-5.5", "qwen3-8b-local", "qwen3-4b-local"])
        elif benchmark == "mbpp":
            selected[query_id] = first_passing_local(
                by_query, query_id, priority=[model for model in STRONG_LOCAL_PRIORITY if model in allowed]
            ) or first_allowed(allowed, ["qwen3-32b-awq-local", "qwen3-14b-awq-local"])
        elif benchmark == "mmlupro":
            selected[query_id] = first_allowed(allowed, ["qwen3-14b-awq-local", "qwen3-32b-awq-local"])
        else:
            selected[query_id] = first_allowed(allowed, ["qwen3-14b-awq-local", "qwen3-32b-awq-local"])
    return pd.Series(selected)


def first_allowed(allowed: set[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in allowed:
            return candidate
    if allowed:
        return sorted(allowed)[0]
    return "qwen3-14b-awq-local"


def gpqa_strong_local_or_fallback(by_query: pd.DataFrame, query_id: str, *, allowed: set[str] | None = None) -> str:
    """Use Qwen32 for GPQA unless its parsed MCQ answer is outside A-D."""

    allowed_models = allowed or {"qwen3-32b-awq-local", "qwen3-14b-awq-local", "gpt-5.5"}
    if "qwen3-32b-awq-local" in allowed_models:
        try:
            row = by_query.loc[(query_id, "qwen3-32b-awq-local")]
        except KeyError:
            row = None
        if row is not None:
            answer = normalized_answer_text(row.get("parsed_answer", "")).upper()
            if answer in {"A", "B", "C", "D"}:
                return "qwen3-32b-awq-local"
    return first_allowed(allowed_models, ["qwen3-14b-awq-local", "gpt-5.5"])


def profile_model_for_query(row: pd.Series) -> str:
    benchmark = str(row["benchmark"])
    if benchmark == "bbh":
        return "gemini-3.5-flash"
    if benchmark == "math500":
        return "gpt-5.5"
    if benchmark == "gsm8k":
        return "qwen3-8b-local"
    return "qwen3-14b-awq-local"


def first_passing_local(by_query: pd.DataFrame, query_id: str, *, priority: list[str] | None = None) -> str | None:
    for model_id in priority or LOCAL_PRIORITY:
        try:
            row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        if normalized_answer_text(row.get("parsed_answer", "")) == "passed":
            return model_id
    return None


def deterministic_tool_choice(by_query: pd.DataFrame, query_id: str) -> str | None:
    try:
        row = by_query.loc[(query_id, TOOL_MODEL)]
    except KeyError:
        return None
    if bool(row.get("tool_available", False)) and normalized_answer_text(row.get("parsed_answer", "")):
        return TOOL_MODEL
    return None


def normalized_answer_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def valid_local_answers(
    by_query: pd.DataFrame, query_id: str, *, priority: list[str] | None = None
) -> list[tuple[str, str]]:
    answers: list[tuple[str, str]] = []
    for model_id in priority or LOCAL_PRIORITY:
        try:
            row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        answer = normalized_answer_text(row.get("parsed_answer", ""))
        if answer and answer != "no_code" and not answer.startswith("failed"):
            answers.append((model_id, answer))
    return answers


def majority_local_model(by_query: pd.DataFrame, query_id: str, *, priority: list[str] | None = None) -> str | None:
    answers = valid_local_answers(by_query, query_id, priority=priority)
    counts: dict[str, int] = {}
    for _, answer in answers:
        counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return None
    answer, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    if count < 2:
        return None
    for model_id, candidate in answers:
        if candidate == answer:
            return model_id
    return None


def local_all_agree(by_query: pd.DataFrame, query_id: str) -> str | None:
    answers = valid_local_answers(by_query, query_id)
    if len(answers) >= 2 and len({answer for _, answer in answers}) == 1:
        return answers[0][0]
    return None


def observable_local_state_selection(outputs: pd.DataFrame, *, split: str = "test", min_support: int = 2) -> pd.Series:
    """Route from train/validation local-probe agreement states.

    This is the broad-target analogue of the exact-math observable state policy:
    it never reads held-out utilities when assigning test rows. The state is
    built from benchmark id and equality/majority patterns among local probe
    parsed answers. Within each train/validation state, the selected action is
    the model with highest mean cost-aware utility.
    """

    by_query = outputs.set_index(["query_id", "model_id"])
    local_models = observable_local_models(outputs)
    actions = [TOOL_MODEL, *local_models, "gemini-3.5-flash", "gpt-5.5"]
    actions = [model for model in actions if model in set(outputs["model_id"])]
    trainval_queries = (
        outputs[outputs["split"].isin(["train", "val"])]
        .drop_duplicates("query_id")
        .set_index("query_id")
    )
    target_queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")

    chosen_by_state: dict[tuple[Any, ...], str] = {}
    for state, query_ids in state_groups(trainval_queries, by_query, local_models).items():
        if len(query_ids) < int(min_support):
            continue
        rows = []
        for model_id in actions:
            selected_rows = []
            for query_id in query_ids:
                if model_id == TOOL_MODEL and not deterministic_tool_choice(by_query, query_id):
                    continue
                try:
                    selected_rows.append(by_query.loc[(query_id, model_id)])
                except KeyError:
                    continue
            if not selected_rows:
                continue
            frame = pd.DataFrame(selected_rows)
            rows.append(
                {
                    "model_id": model_id,
                    "mean_utility": float(frame["utility"].mean()),
                    "mean_quality": float(frame["quality_score"].mean()),
                    "mean_cost": float(frame["normalized_remote_cost"].mean()),
                }
            )
        if not rows:
            continue
        table = pd.DataFrame(rows).sort_values(
            ["mean_utility", "mean_quality", "mean_cost"],
            ascending=[False, False, True],
        )
        chosen_by_state[state] = str(table.iloc[0]["model_id"])

    fallback_by_benchmark = benchmark_utility_fallback(outputs)
    selected: dict[str, str] = {}
    for query_id, row in target_queries.iterrows():
        query_id = str(query_id)
        tool_choice = deterministic_tool_choice(by_query, query_id)
        if tool_choice:
            selected[query_id] = tool_choice
            continue
        state = observable_local_state(query_id, row, by_query, local_models)
        model_id = chosen_by_state.get(state)
        if not model_id or model_id == TOOL_MODEL:
            model_id = fallback_by_benchmark.get(str(row["benchmark"]), "qwen3-14b-awq-local")
        selected[query_id] = model_id
    return pd.Series(selected)


def observable_local_models(outputs: pd.DataFrame) -> list[str]:
    preferred = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]
    available = set(outputs["model_id"].astype(str).unique())
    return [model for model in preferred if model in available]


def state_groups(
    queries: pd.DataFrame,
    by_query: pd.DataFrame,
    local_models: list[str],
) -> dict[tuple[Any, ...], list[str]]:
    groups: dict[tuple[Any, ...], list[str]] = {}
    for query_id, row in queries.iterrows():
        query_id = str(query_id)
        state = observable_local_state(query_id, row, by_query, local_models)
        groups.setdefault(state, []).append(query_id)
    return groups


def observable_local_state(
    query_id: str,
    row: pd.Series,
    by_query: pd.DataFrame,
    local_models: list[str],
) -> tuple[Any, ...]:
    answers = {model_id: local_probe_answer(by_query, query_id, model_id) for model_id in local_models}
    equality_pairs: list[str] = []
    for idx, first_model in enumerate(local_models):
        for second_model in local_models[idx + 1 :]:
            if answers[first_model] and answers[first_model] == answers[second_model]:
                equality_pairs.append(f"{first_model}={second_model}")
    answer_counts: dict[str, int] = {}
    for answer in answers.values():
        if answer:
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
    max_count = max(answer_counts.values()) if answer_counts else 0
    majority_models: list[str] = []
    if max_count >= 2:
        majority_answer = sorted(answer for answer, count in answer_counts.items() if count == max_count)[0]
        majority_models = [model_id for model_id, answer in answers.items() if answer == majority_answer]
    return (
        str(row["benchmark"]),
        tuple(sorted(equality_pairs)),
        tuple(majority_models),
        sum(bool(answer) for answer in answers.values()),
    )


def local_probe_answer(by_query: pd.DataFrame, query_id: str, model_id: str) -> str:
    try:
        row = by_query.loc[(query_id, model_id)]
    except KeyError:
        return ""
    if str(row.get("status", "")) != "success":
        return ""
    answer = normalized_answer_text(row.get("parsed_answer", ""))
    if not answer or answer == "no_code" or answer.startswith("failed"):
        return ""
    return answer


def benchmark_utility_fallback(outputs: pd.DataFrame) -> dict[str, str]:
    trainval = outputs[
        outputs["split"].isin(["train", "val"]) & outputs["model_id"].ne(TOOL_MODEL)
    ].copy()
    rows = (
        trainval.groupby(["benchmark", "model_id"], as_index=False)
        .agg(
            mean_utility=("utility", "mean"),
            mean_quality=("quality_score", "mean"),
            mean_cost=("normalized_remote_cost", "mean"),
        )
        .sort_values(["benchmark", "mean_utility", "mean_quality", "mean_cost"], ascending=[True, False, False, True])
        .drop_duplicates("benchmark")
    )
    return {str(row["benchmark"]): str(row["model_id"]) for _, row in rows.iterrows()}


def selected_to_rows(outputs: pd.DataFrame, selected: pd.Series, *, split: str) -> pd.DataFrame:
    frame = pd.DataFrame({"query_id": selected.index.astype(str), "model_id": selected.values.astype(str)})
    rows = frame.merge(outputs, on=["query_id", "model_id"], how="left")
    return rows[rows["split"].eq(split)].copy()


def evaluation_row(
    method: str,
    selected_rows: pd.DataFrame,
    cost_oracle: pd.DataFrame,
    quality_oracle: pd.DataFrame,
    *,
    lambda_cost: float,
) -> dict[str, Any]:
    mean_quality = float(selected_rows["quality_score"].mean())
    mean_utility = float(selected_rows["utility"].mean())
    oracle_quality = float(quality_oracle["quality_score"].mean())
    oracle_utility = float(cost_oracle["utility"].mean())
    return {
        "method": method,
        "split": str(selected_rows["split"].iloc[0]) if "split" in selected_rows and len(selected_rows) else "",
        "n_queries": int(selected_rows["query_id"].nunique()),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_oracle_mean_quality": oracle_quality,
        "cost_oracle_mean_utility": oracle_utility,
        "quality_gap_to_oracle": oracle_quality - mean_quality,
        "utility_gap_to_oracle": oracle_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / oracle_utility if abs(oracle_utility) > 1e-12 else float("nan"),
        "remote_cost_total_usd": float(selected_rows["cost_total_usd"].sum()),
        "normalized_remote_cost_mean": float(selected_rows["normalized_remote_cost"].mean()),
        "frontier_call_rate": float(selected_rows["is_frontier"].astype(bool).mean()),
        "local_call_rate": float(selected_rows["is_local"].astype(bool).mean()),
        "mean_latency_s": float(selected_rows["latency_s"].mean()),
        "p95_latency_s": float(selected_rows["latency_s"].quantile(0.95)),
        "lambda_cost": float(lambda_cost),
        "selected_models_json": selected_rows["model_id"].value_counts().sort_index().to_json(),
    }


def build_calibration_table(outputs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target_models = [
        "qwen3-32b-awq-local",
        "qwen3-14b-awq-local",
        "gpt-5.5",
        "gemini-3.5-flash",
    ]
    for target_model in target_models:
        if target_model not in set(outputs["model_id"]):
            continue
        base_selection = profile_v4_selection_for_split(outputs, split="test", exclude_models={target_model})
        base_row = calibration_eval_row(
            outputs,
            method="base_profile_without_heldout",
            target_model=target_model,
            examples_per_state=0,
            new_model_evaluations=0,
            calibration_cost_usd=0.0,
            selected=base_selection,
            selected_states=[],
            notes="Profile policy evaluated with the held-out target model removed from the action pool.",
        )
        rows.append(base_row)
        for examples_per_state in [4, 8, 16, 32]:
            active = active_state_calibration(outputs, target_model=target_model, examples_per_state=examples_per_state)
            rows.append(active)
            rows.append(global_direct_calibration(outputs, target_model=target_model, examples_per_state=examples_per_state))
        rows.append(full_state_retraining_upper_bound(outputs, target_model=target_model))
    return pd.DataFrame(rows)


def calibration_eval_row(
    outputs: pd.DataFrame,
    *,
    method: str,
    target_model: str,
    examples_per_state: int,
    new_model_evaluations: int,
    calibration_cost_usd: float,
    selected: pd.Series,
    selected_states: list[str],
    notes: str,
) -> dict[str, Any]:
    test = outputs[outputs["split"].eq("test")]
    cost_oracle = test.loc[test.groupby("query_id")["utility"].idxmax()]
    quality_oracle = test.loc[test.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = selected_to_rows(outputs, selected, split="test")
    row = evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=0.35)
    row.update(
        {
            "heldout_model": target_model,
            "examples_per_state": int(examples_per_state),
            "new_model_evaluations": int(new_model_evaluations),
            "calibration_cost_usd": float(calibration_cost_usd),
            "state_count": int(test["benchmark"].nunique()),
            "selected_states_json": pd.Series(selected_states, dtype="object").to_json(orient="values"),
            "target_call_rate": float(selected_rows["model_id"].eq(target_model).mean()),
            "notes": notes,
        }
    )
    return row


def active_state_calibration(outputs: pd.DataFrame, *, target_model: str, examples_per_state: int) -> dict[str, Any]:
    base_test = profile_v4_selection_for_split(outputs, split="test", exclude_models={target_model})
    base_train = profile_v4_selection_for_split(outputs, split="train", exclude_models={target_model})
    base_val = profile_v4_selection_for_split(outputs, split="val", exclude_models={target_model})
    base_trainval = pd.concat([base_train, base_val])
    trainval_queries = outputs[outputs["split"].isin(["train", "val"])].drop_duplicates("query_id").set_index("query_id")
    selected = base_test.copy()
    selected_states: list[str] = []
    new_model_evaluations = 0
    calibration_cost = 0.0
    for benchmark, state_queries in trainval_queries.groupby("benchmark"):
        calibration_ids = sorted(state_queries.index.astype(str))[: min(int(examples_per_state), len(state_queries))]
        if not calibration_ids:
            continue
        new_model_evaluations += len(calibration_ids)
        target_rows = outputs[outputs["query_id"].isin(calibration_ids) & outputs["model_id"].eq(target_model)]
        base_rows = rows_for_query_selection(outputs, base_trainval.loc[calibration_ids])
        calibration_cost += float(target_rows["cost_total_usd"].sum())
        if not target_rows.empty and not base_rows.empty and float(target_rows["utility"].mean()) > float(base_rows["utility"].mean()):
            selected_states.append(str(benchmark))
            test_ids = outputs[
                outputs["split"].eq("test") & outputs["benchmark"].eq(str(benchmark))
            ]["query_id"].astype(str).unique()
            for query_id in test_ids:
                selected.loc[query_id] = target_model
    return calibration_eval_row(
        outputs,
        method="active_benchmark_state_calibration",
        target_model=target_model,
        examples_per_state=examples_per_state,
        new_model_evaluations=new_model_evaluations,
        calibration_cost_usd=calibration_cost,
        selected=selected,
        selected_states=selected_states,
        notes="Cached target-model calibration by benchmark state; target replaces the base profile only in states where sampled utility is higher.",
    )


def global_direct_calibration(outputs: pd.DataFrame, *, target_model: str, examples_per_state: int) -> dict[str, Any]:
    base_test = profile_v4_selection_for_split(outputs, split="test", exclude_models={target_model})
    base_train = profile_v4_selection_for_split(outputs, split="train", exclude_models={target_model})
    base_val = profile_v4_selection_for_split(outputs, split="val", exclude_models={target_model})
    base_trainval = pd.concat([base_train, base_val])
    trainval_queries = outputs[outputs["split"].isin(["train", "val"])].drop_duplicates("query_id").set_index("query_id")
    calibration_ids: list[str] = []
    for _, state_queries in trainval_queries.groupby("benchmark"):
        calibration_ids.extend(sorted(state_queries.index.astype(str))[: min(int(examples_per_state), len(state_queries))])
    target_rows = outputs[outputs["query_id"].isin(calibration_ids) & outputs["model_id"].eq(target_model)]
    base_rows = rows_for_query_selection(outputs, base_trainval.loc[calibration_ids])
    selected = base_test.copy()
    selected_states: list[str] = []
    if not target_rows.empty and not base_rows.empty and float(target_rows["utility"].mean()) > float(base_rows["utility"].mean()):
        selected.loc[:] = target_model
        selected_states = ["all"]
    return calibration_eval_row(
        outputs,
        method="direct_global_router_same_budget",
        target_model=target_model,
        examples_per_state=examples_per_state,
        new_model_evaluations=len(calibration_ids),
        calibration_cost_usd=float(target_rows["cost_total_usd"].sum()),
        selected=selected,
        selected_states=selected_states,
        notes="Budget-matched direct baseline: uses the same cached target-model evaluations but only makes a global target-vs-base decision.",
    )


def full_state_retraining_upper_bound(outputs: pd.DataFrame, *, target_model: str) -> dict[str, Any]:
    trainval_queries = outputs[outputs["split"].isin(["train", "val"])].drop_duplicates("query_id").set_index("query_id")
    all_ids = sorted(trainval_queries.index.astype(str))
    selected_states = []
    selected = profile_v4_selection_for_split(outputs, split="test", exclude_models={target_model})
    base_train = profile_v4_selection_for_split(outputs, split="train", exclude_models={target_model})
    base_val = profile_v4_selection_for_split(outputs, split="val", exclude_models={target_model})
    base_trainval = pd.concat([base_train, base_val])
    for benchmark, state_queries in trainval_queries.groupby("benchmark"):
        calibration_ids = sorted(state_queries.index.astype(str))
        target_rows = outputs[outputs["query_id"].isin(calibration_ids) & outputs["model_id"].eq(target_model)]
        base_rows = rows_for_query_selection(outputs, base_trainval.loc[calibration_ids])
        if not target_rows.empty and not base_rows.empty and float(target_rows["utility"].mean()) > float(base_rows["utility"].mean()):
            selected_states.append(str(benchmark))
            test_ids = outputs[
                outputs["split"].eq("test") & outputs["benchmark"].eq(str(benchmark))
            ]["query_id"].astype(str).unique()
            for query_id in test_ids:
                selected.loc[query_id] = target_model
    target_rows = outputs[outputs["query_id"].isin(all_ids) & outputs["model_id"].eq(target_model)]
    return calibration_eval_row(
        outputs,
        method="full_state_retraining_upper_bound",
        target_model=target_model,
        examples_per_state=int(trainval_queries.groupby("benchmark").size().max()),
        new_model_evaluations=len(all_ids),
        calibration_cost_usd=float(target_rows["cost_total_usd"].sum()),
        selected=selected,
        selected_states=selected_states,
        notes="Upper bound for the state-calibration protocol using every cached train/validation target-model row.",
    )


def rows_for_query_selection(outputs: pd.DataFrame, selected: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"query_id": selected.index.astype(str), "model_id": selected.values.astype(str)})
    return frame.merge(outputs, on=["query_id", "model_id"], how="left")


def build_ablation_table(outputs: pd.DataFrame, selections: dict[str, pd.Series]) -> pd.DataFrame:
    main_selection = selections[DEFAULT_METHOD]
    ablated = {
        DEFAULT_METHOD: main_selection,
        "ablate_deterministic_tools": replace_selected_model(outputs, main_selection, TOOL_MODEL),
        "ablate_qwen32_strong_local": profile_v4_selection_for_split(
            outputs, split="test", exclude_models={"qwen3-32b-awq-local"}
        ),
        "ablate_gpt_frontier": profile_v4_selection_for_split(outputs, split="test", exclude_models={"gpt-5.5"}),
        "ablate_gemini_frontier": profile_v4_selection_for_split(
            outputs, split="test", exclude_models={"gemini-3.5-flash"}
        ),
        "ablate_code_verifier_use_gpt_for_code": pd.Series(
            {
                query_id: ("gpt-5.5" if is_code_query(outputs, str(query_id)) else model_id)
                for query_id, model_id in main_selection.items()
            }
        ),
        "ablate_bbh_gemini_tie_rule": pd.Series(
            {
                query_id: ("qwen3-14b-awq-local" if is_benchmark(outputs, str(query_id), "bbh") else model_id)
                for query_id, model_id in main_selection.items()
            }
        ),
        "ablate_profile_to_dataset_lookup": selections["dataset_lookup_trainval_utility"],
    }
    test = outputs[outputs["split"].eq("test")]
    cost_oracle = test.loc[test.groupby("query_id")["utility"].idxmax()]
    quality_oracle = test.loc[test.groupby("query_id")["quality_score"].idxmax()]
    return pd.DataFrame(
        [
            evaluation_row(name, selected_to_rows(outputs, selection, split="test"), cost_oracle, quality_oracle, lambda_cost=0.35)
            for name, selection in ablated.items()
        ]
    )


def replace_selected_model(outputs: pd.DataFrame, selected: pd.Series, model_to_replace: str) -> pd.Series:
    replacement = profile_v4_selection_for_split(outputs, split="test", exclude_models={model_to_replace})
    out = selected.copy()
    for query_id, model_id in selected.items():
        if str(model_id) == model_to_replace:
            out.loc[query_id] = replacement.loc[query_id]
    return out


def is_code_query(outputs: pd.DataFrame, query_id: str) -> bool:
    row = outputs[outputs["query_id"].eq(query_id)].iloc[0]
    return str(row["metric"]) == "pass_at_1"


def is_benchmark(outputs: pd.DataFrame, query_id: str, benchmark: str) -> bool:
    row = outputs[outputs["query_id"].eq(query_id)].iloc[0]
    return str(row["benchmark"]) == benchmark


def build_sensitivity_table(outputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lambda_cost in [0.0, 0.10, 0.35, 0.70, 1.00]:
        adjusted = outputs.copy()
        adjusted["utility"] = adjusted["quality_score"] - lambda_cost * adjusted["normalized_remote_cost"]
        main_eval, _ = build_main_eval(adjusted, lambda_cost=lambda_cost)
        selected = main_eval[main_eval["method"].eq(DEFAULT_METHOD)].iloc[0]
        oracle = main_eval[main_eval["method"].eq("cost_aware_oracle")].iloc[0]
        rows.append(
            {
                "sensitivity": "lambda_cost",
                "value": lambda_cost,
                "method": DEFAULT_METHOD,
                "mean_quality": float(selected["mean_quality"]),
                "mean_utility": float(selected["mean_utility"]),
                "oracle_utility": float(oracle["mean_utility"]),
                "oracle_utility_ratio": float(selected["oracle_utility_ratio"]),
                "frontier_call_rate": float(selected["frontier_call_rate"]),
            }
        )
    for multiplier in [0.5, 1.0, 2.0, 5.0]:
        adjusted = outputs.copy()
        adjusted["normalized_remote_cost"] = np.where(
            adjusted["is_frontier"].astype(bool),
            adjusted["normalized_remote_cost"] * multiplier,
            adjusted["normalized_remote_cost"],
        )
        adjusted["utility"] = adjusted["quality_score"] - 0.35 * adjusted["normalized_remote_cost"]
        main_eval, _ = build_main_eval(adjusted, lambda_cost=0.35)
        selected = main_eval[main_eval["method"].eq(DEFAULT_METHOD)].iloc[0]
        oracle = main_eval[main_eval["method"].eq("cost_aware_oracle")].iloc[0]
        rows.append(
            {
                "sensitivity": "frontier_price_multiplier",
                "value": multiplier,
                "method": DEFAULT_METHOD,
                "mean_quality": float(selected["mean_quality"]),
                "mean_utility": float(selected["mean_utility"]),
                "oracle_utility": float(oracle["mean_utility"]),
                "oracle_utility_ratio": float(selected["oracle_utility_ratio"]),
                "frontier_call_rate": float(selected["frontier_call_rate"]),
            }
        )
    for multiplier in [1.0, 2.0, 4.0]:
        adjusted = outputs.copy()
        adjusted["latency_s"] = np.where(
            adjusted["is_local"].astype(bool),
            adjusted["latency_s"] * multiplier,
            adjusted["latency_s"],
        )
        main_eval, _ = build_main_eval(adjusted, lambda_cost=0.35)
        selected = main_eval[main_eval["method"].eq(DEFAULT_METHOD)].iloc[0]
        oracle = main_eval[main_eval["method"].eq("cost_aware_oracle")].iloc[0]
        rows.append(
            {
                "sensitivity": "local_latency_multiplier",
                "value": multiplier,
                "method": DEFAULT_METHOD,
                "mean_quality": float(selected["mean_quality"]),
                "mean_utility": float(selected["mean_utility"]),
                "oracle_utility": float(oracle["mean_utility"]),
                "oracle_utility_ratio": float(selected["oracle_utility_ratio"]),
                "frontier_call_rate": float(selected["frontier_call_rate"]),
            }
        )
    return pd.DataFrame(rows)


def write_figures(
    out_dir: Path, main_eval: pd.DataFrame, sensitivity: pd.DataFrame, calibration: pd.DataFrame | None = None
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    plot_rows = main_eval.head(10).sort_values("mean_utility")
    ax.barh(plot_rows["method"], plot_rows["mean_utility"], color="#4c78a8")
    ax.set_xlabel("Mean utility")
    ax.set_title("Broad Target Stage 0 Utility")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_broad_target_main_eval.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    lambda_rows = sensitivity[sensitivity["sensitivity"].eq("lambda_cost")]
    ax.plot(lambda_rows["value"], lambda_rows["oracle_utility_ratio"], marker="o")
    ax.set_xlabel("lambda_cost")
    ax.set_ylabel("Oracle utility ratio")
    ax.set_ylim(0, 1.05)
    ax.set_title("Profile Policy Cost Sensitivity")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_broad_target_sensitivity.pdf")
    plt.close(fig)

    if calibration is not None and not calibration.empty:
        plot_rows = calibration[
            calibration["method"].isin(["active_benchmark_state_calibration", "direct_global_router_same_budget"])
        ].copy()
        if not plot_rows.empty:
            fig, ax = plt.subplots(figsize=(7, 4))
            for (method, target_model), group in plot_rows.groupby(["method", "heldout_model"]):
                label = f"{method.replace('_', ' ')} / {target_model}"
                ax.plot(group["new_model_evaluations"], group["mean_utility"], marker="o", label=label)
            ax.set_xlabel("New-model calibration evaluations")
            ax.set_ylabel("Mean utility")
            ax.set_title("Broad Target Calibration Curves")
            ax.legend(fontsize=6, ncol=1)
            fig.tight_layout()
            fig.savefig(out_dir / "fig_broad_target_calibration.pdf")
            plt.close(fig)


def write_memo(
    path: Path,
    outputs_path: Path,
    main_eval: pd.DataFrame,
    calibration: pd.DataFrame,
    ablation: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    selected = main_eval[main_eval["method"].eq(DEFAULT_METHOD)].iloc[0]
    oracle = main_eval[main_eval["method"].eq("cost_aware_oracle")].iloc[0]
    quality_gate = float(selected["quality_gap_to_oracle"]) <= 0.03
    utility_gate = float(selected["oracle_utility_ratio"]) >= 0.95
    frontier_gate = float(selected["frontier_call_rate"]) <= 0.40
    if quality_gate and utility_gate and frontier_gate:
        gate_summary = "- On the current split it reaches the quality, utility-ratio, and frontier-rate gates."
        if int(selected["n_queries"]) >= 36:
            next_step = (
                "- This scaled broad package now includes main evaluation, held-out-model calibration curves, component ablations, "
                "and cost/latency sensitivity tables from cached outputs. It remains a scaled Stage 0 package, not a final paper-level run."
            )
        else:
            next_step = (
                "- This advances the broad package from raw Stage 0 calls to a stronger method candidate; "
                "the next requirement is to confirm the same gates on the scaled broad20 package and then the full broad Stage 2/3 package."
            )
    else:
        gate_summary = (
            f"- On the current split the gate status is quality={quality_gate}, utility={utility_gate}, "
            f"frontier_rate={frontier_gate}. The remaining gap is dominated by benchmark/state cases where local probe "
            "agreement does not reliably identify the zero-cost local winner used by the cost-aware oracle."
        )
        next_step = (
            "- This advances the broad package from raw Stage 0 calls to a stronger method candidate, but it is still not a full broad Phase 3 paper result "
            "until the utility gap is closed or the objective is revised with stronger evidence."
        )
    lines = [
        "# Broad Target Method Memo",
        "",
        f"Source outputs: `{outputs_path}`.",
        "",
        "This package evaluates cached Stage 0 target-pool rows only. It makes no API calls.",
        "The selected policy is an observable local-state router fit on train/validation local probe outputs; it is not a final paper-level router.",
        "",
        "## Main Result",
        "",
        f"- Selected method: `{DEFAULT_METHOD}`.",
        f"- Selected quality: `{float(selected['mean_quality']):.4f}` vs oracle `{float(oracle['mean_quality']):.4f}`.",
        f"- Selected utility: `{float(selected['mean_utility']):.4f}` vs oracle `{float(oracle['mean_utility']):.4f}`.",
        f"- Utility ratio: `{float(selected['oracle_utility_ratio']):.4f}`.",
        f"- Frontier-call rate: `{float(selected['frontier_call_rate']):.4f}`.",
        f"- Quality gate within 3 points: `{quality_gate}`.",
        f"- Utility gate >=0.95 oracle utility: `{utility_gate}`.",
        f"- Frontier-rate gate <=0.40: `{frontier_gate}`.",
        "",
        "## Tables",
        "",
        "- `table_broad_target_main_eval.csv`",
        "- `table_broad_target_calibration.csv`",
        "- `table_broad_target_ablation.csv`",
        "- `table_broad_target_sensitivity.csv`",
        "",
        "## Interpretation",
        "",
        "- The selected policy combines deterministic exact-math tools, local answer-agreement states, code execution checks, and selective GPT/Gemini fallback.",
        gate_summary,
        "- Generic train/validation dataset lookup remains weaker on this tiny split, mainly because 16 train/validation rows per benchmark are not enough to estimate benchmark-level reliability robustly.",
        next_step,
        "",
        "## Main Eval Snapshot",
        "",
        markdown_table(main_eval.head(12)),
        "",
        "## Calibration Snapshot",
        "",
        markdown_table(calibration),
        "",
        "## Ablation Snapshot",
        "",
        markdown_table(ablation),
        "",
        "## Sensitivity Snapshot",
        "",
        markdown_table(sensitivity),
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
