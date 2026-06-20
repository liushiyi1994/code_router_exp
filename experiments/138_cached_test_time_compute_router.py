from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"
LOCAL_MODELS = ["qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local", "qwen3-32b-awq-local"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cached test-time-compute gates over broad100 strong actions.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_gemini_strong_solver/model_outputs_with_gemini_strong.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_cached_test_time_compute_router"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--gain-thresholds", default="0.0,0.01,0.03,0.05")
    parser.add_argument("--extra-strong-rate-caps", default="0.10,0.20,0.30,0.40")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    outputs = load_outputs(args.outputs, lambda_cost=float(args.lambda_cost))
    query_cache = build_query_cache(outputs)
    gain_thresholds = [float(item) for item in args.gain_thresholds.split(",") if item.strip()]
    rate_caps = [float(item) for item in args.extra_strong_rate_caps.split(",") if item.strip()]

    all_rows, gate_rows = run_policy_grid(
        package,
        outputs,
        query_cache,
        lambda_cost=float(args.lambda_cost),
        min_support=int(args.min_support),
        gain_thresholds=gain_thresholds,
        rate_caps=rate_caps,
    )
    selected = validation_selected_rows(all_rows)
    all_rows.to_csv(args.output_dir / "table_cached_ttc_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_cached_ttc_policy_selected.csv", index=False)
    gate_rows.to_csv(args.output_dir / "table_cached_ttc_state_gates.csv", index=False)
    write_figure(args.output_dir, all_rows)
    write_memo(args.output_dir / "CACHED_TEST_TIME_COMPUTE_ROUTER_MEMO.md", args.outputs, all_rows, selected, gate_rows)
    print(f"Wrote cached test-time-compute router results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_outputs(path: Path, *, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    gpt_norm = max(
        float(outputs[outputs["model_id"].eq("gpt-5.5")].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )
    outputs["normalized_remote_cost"] = outputs["cost_total_usd"].astype(float) / gpt_norm
    outputs["quality_score"] = pd.to_numeric(outputs["quality_score"], errors="coerce").fillna(0.0)
    outputs["utility"] = outputs["quality_score"].astype(float) - float(lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    return outputs


def run_policy_grid(
    package,
    outputs: pd.DataFrame,
    query_cache: dict[str, dict[str, Any]],
    *,
    lambda_cost: float,
    min_support: int,
    gain_thresholds: list[float],
    rate_caps: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_policies = {
        "observable_local_state_v5": {
            split: normalize_selection(package.observable_local_state_selection(outputs, split=split))
            for split in ["val", "test"]
        },
        "tool_probe_profile_v4": {
            split: normalize_selection(
                package.profile_v4_selection_for_split(outputs, split=split, exclude_models={STRONG_MODEL_ID})
            )
            for split in ["val", "test"]
        },
    }
    rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows.append(evaluate_policy(package, outputs, all_strong_selection(outputs, split), "all_gemini_strong", split, "constant", lambda_cost))
        for base_name, selections in base_policies.items():
            base = selections[split]
            rows.append(evaluate_policy(package, outputs, base, base_name, split, "base", lambda_cost))
            diagnostic = oracle_between_base_and_strong(outputs, base)
            rows.append(
                evaluate_policy(
                    package,
                    outputs,
                    diagnostic,
                    f"{base_name}_oracle_between_base_and_gemini_strong",
                    split,
                    "diagnostic_oracle",
                    lambda_cost,
                )
            )

    for base_name, selections in base_policies.items():
        val_base = selections["val"]
        for state_view in ["benchmark", "benchmark_metric", "benchmark_base_model", "benchmark_local_stats"]:
            state_stats = fit_strong_state_stats(outputs, query_cache, val_base, state_view=state_view, min_support=min_support)
            gate_rows.extend(
                {
                    "base_method": base_name,
                    "state_view": state_view,
                    **row,
                }
                for row in state_stats
            )
            for gain_threshold in gain_thresholds:
                chosen = {
                    row["state"]
                    for row in state_stats
                    if int(row["support"]) >= min_support and float(row["mean_gain_vs_base"]) >= gain_threshold
                }
                method = f"{base_name}_strong_gate_{state_view}_gain{gain_threshold:g}"
                for split in ["val", "test"]:
                    selected = apply_strong_gate(outputs, query_cache, selections[split], state_view=state_view, chosen_states=chosen)
                    rows.append(evaluate_policy(package, outputs, selected, method, split, "state_gain_gate", lambda_cost))
            ranked_states = [
                row["state"]
                for row in sorted(state_stats, key=lambda item: (float(item["mean_gain_vs_base"]), int(item["support"])), reverse=True)
                if int(row["support"]) >= min_support and float(row["mean_gain_vs_base"]) > 0
            ]
            for rate_cap in rate_caps:
                chosen = choose_states_under_rate_cap(outputs, query_cache, selections["val"], ranked_states, state_view=state_view, rate_cap=rate_cap)
                method = f"{base_name}_strong_gate_{state_view}_cap{rate_cap:g}"
                for split in ["val", "test"]:
                    selected = apply_strong_gate(outputs, query_cache, selections[split], state_view=state_view, chosen_states=chosen)
                    rows.append(evaluate_policy(package, outputs, selected, method, split, "state_rate_cap_gate", lambda_cost))

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    return table, pd.DataFrame(gate_rows)


def fit_strong_state_stats(
    outputs: pd.DataFrame,
    query_cache: dict[str, dict[str, Any]],
    base_selection: pd.Series,
    *,
    state_view: str,
    min_support: int,
) -> list[dict[str, Any]]:
    by_query = outputs.set_index(["query_id", "model_id"])
    groups: dict[str, list[dict[str, float]]] = {}
    for query_id, base_model in base_selection.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) not in by_query.index or (query_id, str(base_model)) not in by_query.index:
            continue
        state = state_key(query_cache, query_id, state_view=state_view, base_model=str(base_model))
        base_row = by_query.loc[(query_id, str(base_model))]
        strong_row = by_query.loc[(query_id, STRONG_MODEL_ID)]
        groups.setdefault(state, []).append(
            {
                "base_utility": float(base_row["utility"]),
                "strong_utility": float(strong_row["utility"]),
                "base_quality": float(base_row["quality_score"]),
                "strong_quality": float(strong_row["quality_score"]),
            }
        )
    rows: list[dict[str, Any]] = []
    for state, values in groups.items():
        frame = pd.DataFrame(values)
        rows.append(
            {
                "state": state,
                "support": int(len(frame)),
                "mean_base_utility": float(frame["base_utility"].mean()),
                "mean_strong_utility": float(frame["strong_utility"].mean()),
                "mean_gain_vs_base": float((frame["strong_utility"] - frame["base_utility"]).mean()),
                "mean_quality_gain_vs_base": float((frame["strong_quality"] - frame["base_quality"]).mean()),
                "eligible": bool(len(frame) >= min_support),
            }
        )
    return sorted(rows, key=lambda row: (float(row["mean_gain_vs_base"]), int(row["support"])), reverse=True)


def apply_strong_gate(
    outputs: pd.DataFrame,
    query_cache: dict[str, dict[str, Any]],
    base_selection: pd.Series,
    *,
    state_view: str,
    chosen_states: set[str],
) -> pd.Series:
    available = set(outputs["model_id"].astype(str).unique())
    selected = base_selection.copy()
    if STRONG_MODEL_ID not in available:
        return selected
    for query_id, base_model in base_selection.items():
        state = state_key(query_cache, str(query_id), state_view=state_view, base_model=str(base_model))
        if state in chosen_states:
            selected.loc[query_id] = STRONG_MODEL_ID
    return selected


def oracle_between_base_and_strong(outputs: pd.DataFrame, base_selection: pd.Series) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    selected = base_selection.copy()
    for query_id, base_model in base_selection.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) not in by_query.index or (query_id, str(base_model)) not in by_query.index:
            continue
        base_row = by_query.loc[(query_id, str(base_model))]
        strong_row = by_query.loc[(query_id, STRONG_MODEL_ID)]
        if float(strong_row["utility"]) > float(base_row["utility"]):
            selected.loc[query_id] = STRONG_MODEL_ID
    return selected


def choose_states_under_rate_cap(
    outputs: pd.DataFrame,
    query_cache: dict[str, dict[str, Any]],
    val_base: pd.Series,
    ranked_states: list[str],
    *,
    state_view: str,
    rate_cap: float,
) -> set[str]:
    chosen: set[str] = set()
    total = max(1, len(val_base))
    for state in ranked_states:
        trial = chosen | {state}
        selected = apply_strong_gate(outputs, query_cache, val_base, state_view=state_view, chosen_states=trial)
        extra_rate = float(selected.eq(STRONG_MODEL_ID).mean() - val_base.eq(STRONG_MODEL_ID).mean())
        if extra_rate <= float(rate_cap) + 1e-12:
            chosen = trial
        if len(chosen) >= total:
            break
    return chosen


def evaluate_policy(package, outputs: pd.DataFrame, selected: pd.Series, method: str, split: str, family: str, lambda_cost: float) -> dict[str, Any]:
    split_outputs = outputs[outputs["split"].eq(split)]
    cost_oracle = split_outputs.loc[split_outputs.groupby("query_id")["utility"].idxmax()]
    quality_oracle = split_outputs.loc[split_outputs.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["non_strong_frontier_call_rate"] = float(selected_rows["is_frontier"].astype(bool).mean() - row["strong_call_rate"])
    return row


def all_strong_selection(outputs: pd.DataFrame, split: str) -> pd.Series:
    queries = outputs[outputs["split"].eq(split)].drop_duplicates("query_id").set_index("query_id")
    return pd.Series(STRONG_MODEL_ID, index=queries.index.astype(str))


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    out = out.astype(str)
    out.loc[out.eq("nan")] = "qwen3-14b-awq-local"
    return out


def build_query_cache(outputs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    query_rows = outputs.drop_duplicates("query_id").set_index("query_id")
    by_query = outputs.set_index(["query_id", "model_id"])
    available = set(outputs["model_id"].astype(str).unique())
    cache: dict[str, dict[str, Any]] = {}
    for query_id, query in query_rows.iterrows():
        query_id = str(query_id)
        answers = {model_id: local_answer(by_query, query_id, model_id) for model_id in LOCAL_MODELS if model_id in available}
        nonempty = [answer for answer in answers.values() if answer]
        counts = pd.Series(nonempty).value_counts() if nonempty else pd.Series(dtype=int)
        cache[query_id] = {
            "benchmark": str(query.get("benchmark", "")),
            "domain": str(query.get("domain", "")),
            "metric": str(query.get("metric", "")),
            "valid": len(nonempty),
            "unique": int(len(counts)),
            "majority": int(counts.iloc[0]) if not counts.empty else 0,
            "agree_pairs": agree_pair_count(list(answers.values())),
        }
    return cache


def state_key(query_cache: dict[str, dict[str, Any]], query_id: str, *, state_view: str, base_model: str) -> str:
    item = query_cache[query_id]
    if state_view == "benchmark":
        parts = [item["benchmark"]]
    elif state_view == "benchmark_metric":
        parts = [item["benchmark"], item["metric"]]
    elif state_view == "benchmark_base_model":
        parts = [item["benchmark"], base_model]
    elif state_view == "benchmark_local_stats":
        parts = [
            item["benchmark"],
            str(item["valid"]),
            str(item["unique"]),
            str(item["majority"]),
            str(item["agree_pairs"]),
        ]
    else:
        raise ValueError(f"Unknown state view: {state_view}")
    return "|".join(parts)


def local_answer(by_query: pd.DataFrame, query_id: str, model_id: str) -> str:
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
    return answer


def agree_pair_count(answers: list[str]) -> int:
    total = 0
    for idx, first in enumerate(answers):
        for second in answers[idx + 1 :]:
            if first and first == second:
                total += 1
    return total


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for _, group in table.groupby("family"):
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[group["split"].eq("test") & group["method"].eq(best["method"])]
        if not test.empty:
            rows.append(test.iloc[0])
    return pd.DataFrame(rows)


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(16)
    labels = plot["method"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5f9e6e")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Cached Test-Time Compute Routing On Broad100")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cached_ttc_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, outputs_path: Path, table: pd.DataFrame, selected: pd.DataFrame, gate_rows: pd.DataFrame) -> None:
    best_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(12)
    lines = [
        "# Cached Test-Time Compute Router",
        "",
        f"Source outputs: `{outputs_path}`.",
        "",
        "This run makes no provider API calls. It routes over cached broad100 actions, including the cached Gemini strong-solve action.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Top Validation State Gains",
        "",
        markdown_table(gate_rows.sort_values(['mean_gain_vs_base', 'support'], ascending=False).head(20)),
        "",
        "## Interpretation",
        "",
        "- This tests the BEST-Route-style idea that the action should include a stronger test-time-compute option, not only a model id.",
        "- The strong action is validation-calibrated from cached rows; no test utility is used to decide state gates.",
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
