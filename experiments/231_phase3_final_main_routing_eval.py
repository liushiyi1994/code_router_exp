from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")
FRONTIER_MODELS = {"gpt-5.5", "gemini-3.5-flash", "gemini-3.5-flash-strong-solve"}
GPT_MODELS = {"gpt-5.5"}
GEMINI_MODELS = {"gemini-3.5-flash", "gemini-3.5-flash-strong-solve"}
LITERATURE_BASELINES = {
    "routellm_mf": "RouteLLM matrix-factorization baseline",
    "graphrouter": "GraphRouter through LLMRouter",
    "avengerspro": "Avengers-Pro cluster router",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Phase 3 main-routing eval from cached Broad100 rows.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--random-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    out_dir = Path(config["outputs"]["root"]) / "main_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = load_outputs(Path(config["inputs"]["broad100_outputs"]), float(config["method"]["lambda_cost"]))
    test = outputs[outputs["split"].astype(str).eq("test")].copy()
    current_choices = pd.read_csv(config["inputs"]["broad100_current_best_choices"])
    current_choices = current_choices[current_choices["split"].astype(str).eq("test")].copy()

    oracle = build_oracle(test)
    rows, query_rows = build_eval_rows(test, oracle, current_choices, config, args.random_seeds)
    literature_rows, literature_query_rows = build_literature_eval_rows(test, oracle, config)
    if literature_rows:
        rows = pd.concat([rows, pd.DataFrame(literature_rows)], ignore_index=True)
        rows["meets_3pt_quality"] = rows["quality_gap_to_oracle"] <= 0.03
        rows["meets_95pct_utility"] = rows["oracle_utility_ratio"] >= 0.95
        rows["meets_frontier_cap_0p40"] = rows["frontier_call_rate"] <= 0.40
        rows["meets_primary_gate"] = rows["meets_3pt_quality"] & rows["meets_95pct_utility"] & rows["meets_frontier_cap_0p40"]
        rows = rows.sort_values(["mean_utility", "mean_quality"], ascending=False).reset_index(drop=True)
    if literature_query_rows:
        query_rows = pd.concat([query_rows, *literature_query_rows], ignore_index=True)
    baseline_status = build_literature_baseline_status(config)
    per_benchmark = build_per_benchmark(query_rows, oracle)
    action_mix = build_action_mix(query_rows)

    rows.to_csv(out_dir / "table_main_routing_eval.csv", index=False)
    per_benchmark.to_csv(out_dir / "table_per_benchmark_eval.csv", index=False)
    action_mix.to_csv(out_dir / "table_action_mix.csv", index=False)
    baseline_status.to_csv(out_dir / "table_literature_baseline_status.csv", index=False)
    write_quality_cost_figure(out_dir / "fig_quality_cost_frontier.pdf", rows)
    write_oracle_gap_figure(out_dir / "fig_oracle_gap.pdf", rows)
    write_memo(out_dir / "MAIN_ROUTING_EVAL_MEMO.md", rows, baseline_status, config)
    print(f"Wrote final main routing eval to {out_dir}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_outputs(path: Path, lambda_cost: float) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    outputs = outputs[outputs["status"].astype(str).eq("success")].copy()
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["quality_score"] = outputs["quality_score"].astype(float)
    outputs["normalized_remote_cost"] = outputs["normalized_remote_cost"].astype(float)
    outputs["cost_total_usd"] = outputs["cost_total_usd"].astype(float)
    outputs["latency_s"] = outputs["latency_s"].astype(float)
    outputs["utility"] = outputs["quality_score"] - lambda_cost * outputs["normalized_remote_cost"]
    return outputs


def build_oracle(test: pd.DataFrame) -> pd.DataFrame:
    idx = (
        test.sort_values(["query_id", "utility", "quality_score", "normalized_remote_cost"], ascending=[True, False, False, True])
        .groupby("query_id", sort=False)
        .head(1)
        .index
    )
    oracle = test.loc[idx].copy()
    return oracle.set_index("query_id", drop=False)


def build_eval_rows(
    test: pd.DataFrame,
    oracle: pd.DataFrame,
    current_choices: pd.DataFrame,
    config: dict[str, Any],
    random_seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    query_rows: list[pd.DataFrame] = []
    all_query_ids = sorted(oracle.index.astype(str).tolist())

    rows.append(evaluate_selected("cost_aware_oracle", "oracle", oracle.reset_index(drop=True), oracle, "post_hoc_upper_bound"))
    query_rows.append(as_query_rows("cost_aware_oracle", "oracle", oracle.reset_index(drop=True), oracle))

    for model_id, frame in sorted(test.groupby("model_id"), key=lambda item: item[0]):
        selected = frame.set_index("query_id").reindex(all_query_ids).dropna(subset=["model_id"]).reset_index()
        if len(selected) != len(all_query_ids):
            continue
        role = model_role(model_id)
        method = f"all_{model_id}"
        rows.append(evaluate_selected(method, role, selected, oracle, "single_model_all_queries"))
        query_rows.append(as_query_rows(method, role, selected, oracle))

    local_rows = [row for row in rows if row["method_role"] == "local_model"]
    if local_rows:
        best_local_method = max(local_rows, key=lambda row: row["mean_utility"])["method"]
        selected = query_rows_for_method(query_rows, best_local_method)
        rows.append(evaluate_selected("best_local_single_model", "local_reference", selected, oracle, f"best local single model: {best_local_method}"))
        query_rows.append(as_query_rows("best_local_single_model", "local_reference", selected, oracle))

    if not current_choices.empty:
        selected = current_choices.rename(
            columns={
                "selected_action": "model_id",
                "selected_quality": "quality_score",
                "selected_utility": "utility",
                "selected_normalized_cost": "normalized_remote_cost",
                "selected_latency_s": "latency_s",
            }
        ).copy()
        selected = attach_action_costs(selected, test)
        rows.append(
            evaluate_selected(
                str(config["method"]["current_best_method"]),
                "probecode_statecal",
                selected,
                oracle,
                "validation-selected current best from Broad100 package",
            )
        )
        query_rows.append(as_query_rows(str(config["method"]["current_best_method"]), "probecode_statecal", selected, oracle))

    random_eval, random_queries = evaluate_random(test, oracle, random_seeds)
    rows.append(random_eval)
    query_rows.append(random_queries)

    table = pd.DataFrame(rows)
    table["meets_3pt_quality"] = table["quality_gap_to_oracle"] <= 0.03
    table["meets_95pct_utility"] = table["oracle_utility_ratio"] >= 0.95
    table["meets_frontier_cap_0p40"] = table["frontier_call_rate"] <= 0.40
    table["meets_primary_gate"] = table["meets_3pt_quality"] & table["meets_95pct_utility"] & table["meets_frontier_cap_0p40"]
    return table.sort_values(["mean_utility", "mean_quality"], ascending=False).reset_index(drop=True), pd.concat(query_rows, ignore_index=True)


def build_literature_eval_rows(test: pd.DataFrame, oracle: pd.DataFrame, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    path = Path(config.get("inputs", {}).get("broad100_literature_baseline_choices", ""))
    if not path.exists():
        return [], []
    choices = pd.read_csv(path)
    choices = choices[choices["split"].astype(str).eq("test")].copy()
    if choices.empty:
        return [], []
    lookup = test.set_index(["query_id", "model_id"], drop=False)
    rows: list[dict[str, Any]] = []
    query_rows: list[pd.DataFrame] = []
    for method, frame in choices.groupby("method", sort=False):
        selected_rows = []
        for item in frame.to_dict("records"):
            key = (str(item["query_id"]), str(item["model_id"]))
            if key in lookup.index:
                selected_rows.append(lookup.loc[key])
        if not selected_rows:
            continue
        selected = pd.DataFrame(selected_rows).reset_index(drop=True)
        role = str(frame["method_role"].iloc[0]) if "method_role" in frame else "literature_baseline"
        baseline = str(frame["baseline"].iloc[0]) if "baseline" in frame else "external"
        note = f"Broad100 cached adapter for {baseline}"
        rows.append(evaluate_selected(str(method), role, selected, oracle, note))
        query_rows.append(as_query_rows(str(method), role, selected, oracle))
    return rows, query_rows


def attach_action_costs(selected: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy()
    work["query_id"] = work["query_id"].astype(str)
    work["model_id"] = work["model_id"].astype(str)
    lookup_cols = ["query_id", "model_id", "cost_total_usd"]
    if "provider" in test.columns:
        lookup_cols.append("provider")
    if "is_frontier" in test.columns:
        lookup_cols.append("is_frontier")
    cost_lookup = test[lookup_cols].drop_duplicates(["query_id", "model_id"]).copy()
    merged = work.merge(cost_lookup, on=["query_id", "model_id"], how="left", suffixes=("", "_observed"))
    if "cost_total_usd_observed" in merged.columns:
        merged["cost_total_usd"] = merged["cost_total_usd_observed"].combine_first(merged.get("cost_total_usd"))
        merged = merged.drop(columns=["cost_total_usd_observed"])
    if "cost_total_usd" not in merged.columns:
        merged["cost_total_usd"] = np.nan
    return merged


def model_role(model_id: str) -> str:
    if model_id in GPT_MODELS:
        return "all_gpt"
    if model_id in GEMINI_MODELS:
        return "all_gemini"
    if model_id == "deterministic_math_tool":
        return "verifiable_action"
    return "local_model"


def evaluate_selected(method: str, role: str, selected: pd.DataFrame, oracle: pd.DataFrame, notes: str) -> dict[str, Any]:
    work = selected.copy()
    oracle_aligned = oracle.reindex(work["query_id"].astype(str).tolist())
    mean_quality = float(work["quality_score"].mean())
    mean_utility = float(work["utility"].mean())
    oracle_quality = float(oracle_aligned["quality_score"].mean())
    oracle_utility = float(oracle_aligned["utility"].mean())
    frontier_flags = work["model_id"].astype(str).isin(FRONTIER_MODELS)
    return {
        "method": method,
        "method_role": role,
        "n_queries": int(len(work)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "oracle_mean_quality": oracle_quality,
        "oracle_mean_utility": oracle_utility,
        "quality_gap_to_oracle": oracle_quality - mean_quality,
        "utility_gap_to_oracle": oracle_utility - mean_utility,
        "oracle_utility_ratio": mean_utility / max(oracle_utility, 1e-12),
        "remote_cost_total_usd": float(work["cost_total_usd"].sum(skipna=True)) if "cost_total_usd" in work else np.nan,
        "remote_cost_per_1k_queries": float(work["cost_total_usd"].sum(skipna=True) / max(len(work), 1) * 1000.0) if "cost_total_usd" in work else np.nan,
        "normalized_remote_cost_mean": float(work["normalized_remote_cost"].mean()),
        "frontier_call_rate": float(frontier_flags.mean()),
        "local_call_rate": float((~frontier_flags).mean()),
        "latency_p50": float(work["latency_s"].quantile(0.50)),
        "latency_p95": float(work["latency_s"].quantile(0.95)),
        "selected_actions_json": json.dumps(work["model_id"].astype(str).value_counts().sort_index().to_dict(), sort_keys=True),
        "notes": notes,
    }


def as_query_rows(method: str, role: str, selected: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    work = selected.copy()
    work["method"] = method
    work["method_role"] = role
    oracle_aligned = oracle.reindex(work["query_id"].astype(str).tolist())
    work["oracle_quality"] = oracle_aligned["quality_score"].to_numpy(dtype=float)
    work["oracle_utility"] = oracle_aligned["utility"].to_numpy(dtype=float)
    work["utility_regret"] = work["oracle_utility"].astype(float) - work["utility"].astype(float)
    keep = [
        "method",
        "method_role",
        "query_id",
        "benchmark",
        "model_id",
        "quality_score",
        "utility",
        "normalized_remote_cost",
        "latency_s",
        "cost_total_usd",
        "oracle_quality",
        "oracle_utility",
        "utility_regret",
    ]
    return work[[col for col in keep if col in work.columns]].copy()


def query_rows_for_method(query_rows: list[pd.DataFrame], method: str) -> pd.DataFrame:
    for frame in query_rows:
        if not frame.empty and str(frame["method"].iloc[0]) == method:
            return frame.rename(columns={"method": "_old_method", "method_role": "_old_role"}).copy()
    raise RuntimeError(f"No query rows for method {method}")


def evaluate_random(test: pd.DataFrame, oracle: pd.DataFrame, seeds: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    grouped = {qid: frame.copy() for qid, frame in test.groupby("query_id")}
    rows = []
    query_frames = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        selected = []
        for qid in sorted(grouped):
            frame = grouped[qid]
            selected.append(frame.iloc[int(rng.integers(0, len(frame)))])
        selected_df = pd.DataFrame(selected).reset_index(drop=True)
        rows.append(evaluate_selected(f"random_assignment_seed{seed}", "random", selected_df, oracle, "random action assignment"))
        query_frames.append(as_query_rows(f"random_assignment_seed{seed}", "random", selected_df, oracle))
    table = pd.DataFrame(rows)
    mean_row = {
        "method": "random_assignment_mean",
        "method_role": "random",
        "n_queries": int(table["n_queries"].iloc[0]),
        "mean_quality": float(table["mean_quality"].mean()),
        "mean_utility": float(table["mean_utility"].mean()),
        "oracle_mean_quality": float(table["oracle_mean_quality"].mean()),
        "oracle_mean_utility": float(table["oracle_mean_utility"].mean()),
        "quality_gap_to_oracle": float(table["quality_gap_to_oracle"].mean()),
        "utility_gap_to_oracle": float(table["utility_gap_to_oracle"].mean()),
        "oracle_utility_ratio": float(table["oracle_utility_ratio"].mean()),
        "remote_cost_total_usd": float(table["remote_cost_total_usd"].mean()),
        "remote_cost_per_1k_queries": float(table["remote_cost_per_1k_queries"].mean()),
        "normalized_remote_cost_mean": float(table["normalized_remote_cost_mean"].mean()),
        "frontier_call_rate": float(table["frontier_call_rate"].mean()),
        "local_call_rate": float(table["local_call_rate"].mean()),
        "latency_p50": float(table["latency_p50"].mean()),
        "latency_p95": float(table["latency_p95"].mean()),
        "selected_actions_json": "{}",
        "notes": f"mean over seeds {seeds}",
    }
    return mean_row, pd.concat(query_frames, ignore_index=True)


def build_literature_baseline_status(config: dict[str, Any]) -> pd.DataFrame:
    final_status_path = Path(config.get("inputs", {}).get("broad100_literature_baseline_status", ""))
    if final_status_path.exists():
        status = pd.read_csv(final_status_path)
        required = {"baseline", "description", "status", "broad100_final_eval_included", "notes"}
        missing = required - set(status.columns)
        if not missing:
            return status
    readiness_path = Path(config["inputs"].get("external_readiness", ""))
    readiness = pd.read_csv(readiness_path) if readiness_path.exists() else pd.DataFrame()
    rows = []
    mapping = {
        "routellm_mf": "routecode_local_routellm_mf_metric",
        "graphrouter": "graphrouter",
        "avengerspro": "avengerspro",
    }
    for baseline, label in LITERATURE_BASELINES.items():
        subset = readiness[readiness.astype(str).apply(lambda col: col.str.contains(mapping[baseline], case=False, regex=False, na=False)).any(axis=1)] if not readiness.empty else pd.DataFrame()
        status = "pending_broad100_adapter"
        evidence = ""
        if not subset.empty:
            available = subset[subset["status"].astype(str).str.contains("available", case=False, na=False)]
            status = "available_on_llmrouterbench_broad20" if not available.empty else str(subset.iloc[0].get("status", "documented"))
            evidence = str(subset.iloc[0].get("command", ""))
        rows.append(
            {
                "baseline": baseline,
                "description": label,
                "status": status,
                "broad100_final_eval_included": False,
                "evidence": evidence,
                "notes": "Not yet run on the final cached Broad100 action matrix; required before paper-level baseline claim.",
            }
        )
    return pd.DataFrame(rows)


def build_per_benchmark(query_rows: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, role, benchmark), frame in query_rows.groupby(["method", "method_role", "benchmark"]):
        rows.append(
            {
                "method": method,
                "method_role": role,
                "benchmark": benchmark,
                "n_queries": int(len(frame)),
                "mean_quality": float(frame["quality_score"].mean()),
                "mean_utility": float(frame["utility"].mean()),
                "mean_oracle_utility": float(frame["oracle_utility"].mean()),
                "oracle_utility_ratio": float(frame["utility"].mean() / max(frame["oracle_utility"].mean(), 1e-12)),
                "mean_utility_regret": float(frame["utility_regret"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["benchmark", "mean_utility"], ascending=[True, False]).reset_index(drop=True)


def build_action_mix(query_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        query_rows.groupby(["method", "method_role", "model_id"], as_index=False)
        .agg(n_queries=("query_id", "size"), mean_quality=("quality_score", "mean"), mean_utility=("utility", "mean"))
        .sort_values(["method", "n_queries"], ascending=[True, False])
    )


def write_quality_cost_figure(path: Path, rows: pd.DataFrame) -> None:
    plot = rows[~rows["method_role"].eq("oracle")].copy()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(plot["normalized_remote_cost_mean"], plot["mean_quality"], s=50)
    for _, row in plot.iterrows():
        if row["method_role"] in {"probecode_statecal", "all_gpt", "all_gemini", "local_reference", "random"}:
            ax.annotate(row["method"], (row["normalized_remote_cost_mean"], row["mean_quality"]), fontsize=7)
    ax.set_xlabel("Mean normalized remote cost")
    ax.set_ylabel("Mean quality")
    ax.set_title("Final Broad100 Quality-Cost Frontier")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_oracle_gap_figure(path: Path, rows: pd.DataFrame) -> None:
    plot = rows[rows["method_role"].isin(["probecode_statecal", "all_gpt", "all_gemini", "local_reference", "random", "local_model"])].copy()
    plot = plot.sort_values("utility_gap_to_oracle", ascending=False).tail(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(plot["method"], plot["utility_gap_to_oracle"], color="#426b69")
    ax.set_xlabel("Utility gap to oracle")
    ax.set_title("Final Broad100 Oracle Utility Gap")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_memo(path: Path, rows: pd.DataFrame, baseline_status: pd.DataFrame, config: dict[str, Any]) -> None:
    current = rows[rows["method"].eq(str(config["method"]["current_best_method"]))].iloc[0]
    oracle = rows[rows["method_role"].eq("oracle")].iloc[0]
    lines = [
        "# Final Main Routing Evaluation",
        "",
        "This is a cache-backed Broad100 final-evaluation pass. It makes no provider, vLLM, or local generation calls.",
        "",
        "## Current Method",
        "",
        f"- Method: `{current['method']}`",
        f"- Mean quality: `{current['mean_quality']:.4f}`",
        f"- Mean utility: `{current['mean_utility']:.4f}`",
        f"- Quality gap to oracle: `{current['quality_gap_to_oracle']:.4f}`",
        f"- Oracle utility ratio: `{current['oracle_utility_ratio']:.4f}`",
        f"- Frontier-call rate: `{current['frontier_call_rate']:.4f}`",
        "",
        "## Oracle",
        "",
        f"- Oracle mean quality: `{oracle['mean_quality']:.4f}`",
        f"- Oracle mean utility: `{oracle['mean_utility']:.4f}`",
        "",
        "## Literature Baseline Status",
        "",
    ]
    for row in baseline_status.to_dict("records"):
        lines.append(f"- `{row['baseline']}`: `{row['status']}`. {row['notes']}")
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "The three literature baselines are not yet included in this Broad100 final table. Existing LLMRouterBench broad20 baseline artifacts are documented separately, but a final split-aligned adapter pass is still required.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
