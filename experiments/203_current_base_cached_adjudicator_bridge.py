from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SOURCES = {
    "gpt_with_frontier": "results/controlled/broad100_answer_adjudicator/table_broad_answer_adjudications.csv",
    "gemini_with_frontier": "results/controlled/broad100_answer_adjudicator_gemini/table_broad_answer_adjudications.csv",
    "gpt_local_only": "results/controlled/broad100_answer_adjudicator_gpt_local_only/table_broad_answer_adjudications.csv",
    "medium_with_frontier": "results/controlled/broad100_answer_adjudicator_medium/table_broad_answer_adjudications.csv",
}
THRESHOLDS = [0.0, 0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98]
MODES = [
    "always",
    "if_base_frontier",
    "if_base_local",
    "if_adjudicator_frontier",
    "if_adjudicator_local",
    "if_different",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay cached broad adjudicator decisions as an override bridge on top of the "
            "current best concrete Broad100 policy. No provider, local, or vLLM calls are made."
        )
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--base-query-choices",
        type=Path,
        default=Path(
            "results/controlled/broad100_current_policy_variable_verifier_fusion/"
            "table_current_policy_variable_verifier_query_choices.csv"
        ),
    )
    parser.add_argument("--base-policy", default="base_current_policy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_current_base_cached_adjudicator_bridge"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    matrix = build_matrix(outputs)
    base = load_base(args.base_query_choices, args.base_policy, matrix)
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []

    base_choices = base_choice_frame(base, matrix, "current_base_no_adjudicator")
    rows.extend(metric_rows(base_choices, "current_base_no_adjudicator", "reference", args))
    details.append(base_choices)

    for source, path in DEFAULT_SOURCES.items():
        source_path = Path(path)
        if not source_path.exists():
            continue
        adjudicator = pd.read_csv(source_path)
        for threshold in THRESHOLDS:
            for mode in MODES:
                choices = apply_bridge(base, adjudicator, matrix, source, threshold, mode, lambda_cost=float(args.lambda_cost))
                method = f"{source}_thr{threshold:g}_{mode}"
                rows.extend(metric_rows(choices, method, "cached_adjudicator_bridge", args))
                details.append(choices.assign(method=method, family="cached_adjudicator_bridge"))

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility_with_probe_cost"], ascending=[True, False])
    selected = selected_rows(table)
    selected_methods = set(selected["method"].astype(str).tolist())
    query_choices = (
        pd.concat(details, ignore_index=True)
        if details
        else pd.DataFrame()
    )
    if not query_choices.empty:
        query_choices = query_choices[query_choices["method"].astype(str).isin(selected_methods)].copy()

    table.to_csv(args.output_dir / "table_cached_adjudicator_bridge_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_cached_adjudicator_bridge_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_cached_adjudicator_bridge_query_choices.csv", index=False)
    write_figure(args.output_dir, table)
    write_memo(args.output_dir / "CACHED_ADJUDICATOR_BRIDGE_MEMO.md", args, table, selected)
    print(f"Wrote cached adjudicator bridge results to {args.output_dir}")


def build_matrix(outputs: pd.DataFrame) -> dict[str, Any]:
    model_ids = sorted(outputs["model_id"].astype(str).unique().tolist())
    utility = outputs.pivot(index="query_id", columns="model_id", values="utility").reindex(columns=model_ids)
    quality = outputs.pivot(index="query_id", columns="model_id", values="quality_score").reindex(columns=model_ids)
    cost = outputs.pivot(index="query_id", columns="model_id", values="normalized_remote_cost").reindex(columns=model_ids)
    frontier = outputs.pivot(index="query_id", columns="model_id", values="is_frontier").fillna(False).astype(bool)
    meta = outputs.drop_duplicates("query_id").set_index("query_id")[["query_text", "split", "benchmark", "domain", "metric"]]
    oracle_idx = utility.to_numpy().argmax(axis=1)
    oracle_utility = pd.Series(
        utility.to_numpy()[np.arange(len(utility)), oracle_idx],
        index=utility.index,
        name="oracle_utility",
    )
    oracle_quality = pd.Series(
        quality.to_numpy()[np.arange(len(quality)), oracle_idx],
        index=quality.index,
        name="oracle_quality",
    )
    gpt = outputs[outputs["model_id"].astype(str).eq("gpt-5.5")]
    gpt_cost = max(float(gpt.groupby("query_id")["cost_total_usd"].mean().mean()), 1e-12) if not gpt.empty else 1.0
    return {
        "model_ids": model_ids,
        "utility": utility,
        "quality": quality,
        "cost": cost,
        "frontier": frontier,
        "meta": meta,
        "oracle_utility": oracle_utility,
        "oracle_quality": oracle_quality,
        "gpt_cost_usd": gpt_cost,
    }


def load_base(path: Path, policy: str, matrix: dict[str, Any]) -> pd.DataFrame:
    base = pd.read_csv(path)
    base = base[base["policy"].astype(str).eq(str(policy))].copy()
    if base.empty:
        raise RuntimeError(f"Base policy {policy!r} not found in {path}.")
    selected_col = "selected_model" if "selected_model" in base.columns else "selected_model_id"
    base = base[["query_id", selected_col]].rename(columns={selected_col: "base_model"})
    meta = matrix["meta"].reset_index()
    return meta.merge(base, on="query_id", how="inner")


def base_choice_frame(base: pd.DataFrame, matrix: dict[str, Any], method: str) -> pd.DataFrame:
    rows = []
    for row in base.itertuples(index=False):
        rows.append(choice_row(matrix, str(row.query_id), str(row.base_model), method, False, 0.0, "", np.nan, False))
    return pd.DataFrame(rows)


def apply_bridge(
    base: pd.DataFrame,
    adjudicator: pd.DataFrame,
    matrix: dict[str, Any],
    source: str,
    threshold: float,
    mode: str,
    *,
    lambda_cost: float,
) -> pd.DataFrame:
    adj = adjudicator.set_index("query_id")
    rows: list[dict[str, Any]] = []
    for row in base.itertuples(index=False):
        query_id = str(row.query_id)
        base_model = str(row.base_model)
        selected_model = base_model
        confidence = 0.0
        suggested = ""
        probe_called = probe_called_for_mode(mode, base_model, matrix, query_id)
        if query_id in adj.index and probe_called:
            item = adj.loc[query_id]
            suggested = str(item.get("selected_model", ""))
            confidence = float(item.get("selected_confidence", 0.0) or 0.0)
            if suggested not in matrix["model_ids"]:
                suggested = base_model
            active = confidence >= float(threshold)
            if mode == "if_adjudicator_frontier":
                active = active and bool(matrix["frontier"].loc[query_id, suggested])
            elif mode == "if_adjudicator_local":
                active = active and not bool(matrix["frontier"].loc[query_id, suggested])
            elif mode == "if_different":
                active = active and suggested != base_model
            if active:
                selected_model = suggested
            adjudicator_cost = float(item.get("adjudicator_cost", 0.0) or 0.0)
        else:
            adjudicator_cost = 0.0
        probe_norm_cost = adjudicator_cost / max(float(matrix["gpt_cost_usd"]), 1e-12) if probe_called else 0.0
        rows.append(
            choice_row(
                matrix,
                query_id,
                selected_model,
                f"{source}_thr{threshold:g}_{mode}",
                probe_called,
                probe_norm_cost,
                suggested,
                confidence,
                selected_model != base_model,
                lambda_cost=lambda_cost,
            )
        )
    return pd.DataFrame(rows)


def probe_called_for_mode(mode: str, base_model: str, matrix: dict[str, Any], query_id: str) -> bool:
    if mode == "if_base_frontier":
        return bool(matrix["frontier"].loc[query_id, base_model])
    if mode == "if_base_local":
        return not bool(matrix["frontier"].loc[query_id, base_model])
    return True


def choice_row(
    matrix: dict[str, Any],
    query_id: str,
    selected_model: str,
    method: str,
    probe_called: bool,
    probe_norm_cost: float,
    adjudicator_model: str,
    adjudicator_confidence: float,
    changed: bool,
    *,
    lambda_cost: float = 0.35,
) -> dict[str, Any]:
    meta = matrix["meta"].loc[query_id]
    utility = float(matrix["utility"].loc[query_id, selected_model])
    quality = float(matrix["quality"].loc[query_id, selected_model])
    route_cost = float(matrix["cost"].loc[query_id, selected_model])
    return {
        "query_id": query_id,
        "query_text": str(meta["query_text"]),
        "split": str(meta["split"]),
        "benchmark": str(meta["benchmark"]),
        "domain": str(meta["domain"]),
        "metric": str(meta["metric"]),
        "method": method,
        "selected_model": selected_model,
        "selected_quality": quality,
        "selected_utility": utility,
        "selected_normalized_cost": route_cost,
        "selected_utility_with_probe_cost": quality - float(lambda_cost) * (route_cost + float(probe_norm_cost)),
        "selected_frontier": bool(matrix["frontier"].loc[query_id, selected_model]),
        "probe_called": bool(probe_called),
        "probe_norm_cost": float(probe_norm_cost),
        "adjudicator_model": adjudicator_model,
        "adjudicator_confidence": float(adjudicator_confidence) if not pd.isna(adjudicator_confidence) else np.nan,
        "changed": bool(changed),
        "oracle_utility": float(matrix["oracle_utility"].loc[query_id]),
        "oracle_quality": float(matrix["oracle_quality"].loc[query_id]),
    }


def metric_rows(choices: pd.DataFrame, method: str, family: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, group in choices.groupby("split", sort=False):
        utility = group["selected_utility"].astype(float).to_numpy()
        utility_with_probe = group["selected_utility_with_probe_cost"].astype(float).to_numpy()
        ci_low, ci_high = bootstrap_ci(utility, int(args.bootstrap_samples), int(args.seed))
        cost_ci_low, cost_ci_high = bootstrap_ci(utility_with_probe, int(args.bootstrap_samples), int(args.seed))
        oracle_utility = float(group["oracle_utility"].astype(float).mean())
        mean_utility = float(utility.mean())
        mean_utility_with_probe = float(utility_with_probe.mean())
        rows.append(
            {
                "method": method,
                "family": family,
                "split": split,
                "n_queries": int(len(group)),
                "mean_quality": float(group["selected_quality"].astype(float).mean()),
                "mean_utility": mean_utility,
                "mean_utility_ci_low": ci_low,
                "mean_utility_ci_high": ci_high,
                "mean_utility_with_probe_cost": mean_utility_with_probe,
                "mean_utility_with_probe_cost_ci_low": cost_ci_low,
                "mean_utility_with_probe_cost_ci_high": cost_ci_high,
                "mean_normalized_cost": float(group["selected_normalized_cost"].astype(float).mean()),
                "extra_probe_norm_cost_mean": float(group["probe_norm_cost"].astype(float).mean()),
                "oracle_mean_quality": float(group["oracle_quality"].astype(float).mean()),
                "oracle_mean_utility": oracle_utility,
                "oracle_utility_ratio": mean_utility / max(oracle_utility, 1e-12),
                "oracle_utility_ratio_with_probe_cost": mean_utility_with_probe / max(oracle_utility, 1e-12),
                "utility_gap_to_oracle": oracle_utility - mean_utility,
                "quality_gap_to_oracle": float(group["oracle_quality"].astype(float).mean())
                - float(group["selected_quality"].astype(float).mean()),
                "frontier_call_rate": float(group["selected_frontier"].astype(bool).mean()),
                "probe_call_rate": float(group["probe_called"].astype(bool).mean()),
                "override_rate": float(group["changed"].astype(bool).mean()),
                "selected_models_json": json.dumps(group["selected_model"].value_counts().sort_index().to_dict(), sort_keys=True),
            }
        )
    return rows


def selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    val = table[table["split"].astype(str).eq("val")].copy()
    test = table[table["split"].astype(str).eq("test")].copy()
    for metric, rule in [
        ("mean_utility", "val_best_raw_utility"),
        ("mean_utility_with_probe_cost", "val_best_probe_cost_utility"),
    ]:
        if val.empty:
            continue
        best = val.sort_values([metric, "frontier_call_rate", "probe_call_rate"], ascending=[False, True, True]).iloc[0]
        best_row = best.copy()
        best_row["selection_rule"] = rule
        rows.append(best_row)
        match = test[test["method"].astype(str).eq(str(best["method"]))]
        if not match.empty:
            test_row = match.iloc[0].copy()
            test_row["selection_rule"] = f"{rule}_test"
            rows.append(test_row)
    for _, row in test.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).head(8).iterrows():
        diagnostic = row.copy()
        diagnostic["selection_rule"] = "top_test_raw_diagnostic"
        rows.append(diagnostic)
    for _, row in test.sort_values(["mean_utility_with_probe_cost", "frontier_call_rate"], ascending=[False, True]).head(8).iterrows():
        diagnostic = row.copy()
        diagnostic["selection_rule"] = "top_test_probe_cost_diagnostic"
        rows.append(diagnostic)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["selection_rule", "method", "split"], keep="first")


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values("mean_utility", ascending=False).head(18)
    labels = plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#566c78", label="raw")
    ax.scatter(plot["mean_utility_with_probe_cost"].iloc[::-1], labels.iloc[::-1], color="#a05a3b", s=22, label="with probe cost")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Current Base + Cached Adjudicator Bridge")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_cached_adjudicator_bridge_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_with_probe_cost",
        "oracle_utility_ratio",
        "oracle_utility_ratio_with_probe_cost",
        "frontier_call_rate",
        "probe_call_rate",
        "override_rate",
        "selection_rule",
    ]
    lines = [
        "# Current Base + Cached Adjudicator Bridge",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/203_current_base_cached_adjudicator_bridge.py",
        "```",
        "",
        "## What This Tests",
        "",
        "- Replays cached broad GPT/Gemini adjudicator outputs as override signals on the current best concrete Broad100 policy.",
        "- Makes no new provider, local generation, or vLLM calls.",
        "- This is a diagnostic bridge, not the main benchmark-agnostic local-probe method, because some cached adjudicator prompts included benchmark metadata.",
        "- Reports raw route utility and route utility after charging adjudicator probe cost on the normalized GPT-cost scale.",
        "",
        "## Inputs",
        "",
        f"- Outputs: `{args.outputs}`",
        f"- Base choices: `{args.base_query_choices}`",
        f"- Base policy: `{args.base_policy}`",
        "",
        "## Selected Rows",
        "",
        "```csv",
        selected[[column for column in cols if column in selected.columns]].to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        "- If validation selects the base row after probe-cost accounting, the cached adjudicator is not worth using as a route-time bridge.",
        "- Top-test diagnostic rows show whether there is any residual signal worth revisiting with a cheaper, cleaner, benchmark-agnostic probe.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
