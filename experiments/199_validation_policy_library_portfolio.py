from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_COLUMNS = (
    "selected_model",
    "selected_model_id",
    "model_id",
    "model_id_x",
    "fused_model",
    "patched_model",
)
METHOD_COLUMNS = ("policy", "method")
BLOCKED_TERMS = (
    "oracle",
    "diagnostic",
    "posthoc",
    "same_answer",
    "evidence_ceiling",
    "full_cost_aware",
    "best_large_action",
    "best_local_action",
    "target_best",
)
BLOCKED_PATH_TERMS = (
    "residual_action_identity_audit",
    "validation_policy_library_portfolio",
)
STRONG_LOCAL = {"qwen3-32b-awq-local", "qwen3-32b-awq-selfconsistency-n3-local"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose existing cached deployable/cheap policies by validation benchmark utility."
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=Path("results/controlled"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_validation_policy_library_portfolio"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_outputs = pd.read_parquet(args.outputs).copy()
    all_outputs["utility"] = (
        all_outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * all_outputs["normalized_remote_cost"].astype(float)
    )
    train_priors = model_train_priors(all_outputs)
    outputs = all_outputs[all_outputs["split"].astype(str).isin(["val", "test"])].copy()

    query_meta = (
        outputs[["query_id", "query_text", "benchmark", "split"]]
        .drop_duplicates("query_id")
        .reset_index(drop=True)
    )
    query_counts = (
        query_meta.groupby(["split", "benchmark"], as_index=False)
        .agg(required_queries=("query_id", "nunique"))
    )
    action_rows = {
        (str(row.query_id), str(row.model_id)): row._asdict()
        for row in outputs.itertuples(index=False)
    }
    oracle = query_oracle(outputs)

    library = load_policy_library(args.policy_root, outputs, query_meta, action_rows, oracle, train_priors)
    direct = direct_action_candidates(outputs, oracle, train_priors)
    library = pd.concat([library, direct], ignore_index=True)
    library = library.drop_duplicates(["candidate_id", "query_id"], keep="last")

    summary = summarize_candidates(library, query_counts)
    full_coverage = summary[summary["full_coverage"].astype(bool)].copy()
    portfolios, query_choices = build_portfolios(library, full_coverage, query_counts, oracle, args)
    selected = selected_rows(portfolios)

    library_manifest = (
        library[["candidate_id", "source_path", "source_method"]]
        .drop_duplicates()
        .sort_values("candidate_id")
    )
    library_manifest.to_csv(args.output_dir / "table_policy_library_manifest.csv", index=False)
    summary.to_csv(args.output_dir / "table_policy_library_candidate_summary.csv", index=False)
    portfolios.to_csv(args.output_dir / "table_policy_library_portfolio_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_policy_library_portfolio_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_policy_library_portfolio_query_choices.csv", index=False)
    write_memo(args.output_dir / "POLICY_LIBRARY_PORTFOLIO_MEMO.md", args, library_manifest, selected)
    print(f"Wrote validation policy-library portfolio to {args.output_dir}")


def query_oracle(outputs: pd.DataFrame) -> pd.DataFrame:
    idx = outputs.groupby("query_id")["utility"].idxmax()
    return outputs.loc[idx, ["query_id", "model_id", "utility", "quality_score"]].rename(
        columns={
            "model_id": "oracle_model",
            "utility": "oracle_utility",
            "quality_score": "oracle_quality",
        }
    )


def model_train_priors(outputs: pd.DataFrame) -> dict[tuple[str, str], dict[str, float]]:
    train = outputs[outputs["split"].astype(str).eq("train")].copy()
    priors: dict[tuple[str, str], dict[str, float]] = {}
    if train.empty:
        return priors
    grouped = train.groupby(["benchmark", "model_id"], as_index=False).agg(
        train_prior_utility=("utility", "mean"),
        train_prior_quality=("quality_score", "mean"),
    )
    benchmark_fallback = train.groupby("benchmark", as_index=False).agg(
        train_prior_utility=("utility", "mean"),
        train_prior_quality=("quality_score", "mean"),
    )
    for row in grouped.itertuples(index=False):
        priors[(str(row.benchmark), str(row.model_id))] = {
            "train_prior_utility": float(row.train_prior_utility),
            "train_prior_quality": float(row.train_prior_quality),
        }
    for row in benchmark_fallback.itertuples(index=False):
        priors[(str(row.benchmark), "__fallback__")] = {
            "train_prior_utility": float(row.train_prior_utility),
            "train_prior_quality": float(row.train_prior_quality),
        }
    return priors


def load_policy_library(
    root: Path,
    outputs: pd.DataFrame,
    query_meta: pd.DataFrame,
    action_rows: dict[tuple[str, str], dict[str, Any]],
    oracle: pd.DataFrame,
    train_priors: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    files = sorted(
        {
            *root.glob("**/*query_choices*.csv"),
            *root.glob("**/*query_choice*.csv"),
        }
    )
    output_query_ids = set(query_meta["query_id"].astype(str))
    for path in files:
        path_text = str(path).lower()
        if any(term in path_text for term in BLOCKED_PATH_TERMS):
            continue
        try:
            raw = pd.read_csv(path)
        except Exception:
            continue
        if "query_id" not in raw.columns:
            continue
        model_col = first_existing(raw.columns, MODEL_COLUMNS)
        if not model_col:
            continue
        method_col = first_existing(raw.columns, METHOD_COLUMNS)
        method = raw[method_col].astype(str) if method_col else pd.Series([path.stem] * len(raw))
        family = raw["family"].astype(str) if "family" in raw.columns else pd.Series([""] * len(raw))
        selection_rule = raw["selection_rule"].astype(str) if "selection_rule" in raw.columns else pd.Series([""] * len(raw))
        method_key = method.str.cat(family, sep=" ").str.cat(selection_rule, sep=" ").str.lower()
        keep = ~method_key.map(is_blocked_method)
        frame = raw.loc[keep, ["query_id", model_col]].copy()
        frame["query_id"] = frame["query_id"].astype(str)
        frame["model_id"] = frame[model_col].astype(str)
        frame["source_method"] = method.loc[keep].astype(str).values
        frame = frame[frame["query_id"].isin(output_query_ids)].copy()
        if frame.empty:
            continue
        frame["candidate_id"] = [
            sanitize_candidate_id(path.parent.name, value) for value in frame["source_method"]
        ]
        frame["source_path"] = str(path)
        frame = frame.drop_duplicates(["candidate_id", "query_id"], keep="last")
        rows = materialize_choices(frame, query_meta, action_rows, oracle, train_priors)
        if not rows.empty:
            frames.append(rows)
    if not frames:
        raise RuntimeError(f"No usable query-choice policy files found under {root}.")
    return pd.concat(frames, ignore_index=True)


def direct_action_candidates(
    outputs: pd.DataFrame,
    oracle: pd.DataFrame,
    train_priors: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for model_id, group in outputs.groupby("model_id", sort=False):
        frame = group.copy()
        frame["candidate_id"] = "direct_action:" + sanitize(str(model_id))
        frame["source_path"] = "direct_action_from_outputs"
        frame["source_method"] = str(model_id)
        frame["train_prior_utility"] = [
            train_priors.get((str(row.benchmark), str(model_id)), {}).get("train_prior_utility", np.nan)
            for row in frame.itertuples(index=False)
        ]
        frame["train_prior_quality"] = [
            train_priors.get((str(row.benchmark), str(model_id)), {}).get("train_prior_quality", np.nan)
            for row in frame.itertuples(index=False)
        ]
        rows.append(
            frame[
                [
                    "candidate_id",
                    "source_path",
                    "source_method",
                    "query_id",
                    "query_text",
                    "benchmark",
                    "split",
                    "model_id",
                    "quality_score",
                    "utility",
                    "train_prior_utility",
                    "train_prior_quality",
                    "normalized_remote_cost",
                    "is_frontier",
                ]
            ].merge(oracle, on="query_id", how="left")
        )
    return pd.concat(rows, ignore_index=True)


def materialize_choices(
    frame: pd.DataFrame,
    query_meta: pd.DataFrame,
    action_rows: dict[tuple[str, str], dict[str, Any]],
    oracle: pd.DataFrame,
    train_priors: dict[tuple[str, str], dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta = query_meta.set_index("query_id").to_dict("index")
    oracle_map = oracle.set_index("query_id").to_dict("index")
    for row in frame.itertuples(index=False):
        query_id = str(row.query_id)
        model_id = str(row.model_id)
        action = action_rows.get((query_id, model_id))
        if not action:
            continue
        info = meta.get(query_id, {})
        oracle_info = oracle_map.get(query_id, {})
        benchmark = str(info.get("benchmark", action.get("benchmark", "")))
        prior = train_priors.get((benchmark, model_id), train_priors.get((benchmark, "__fallback__"), {}))
        rows.append(
            {
                "candidate_id": str(row.candidate_id),
                "source_path": str(row.source_path),
                "source_method": str(row.source_method),
                "query_id": query_id,
                "query_text": str(info.get("query_text", action.get("query_text", ""))),
                "benchmark": benchmark,
                "split": str(info.get("split", action.get("split", ""))),
                "model_id": model_id,
                "quality_score": float(action.get("quality_score", 0.0)),
                "utility": float(action.get("utility", 0.0)),
                "train_prior_utility": float(prior.get("train_prior_utility", np.nan)),
                "train_prior_quality": float(prior.get("train_prior_quality", np.nan)),
                "normalized_remote_cost": float(action.get("normalized_remote_cost", 0.0) or 0.0),
                "is_frontier": bool(action.get("is_frontier", False)),
                "oracle_model": str(oracle_info.get("oracle_model", "")),
                "oracle_utility": float(oracle_info.get("oracle_utility", np.nan)),
                "oracle_quality": float(oracle_info.get("oracle_quality", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def summarize_candidates(library: pd.DataFrame, query_counts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required = {
        (str(row.split), str(row.benchmark)): int(row.required_queries)
        for row in query_counts.itertuples(index=False)
    }
    for (candidate_id, split, benchmark), group in library.groupby(["candidate_id", "split", "benchmark"], sort=False):
        n = int(group["query_id"].nunique())
        req = int(required.get((str(split), str(benchmark)), 0))
        rows.append(
            {
                "candidate_id": str(candidate_id),
                "split": str(split),
                "benchmark": str(benchmark),
                "n_queries": n,
                "required_queries": req,
                "full_coverage": bool(req > 0 and n >= req),
                "mean_quality": float(group["quality_score"].astype(float).mean()),
                "mean_utility": float(group["utility"].astype(float).mean()),
                "mean_train_prior_quality": float(group["train_prior_quality"].astype(float).mean()),
                "mean_train_prior_utility": float(group["train_prior_utility"].astype(float).mean()),
                "val_train_prior_utility_score": float(
                    0.5 * group["utility"].astype(float).mean()
                    + 0.5 * group["train_prior_utility"].astype(float).mean()
                ),
                "val_train_prior_quality_score": float(
                    0.5 * group["quality_score"].astype(float).mean()
                    + 0.5 * group["train_prior_quality"].astype(float).mean()
                ),
                "oracle_mean_quality": float(group["oracle_quality"].astype(float).mean()),
                "oracle_mean_utility": float(group["oracle_utility"].astype(float).mean()),
                "oracle_utility_ratio": float(group["utility"].astype(float).mean())
                / max(float(group["oracle_utility"].astype(float).mean()), 1e-12),
                "frontier_call_rate": float(group["is_frontier"].astype(bool).mean()),
                "strong_or_frontier_call_rate": float(
                    np.mean(
                        group["is_frontier"].astype(bool).to_numpy()
                        | group["model_id"].astype(str).isin(STRONG_LOCAL).to_numpy()
                    )
                ),
                "models_json": json.dumps(group["model_id"].value_counts().sort_index().to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_portfolios(
    library: pd.DataFrame,
    summary: pd.DataFrame,
    query_counts: pd.DataFrame,
    oracle: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    choices: list[pd.DataFrame] = []
    for objective, metric in [
        ("utility", "mean_utility"),
        ("utility_trainprior", "val_train_prior_utility_score"),
    ]:
        for cap in [0.25, 0.30, 0.35, 0.40, 0.45, 1.00]:
            selected = select_by_benchmark(summary, split="val", cap=cap, metric=metric)
            for split in ["val", "test"]:
                picked = apply_benchmark_selection(library, selected, split)
                method = f"per_benchmark_val_{objective}_frontiercap{cap:g}"
                rows.append(evaluate_choice_frame(method, split, picked, oracle, args))
                choices.append(picked.assign(portfolio_method=method))
    global_selected = select_global(summary, split="val")
    for split in ["val", "test"]:
        picked = apply_global_selection(library, global_selected, split)
        method = "global_val_best_candidate"
        rows.append(evaluate_choice_frame(method, split, picked, oracle, args))
        choices.append(picked.assign(portfolio_method=method))
    test_oracle = select_by_benchmark(summary, split="test", cap=1.0, metric="mean_utility")
    picked = apply_benchmark_selection(library, test_oracle, "test")
    rows.append(evaluate_choice_frame("diagnostic_test_best_per_benchmark", "test", picked, oracle, args))
    choices.append(picked.assign(portfolio_method="diagnostic_test_best_per_benchmark"))
    return pd.DataFrame(rows), pd.concat(choices, ignore_index=True)


def select_by_benchmark(summary: pd.DataFrame, *, split: str, cap: float, metric: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    val = summary[summary["split"].astype(str).eq(split) & summary["full_coverage"].astype(bool)].copy()
    for benchmark, group in val.groupby("benchmark", sort=False):
        feasible = group[group["frontier_call_rate"].astype(float) <= float(cap)].copy()
        if feasible.empty:
            feasible = group
        best = feasible.sort_values([metric, "mean_quality", "frontier_call_rate"], ascending=[False, False, True]).iloc[0]
        selected[str(benchmark)] = str(best["candidate_id"])
    return selected


def select_global(summary: pd.DataFrame, *, split: str) -> str:
    frame = summary[summary["split"].astype(str).eq(split) & summary["full_coverage"].astype(bool)].copy()
    agg = (
        frame.groupby("candidate_id", as_index=False)
        .agg(
            n_benchmarks=("benchmark", "nunique"),
            mean_utility=("mean_utility", "mean"),
            mean_quality=("mean_quality", "mean"),
            frontier_call_rate=("frontier_call_rate", "mean"),
        )
        .sort_values(["n_benchmarks", "mean_utility", "mean_quality"], ascending=[False, False, False])
    )
    return str(agg.iloc[0]["candidate_id"])


def apply_benchmark_selection(library: pd.DataFrame, selected: dict[str, str], split: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for benchmark, candidate_id in selected.items():
        part = library[
            library["split"].astype(str).eq(split)
            & library["benchmark"].astype(str).eq(benchmark)
            & library["candidate_id"].astype(str).eq(candidate_id)
        ].copy()
        part["chosen_candidate_id"] = candidate_id
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def apply_global_selection(library: pd.DataFrame, candidate_id: str, split: str) -> pd.DataFrame:
    out = library[
        library["split"].astype(str).eq(split)
        & library["candidate_id"].astype(str).eq(candidate_id)
    ].copy()
    out["chosen_candidate_id"] = candidate_id
    return out


def evaluate_choice_frame(
    method: str,
    split: str,
    frame: pd.DataFrame,
    oracle: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, Any]:
    frame = frame.drop_duplicates("query_id", keep="last").copy()
    if frame.empty:
        return {
            "method": method,
            "split": split,
            "n_queries": 0,
        }
    values = frame["utility"].astype(float).to_numpy()
    ci_low, ci_high = bootstrap_ci(values, int(args.bootstrap_samples), int(args.seed))
    oracle_mean_utility = float(frame["oracle_utility"].astype(float).mean())
    oracle_mean_quality = float(frame["oracle_quality"].astype(float).mean())
    return {
        "method": method,
        "split": split,
        "n_queries": int(len(frame)),
        "mean_quality": float(frame["quality_score"].astype(float).mean()),
        "mean_utility": float(values.mean()),
        "mean_utility_ci_low": ci_low,
        "mean_utility_ci_high": ci_high,
        "oracle_mean_quality": oracle_mean_quality,
        "oracle_mean_utility": oracle_mean_utility,
        "oracle_utility_ratio": float(values.mean()) / max(oracle_mean_utility, 1e-12),
        "quality_gap_to_oracle": oracle_mean_quality - float(frame["quality_score"].astype(float).mean()),
        "utility_gap_to_oracle": oracle_mean_utility - float(values.mean()),
        "frontier_call_rate": float(frame["is_frontier"].astype(bool).mean()),
        "strong_or_frontier_call_rate": float(
            np.mean(
                frame["is_frontier"].astype(bool).to_numpy()
                | frame["model_id"].astype(str).isin(STRONG_LOCAL).to_numpy()
            )
        ),
        "selected_models_json": json.dumps(frame["model_id"].value_counts().sort_index().to_dict(), sort_keys=True),
        "chosen_candidates_json": json.dumps(
            frame["chosen_candidate_id"].value_counts().sort_index().to_dict(), sort_keys=True
        ),
    }


def selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[table["split"].astype(str).eq("val")].copy()
    for cap_name, cap in [("frontier_le_0.40", 0.40), ("frontier_le_0.35", 0.35), ("frontier_le_0.30", 0.30)]:
        feasible = val[val["frontier_call_rate"].astype(float) <= cap].sort_values(
            ["mean_utility", "mean_quality"], ascending=[False, False]
        )
        if feasible.empty:
            continue
        method = str(feasible.iloc[0]["method"])
        rows.append(feasible.head(1).assign(selection_rule=f"val_best_{cap_name}"))
        rows.append(
            table[table["method"].astype(str).eq(method) & table["split"].astype(str).eq("test")].assign(
                selection_rule=f"val_best_{cap_name}_test"
            )
        )
    if not val.empty:
        method = str(val.sort_values(["mean_utility", "mean_quality"], ascending=[False, False]).iloc[0]["method"])
        rows.append(val[val["method"].astype(str).eq(method)].head(1).assign(selection_rule="val_best_unconstrained"))
        rows.append(
            table[table["method"].astype(str).eq(method) & table["split"].astype(str).eq("test")].assign(
                selection_rule="val_best_unconstrained_test"
            )
        )
    for cap in [0.30, 0.40]:
        method = f"per_benchmark_val_utility_trainprior_frontiercap{cap:g}"
        rows.append(
            table[table["method"].astype(str).eq(method) & table["split"].astype(str).eq("val")].assign(
                selection_rule=f"trainprior_prespecified_cap{cap:g}"
            )
        )
        rows.append(
            table[table["method"].astype(str).eq(method) & table["split"].astype(str).eq("test")].assign(
                selection_rule=f"trainprior_prespecified_cap{cap:g}_test"
            )
        )
    diagnostic = table[table["method"].astype(str).eq("diagnostic_test_best_per_benchmark")]
    if not diagnostic.empty:
        rows.append(diagnostic.assign(selection_rule="diagnostic_test_best_per_benchmark"))
    return pd.concat(rows, ignore_index=True).drop_duplicates(["selection_rule", "method", "split"]) if rows else pd.DataFrame()


def first_existing(columns: Any, candidates: tuple[str, ...]) -> str:
    present = set(columns)
    for candidate in candidates:
        if candidate in present:
            return candidate
    return ""


def is_blocked_method(value: str) -> bool:
    return any(term in value for term in BLOCKED_TERMS)


def sanitize_candidate_id(prefix: str, method: str) -> str:
    return sanitize(prefix) + ":" + sanitize(method)[:160]


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", str(value).strip())
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unnamed"


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [float(values[rng.integers(0, len(values), len(values))].mean()) for _ in range(max(1, samples))]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_memo(path: Path, args: argparse.Namespace, manifest: pd.DataFrame, selected: pd.DataFrame) -> None:
    cols = [
        "selection_rule",
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_mean_quality",
        "oracle_mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "quality_gap_to_oracle",
        "utility_gap_to_oracle",
    ]
    lines = [
        "# Validation Policy-Library Portfolio",
        "",
        "This no-call experiment composes existing cached RouteCode/ProbeRoute policies.",
        "It excludes oracle, post-hoc, same-answer, and diagnostic methods, chooses a candidate policy per benchmark on validation only, and reports held-out test.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/199_validation_policy_library_portfolio.py",
        "```",
        "",
        "## Inputs",
        "",
        f"- Action matrix: `{args.outputs}`",
        f"- Policy library root: `{args.policy_root}`",
        f"- Candidate source/method pairs loaded: `{len(manifest)}`",
        "- No GPT, Gemini, Claude, local generation, or vLLM serving calls are made.",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[column for column in cols if column in selected.columns]]) if not selected.empty else "No selected rows.",
        "",
        "## Interpretation",
        "",
        "- A positive deployable result would improve held-out utility and quality under validation-selected frontier caps.",
        "- If the diagnostic test-best portfolio is much better than validation-selected portfolios, the policy library has useful pieces but current validation composition is unstable.",
        "- This is still policy composition, not a final trained router or paper claim.",
        "",
        "## Artifacts",
        "",
        f"- Manifest: `{path.parent / 'table_policy_library_manifest.csv'}`",
        f"- Candidate summary: `{path.parent / 'table_policy_library_candidate_summary.csv'}`",
        f"- Portfolio table: `{path.parent / 'table_policy_library_portfolio_all.csv'}`",
        f"- Selected table: `{path.parent / 'table_policy_library_portfolio_selected.csv'}`",
        f"- Query choices: `{path.parent / 'table_policy_library_portfolio_query_choices.csv'}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
