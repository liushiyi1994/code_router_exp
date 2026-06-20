from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TOOL = "deterministic_math_tool"
LOCAL_POOL = (
    TOOL,
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
)
FRONTIER_POOL = ("gemini-3.5-flash", "gpt-5.5", "gemini-3.5-flash-strong-solve")
STRONG_OR_FRONTIER_POOL = (
    "qwen3-32b-awq-local",
    "qwen3-32b-awq-selfconsistency-n3-local",
    *FRONTIER_POOL,
)


@dataclass(frozen=True)
class AnswerGroupConfig:
    reliability_mode: str
    support_weight: float
    fallback_kind: str
    threshold: float
    force_local_benchmarks: tuple[str, ...]
    force_frontier_benchmarks: tuple[str, ...]

    @property
    def method(self) -> str:
        local = "-".join(self.force_local_benchmarks) or "none"
        frontier = "-".join(self.force_frontier_benchmarks) or "none"
        return (
            f"answer_group_{self.reliability_mode}_w{self.support_weight:g}"
            f"_fb{self.fallback_kind}_thr{self.threshold:g}"
            f"_local{local}_frontier{frontier}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train-only local answer-group verifier policy for broad100.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_vllm_self_consistency_probe/model_outputs_with_self_consistency.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_answer_group_verifier_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp172 = load_module("experiments/172_tool_aware_deployed_action_policy.py", "deployed_172")
    outputs = exp172.prepare_outputs(pd.read_parquet(args.outputs))
    target = pd.read_csv(args.target_table)

    context = build_context(outputs, target)
    configs = candidate_configs()
    policy_internal, selected_choices = evaluate_configs(configs, context, outputs, target, exp172, lambda_cost=float(args.lambda_cost))
    policy_table = exp172.add_bootstrap_ci(policy_internal, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    policy_table = policy_table.drop(columns=["_utility_values"], errors="ignore")
    selected = selected_rows(policy_internal, exp172, bootstrap_samples=int(args.bootstrap_samples), seed=int(args.seed))
    query_choices = query_choice_rows(selected, selected_choices, outputs, target)
    reliability = reliability_table(context)

    policy_table.to_csv(args.output_dir / "table_answer_group_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_answer_group_policy_selected.csv", index=False)
    query_choices.to_csv(args.output_dir / "table_answer_group_query_choices.csv", index=False)
    reliability.to_csv(args.output_dir / "table_answer_group_reliability.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(args.output_dir / "ANSWER_GROUP_VERIFIER_POLICY_MEMO.md", args, policy_table, selected)
    print(f"Wrote answer-group verifier policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def normalized_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in {"nan", "none", "no_code"} or text.startswith("failed"):
        return ""
    return text


def candidate_configs() -> list[AnswerGroupConfig]:
    force_local_sets = [
        (),
        ("humaneval", "mbpp"),
        ("humaneval", "mbpp", "bbh", "aime", "livemathbench"),
    ]
    force_frontier_sets = [
        (),
        ("gpqa",),
        ("mmlupro",),
        ("gpqa", "mmlupro"),
        ("gpqa", "mmlupro", "math500"),
    ]
    configs: list[AnswerGroupConfig] = []
    for mode in ["bench_support_strong", "bench_support", "support"]:
        for weight in [0.0, 0.05]:
            for fallback in ["overall", "frontier"]:
                for threshold in [0.0, 0.2, 0.4, 0.6, 0.65, 0.75, 0.9, 1.0]:
                    for force_local in force_local_sets:
                        for force_frontier in force_frontier_sets:
                            configs.append(
                                AnswerGroupConfig(
                                    reliability_mode=mode,
                                    support_weight=float(weight),
                                    fallback_kind=fallback,
                                    threshold=float(threshold),
                                    force_local_benchmarks=tuple(force_local),
                                    force_frontier_benchmarks=tuple(force_frontier),
                                )
                            )
    return configs


def build_context(outputs: pd.DataFrame, target: pd.DataFrame) -> dict[str, Any]:
    frame = outputs.copy()
    frame["answer_norm_group"] = frame["parsed_answer"].map(normalized_answer)
    model_prior = (
        frame[frame["split"].astype(str).eq("train")]
        .groupby(["benchmark", "model_id"], as_index=False)
        .agg(
            train_model_quality=("quality_score", "mean"),
            train_model_utility=("utility", "mean"),
        )
    )
    prior_utility = {
        (str(row.benchmark), str(row.model_id)): float(row.train_model_utility)
        for row in model_prior.itertuples(index=False)
    }
    best_by_kind: dict[str, dict[str, str]] = {}
    for kind, pool in {
        "overall": None,
        "local": LOCAL_POOL,
        "frontier": FRONTIER_POOL,
    }.items():
        table: dict[str, str] = {}
        for benchmark, group in model_prior.groupby("benchmark", sort=False):
            candidates = group if pool is None else group[group["model_id"].astype(str).isin(pool)]
            if candidates.empty:
                candidates = group
            best = candidates.sort_values(
                ["train_model_utility", "train_model_quality"], ascending=[False, False]
            ).iloc[0]
            table[str(benchmark)] = str(best["model_id"])
        best_by_kind[kind] = table

    local = frame[frame["model_id"].astype(str).isin(LOCAL_POOL)].copy()
    groups = build_answer_groups(local)
    train_groups = groups[groups["split"].astype(str).eq("train")].copy()
    global_mean = float(train_groups["correct"].mean()) if not train_groups.empty else 0.0
    benchmark_mean = train_groups.groupby("benchmark")["correct"].mean().astype(float).to_dict()
    rels = build_reliability_maps(train_groups, global_mean, benchmark_mean)
    choices = precompute_local_choices(groups, local, rels, benchmark_mean, global_mean, prior_utility)
    rows_by_query = {
        str(query_id): group.set_index("model_id").to_dict("index")
        for query_id, group in frame.groupby("query_id", sort=False)
    }
    query_meta = target.drop_duplicates("query_id").set_index("query_id").to_dict("index")
    return {
        "outputs": frame,
        "groups": groups,
        "train_groups": train_groups,
        "reliability_maps": rels,
        "benchmark_mean": benchmark_mean,
        "global_mean": global_mean,
        "choices": choices,
        "rows_by_query": rows_by_query,
        "query_meta": query_meta,
        "best_by_kind": best_by_kind,
    }


def build_answer_groups(local: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (query_id, answer), group in local.groupby(["query_id", "answer_norm_group"], sort=False):
        if not str(answer):
            continue
        models = tuple(sorted(str(model) for model in group["model_id"].tolist()))
        first = group.iloc[0]
        rows.append(
            {
                "query_id": str(query_id),
                "split": str(first["split"]),
                "benchmark": str(first["benchmark"]),
                "answer_norm_group": str(answer),
                "support": int(len(group)),
                "models": models,
                "has_tool": int(TOOL in models),
                "has_14b": int("qwen3-14b-awq-local" in models),
                "has_32b": int("qwen3-32b-awq-local" in models),
                "has_self": int("qwen3-32b-awq-selfconsistency-n3-local" in models),
                "correct": float(group["quality_score"].max()),
            }
        )
    return pd.DataFrame(rows)


def reliability_key(row: pd.Series | Any, mode: str) -> tuple[Any, ...]:
    benchmark = str(row.benchmark)
    support = int(row.support)
    if mode == "bench_support_strong":
        return (benchmark, support, int(row.has_tool), int(row.has_14b), int(row.has_32b), int(row.has_self))
    if mode == "bench_support":
        return (benchmark, support)
    if mode == "support":
        return (support,)
    raise ValueError(f"Unknown reliability mode: {mode}")


def build_reliability_maps(
    train_groups: pd.DataFrame, global_mean: float, benchmark_mean: dict[str, float]
) -> dict[str, dict[tuple[Any, ...], float]]:
    maps: dict[str, dict[tuple[Any, ...], float]] = {}
    for mode in ["bench_support_strong", "bench_support", "support"]:
        table: dict[tuple[Any, ...], float] = {}
        if train_groups.empty:
            maps[mode] = table
            continue
        keys = train_groups.apply(lambda row: reliability_key(row, mode), axis=1)
        for key, indices in keys.groupby(keys).groups.items():
            group = train_groups.loc[indices]
            prior = benchmark_mean.get(str(group.iloc[0]["benchmark"]), global_mean) if mode.startswith("bench") else global_mean
            table[key] = float((group["correct"].sum() + 2.0 * prior) / (len(group) + 2.0))
        maps[mode] = table
    return maps


def group_reliability(row: pd.Series | Any, mode: str, maps: dict[str, dict[tuple[Any, ...], float]], benchmark_mean: dict[str, float], global_mean: float) -> float:
    for candidate_mode in [mode, "bench_support", "support"]:
        key = reliability_key(row, candidate_mode)
        if key in maps.get(candidate_mode, {}):
            return float(maps[candidate_mode][key])
    return float(benchmark_mean.get(str(row.benchmark), global_mean))


def precompute_local_choices(
    groups: pd.DataFrame,
    local: pd.DataFrame,
    rels: dict[str, dict[tuple[Any, ...], float]],
    benchmark_mean: dict[str, float],
    global_mean: float,
    prior_utility: dict[tuple[str, str], float],
) -> dict[tuple[str, float], dict[str, dict[str, Any]]]:
    choices: dict[tuple[str, float], dict[str, dict[str, Any]]] = {}
    if groups.empty:
        return choices
    local_by_query_answer = {
        (str(query_id), str(answer)): group
        for (query_id, answer), group in local.groupby(["query_id", "answer_norm_group"], sort=False)
    }
    for mode in ["bench_support_strong", "bench_support", "support"]:
        scored = groups.copy()
        scored["answer_group_score"] = [
            group_reliability(row, mode, rels, benchmark_mean, global_mean) for row in scored.itertuples(index=False)
        ]
        for weight in [0.0, 0.05]:
            out: dict[str, dict[str, Any]] = {}
            scored["rank_score"] = scored["answer_group_score"] + float(weight) * scored["support"].astype(float)
            for query_id, group in scored.groupby("query_id", sort=False):
                best_group = group.sort_values(
                    ["rank_score", "answer_group_score", "support"], ascending=[False, False, False]
                ).iloc[0]
                local_rows = local_by_query_answer[(str(query_id), str(best_group["answer_norm_group"]))]
                best_model = choose_action_for_group(local_rows, prior_utility)
                out[str(query_id)] = {
                    "model_id": str(best_model),
                    "answer_group_score": float(best_group["answer_group_score"]),
                    "answer_group_support": int(best_group["support"]),
                    "answer_group": str(best_group["answer_norm_group"]),
                }
            choices[(mode, float(weight))] = out
    return choices


def choose_action_for_group(local_rows: pd.DataFrame, prior_utility: dict[tuple[str, str], float]) -> str:
    best_model = str(local_rows.iloc[0]["model_id"])
    best_key = (-1e9, -1e9)
    for row in local_rows.itertuples(index=False):
        key = (
            float(prior_utility.get((str(row.benchmark), str(row.model_id)), 0.0)),
            float(row.quality_score),
        )
        if key > best_key:
            best_key = key
            best_model = str(row.model_id)
    return best_model


def evaluate_configs(
    configs: list[AnswerGroupConfig],
    context: dict[str, Any],
    outputs: pd.DataFrame,
    target: pd.DataFrame,
    exp172,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    frontiers = set(outputs[outputs["is_frontier"].astype(bool)]["model_id"].astype(str))
    rows: list[dict[str, Any]] = []
    choices: dict[tuple[str, str], pd.DataFrame] = {}
    for config in configs:
        for split in ["val", "test"]:
            frame = target[target["split"].astype(str).eq(split)].copy()
            selected = select_actions(frame, config, context)
            selected_rows = selected.merge(outputs, on=["query_id", "model_id"], how="left")
            selected_rows = selected_rows[selected_rows["split"].astype(str).eq(split)].copy()
            row = exp172.evaluate_selected_rows(
                config.method,
                "answer_group_verifier",
                split,
                selected_rows,
                outputs,
                target=frame,
                frontiers=frontiers,
                lambda_cost=lambda_cost,
            )
            row.update(
                {
                    "reliability_mode": config.reliability_mode,
                    "support_weight": config.support_weight,
                    "fallback_kind": config.fallback_kind,
                    "threshold": config.threshold,
                    "force_local_benchmarks": ",".join(config.force_local_benchmarks),
                    "force_frontier_benchmarks": ",".join(config.force_frontier_benchmarks),
                    "answer_group_local_rate": float(selected["used_answer_group"].mean()),
                    "answer_group_mean_score": float(selected["answer_group_score"].replace(-1, np.nan).mean()),
                    "answer_group_mean_support": float(selected["answer_group_support"].replace(0, np.nan).mean()),
                }
            )
            rows.append(row)
            choices[(config.method, split)] = selected
    return pd.DataFrame(rows).sort_values(["split", "mean_utility"], ascending=[True, False]), choices


def select_actions(frame: pd.DataFrame, config: AnswerGroupConfig, context: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    choices = context["choices"].get((config.reliability_mode, float(config.support_weight)), {})
    rows_by_query = context["rows_by_query"]
    best_by_kind = context["best_by_kind"]
    for row in frame.itertuples(index=False):
        query_id = str(row.query_id)
        benchmark = str(row.benchmark)
        choice = choices.get(query_id)
        score = float(choice["answer_group_score"]) if choice else -1.0
        support = int(choice["answer_group_support"]) if choice else 0
        use_answer_group = bool(choice and (score >= config.threshold or benchmark in config.force_local_benchmarks))
        if benchmark in config.force_frontier_benchmarks:
            use_answer_group = False
        if use_answer_group and choice:
            model_id = str(choice["model_id"])
        else:
            model_id = fallback_model(config.fallback_kind, benchmark, best_by_kind)
        if model_id not in rows_by_query.get(query_id, {}):
            model_id = first_available(rows_by_query.get(query_id, {}), (*LOCAL_POOL, *FRONTIER_POOL))
        rows.append(
            {
                "query_id": query_id,
                "model_id": str(model_id),
                "answer_group_score": score,
                "answer_group_support": support,
                "answer_group": str(choice["answer_group"]) if choice else "",
                "used_answer_group": bool(use_answer_group),
            }
        )
    return pd.DataFrame(rows)


def fallback_model(kind: str, benchmark: str, best_by_kind: dict[str, dict[str, str]]) -> str:
    if kind in best_by_kind:
        return best_by_kind[kind].get(benchmark, "qwen3-14b-awq-local")
    return kind


def first_available(actions: dict[str, Any], pool: tuple[str, ...]) -> str:
    for model_id in pool:
        if model_id in actions:
            return str(model_id)
    return str(next(iter(actions.keys())))


def selected_rows(table: pd.DataFrame, exp172, *, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    val = table[table["split"].eq("val")].copy()
    for cap in [0.40, 0.45]:
        feasible = val[val["frontier_call_rate"] <= cap].sort_values(
            ["mean_utility", "mean_quality"], ascending=False
        )
        if not feasible.empty:
            method = str(feasible.iloc[0]["method"])
            rows.append(feasible.head(1).assign(selection_rule=f"val_best_utility_frontier_le_{cap:g}"))
            rows.append(
                table[table["method"].eq(method) & table["split"].eq("test")].assign(
                    selection_rule=f"val_best_utility_frontier_le_{cap:g}_test"
                )
            )
    if not val.empty:
        method = str(val.sort_values(["mean_utility", "mean_quality"], ascending=False).iloc[0]["method"])
        rows.append(val[val["method"].eq(method)].head(1).assign(selection_rule="val_best_utility_unconstrained"))
        rows.append(table[table["method"].eq(method) & table["split"].eq("test")].assign(selection_rule="val_best_utility_unconstrained_test"))
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if selected.empty:
        return selected
    values = table[["method", "split", "_utility_values"]]
    selected = selected.drop(columns=["_utility_values"], errors="ignore").merge(values, on=["method", "split"], how="left")
    selected = exp172.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)
    return selected.drop(columns=["_utility_values"], errors="ignore")


def query_choice_rows(selected: pd.DataFrame, choices: dict[tuple[str, str], pd.DataFrame], outputs: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    methods = set(selected["method"].astype(str).tolist())
    frames: list[pd.DataFrame] = []
    for (method, split), frame in choices.items():
        if method not in methods or split != "test":
            continue
        merged = (
            frame.merge(outputs, on=["query_id", "model_id"], how="left")
            .merge(target[["query_id", "best_local_action", "best_large_action", "local_quality", "large_quality"]], on="query_id", how="left")
        )
        merged["method"] = method
        frames.append(
            merged[
                [
                    "method",
                    "query_id",
                    "benchmark",
                    "model_id",
                    "quality_score",
                    "utility",
                    "normalized_remote_cost",
                    "is_frontier",
                    "parsed_answer",
                    "answer_group_score",
                    "answer_group_support",
                    "answer_group",
                    "used_answer_group",
                    "best_local_action",
                    "best_large_action",
                    "local_quality",
                    "large_quality",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True).drop_duplicates(["method", "query_id"]) if frames else pd.DataFrame()


def reliability_table(context: dict[str, Any]) -> pd.DataFrame:
    train_groups = context["train_groups"]
    if train_groups.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for mode in ["bench_support_strong", "bench_support", "support"]:
        scored = train_groups.copy()
        scored["reliability"] = [
            group_reliability(
                row,
                mode,
                context["reliability_maps"],
                context["benchmark_mean"],
                context["global_mean"],
            )
            for row in scored.itertuples(index=False)
        ]
        summary = (
            scored.groupby(["benchmark", "support"], as_index=False)
            .agg(n_groups=("query_id", "size"), empirical_correct=("correct", "mean"), mean_reliability=("reliability", "mean"))
        )
        summary["reliability_mode"] = mode
        rows.extend(summary.to_dict("records"))
    return pd.DataFrame(rows)


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot["method"].iloc[::-1], plot["mean_utility"].iloc[::-1], color="#597a68")
    ax.set_xlabel("Held-out test utility")
    ax.set_title("Answer-Group Verifier Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_answer_group_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    selected_cols = [
        "method",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "within_3pct_oracle_utility",
        "within_3pt_oracle_quality",
        "frontier_call_rate",
        "strong_or_frontier_call_rate",
        "answer_group_local_rate",
        "answer_group_mean_score",
        "answer_group_mean_support",
        "selection_rule",
    ]
    lines = [
        "# Answer-Group Verifier Policy",
        "",
        "This cached experiment tests a train-only local answer-group verifier. It calibrates local answer-group reliability on train rows, selects thresholds on validation, and reports held-out test. It makes no GPT, Gemini, Claude, vLLM, or local model calls.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/178_answer_group_verifier_policy.py",
        (
            "PYTHONPATH=src python experiments/178_answer_group_verifier_policy.py "
            f"--target-table {args.target_table} "
            f"--outputs {args.outputs} "
            f"--output-dir {args.output_dir}"
        ),
        "```",
        "",
        "## Selected Rows",
        "",
        markdown_table(selected[[col for col in selected_cols if col in selected.columns]]),
        "",
        "## Best Held-Out Diagnostics",
        "",
        markdown_table(
            table[table["split"].eq("test")]
            .sort_values(["mean_utility", "mean_quality"], ascending=False)
            .head(12)[[col for col in selected_cols if col in table.columns]]
        ),
        "",
        "## Interpretation",
        "",
        "- Local answer-group reliability is a plausible verifier signal, but validation-selected thresholds do not close the broad100 oracle gap.",
        "- This result supports the current bottleneck story: local answer agreement is useful on some slices but too noisy for GPQA, MMLUPro, AIME, and exact math.",
        "- The next probe should add stronger task-specific checker evidence rather than another shallow answer-support threshold.",
        "",
        "## Artifacts",
        "",
        f"- All policy table: `{args.output_dir / 'table_answer_group_policy_all.csv'}`",
        f"- Selected policy table: `{args.output_dir / 'table_answer_group_policy_selected.csv'}`",
        f"- Query choices: `{args.output_dir / 'table_answer_group_query_choices.csv'}`",
        f"- Reliability table: `{args.output_dir / 'table_answer_group_reliability.csv'}`",
        f"- Figure: `{args.output_dir / 'fig_answer_group_policy_utility.pdf'}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            elif isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
