from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import load_env_values, resolve_key


GEMINI_STRONG = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retry Gemini strong-solve on Broad100 deterministic-tool residual rows "
            "with a larger thinking budget, then evaluate the no-tool/full oracle bound."
        )
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
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--query-ids-file",
        type=Path,
        default=Path("results/controlled/broad100_gpt_strong_residual2048/residual_tool_query_ids.txt"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_gemini_residual_thinking"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-api-spend-usd", type=float, default=4.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp129 = load_module("experiments/129_broad_gemini_strong_solver.py", "exp129_for_220")
    exp213 = load_module("experiments/213_broad100_target_method_package.py", "exp213_for_220")

    outputs = pd.read_parquet(args.outputs).copy()
    outputs = normalize_outputs(outputs, lambda_cost=float(args.lambda_cost))
    query_ids = load_query_ids(args.query_ids_file)
    queries = (
        outputs[outputs["query_id"].astype(str).isin(query_ids)]
        .drop_duplicates("query_id")
        .sort_values(["split", "benchmark", "query_id"])
        .reset_index(drop=True)
    )
    estimate = estimate_table(exp129, queries, args)
    estimate.to_csv(args.output_dir / "table_gemini_residual_thinking_cost_estimate.csv", index=False)
    queries[
        ["query_id", "query_text", "split", "benchmark", "domain", "metric", "gold_answer"]
    ].to_csv(args.output_dir / "table_gemini_residual_thinking_manifest.csv", index=False)
    if bool(estimate.iloc[0]["exceeds_spend_cap"]):
        raise RuntimeError(
            f"Estimated Gemini spend ${float(estimate.iloc[0]['estimated_uncached_cost_usd']):.4f} "
            f"exceeds cap ${float(args.max_api_spend_usd):.4f}."
        )
    if args.dry_run:
        write_dry_run_memo(args.output_dir / "GEMINI_RESIDUAL_THINKING_MEMO.md", estimate)
        print(f"Wrote dry-run Gemini residual estimate to {args.output_dir}")
        return

    api_key = resolve_key(
        load_env_values(args.env_file),
        ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"],
    )
    if not api_key:
        raise RuntimeError("Missing Gemini API key.")
    strong = exp129.collect_rows(
        queries,
        args.output_dir,
        api_key=api_key,
        max_output_tokens=int(args.max_output_tokens),
        thinking_budget=int(args.thinking_budget),
        temperature=float(args.temperature),
        max_api_spend_usd=float(args.max_api_spend_usd),
        concurrency=int(args.concurrency),
    )
    strong.to_csv(args.output_dir / "table_gemini_residual_thinking_outputs.csv", index=False)
    augmented = replace_gemini_strong_rows(exp129, outputs, strong, lambda_cost=float(args.lambda_cost))
    augmented.to_parquet(args.output_dir / "model_outputs_with_gemini_residual_thinking.parquet", index=False)

    bounds, choices = evaluate_bounds(
        exp213,
        original=outputs,
        augmented=augmented,
        target_table=pd.read_csv(args.target_table),
        lambda_cost=float(args.lambda_cost),
    )
    bounds.to_csv(args.output_dir / "table_gemini_residual_thinking_bounds.csv", index=False)
    choices.to_csv(args.output_dir / "table_gemini_residual_thinking_query_choices.csv", index=False)
    write_figure(args.output_dir, bounds)
    write_memo(args.output_dir / "GEMINI_RESIDUAL_THINKING_MEMO.md", args, estimate, strong, bounds)
    print(f"Wrote Gemini residual-thinking results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_outputs(outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    out = outputs.copy()
    for column in ["quality_score", "normalized_remote_cost", "cost_total_usd", "latency_s"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["query_id"] = out["query_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    out["utility"] = out["quality_score"] - float(lambda_cost) * out["normalized_remote_cost"]
    return out


def load_query_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def estimate_table(exp129: Any, queries: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    cache_dir = (
        Path(args.output_dir)
        / "raw_gemini_strong_solver"
        / exp129.GEMINI_MODEL
        / f"think_{int(args.thinking_budget)}_max_{int(args.max_output_tokens)}"
    )
    prompts = [exp129.prompt_for(row) for _, row in queries.iterrows()]
    missing = [
        prompt
        for prompt, (_, row) in zip(prompts, queries.iterrows())
        if exp129.load_json_or_none(
            cache_dir / exp129.cache_name(str(row["query_id"]), int(args.thinking_budget), int(args.max_output_tokens))
        )
        is None
    ]
    estimated = exp129.estimate_missing_cost(missing, int(args.max_output_tokens), int(args.thinking_budget))
    return pd.DataFrame(
        [
            {
                "model_id": GEMINI_STRONG,
                "selected_queries": int(len(queries)),
                "cached_queries": int(len(queries) - len(missing)),
                "uncached_queries": int(len(missing)),
                "thinking_budget": int(args.thinking_budget),
                "max_output_tokens": int(args.max_output_tokens),
                "estimated_uncached_cost_usd": float(estimated),
                "max_api_spend_usd": float(args.max_api_spend_usd),
                "within_spend_cap": bool(estimated <= float(args.max_api_spend_usd) + 1e-12),
                "exceeds_spend_cap": bool(estimated > float(args.max_api_spend_usd) + 1e-12),
            }
        ]
    )


def replace_gemini_strong_rows(
    exp129: Any,
    outputs: pd.DataFrame,
    strong: pd.DataFrame,
    *,
    lambda_cost: float,
) -> pd.DataFrame:
    query_ids = set(str(query_id) for query_id in strong["query_id"])
    base = outputs[
        ~(
            outputs["query_id"].astype(str).isin(query_ids)
            & outputs["model_id"].astype(str).eq(GEMINI_STRONG)
        )
    ].copy()
    return exp129.append_strong_rows(base, strong, lambda_cost=lambda_cost)


def evaluate_bounds(
    exp213: Any,
    *,
    original: pd.DataFrame,
    augmented: pd.DataFrame,
    target_table: pd.DataFrame,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_original = exp213.rebuild_target_pool(
        target_table, original, exp213.FULL_LOCAL_ACTIONS, exp213.LARGE_ACTIONS, lambda_cost
    )
    no_tool_original = exp213.rebuild_target_pool(
        target_table, original, exp213.NO_TOOL_LOCAL_ACTIONS, exp213.LARGE_ACTIONS, lambda_cost
    )
    full_augmented = exp213.rebuild_target_pool(
        target_table, augmented, exp213.FULL_LOCAL_ACTIONS, exp213.LARGE_ACTIONS, lambda_cost
    )
    no_tool_augmented = exp213.rebuild_target_pool(
        target_table, augmented, exp213.NO_TOOL_LOCAL_ACTIONS, exp213.LARGE_ACTIONS, lambda_cost
    )
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        full_ref = full_original[full_original["split"].eq(split)].copy()
        frames = [
            ("full_original_oracle", "full_original", full_ref, full_ref),
            (
                "no_tool_original_oracle_vs_full",
                "no_tool_original_vs_full",
                no_tool_original[no_tool_original["split"].eq(split)].copy(),
                full_ref,
            ),
            (
                "no_tool_gemini_residual_oracle_vs_full",
                "no_tool_gemini_residual_vs_full",
                no_tool_augmented[no_tool_augmented["split"].eq(split)].copy(),
                full_ref,
            ),
            (
                "no_tool_gemini_residual_oracle_self",
                "no_tool_gemini_residual_self",
                no_tool_augmented[no_tool_augmented["split"].eq(split)].copy(),
                no_tool_augmented[no_tool_augmented["split"].eq(split)].copy(),
            ),
            (
                "full_gemini_residual_augmented_oracle",
                "full_augmented",
                full_augmented[full_augmented["split"].eq(split)].copy(),
                full_augmented[full_augmented["split"].eq(split)].copy(),
            ),
        ]
        for method, role, frame, reference in frames:
            choose = frame["large_utility"].to_numpy(dtype=float) >= frame["local_utility"].to_numpy(dtype=float)
            row, detail = exp213.evaluate_policy(
                frame,
                choose,
                oracle_reference=reference,
                split=split,
                method=method,
                family="gemini_residual_thinking_bound",
                action_pool_variant="gemini_residual_thinking",
                lambda_cost=lambda_cost,
            )
            row["bound_role"] = role
            rows.append(row)
            details.append(detail.assign(bound_role=role))
    return exp213.add_target_gates(pd.DataFrame(rows)), pd.concat(details, ignore_index=True)


def write_figure(output_dir: Path, bounds: pd.DataFrame) -> None:
    plot = bounds[
        bounds["split"].eq("test")
        & bounds["method"].isin(
            [
                "full_original_oracle",
                "no_tool_original_oracle_vs_full",
                "no_tool_gemini_residual_oracle_vs_full",
                "full_gemini_residual_augmented_oracle",
            ]
        )
    ][["method", "mean_utility"]].copy()
    plot = plot.sort_values("mean_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(plot["method"], plot["mean_utility"], color="#6f7f3f")
    ax.set_xlabel("Held-out Broad100 test mean utility")
    ax.set_title("Gemini Residual Thinking Action Feasibility")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_gemini_residual_thinking_utility.pdf")
    plt.close(fig)


def write_dry_run_memo(path: Path, estimate: pd.DataFrame) -> None:
    lines = [
        "# Broad100 Gemini Residual Thinking Action",
        "",
        "Dry run only. No provider calls were made.",
        "",
        "```csv",
        estimate.to_csv(index=False).strip(),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_memo(
    path: Path,
    args: argparse.Namespace,
    estimate: pd.DataFrame,
    strong: pd.DataFrame,
    bounds: pd.DataFrame,
) -> None:
    test = bounds[bounds["split"].eq("test")].copy()
    by_benchmark = strong.groupby(["split", "benchmark"])["quality_score"].mean().reset_index()
    lines = [
        "# Broad100 Gemini Residual Thinking Action",
        "",
        "This experiment retries `gemini-3.5-flash-strong-solve` only on deterministic-tool residual rows.",
        f"Thinking budget: `{int(args.thinking_budget)}`. Max output tokens: `{int(args.max_output_tokens)}`.",
        "Claude is not used.",
        "",
        "## Cost Guard",
        "",
        "```csv",
        estimate.to_csv(index=False).strip(),
        "```",
        f"Actual recorded Gemini residual cost: `${float(strong['cost_total_usd'].sum()):.4f}`.",
        f"Rows: `{len(strong)}`; successful rows: `{int(strong['status'].eq('success').sum())}`.",
        "",
        "## Residual Quality By Benchmark",
        "",
        "```csv",
        by_benchmark.to_csv(index=False).strip(),
        "```",
        "",
        "## Held-Out Test Bounds",
        "",
        "```csv",
        test.to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        "- This is an action-pool feasibility test, not a deployable selector.",
        "- Passing here would mean a cheaper non-tool action can substitute for some deterministic-tool oracle wins.",
        "- Failing here means the clean no-tool gap remains an action-quality or cost-utility problem.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
