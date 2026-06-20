from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    score_output,
    usage_from_openai,
)


GPT = "gpt-5.5"
STRONG_MODEL_ID = "gpt-5.5-strong-solve"
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00
DEFAULT_EXACT_BENCHMARKS = ("aime", "gsm8k", "livemathbench", "math500")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect/evaluate a cached GPT-5.5 strong-solve action for Broad100 exact-answer rows. "
            "This tests whether a non-tool model action can substitute for deterministic-tool oracle wins."
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
        "--scores",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_scores.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_gpt_strong_math_action"),
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--benchmarks", default=",".join(DEFAULT_EXACT_BENCHMARKS))
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-api-spend-usd", type=float, default=12.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument(
        "--query-ids",
        default="",
        help="Optional comma-separated query_id allowlist. Useful for residual-only retries.",
    )
    parser.add_argument(
        "--query-ids-file",
        type=Path,
        default=None,
        help="Optional text file with one query_id per line. Combined with --query-ids if both are set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only write the query manifest and cost estimate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp213 = load_module("experiments/213_broad100_target_method_package.py", "exp213_for_219")
    outputs = pd.read_parquet(args.outputs).copy()
    outputs = normalize_outputs(outputs, lambda_cost=float(args.lambda_cost))
    queries = select_queries(outputs, args)
    prompts = [prompt_for(row) for _, row in queries.iterrows()]
    cache_dir = cache_directory(args)
    missing_prompts = [
        prompt
        for prompt, (_, row) in zip(prompts, queries.iterrows())
        if not (cache_dir / cache_name(str(row["query_id"]), str(args.reasoning_effort), int(args.max_output_tokens))).exists()
    ]
    estimate = estimate_cost(missing_prompts, int(args.max_output_tokens))
    estimate_table = pd.DataFrame(
        [
            {
                "model_id": STRONG_MODEL_ID,
                "selected_queries": int(len(queries)),
                "cached_queries": int(len(queries) - len(missing_prompts)),
                "uncached_queries": int(len(missing_prompts)),
                "estimated_uncached_cost_usd": estimate,
                "max_api_spend_usd": float(args.max_api_spend_usd),
                "within_spend_cap": bool(estimate <= float(args.max_api_spend_usd) + 1e-12),
            }
        ]
    )
    queries.to_csv(args.output_dir / "table_gpt_strong_math_action_manifest.csv", index=False)
    estimate_table.to_csv(args.output_dir / "table_gpt_strong_math_action_cost_estimate.csv", index=False)
    if estimate > float(args.max_api_spend_usd) + 1e-12:
        raise RuntimeError(
            f"Estimated uncached GPT strong-solve spend ${estimate:.4f} exceeds cap ${float(args.max_api_spend_usd):.4f}."
        )
    if args.dry_run:
        write_dry_run_memo(args.output_dir / "GPT_STRONG_MATH_ACTION_MEMO.md", args, estimate_table)
        print(f"Wrote dry-run cost estimate to {args.output_dir}")
        return

    api_key = resolve_key(load_env_values(args.env_file), ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")
    strong = collect_rows(queries, prompts, args, api_key=api_key)
    strong.to_csv(args.output_dir / "table_gpt_strong_math_action_outputs.csv", index=False)
    augmented = append_strong_rows(outputs, strong, lambda_cost=float(args.lambda_cost))
    augmented.to_parquet(args.output_dir / "model_outputs_with_gpt_strong_math_action.parquet", index=False)

    bounds, selected, choices = evaluate_augmented_action_pool(
        exp213,
        augmented,
        pd.read_csv(args.target_table),
        pd.read_csv(args.scores),
        lambda_cost=float(args.lambda_cost),
    )
    bounds.to_csv(args.output_dir / "table_gpt_strong_math_action_bounds.csv", index=False)
    selected.to_csv(args.output_dir / "table_gpt_strong_math_action_policy_selected.csv", index=False)
    choices.to_csv(args.output_dir / "table_gpt_strong_math_action_query_choices.csv", index=False)
    write_figure(args.output_dir, bounds, selected)
    write_memo(args.output_dir / "GPT_STRONG_MATH_ACTION_MEMO.md", args, estimate_table, strong, bounds, selected)
    print(f"Wrote GPT strong math-action results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def normalize_outputs(outputs: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    out = outputs.copy()
    out["query_id"] = out["query_id"].astype(str)
    out["model_id"] = out["model_id"].astype(str)
    for column in ["quality_score", "normalized_remote_cost", "cost_total_usd", "latency_s"]:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["utility"] = out["quality_score"] - float(lambda_cost) * out["normalized_remote_cost"]
    return out


def select_queries(outputs: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    benchmarks = {item.strip() for item in str(args.benchmarks).split(",") if item.strip()}
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    allowed_query_ids = query_id_allowlist(args)
    meta = outputs.drop_duplicates("query_id").copy()
    selected = meta[meta["split"].isin(splits) & meta["benchmark"].isin(benchmarks)].copy()
    if allowed_query_ids is not None:
        selected = selected[selected["query_id"].astype(str).isin(allowed_query_ids)].copy()
    selected = selected.sort_values(["split", "benchmark", "query_id"]).reset_index(drop=True)
    return selected[
        ["query_id", "query_text", "split", "benchmark", "domain", "metric", "gold_answer"]
    ].copy()


def query_id_allowlist(args: argparse.Namespace) -> set[str] | None:
    values = {item.strip() for item in str(args.query_ids or "").split(",") if item.strip()}
    if args.query_ids_file is not None:
        values.update(
            line.strip()
            for line in args.query_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return values or None


def prompt_for(row: pd.Series) -> str:
    query = str(row["query_text"]).strip()
    return (
        "Solve the task carefully. You may use private scratch work, but the visible response must be compact.\n"
        "Return JSON only, with the final exact answer as a string: {\"answer\":\"...\"}\n\n"
        f"Task:\n{query}"
    )


def cache_directory(args: argparse.Namespace) -> Path:
    return (
        Path(args.output_dir)
        / "raw_gpt_strong_math_action"
        / GPT
        / f"effort_{args.reasoning_effort}_max_{int(args.max_output_tokens)}"
    )


def cache_name(query_id: str, effort: str, max_output_tokens: int) -> str:
    digest = hashlib.sha1(f"{query_id}:{effort}:{max_output_tokens}".encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def estimate_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * int(max_output_tokens)
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def token_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def openai_payloads(prompt: str, max_output_tokens: int, reasoning_effort: str) -> list[dict[str, Any]]:
    base = {
        "model": GPT,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "text": {"verbosity": "low"},
    }
    payloads: list[dict[str, Any]] = []
    if reasoning_effort:
        payloads.append(base | {"reasoning": {"effort": reasoning_effort}})
    payloads.extend([base | {"reasoning": {"effort": "minimal"}}, dict(base)])
    return payloads


def call_openai(prompt: str, api_key: str, max_output_tokens: int, reasoning_effort: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error = ""
    last_response: dict[str, Any] | None = None
    for payload in openai_payloads(prompt, max_output_tokens, reasoning_effort):
        try:
            response = post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s=240.0)
            last_response = response
            if str(response.get("status", "")).lower() == "incomplete" and not extract_openai_text(response):
                last_error = f"incomplete_response:{response.get('incomplete_details')}"
                continue
            return response
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:600]
            last_error = f"HTTP {exc.code}: {body}"
            if exc.code == 400:
                continue
            raise
    if last_response is not None:
        return last_response
    raise RuntimeError(last_error or "OpenAI request failed.")


def parse_answer(text: object) -> str:
    raw = str(text or "").strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            if "answer" in payload:
                return str(payload["answer"]).strip()
        except json.JSONDecodeError:
            pass
    return raw


def collect_rows(queries: pd.DataFrame, prompts: list[str], args: argparse.Namespace, *, api_key: str) -> pd.DataFrame:
    cache_dir = cache_directory(args)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def one(row: pd.Series, prompt: str) -> dict[str, Any]:
        query_id = str(row["query_id"])
        raw_path = cache_dir / cache_name(query_id, str(args.reasoning_effort), int(args.max_output_tokens))
        cache_hit = raw_path.exists()
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_openai(prompt, api_key, int(args.max_output_tokens), str(args.reasoning_effort))
            except Exception as exc:  # noqa: BLE001
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:1000], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - started
            write_json_atomic(raw_path, payload)

        row_status = str(payload.get("_status", status))
        text = extract_openai_text(payload) if row_status == "success" else ""
        answer = parse_answer(text)
        metric = str(row.get("metric", "exact_final_answer") or "exact_final_answer")
        parsed, quality = score_output(answer, str(row["gold_answer"]), metric)
        if row_status != "success":
            quality = np.nan
        input_tokens, output_tokens = usage_from_openai(
            payload,
            max(1, len(prompt) // 4),
            int(args.max_output_tokens),
        ) if row_status == "success" else (0, 0)
        return {
            "query_id": query_id,
            "split": str(row["split"]),
            "benchmark": str(row["benchmark"]),
            "domain": str(row["domain"]),
            "metric": metric,
            "query_text": str(row["query_text"]),
            "gold_answer": str(row["gold_answer"]),
            "status": row_status,
            "error_type": str(payload.get("_error_type", error_type)),
            "raw_text": text,
            "parsed_answer": parsed,
            "quality_score": float(quality) if not pd.isna(quality) else np.nan,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost_total_usd": token_cost(int(input_tokens), int(output_tokens)),
            "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
            "cache_hit": bool(cache_hit),
            "raw_output_path": str(raw_path),
        }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as executor:
        futures = [executor.submit(one, row, prompt) for (_, row), prompt in zip(queries.iterrows(), prompts)]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"GPT strong math-action rows {index}/{len(futures)}")
    return pd.DataFrame(rows)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def append_strong_rows(outputs: pd.DataFrame, strong: pd.DataFrame, *, lambda_cost: float) -> pd.DataFrame:
    template_cols = list(outputs.columns)
    rows: list[dict[str, Any]] = []
    for row in strong.itertuples(index=False):
        rows.append(
            {
                "run_id": "controlled_broad100_gpt_strong_math_action",
                "query_id": row.query_id,
                "query_text": row.query_text,
                "benchmark": row.benchmark,
                "domain": row.domain,
                "model_id": STRONG_MODEL_ID,
                "provider": "openai",
                "is_local": False,
                "is_frontier": True,
                "is_probe": False,
                "prompt_template_version": "gpt_strong_math_action_v1",
                "prompt_hash": "",
                "input_tokens": int(row.input_tokens),
                "output_tokens": int(row.output_tokens),
                "max_output_tokens": 0,
                "start_time_unix": 0.0,
                "end_time_unix": 0.0,
                "latency_s": float(row.latency_s),
                "model_load_time_s": 0.0,
                "warmup_time_s": 0.0,
                "latency_excludes_load_warmup": True,
                "load_mode": "api",
                "status": row.status,
                "error_type": row.error_type,
                "raw_output_path": row.raw_output_path,
                "parsed_answer": row.parsed_answer,
                "gold_answer": row.gold_answer,
                "quality_score": float(row.quality_score) if not pd.isna(row.quality_score) else 0.0,
                "cost_input_usd": 0.0,
                "cost_output_usd": float(row.cost_total_usd),
                "cost_total_usd": float(row.cost_total_usd),
                "cache_hit": bool(row.cache_hit),
                "server_backend": "api",
                "server_config_json": json.dumps({"model": GPT, "strong_solver": True}, sort_keys=True),
                "hardware_id": "remote_api",
                "metric": row.metric,
                "rank_in_benchmark": 0,
                "split": row.split,
                "normalized_remote_cost": 0.0,
                "utility": 0.0,
                "tool_available": False,
            }
        )
    strong_rows = pd.DataFrame(rows)
    for column in template_cols:
        if column not in strong_rows.columns:
            strong_rows[column] = np.nan
    appended = pd.concat([outputs[template_cols], strong_rows[template_cols]], ignore_index=True)
    gpt_norm = max(
        float(appended[appended["model_id"].eq(GPT)].groupby("query_id")["cost_total_usd"].mean().mean()),
        1e-12,
    )
    appended["normalized_remote_cost"] = appended["cost_total_usd"].astype(float) / gpt_norm
    appended["quality_score"] = pd.to_numeric(appended["quality_score"], errors="coerce").fillna(0.0)
    appended["utility"] = appended["quality_score"].astype(float) - float(lambda_cost) * appended["normalized_remote_cost"].astype(float)
    return appended


def evaluate_augmented_action_pool(
    exp213: Any,
    outputs: pd.DataFrame,
    base_target: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_original = exp213.rebuild_target_pool(base_target, outputs, exp213.FULL_LOCAL_ACTIONS, exp213.LARGE_ACTIONS, lambda_cost)
    no_tool_original = exp213.rebuild_target_pool(base_target, outputs, exp213.NO_TOOL_LOCAL_ACTIONS, exp213.LARGE_ACTIONS, lambda_cost)
    strong_large_actions = tuple(dict.fromkeys((*exp213.LARGE_ACTIONS, STRONG_MODEL_ID)))
    no_tool_strong = exp213.rebuild_target_pool(base_target, outputs, exp213.NO_TOOL_LOCAL_ACTIONS, strong_large_actions, lambda_cost)
    full_strong = exp213.rebuild_target_pool(base_target, outputs, exp213.FULL_LOCAL_ACTIONS, strong_large_actions, lambda_cost)

    bound_rows: list[dict[str, Any]] = []
    detail_rows: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        full_ref = full_original[full_original["split"].eq(split)].copy()
        full_aug = full_strong[full_strong["split"].eq(split)].copy()
        no_orig = no_tool_original[no_tool_original["split"].eq(split)].copy()
        no_strong = no_tool_strong[no_tool_strong["split"].eq(split)].copy()
        for frame, reference, method, role in [
            (full_ref, full_ref, "full_original_oracle", "full_original_oracle"),
            (no_orig, full_ref, "no_tool_original_oracle_vs_full", "no_tool_original_vs_full"),
            (no_strong, full_ref, "no_tool_gpt_strong_oracle_vs_full", "no_tool_gpt_strong_vs_full"),
            (no_strong, no_strong, "no_tool_gpt_strong_oracle_self", "no_tool_gpt_strong_self"),
            (full_aug, full_aug, "full_gpt_strong_augmented_oracle", "full_augmented_oracle"),
        ]:
            choose = frame["large_utility"].to_numpy(dtype=float) >= frame["local_utility"].to_numpy(dtype=float)
            row, detail = exp213.evaluate_policy(
                frame,
                choose,
                oracle_reference=reference,
                split=split,
                method=method,
                family="gpt_strong_math_action_bound",
                action_pool_variant="gpt_strong_math_action",
                lambda_cost=lambda_cost,
            )
            row["bound_role"] = role
            bound_rows.append(row)
            detail_rows.append(detail.assign(bound_role=role))

    bound_table = exp213.add_target_gates(pd.DataFrame(bound_rows))
    policy_all, policy_choices = evaluate_threshold_policies(exp213, no_tool_strong, full_original, scores, lambda_cost=lambda_cost)
    selected = select_policy_rows(policy_all)
    choices = choices_for_selected(policy_choices, selected)
    return pd.concat([bound_table, policy_all], ignore_index=True), selected, choices


def evaluate_threshold_policies(
    exp213: Any,
    target: pd.DataFrame,
    oracle_ref: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    score_table = compact_scores(scores)
    rows: list[dict[str, Any]] = []
    details: dict[tuple[str, str], pd.DataFrame] = {}
    for classifier, score_col in [
        ("extratrees_d3_leaf8", "pred_verifiability_score_extratrees_d3_leaf8"),
        ("gb_depth2", "pred_verifiability_score_gb_depth2"),
        ("logreg_c0.3", "pred_verifiability_score_logreg_c0.3"),
    ]:
        if score_col not in score_table.columns:
            continue
        val_scores = score_table.loc[score_table["split"].eq("val"), score_col].dropna().to_numpy(dtype=float)
        if len(val_scores) == 0:
            continue
        thresholds = sorted(set(np.quantile(val_scores, np.linspace(0.05, 0.95, 19)).round(6).tolist()))
        for threshold in thresholds:
            method = f"{classifier}_thr{threshold:.4f}_gpt_strong_math_else_local"
            for split in ["val", "test"]:
                frame = target[target["split"].eq(split)].merge(
                    score_table[["query_id", score_col]],
                    on="query_id",
                    how="left",
                )
                reference = oracle_ref[oracle_ref["split"].eq(split)].copy()
                choose = frame[score_col].fillna(-np.inf).to_numpy(dtype=float) >= float(threshold)
                row, detail = exp213.evaluate_policy(
                    frame,
                    choose,
                    oracle_reference=reference,
                    split=split,
                    method=method,
                    family="gpt_strong_math_threshold_policy",
                    action_pool_variant="no_tool_gpt_strong_math_action",
                    lambda_cost=lambda_cost,
                )
                row.update({"classifier": classifier, "threshold": float(threshold)})
                rows.append(row)
                details[(method, split)] = detail
    table = exp213.add_target_gates(pd.DataFrame(rows)) if rows else pd.DataFrame()
    return table, details


def compact_scores(scores: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "query_id",
        "split",
        "benchmark",
        "pred_verifiability_score_extratrees_d3_leaf8",
        "pred_verifiability_score_gb_depth2",
        "pred_verifiability_score_logreg_c0.3",
    ]
    present = [col for col in cols if col in scores.columns]
    compact = scores[present].copy()
    return compact.groupby("query_id", as_index=False).first()


def select_policy_rows(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    val = table[table["split"].eq("val")].copy()
    feasible = val[val["frontier_call_rate"].le(0.40)].copy()
    pool = feasible if not feasible.empty else val
    best = pool.sort_values(
        ["meets_primary_numeric_target", "mean_utility", "mean_quality", "frontier_call_rate"],
        ascending=[False, False, False, True],
    ).head(1)
    rows = [best.assign(selection_rule="val_frontier_cap_best_utility")]
    method = str(best.iloc[0]["method"])
    test = table[table["split"].eq("test") & table["method"].eq(method)].copy()
    if not test.empty:
        rows.append(test.assign(selection_rule="val_frontier_cap_best_utility_test"))
    return pd.concat(rows, ignore_index=True)


def choices_for_selected(choices: dict[tuple[str, str], pd.DataFrame], selected: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in selected.itertuples(index=False):
        key = (str(row.method), str(row.split))
        if key in choices:
            frames.append(choices[key].assign(selection_rule=str(row.selection_rule)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_figure(output_dir: Path, bounds: pd.DataFrame, selected: pd.DataFrame) -> None:
    test_bounds = bounds[bounds["split"].eq("test")].copy()
    plot = test_bounds[
        test_bounds["method"].isin(
            [
                "full_original_oracle",
                "no_tool_original_oracle_vs_full",
                "no_tool_gpt_strong_oracle_vs_full",
                "full_gpt_strong_augmented_oracle",
            ]
        )
    ][["method", "mean_utility"]].copy()
    if not selected.empty:
        plot = pd.concat(
            [
                plot,
                selected[selected["split"].eq("test")][["method", "mean_utility"]],
            ],
            ignore_index=True,
        )
    plot = plot.sort_values("mean_utility", ascending=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.barh(plot["method"], plot["mean_utility"], color="#3f7f5f")
    ax.set_xlabel("Held-out Broad100 test mean utility")
    ax.set_title("GPT-5.5 Strong Math Action Feasibility")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_gpt_strong_math_action_utility.pdf")
    plt.close(fig)


def write_dry_run_memo(path: Path, args: argparse.Namespace, estimate: pd.DataFrame) -> None:
    lines = [
        "# Broad100 GPT Strong Math Action",
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
    selected: pd.DataFrame,
) -> None:
    test = bounds[bounds["split"].eq("test")].copy()
    by_benchmark = strong.groupby(["split", "benchmark"])["quality_score"].mean().reset_index()
    lines = [
        "# Broad100 GPT Strong Math Action",
        "",
        "This experiment adds a non-tool `gpt-5.5-strong-solve` action on exact-answer Broad100 math rows.",
        "It is meant to test whether a model action can substitute for deterministic-tool oracle wins without using Claude or benchmark-specific checkers.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python experiments/219_broad100_gpt_strong_math_action.py",
        "```",
        "",
        "## Cost Guard",
        "",
        "```csv",
        estimate.to_csv(index=False).strip(),
        "```",
        f"Actual recorded GPT strong-solve cost: `${float(strong['cost_total_usd'].sum()):.4f}`.",
        f"Rows: `{len(strong)}`; cache hits: `{int(strong['cache_hit'].sum())}`.",
        "",
        "## Strong Action Quality By Benchmark",
        "",
        "```csv",
        by_benchmark.to_csv(index=False).strip(),
        "```",
        "",
        "## Held-Out Test Bounds And Selected Policy",
        "",
        "```csv",
        test.to_csv(index=False).strip(),
        "```",
    ]
    if not selected.empty:
        lines.extend(
            [
                "",
                "## Validation-Selected Threshold Policy",
                "",
                "```csv",
                selected.to_csv(index=False).strip(),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If `no_tool_gpt_strong_oracle_vs_full` still misses the full oracle target, the gap is not fixed by this model action.",
            "- If the bound passes but the selected threshold policy misses, the action pool is feasible but observability is still the bottleneck.",
            "- This is a non-tool model-action test; it should not be described as a custom verifier.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
