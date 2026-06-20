from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import (
    extract_openai_text,
    load_env_values,
    post_json,
    resolve_key,
    usage_from_openai,
)


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
GEMINI = "gemini-3.5-flash"
GPT = "gpt-5.5"
ROUTE_MODEL = "gpt-5.5"
ACTION_LABELS = {
    "USE_QWEN06": "qwen3-0.6b-probe",
    "USE_QWEN4": "qwen3-4b-local",
    "USE_QWEN8": "qwen3-8b-local",
    "USE_QWEN14": "qwen3-14b-awq-local",
    "USE_GEMINI": GEMINI,
    "USE_GPT": GPT,
}
MODEL_TO_LABEL = {value: key for key, value in ACTION_LABELS.items()}
INPUT_PER_MTOK = 5.00
OUTPUT_PER_MTOK = 30.00


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use GPT-5.5 as a cached route-labeler for the expanded local pool.")
    parser.add_argument(
        "--query-table",
        default="results/controlled/expanded_local_pool_qwen14/query_table_expanded_local_pool.csv",
    )
    parser.add_argument("--output-dir", default="results/controlled/gpt_route_labeler_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--examples-per-label", type=int, default=3)
    parser.add_argument("--prompt-mode", choices=["few_shot", "solve_check"], default="few_shot")
    parser.add_argument("--max-output-tokens", type=int, default=72)
    parser.add_argument("--max-api-spend-usd", type=float, default=3.00)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def truncate(text: object, limit: int = 900) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def add_oracle_targets(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    local_utility = table[[f"{model_id}_quality" for model_id in LOCAL_MODELS]].copy()
    local_utility.columns = LOCAL_MODELS
    frontier_utility = table[[f"{GEMINI}_utility_selected_cost", f"{GPT}_utility_selected_cost"]].copy()
    frontier_utility.columns = [GEMINI, GPT]
    cost_utility = pd.concat([local_utility, frontier_utility], axis=1)
    quality = table[[f"{model_id}_quality" for model_id in LOCAL_MODELS + [GEMINI, GPT]]].copy()
    quality.columns = LOCAL_MODELS + [GEMINI, GPT]
    table["expanded_cost_oracle_model"] = cost_utility.idxmax(axis=1)
    table["expanded_cost_oracle_utility"] = cost_utility.max(axis=1)
    table["expanded_quality_oracle_model"] = quality.idxmax(axis=1)
    table["expanded_quality_oracle_value"] = quality.max(axis=1)
    return table


def build_few_shot_block(train: pd.DataFrame, examples_per_label: int) -> str:
    rows: list[str] = []
    train = train.copy()
    for model_id in LOCAL_MODELS + [GEMINI, GPT]:
        candidates = train[train["expanded_cost_oracle_model"].eq(model_id)].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["dataset", "query_len", "query_id"]).head(examples_per_label)
        for _, row in candidates.iterrows():
            rows.append(
                "\n".join(
                    [
                        f"Example action: {MODEL_TO_LABEL[model_id]}",
                        f"Dataset: {row['dataset']}",
                        f"Problem: {truncate(row['query_text'], 420)}",
                        f"Qwen0.6 answer: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 80)}",
                        f"Qwen4 answer: {truncate(row.get('qwen3-4b-local_answer', ''), 80)}",
                        f"Qwen8 answer: {truncate(row.get('qwen3-8b-local_answer', ''), 80)}",
                        f"Qwen14 answer: {truncate(row.get('qwen3-14b-awq-local_answer', ''), 80)}",
                    ]
                )
            )
    return "\n\n".join(rows)


def prompt_for(row: pd.Series, few_shots: str, prompt_mode: str) -> str:
    if prompt_mode == "solve_check":
        return f"""You are a RouteCode routing judge for exact-answer math.

Privately estimate the final answer or the likely difficulty. Then choose exactly one action:
- USE_QWEN06, USE_QWEN4, USE_QWEN8, USE_QWEN14 if that local answer looks correct, or if all paid solvers look unlikely to help.
- USE_GEMINI if the query is standard enough for Gemini 3.5 Flash and no local answer is clearly reliable.
- USE_GPT if this is a hard AIME/olympiad/combinatorics/geometry/algebra rescue case where GPT-5.5 is likely needed.

Important:
- For AIME-style hard problems with all local answers inconsistent, prefer USE_GPT unless the row looks hopeless.
- If two local models agree on a simple exact answer and it fits the problem, prefer the stronger agreeing local model.
- Do not choose a local answer just because it is short.

Return compact JSON only:
{{"action":"USE_QWEN06|USE_QWEN4|USE_QWEN8|USE_QWEN14|USE_GEMINI|USE_GPT","confidence":0.0}}

Dataset: {row['dataset']}
Problem: {truncate(row['query_text'], 1300)}
Local candidate answers:
- Qwen0.6: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 160)}
- Qwen4: {truncate(row.get('qwen3-4b-local_answer', ''), 160)}
- Qwen8: {truncate(row.get('qwen3-8b-local_answer', ''), 160)}
- Qwen14: {truncate(row.get('qwen3-14b-awq-local_answer', ''), 160)}
Local agreement max vote: {row.get('local_max_vote', '')}
"""
    return f"""You are a calibrated RouteCode route-labeler for exact-answer math model routing.

Choose exactly one action label:
- USE_QWEN06: use the Qwen3-0.6B local answer.
- USE_QWEN4: use the Qwen3-4B local answer.
- USE_QWEN8: use the Qwen3-8B local answer.
- USE_QWEN14: use the Qwen3-14B-AWQ local answer.
- USE_GEMINI: pay for Gemini 3.5 Flash as final solver.
- USE_GPT: pay for GPT-5.5 as final solver.

The labels are utility-aware: if the row looks hopeless, prefer a local action rather than paying a frontier solver. If a local answer is likely correct, prefer that local action. Use GPT only when the query likely needs GPT-specific rescue.

Calibrated examples from training:

{few_shots}

Classify the new query. Return compact JSON only:
{{"action":"USE_QWEN06|USE_QWEN4|USE_QWEN8|USE_QWEN14|USE_GEMINI|USE_GPT","confidence":0.0}}

Dataset: {row['dataset']}
Problem: {truncate(row['query_text'], 1100)}
Local answers:
- Qwen0.6: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 140)}
- Qwen4: {truncate(row.get('qwen3-4b-local_answer', ''), 140)}
- Qwen8: {truncate(row.get('qwen3-8b-local_answer', ''), 140)}
- Qwen14: {truncate(row.get('qwen3-14b-awq-local_answer', ''), 140)}
Local agreement max vote: {row.get('local_max_vote', '')}
"""


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * max_output_tokens
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def route_call_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * (INPUT_PER_MTOK / 1_000_000) + output_tokens * (OUTPUT_PER_MTOK / 1_000_000)


def call_openai_route(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 90.0) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": ROUTE_MODEL,
        "input": prompt,
        "max_output_tokens": int(max_output_tokens),
        "reasoning": {"effort": "none"},
        "text": {"verbosity": "low"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            payload.pop("text", None)
            return post_json("https://api.openai.com/v1/responses", payload, headers, timeout_s)
        raise


def parse_action(text: object) -> tuple[str, float]:
    raw = str(text or "").strip()
    action = ""
    confidence = np.nan
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            payload = json.loads(match.group(0))
            action = str(payload.get("action", "")).strip().upper()
            confidence = float(payload.get("confidence", np.nan))
        except Exception:
            pass
    if action not in ACTION_LABELS:
        upper = raw.upper()
        for candidate in ACTION_LABELS:
            if candidate in upper:
                action = candidate
                break
    if action not in ACTION_LABELS:
        action = "USE_GEMINI"
    return action, confidence


def cache_name(query_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)


def collect_routes(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    few_shots: str,
    prompt_mode: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_route_labels" / f"{ROUTE_MODEL}_{prompt_mode}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = [prompt_for(row, few_shots, prompt_mode) for _, row in frame.iterrows()]
    missing = [
        prompt
        for prompt, (_, row) in zip(prompts, frame.iterrows())
        if not (cache_dir / f"{cache_name(str(row['query_id']))}.json").exists()
    ]
    estimated = estimate_prompt_cost(missing, max_output_tokens)
    if estimated > max_api_spend_usd:
        raise RuntimeError(f"Estimated uncached route-label spend ${estimated:.4f} exceeds cap ${max_api_spend_usd:.4f}.")

    def one(item: tuple[pd.Series, str]) -> dict[str, object]:
        row, prompt = item
        query_id = str(row["query_id"])
        cache_path = cache_dir / f"{cache_name(query_id)}.json"
        cache_hit = cache_path.exists()
        status = "success"
        error_type = ""
        start = time.time()
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_openai_route(prompt, api_key, max_output_tokens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        row_status = str(payload.get("_status", status))
        text = extract_openai_text(payload) if row_status == "success" else ""
        action, confidence = parse_action(text)
        if row_status == "success":
            input_tokens, output_tokens = usage_from_openai(payload, max(1, len(prompt) // 4), max_output_tokens)
        else:
            input_tokens, output_tokens = 0, 0
        return {
            "query_id": query_id,
            "route_status": row_status,
            "route_error_type": str(payload.get("_error_type", error_type)),
            "route_text": text,
            "route_action_label": action,
            "route_model_action": ACTION_LABELS[action],
            "route_confidence": confidence,
            "route_input_tokens": int(input_tokens),
            "route_output_tokens": int(output_tokens),
            "route_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "route_cache_hit": cache_hit,
            "route_raw_path": str(cache_path),
            "route_cost": route_call_cost(int(input_tokens), int(output_tokens)),
        }

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, item) for item in zip([row for _, row in frame.iterrows()], prompts)]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def evaluate_actions(frame: pd.DataFrame, actions: pd.Series, lambda_cost: float) -> dict[str, object]:
    qualities: list[float] = []
    solver_costs: list[float] = []
    total_costs: list[float] = []
    solver_frontier: list[bool] = []
    gpt_solver: list[bool] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        route_cost = float(row.get("route_cost", 0.0) or 0.0)
        if action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            solver_cost = 0.0
            frontier = False
            gpt = False
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            solver_cost = float(row[f"{GEMINI}_cost"])
            frontier = True
            gpt = False
        elif action == GPT:
            quality = float(row[f"{GPT}_quality"])
            solver_cost = float(row[f"{GPT}_cost"])
            frontier = True
            gpt = True
        else:
            raise ValueError(action)
        qualities.append(quality)
        solver_costs.append(solver_cost)
        total_costs.append(solver_cost + route_cost)
        solver_frontier.append(frontier)
        gpt_solver.append(gpt)
    all_gpt_cost = max(float(frame[f"{GPT}_cost"].sum()), 1e-12)
    mean_quality = float(np.mean(qualities))
    solver_normalized_cost = float(np.sum(solver_costs) / all_gpt_cost)
    total_normalized_cost = float(np.sum(total_costs) / all_gpt_cost)
    solver_utility = float(mean_quality - lambda_cost * solver_normalized_cost)
    total_utility = float(mean_quality - lambda_cost * total_normalized_cost)
    oracle_utility = float(frame["expanded_cost_oracle_utility"].mean())
    return {
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "quality_gap_to_expanded_oracle": float(frame["expanded_quality_oracle"].mean() - mean_quality),
        "solver_utility": solver_utility,
        "total_utility_with_route_cost": total_utility,
        "solver_utility_ratio_to_expanded_cost_oracle": float(solver_utility / oracle_utility) if oracle_utility else np.nan,
        "total_utility_ratio_to_expanded_cost_oracle": float(total_utility / oracle_utility) if oracle_utility else np.nan,
        "normalized_solver_cost_vs_all_gpt": solver_normalized_cost,
        "normalized_total_cost_vs_all_gpt": total_normalized_cost,
        "route_cost_total_usd": float(frame["route_cost"].sum()),
        "solver_frontier_call_rate": float(np.mean(solver_frontier)),
        "total_remote_call_rate_including_router": 1.0,
        "gpt_solver_call_rate": float(np.mean(gpt_solver)),
        "action_counts": json.dumps({str(k): int(v) for k, v in actions.value_counts().to_dict().items()}, sort_keys=True),
    }


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
    table = add_oracle_targets(pd.read_csv(args.query_table))
    few_shots = build_few_shot_block(table[table["split"].eq("train")], args.examples_per_label)
    (output_dir / "route_label_few_shots.md").write_text(few_shots + "\n", encoding="utf-8")
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    eval_table = table[table["split"].astype(str).isin(splits)].copy() if splits else table.copy()
    if args.max_rows:
        eval_table = eval_table.head(int(args.max_rows)).copy()
    api_key = resolve_key(load_env_values(args.env_file), ["OPENAI_API_KEY", "openai_api_key"])
    if not api_key:
        raise RuntimeError("OpenAI API key not found.")
    routes = collect_routes(
        eval_table,
        output_dir,
        api_key=api_key,
        few_shots=few_shots,
        prompt_mode=args.prompt_mode,
        max_output_tokens=args.max_output_tokens,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    routes_path = output_dir / "table_gpt_route_labeler_outputs.csv"
    routes.to_csv(routes_path, index=False)
    merged = eval_table.merge(routes, on="query_id", how="left")
    merged_path = output_dir / "query_table_with_gpt_route_labels.csv"
    merged.to_csv(merged_path, index=False)
    rows = []
    for split, frame in merged.groupby("split", sort=False):
        row = evaluate_actions(frame, frame["route_model_action"], args.lambda_cost)
        row["method"] = "gpt_route_labeler_direct"
        rows.append(row)
    results = pd.DataFrame(rows)
    results_path = output_dir / "table_gpt_route_labeler_gate.csv"
    results.to_csv(results_path, index=False)
    label_counts = pd.crosstab(merged["split"], merged["route_action_label"]).reset_index()
    memo_path = output_dir / "GPT_ROUTE_LABELER_GATE_MEMO.md"
    memo = [
        "# GPT Route Labeler Gate Memo",
        "",
        f"Rows evaluated: `{len(merged)}` across splits `{args.splits}`.",
        f"Route model: `{ROUTE_MODEL}` with `reasoning.effort=none`, prompt mode `{args.prompt_mode}`, cached raw JSON, max output tokens `{args.max_output_tokens}`.",
        "Solver outputs are reused from cache. This script calls GPT only for route labeling when cache entries are missing.",
        "Accounting reports solver frontier-call rate separately from total remote call rate including the route labeler.",
        "",
        "## Results",
        "",
        markdown_table(results),
        "",
        "## Label Counts",
        "",
        markdown_table(label_counts),
        "",
        "## Files",
        "",
        f"- `{results_path}`",
        f"- `{routes_path}`",
        f"- `{merged_path}`",
        f"- `{output_dir / 'route_label_few_shots.md'}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {results_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
