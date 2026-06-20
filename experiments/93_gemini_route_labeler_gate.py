from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import load_env_values, resolve_key


LOCAL_MODEL = "qwen3-8b-local"
GEMINI_MODEL = "gemini-3.5-flash"
GPT_MODEL = "gpt-5.5"
ROUTE_MODEL = "gemini-3.5-flash"
ROUTE_LABELS = {
    "local": "LOCAL_SAVE",
    "gemini": "GEMINI_SOLVE",
    "gpt": "GPT_RESCUE",
    "gemini_then_gpt_guarded": "GPT_RESCUE",
}
LABEL_TO_ACTION = {
    "LOCAL_SAVE": "local",
    "GEMINI_SOLVE": "gemini",
    "GPT_RESCUE": "gpt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a cheap Gemini route-label classifier gate.")
    parser.add_argument("--query-table", default="results/controlled/gemini_metadata_gate/query_table_with_gemini_metadata.csv")
    parser.add_argument("--output-dir", default="results/controlled/gemini_route_labeler_gate")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--examples-per-label", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=48)
    parser.add_argument("--max-api-spend-usd", type=float, default=2.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    return parser.parse_args()


def selected_cost_action(row: pd.Series) -> str:
    utilities = {
        "local": float(row[f"{LOCAL_MODEL}_utility_selected_cost"]),
        "gemini": float(row[f"{GEMINI_MODEL}_utility_selected_cost"]),
        "gpt": float(row[f"{GPT_MODEL}_utility_selected_cost"]),
    }
    return max(utilities, key=utilities.get)


def truncate(text: object, limit: int = 520) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 3] + "..."


def build_few_shot_block(train: pd.DataFrame, examples_per_label: int) -> str:
    rows: list[str] = []
    train = train.copy()
    train["_action"] = train.apply(selected_cost_action, axis=1)
    for action in ["local", "gemini", "gpt"]:
        label = ROUTE_LABELS[action]
        candidates = train[train["_action"].eq(action)].copy()
        if candidates.empty:
            continue
        candidates = candidates.sort_values(["dataset", "query_len", "query_id"]).head(examples_per_label)
        for _, row in candidates.iterrows():
            rows.append(
                "\n".join(
                    [
                        f"Example route: {label}",
                        f"Dataset: {row['dataset']}",
                        f"Problem: {truncate(row['query_text'])}",
                        f"Qwen8 answer: {truncate(row[f'{LOCAL_MODEL}_answer'], 120)}",
                    ]
                )
            )
    return "\n\n".join(rows)


def prompt_for(row: pd.Series, few_shots: str) -> str:
    return f"""You are a calibrated RouteCode route-label classifier for exact-answer math routing.

Choose exactly one route label:
- LOCAL_SAVE: use local/no-remote. This includes easy local-correct rows and hopeless rows where paying remote models is not utility-optimal.
- GEMINI_SOLVE: use Gemini 3.5 Flash as the final solver.
- GPT_RESCUE: use GPT-5.5 as the final solver.

The labels are utility-aware, not topic labels. Use the calibrated examples below.

{few_shots}

Now classify the new query. Return compact JSON only: {{"route":"LOCAL_SAVE|GEMINI_SOLVE|GPT_RESCUE","confidence":0.0}}

Dataset: {row['dataset']}
Problem: {truncate(row['query_text'], 900)}
Qwen8 answer: {truncate(row[f'{LOCAL_MODEL}_answer'], 160)}
Qwen4 answer: {truncate(row.get('qwen3-4b-local_answer', ''), 160)}
Qwen0.6B answer: {truncate(row.get('qwen3-0.6b-probe_answer', ''), 160)}
"""


def extract_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(str(part.get("text", "")) for part in parts)


def parse_route(text: object) -> tuple[str, float]:
    raw = str(text or "").strip()
    route = ""
    confidence = np.nan
    json_match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            route = str(payload.get("route", "")).strip().upper()
            confidence = float(payload.get("confidence", np.nan))
        except Exception:
            pass
    if route not in LABEL_TO_ACTION:
        upper = raw.upper()
        for candidate in LABEL_TO_ACTION:
            if candidate in upper:
                route = candidate
                break
    if route not in LABEL_TO_ACTION:
        route = "GEMINI_SOLVE"
    return route, confidence


def route_call_cost(row: dict[str, object]) -> float:
    input_price = 1.50 / 1_000_000
    output_price = 9.00 / 1_000_000
    return float(row.get("route_input_tokens", 0) or 0) * input_price + float(row.get("route_output_tokens", 0) or 0) * output_price


def estimate_prompt_cost(prompts: list[str], max_output_tokens: int) -> float:
    # Conservative cheap estimate: 1 token ~= 4 chars.
    input_tokens = sum(max(1, len(prompt) // 4) for prompt in prompts)
    output_tokens = len(prompts) * max_output_tokens
    return input_tokens * (1.50 / 1_000_000) + output_tokens * (9.00 / 1_000_000)


def call_gemini_route(prompt: str, api_key: str, max_output_tokens: int, timeout_s: float = 60.0) -> dict[str, Any]:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": int(max_output_tokens),
            "temperature": 0,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{ROUTE_MODEL}:generateContent",
        data=json.dumps(payload).encode("utf-8"),
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def collect_routes(
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    api_key: str,
    few_shots: str,
    max_output_tokens: int,
    max_api_spend_usd: float,
    concurrency: int,
) -> pd.DataFrame:
    cache_dir = output_dir / "raw_route_labels" / ROUTE_MODEL
    cache_dir.mkdir(parents=True, exist_ok=True)
    prompts = [prompt_for(row, few_shots) for _, row in frame.iterrows()]
    missing_prompts = [
        prompt
        for prompt, (_, row) in zip(prompts, frame.iterrows())
        if not (cache_dir / f"{str(row['query_id']).replace(':', '_')}.json").exists()
    ]
    estimated_cost = estimate_prompt_cost(missing_prompts, max_output_tokens)
    if estimated_cost > max_api_spend_usd:
        raise RuntimeError(
            f"Estimated uncached route-label spend ${estimated_cost:.4f} exceeds cap ${max_api_spend_usd:.4f}."
        )

    def one(item: tuple[int, pd.Series, str]) -> dict[str, object]:
        _, row, prompt = item
        query_id = str(row["query_id"])
        cache_path = cache_dir / f"{query_id.replace(':', '_')}.json"
        cache_hit = cache_path.exists()
        start = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            try:
                payload = call_gemini_route(prompt, api_key, max_output_tokens)
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error": str(exc)[:500], "error_type": error_type}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        text = extract_text(payload) if payload.get("_status", status) == "success" else ""
        route, confidence = parse_route(text)
        usage = payload.get("usageMetadata", {}) if isinstance(payload, dict) else {}
        out = {
            "query_id": query_id,
            "route_status": str(payload.get("_status", status)),
            "route_error_type": str(payload.get("_error_type", error_type)),
            "route_text": text,
            "route_label": route,
            "route_action": LABEL_TO_ACTION[route],
            "route_confidence": confidence,
            "route_input_tokens": int(usage.get("promptTokenCount", 0) or 0),
            "route_output_tokens": int(usage.get("candidatesTokenCount", 0) or 0),
            "route_thoughts_tokens": int(usage.get("thoughtsTokenCount", 0) or 0),
            "route_latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
            "route_cache_hit": cache_hit,
            "route_raw_path": str(cache_path),
        }
        out["route_cost"] = route_call_cost(out)
        return out

    rows: list[dict[str, object]] = []
    items = [(idx, row, prompt) for (idx, row), prompt in zip(frame.iterrows(), prompts)]
    with ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as executor:
        futures = [executor.submit(one, item) for item in items]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def evaluate_actions(table: pd.DataFrame, actions: pd.Series, lambda_cost: float, cost_norm: float, split: str) -> dict[str, object]:
    qualities: list[float] = []
    costs: list[float] = []
    frontier_solver_calls: list[bool] = []
    gpt_calls: list[bool] = []
    local_final: list[bool] = []
    for idx, row in table.iterrows():
        action = str(actions.loc[idx])
        route_cost = float(row.get("route_cost", 0.0) or 0.0)
        if action == "local":
            quality = float(row[f"{LOCAL_MODEL}_quality"])
            cost = route_cost
            solver_frontier = False
            gpt = False
            local = True
        elif action == "gemini":
            quality = float(row[f"{GEMINI_MODEL}_quality"])
            cost = route_cost + float(row[f"{GEMINI_MODEL}_cost"])
            solver_frontier = True
            gpt = False
            local = False
        elif action == "gpt":
            quality = float(row[f"{GPT_MODEL}_quality"])
            cost = route_cost + float(row[f"{GPT_MODEL}_cost"])
            solver_frontier = True
            gpt = True
            local = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        frontier_solver_calls.append(solver_frontier)
        gpt_calls.append(gpt)
        local_final.append(local)
    quality_oracle = table[[f"{LOCAL_MODEL}_quality", f"{GEMINI_MODEL}_quality", f"{GPT_MODEL}_quality"]].max(axis=1)
    cost_oracle = table[
        [
            f"{LOCAL_MODEL}_utility_selected_cost",
            f"{GEMINI_MODEL}_utility_selected_cost",
            f"{GPT_MODEL}_utility_selected_cost",
        ]
    ].max(axis=1)
    mean_quality = float(np.mean(qualities))
    mean_utility = float(mean_quality - lambda_cost * (np.mean(costs) / cost_norm))
    return {
        "method": "gemini_route_labeler_gate",
        "split": split,
        "n_queries": int(len(table)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_oracle": float(quality_oracle.mean() - mean_quality),
        "utility_ratio_to_cost_oracle": float(mean_utility / cost_oracle.mean())
        if abs(float(cost_oracle.mean())) > 1e-12
        else np.nan,
        "normalized_remote_cost_vs_all_gpt": float(np.sum(costs) / table[f"{GPT_MODEL}_cost"].astype(float).sum()),
        "route_call_rate": 1.0,
        "solver_frontier_call_rate": float(np.mean(frontier_solver_calls)),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "local_final_rate": float(np.mean(local_final)),
        "route_cost_total_usd": float(table["route_cost"].sum()),
        "remote_cost_total_usd": float(np.sum(costs)),
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
    table = pd.read_csv(args.query_table)
    few_shots = build_few_shot_block(table[table["split"].eq("train")].copy(), args.examples_per_label)
    (output_dir / "route_label_few_shots.md").write_text(few_shots + "\n", encoding="utf-8")
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    eval_table = table[table["split"].astype(str).isin(splits)].copy() if splits else table.copy()
    if args.max_rows:
        eval_table = eval_table.head(int(args.max_rows)).copy()
    api_key = resolve_key(load_env_values(args.env_file), ["GEMINI_API_KEY", "GOOGLE_API_KEY", "gemini_api_key", "google_api_key"])
    if not api_key:
        raise RuntimeError("Gemini API key not found.")
    routes = collect_routes(
        eval_table,
        output_dir,
        api_key=api_key,
        few_shots=few_shots,
        max_output_tokens=args.max_output_tokens,
        max_api_spend_usd=args.max_api_spend_usd,
        concurrency=args.concurrency,
    )
    routes_path = output_dir / "table_gemini_route_labeler_outputs.csv"
    routes.to_csv(routes_path, index=False)
    merged = eval_table.merge(routes, on="query_id", how="left")
    merged_path = output_dir / "query_table_with_gemini_route_labels.csv"
    merged.to_csv(merged_path, index=False)
    cost_norm = max(float(table[f"{GPT_MODEL}_cost"].mean()), 1e-12)
    rows = []
    for split, frame in merged.groupby("split", sort=False):
        rows.append(evaluate_actions(frame, frame["route_action"], args.lambda_cost, cost_norm, str(split)))
    results = pd.DataFrame(rows)
    results_path = output_dir / "table_gemini_route_labeler_gate.csv"
    results.to_csv(results_path, index=False)
    memo_path = output_dir / "GEMINI_ROUTE_LABELER_GATE_MEMO.md"
    memo = [
        "# Gemini Route Labeler Gate Memo",
        "",
        f"Rows evaluated: `{len(merged)}`. Route calls use `{ROUTE_MODEL}` with thinking disabled and are cached.",
        f"Route-label cost total: `${merged['route_cost'].sum():.4f}`.",
        "Solver outputs are reused from cache; this script does not call GPT.",
        "",
        "Important accounting note: `route_call_rate` is 1.0 for evaluated rows because the cheap Gemini labeler is itself a remote call. `solver_frontier_call_rate` counts only the final solver selected after routing.",
        "",
        "## Results",
        "",
        markdown_table(results),
        "",
        "## Label Counts",
        "",
        markdown_table(pd.crosstab(merged["split"], merged["route_label"]).reset_index()),
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
