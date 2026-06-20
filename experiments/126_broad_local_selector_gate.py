from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

LOCAL_CANDIDATES = [
    "qwen3-4b-local",
    "qwen3-8b-local",
    "qwen3-14b-awq-local",
    "qwen3-32b-awq-local",
    "deterministic_math_tool",
]
FRONTIER_CANDIDATES = ["gemini-3.5-flash", "gpt-5.5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use local Qwen32 as a broad100 candidate-answer selector/gate.")
    parser.add_argument("--outputs", type=Path, default=Path("results/controlled/live_broad100_stage0/model_outputs.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/controlled/broad100_local_selector_gate"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--request-timeout-s", type=float, default=90.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw_selector"
    raw_dir.mkdir(parents=True, exist_ok=True)

    package = load_broad_package()
    outputs = package.load_outputs(args.outputs, lambda_cost=args.lambda_cost)
    routes = collect_routes(outputs, args=args, raw_dir=raw_dir, package=package)
    grid = evaluate_grid(outputs, routes, lambda_cost=args.lambda_cost, package=package)
    selected = select_val_threshold(grid)

    routes.to_csv(args.output_dir / "table_broad_local_selector_routes.csv", index=False)
    grid.to_csv(args.output_dir / "table_broad_local_selector_gate.csv", index=False)
    selected.to_csv(args.output_dir / "table_broad_local_selector_selected.csv", index=False)
    write_memo(args.output_dir / "BROAD_LOCAL_SELECTOR_MEMO.md", args.outputs, grid, selected)
    print(f"Wrote broad local selector gate to {args.output_dir}")


def load_broad_package():
    path = Path("experiments/125_phase3_broad_target_method_package.py")
    spec = importlib.util.spec_from_file_location("broad_package", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect_routes(outputs: pd.DataFrame, *, args: argparse.Namespace, raw_dir: Path, package) -> pd.DataFrame:
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    queries = outputs[outputs["split"].astype(str).isin(splits)].drop_duplicates("query_id").copy()
    if args.max_rows is not None:
        queries = queries.head(int(args.max_rows)).copy()
    by_query = outputs.set_index(["query_id", "model_id"])
    rows: list[dict[str, Any]] = []
    total = len(queries)
    for index, (_, row) in enumerate(queries.iterrows(), start=1):
        query_id = str(row["query_id"])
        payload = call_selector(row, by_query, args=args, raw_dir=raw_dir, package=package)
        parsed = parse_selector_text(str(payload.get("_text", "")))
        chosen = normalize_choice(parsed.get("choice", ""))
        rows.append(
            {
                "query_id": query_id,
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "domain": str(row["domain"]),
                "selector_choice": chosen,
                "selector_model": chosen if chosen in LOCAL_CANDIDATES else "",
                "selector_confidence": float(parsed.get("confidence", 0.0)),
                "selector_need_frontier": bool(parsed.get("need_frontier", False)),
                "selector_status_code": int(payload.get("_status_code", 0) or 0),
                "selector_latency_s": float(payload.get("_latency_s", 0.0) or 0.0),
                "selector_input_tokens": int(payload.get("_input_tokens", 0) or 0),
                "selector_output_tokens": int(payload.get("_output_tokens", 0) or 0),
                "selector_cache_hit": bool(payload.get("_cache_hit", False)),
                "selector_raw_path": str(raw_dir / cache_name(query_id)),
            }
        )
        if index % 25 == 0 or index == total:
            print(f"selector rows {index}/{total}")
    return pd.DataFrame(rows)


def call_selector(row: pd.Series, by_query: pd.DataFrame, *, args: argparse.Namespace, raw_dir: Path, package) -> dict[str, Any]:
    query_id = str(row["query_id"])
    raw_path = raw_dir / cache_name(query_id)
    if raw_path.exists() and not args.force:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        payload["_cache_hit"] = True
        return payload
    prompt = build_prompt(row, by_query, package=package)
    started = time.time()
    response = requests.post(
        f"{args.base_url.rstrip('/')}/chat/completions",
        json={
            "model": args.served_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": int(args.max_output_tokens),
        },
        timeout=float(args.request_timeout_s),
    )
    latency_s = time.time() - started
    payload: dict[str, Any] = {
        "_status_code": response.status_code,
        "_latency_s": latency_s,
        "_query_id": query_id,
    }
    if response.status_code == 200:
        body = response.json()
        payload["_response"] = body
        payload["_text"] = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage", {})
        if isinstance(usage, dict):
            payload["_input_tokens"] = int(usage.get("prompt_tokens", 0) or 0)
            payload["_output_tokens"] = int(usage.get("completion_tokens", 0) or 0)
    else:
        payload["_text"] = response.text[:2000]
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["_cache_hit"] = False
    return payload


def build_prompt(row: pd.Series, by_query: pd.DataFrame, *, package) -> str:
    query_id = str(row["query_id"])
    candidate_lines = []
    for model_id in LOCAL_CANDIDATES:
        try:
            model_row = by_query.loc[(query_id, model_id)]
        except KeyError:
            continue
        if model_id == "deterministic_math_tool" and not package.deterministic_tool_choice(by_query, query_id):
            continue
        answer = clean_answer(model_row.get("parsed_answer", ""))
        status = str(model_row.get("status", ""))
        candidate_lines.append(f"- {model_id}: status={status}; answer={answer[:220] if answer else '[empty]'}")
    if not candidate_lines:
        candidate_lines.append("- no valid local candidate answers")
    query = compact_text(str(row["query_text"]), max_chars=1800)
    return (
        "You are a local route verifier. You must choose which cached local candidate answer is most likely correct.\n"
        "You do not see the gold answer. Prefer a local candidate only when its answer is credible for the task.\n"
        "If all local candidates look unreliable, choose FRONTIER so a remote model can solve it.\n"
        "Return only JSON with keys choice, confidence, need_frontier.\n"
        "choice must be one of: "
        + ", ".join([*LOCAL_CANDIDATES, "FRONTIER"])
        + ". confidence is 0 to 1.\n\n"
        f"Benchmark: {row['benchmark']}\n"
        f"Metric: {row['metric']}\n"
        f"Question:\n{query}\n\n"
        "Local candidate final answers:\n"
        + "\n".join(candidate_lines)
        + '\n\nExample: {"choice":"qwen3-4b-local","confidence":0.72,"need_frontier":false}\n/no_think'
    )


def clean_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def compact_text(text: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.70)].rstrip()
    tail = text[-int(max_chars * 0.25) :].lstrip()
    return f"{head}\n...[truncated]...\n{tail}"


def cache_name(query_id: str) -> str:
    digest = hashlib.sha1(query_id.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def parse_selector_text(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    choice = normalize_choice(parsed.get("choice", ""))
    if choice == "PARSE_FAIL":
        loose = re.search(
            r"\b(qwen3-4b-local|qwen3-8b-local|qwen3-14b-awq-local|qwen3-32b-awq-local|deterministic_math_tool|frontier)\b",
            clean,
            flags=re.I,
        )
        choice = normalize_choice(loose.group(1) if loose else "")
    try:
        confidence = float(parsed.get("confidence", np.nan))
    except (TypeError, ValueError):
        conf_match = re.search(r'"?confidence"?\s*[:=]\s*([0-9.]+)', clean, flags=re.I)
        confidence = float(conf_match.group(1)) if conf_match else np.nan
    if np.isnan(confidence):
        confidence = 0.0
    need_raw = parsed.get("need_frontier", choice == "FRONTIER")
    if isinstance(need_raw, str):
        need_frontier = need_raw.strip().lower() in {"true", "yes", "1"}
    else:
        need_frontier = bool(need_raw)
    return {
        "choice": choice,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "need_frontier": bool(need_frontier or choice == "FRONTIER"),
    }


def normalize_choice(value: object) -> str:
    choice = str(value).strip()
    lower = choice.lower()
    mapping = {model.lower(): model for model in LOCAL_CANDIDATES}
    if lower in mapping:
        return mapping[lower]
    if lower in {"frontier", "remote", "api", "none"}:
        return "FRONTIER"
    return "PARSE_FAIL"


def evaluate_grid(pd_outputs: pd.DataFrame, routes: pd.DataFrame, *, lambda_cost: float, package) -> pd.DataFrame:
    outputs = pd_outputs.copy()
    trainval = outputs[outputs["split"].isin(["train", "val"])]
    fallback = frontier_fallback_by_benchmark(trainval, lambda_cost=lambda_cost)
    rows: list[dict[str, Any]] = []
    for threshold in [0.0, 0.25, 0.4, 0.55, 0.7, 0.85]:
        for use_need_frontier in [False, True]:
            for split in ["val", "test"]:
                split_routes = routes[routes["split"].eq(split)]
                selected = selector_policy(split_routes, fallback, threshold=threshold, use_need_frontier=use_need_frontier)
                selected_rows = package.selected_to_rows(outputs, selected, split=split)
                if selected_rows.empty:
                    continue
                test_like = outputs[outputs["split"].eq(split)]
                cost_oracle = test_like.loc[test_like.groupby("query_id")["utility"].idxmax()]
                quality_oracle = test_like.loc[test_like.groupby("query_id")["quality_score"].idxmax()]
                row = package.evaluation_row(
                    f"qwen32_local_selector_t{threshold:g}_need{int(use_need_frontier)}",
                    selected_rows,
                    cost_oracle,
                    quality_oracle,
                    lambda_cost=lambda_cost,
                )
                row["selector_confidence_threshold"] = threshold
                row["selector_need_frontier_enabled"] = bool(use_need_frontier)
                route_lookup = split_routes.set_index("query_id")
                row["selector_probe_rate"] = 1.0
                row["selector_latency_mean_s"] = float(route_lookup["selector_latency_s"].mean())
                row["selector_latency_p95_s"] = float(route_lookup["selector_latency_s"].quantile(0.95))
                row["selector_input_tokens_mean"] = float(route_lookup["selector_input_tokens"].mean())
                row["selector_output_tokens_mean"] = float(route_lookup["selector_output_tokens"].mean())
                row["fallback_frontier_json"] = json.dumps(fallback, sort_keys=True)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])


def frontier_fallback_by_benchmark(trainval: pd.DataFrame, *, lambda_cost: float) -> dict[str, str]:
    del lambda_cost
    frontier = trainval[trainval["model_id"].isin(FRONTIER_CANDIDATES)].copy()
    table = (
        frontier.groupby(["benchmark", "model_id"], as_index=False)
        .agg(mean_utility=("utility", "mean"), mean_quality=("quality_score", "mean"), mean_cost=("normalized_remote_cost", "mean"))
        .sort_values(["benchmark", "mean_utility", "mean_quality", "mean_cost"], ascending=[True, False, False, True])
        .drop_duplicates("benchmark")
    )
    return {str(row["benchmark"]): str(row["model_id"]) for _, row in table.iterrows()}


def selector_policy(
    routes: pd.DataFrame,
    fallback: dict[str, str],
    *,
    threshold: float,
    use_need_frontier: bool,
) -> pd.Series:
    selected: dict[str, str] = {}
    for _, row in routes.iterrows():
        choice = str(row["selector_choice"])
        confidence = float(row["selector_confidence"])
        use_frontier = choice not in LOCAL_CANDIDATES or confidence < threshold
        if use_need_frontier:
            use_frontier = use_frontier or bool(row["selector_need_frontier"])
        selected[str(row["query_id"])] = fallback.get(str(row["benchmark"]), "gpt-5.5") if use_frontier else choice
    return pd.Series(selected)


def select_val_threshold(grid: pd.DataFrame) -> pd.DataFrame:
    val = grid[grid["split"].eq("val")].copy()
    if val.empty:
        return pd.DataFrame()
    picked = val.sort_values(["mean_utility", "mean_quality", "frontier_call_rate"], ascending=[False, False, True]).head(1)
    threshold = float(picked.iloc[0]["selector_confidence_threshold"])
    need = bool(picked.iloc[0]["selector_need_frontier_enabled"])
    return grid[
        grid["selector_confidence_threshold"].eq(threshold)
        & grid["selector_need_frontier_enabled"].eq(need)
    ].copy()


def write_memo(path: Path, outputs_path: Path, grid: pd.DataFrame, selected: pd.DataFrame) -> None:
    lines = [
        "# Broad100 Local Selector Gate",
        "",
        f"Source outputs: `{outputs_path}`",
        "",
        "This diagnostic uses local Qwen3-32B-AWQ as a candidate-answer selector over local model answers.",
        "It does not send prompts to GPT, Gemini, or Claude. Frontier fallback rows are evaluated from cached solver outputs.",
        "",
    ]
    if not selected.empty:
        lines.extend(
            [
                "## Validation-Selected Threshold",
                "",
                "```csv",
                selected.to_csv(index=False).strip(),
                "```",
                "",
            ]
        )
    test = grid[grid["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(10)
    if not test.empty:
        lines.extend(["## Top Test Diagnostics", "", "```csv", test.to_csv(index=False).strip(), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
