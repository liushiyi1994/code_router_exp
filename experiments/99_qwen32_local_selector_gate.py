from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


LOCAL_MODELS = ["qwen3-0.6b-probe", "qwen3-4b-local", "qwen3-8b-local", "qwen3-14b-awq-local"]
FRONTIER_MODELS = ["gemini-3.5-flash", "gpt-5.5"]
GEMINI = "gemini-3.5-flash"
GPT = "gpt-5.5"
CHOICES = dict(zip(["A", "B", "C", "D"], LOCAL_MODELS, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use local Qwen32-AWQ as a candidate-answer selector/gate.")
    parser.add_argument("--query-table", default="results/controlled/expanded_local_pool_qwen14/query_table_expanded_local_pool.csv")
    parser.add_argument("--output-dir", default="results/controlled/qwen32_local_selector_gate")
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits to label.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--request-timeout-s", type=float, default=90.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--quality-gap-target", type=float, default=0.03)
    parser.add_argument("--frontier-rate-target", type=float, default=0.40)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def answer_value(row: pd.Series, model_id: str) -> str:
    norm_col = f"{model_id}_answer_norm"
    raw_col = f"{model_id}_answer"
    value = row.get(norm_col, row.get(raw_col, ""))
    return "" if pd.isna(value) else str(value)


def build_prompt(row: pd.Series) -> str:
    candidates = []
    for letter, model_id in CHOICES.items():
        answer = answer_value(row, model_id)
        candidates.append(f"{letter}. {answer[:180] if answer else '[empty]'}")
    query = str(row["query_text"])
    return (
        "You are a local verifier for exact-answer math routing.\n"
        "Given a problem and four candidate final answers from local models, choose the candidate most likely to be exactly correct.\n"
        "If the candidates are all unreliable or the problem likely needs a frontier solver, choose NONE.\n"
        "Return only compact JSON with keys choice, confidence, need_frontier.\n"
        "choice must be A, B, C, D, or NONE. confidence must be a number from 0 to 1.\n\n"
        f"Dataset: {row['dataset']}\n"
        f"Problem:\n{query[:1700]}\n\n"
        "Candidate final answers:\n"
        + "\n".join(candidates)
        + '\n\nExample output: {"choice":"B","confidence":0.73,"need_frontier":false}\n/no_think'
    )


def cache_name(query_id: str) -> str:
    digest = hashlib.sha1(query_id.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", query_id)[:80]
    return f"{safe}_{digest}.json"


def call_qwen32(
    row: pd.Series,
    *,
    base_url: str,
    model: str,
    raw_dir: Path,
    max_output_tokens: int,
    timeout_s: float,
    force: bool,
) -> dict[str, Any]:
    raw_path = raw_dir / cache_name(str(row["query_id"]))
    if raw_path.exists() and not force:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
        payload["_cache_hit"] = True
        return payload

    prompt = build_prompt(row)
    started = time.time()
    response = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_output_tokens,
        },
        timeout=timeout_s,
    )
    latency_s = time.time() - started
    payload: dict[str, Any] = {
        "_status_code": response.status_code,
        "_latency_s": latency_s,
        "_query_id": str(row["query_id"]),
    }
    if response.status_code == 200:
        body = response.json()
        payload["_response"] = body
        payload["_text"] = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = body.get("usage", {})
        if isinstance(usage, dict):
            payload["_input_tokens"] = usage.get("prompt_tokens")
            payload["_output_tokens"] = usage.get("completion_tokens")
    else:
        payload["_text"] = response.text[:2000]
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    payload["_cache_hit"] = False
    return payload


def parse_selector_text(text: str) -> dict[str, Any]:
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    parsed: dict[str, Any] = {}
    match = re.search(r"\{.*?\}", clean, flags=re.S)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = {}
    choice = str(parsed.get("choice", "")).upper().strip()
    if choice not in {*CHOICES.keys(), "NONE"}:
        choice_match = re.search(r'"?choice"?\s*[:=]\s*"?(A|B|C|D|NONE)"?', clean, flags=re.I)
        if choice_match:
            choice = choice_match.group(1).upper()
        else:
            loose = re.search(r"\b(A|B|C|D|NONE)\b", clean, flags=re.I)
            choice = loose.group(1).upper() if loose else "PARSE_FAIL"
    try:
        confidence = float(parsed.get("confidence", np.nan))
    except (TypeError, ValueError):
        conf_match = re.search(r'"?confidence"?\s*[:=]\s*([0-9.]+)', clean, flags=re.I)
        confidence = float(conf_match.group(1)) if conf_match else np.nan
    if np.isnan(confidence):
        confidence = 0.0
    confidence = float(np.clip(confidence, 0.0, 1.0))
    need_raw = parsed.get("need_frontier", False)
    if isinstance(need_raw, str):
        need_frontier = need_raw.strip().lower() in {"true", "yes", "1"}
    else:
        need_frontier = bool(need_raw)
    if "need_frontier" not in parsed:
        need_match = re.search(r'"?need_frontier"?\s*[:=]\s*(true|false|yes|no|0|1)', clean, flags=re.I)
        if need_match:
            need_frontier = need_match.group(1).lower() in {"true", "yes", "1"}
    return {"choice": choice, "confidence": confidence, "need_frontier": need_frontier}


def frontier_action_table(train: pd.DataFrame, lambda_cost: float, cost_norm: float, strategy: str) -> dict[str, str]:
    actions: dict[str, str] = {}
    for dataset, frame in train.groupby("dataset"):
        if strategy == "gemini":
            actions[str(dataset)] = GEMINI
            continue
        if strategy == "gpt":
            actions[str(dataset)] = GPT
            continue
        if strategy == "train_quality":
            actions[str(dataset)] = (
                GPT
                if float(frame[f"{GPT}_quality"].astype(float).mean())
                >= float(frame[f"{GEMINI}_quality"].astype(float).mean())
                else GEMINI
            )
            continue
        if strategy != "train_utility":
            raise ValueError(strategy)
        gemini_utility = frame[f"{GEMINI}_quality"].astype(float) - lambda_cost * (
            frame[f"{GEMINI}_cost"].astype(float) / cost_norm
        )
        gpt_utility = frame[f"{GPT}_quality"].astype(float) - lambda_cost * (
            frame[f"{GPT}_cost"].astype(float) / cost_norm
        )
        actions[str(dataset)] = GPT if float(gpt_utility.mean()) >= float(gemini_utility.mean()) else GEMINI
    return actions


def local_fallback_table(train: pd.DataFrame) -> dict[str, str]:
    actions: dict[str, str] = {}
    for dataset, frame in train.groupby("dataset"):
        qualities = {model_id: float(frame[f"{model_id}_quality"].mean()) for model_id in LOCAL_MODELS}
        actions[str(dataset)] = max(qualities, key=qualities.get)
    return actions


def selector_rows(table: pd.DataFrame, args: argparse.Namespace, raw_dir: Path) -> pd.DataFrame:
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    frame = table[table["split"].astype(str).isin(splits)].copy()
    if args.max_rows is not None:
        frame = frame.head(int(args.max_rows)).copy()
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(frame.iterrows(), start=1):
        payload = call_qwen32(
            row,
            base_url=args.base_url,
            model=args.served_model_name,
            raw_dir=raw_dir,
            max_output_tokens=args.max_output_tokens,
            timeout_s=args.request_timeout_s,
            force=args.force,
        )
        parsed = parse_selector_text(str(payload.get("_text", "")))
        chosen_model = CHOICES.get(parsed["choice"], "")
        chosen_quality = float(row[f"{chosen_model}_quality"]) if chosen_model else 0.0
        rows.append(
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "dataset": str(row["dataset"]),
                "choice": parsed["choice"],
                "chosen_local_model": chosen_model or parsed["choice"],
                "selector_confidence": parsed["confidence"],
                "selector_need_frontier": parsed["need_frontier"],
                "selector_local_quality": chosen_quality,
                "selector_latency_s": float(payload.get("_latency_s", 0.0) or 0.0),
                "selector_input_tokens": payload.get("_input_tokens"),
                "selector_output_tokens": payload.get("_output_tokens"),
                "selector_cache_hit": bool(payload.get("_cache_hit", False)),
                "selector_text": str(payload.get("_text", ""))[:500],
            }
        )
        if i % 20 == 0:
            print(f"Labeled {i}/{len(frame)} rows")
    return pd.DataFrame(rows)


def add_oracle_utility(table: pd.DataFrame, lambda_cost: float, cost_norm: float) -> pd.DataFrame:
    table = table.copy()
    utility_cols = []
    for model_id in LOCAL_MODELS:
        col = f"{model_id}_utility_for_qwen32_gate"
        table[col] = table[f"{model_id}_quality"].astype(float)
        utility_cols.append(col)
    for model_id in FRONTIER_MODELS:
        col = f"{model_id}_utility_for_qwen32_gate"
        table[col] = table[f"{model_id}_quality"].astype(float) - lambda_cost * (
            table[f"{model_id}_cost"].astype(float) / cost_norm
        )
        utility_cols.append(col)
    table["qwen32_gate_cost_oracle_utility"] = table[utility_cols].max(axis=1)
    table["qwen32_gate_quality_oracle"] = table[[f"{model_id}_quality" for model_id in LOCAL_MODELS + FRONTIER_MODELS]].max(axis=1)
    return table


def build_actions(
    frame: pd.DataFrame,
    *,
    frontier_by_dataset: dict[str, str],
    local_by_dataset: dict[str, str],
    mode: str,
    threshold: float | None,
    budget_rate: float | None,
) -> pd.Series:
    local_actions = []
    need_scores = []
    for _, row in frame.iterrows():
        chosen = str(row["chosen_local_model"])
        if chosen in LOCAL_MODELS:
            local_action = chosen
        else:
            local_action = local_by_dataset.get(str(row["dataset"]), "qwen3-8b-local")
        local_actions.append(local_action)
        confidence = float(row["selector_confidence"])
        need = bool(row["selector_need_frontier"])
        none = str(row["choice"]).upper() == "NONE"
        need_score = (1.0 - confidence) + (0.75 if need else 0.0) + (0.5 if none else 0.0)
        need_scores.append(need_score)
    actions = pd.Series(local_actions, index=frame.index)
    frontier_actions = frame["dataset"].map(lambda dataset: frontier_by_dataset.get(str(dataset), GEMINI))
    need_scores_series = pd.Series(need_scores, index=frame.index)
    if mode == "threshold":
        assert threshold is not None
        frontier_index = need_scores_series[need_scores_series.ge(threshold)].index
    elif mode == "budget":
        assert budget_rate is not None
        budget = int(np.floor(budget_rate * len(frame)))
        frontier_index = need_scores_series.sort_values(ascending=False).head(budget).index if budget > 0 else []
    elif mode == "need_flag":
        frontier_index = frame[frame["selector_need_frontier"].astype(bool)].index
    elif mode == "none_or_need":
        frontier_index = frame[frame["selector_need_frontier"].astype(bool) | frame["choice"].astype(str).str.upper().eq("NONE")].index
    else:
        raise ValueError(mode)
    actions.loc[frontier_index] = frontier_actions.loc[frontier_index]
    return actions


def evaluate_actions(
    frame: pd.DataFrame,
    actions: pd.Series,
    *,
    method: str,
    lambda_cost: float,
) -> dict[str, Any]:
    qualities: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    gpt_calls: list[bool] = []
    gemini_calls: list[bool] = []
    for idx, row in frame.iterrows():
        action = str(actions.loc[idx])
        selector_latency = float(row.get("selector_latency_s", 0.0) or 0.0)
        if action in LOCAL_MODELS:
            quality = float(row[f"{action}_quality"])
            cost = 0.0
            latency = selector_latency
            gpt = False
            gemini = False
        elif action == GEMINI:
            quality = float(row[f"{GEMINI}_quality"])
            cost = float(row[f"{GEMINI}_cost"])
            latency = selector_latency + float(row[f"{GEMINI}_latency"])
            gpt = False
            gemini = True
        elif action == GPT:
            quality = float(row[f"{GPT}_quality"])
            cost = float(row[f"{GPT}_cost"])
            latency = selector_latency + float(row[f"{GPT}_latency"])
            gpt = True
            gemini = False
        else:
            raise ValueError(action)
        qualities.append(quality)
        costs.append(cost)
        latencies.append(latency)
        gpt_calls.append(gpt)
        gemini_calls.append(gemini)
    mean_quality = float(np.mean(qualities))
    normalized_cost = float(np.sum(costs) / frame[f"{GPT}_cost"].astype(float).sum())
    mean_utility = float(mean_quality - lambda_cost * normalized_cost)
    return {
        "method": method,
        "split": str(frame["split"].iloc[0]),
        "n_queries": int(len(frame)),
        "mean_quality": mean_quality,
        "mean_utility": mean_utility,
        "quality_gap_to_expanded_oracle": float(frame["qwen32_gate_quality_oracle"].mean() - mean_quality),
        "utility_ratio_to_expanded_cost_oracle": float(mean_utility / frame["qwen32_gate_cost_oracle_utility"].mean()),
        "normalized_remote_cost_vs_all_gpt": normalized_cost,
        "frontier_call_rate": float(np.mean([g or h for g, h in zip(gpt_calls, gemini_calls)])),
        "gpt_call_rate": float(np.mean(gpt_calls)),
        "gemini_call_rate": float(np.mean(gemini_calls)),
        "selector_mean_latency_s": float(frame["selector_latency_s"].mean()),
        "p95_latency_s": float(np.quantile(latencies, 0.95)),
        "action_counts": json.dumps({str(key): int(value) for key, value in actions.value_counts().to_dict().items()}, sort_keys=True),
    }


def evaluate(table: pd.DataFrame, lambda_cost: float, frontier_rate_target: float) -> pd.DataFrame:
    cost_norm = max(float(table[f"{GPT}_cost"].mean()), 1e-12)
    table = add_oracle_utility(table, lambda_cost, cost_norm)
    train = table[table["split"].eq("train")].copy()
    local_by_dataset = local_fallback_table(train)
    rows: list[dict[str, Any]] = []
    for frontier_strategy in ["gemini", "gpt", "train_quality", "train_utility"]:
        frontier_by_dataset = frontier_action_table(train, lambda_cost, cost_norm, frontier_strategy)
        for split, frame in table[table["split"].isin(["val", "test"])].groupby("split", sort=False):
            for mode in ["need_flag", "none_or_need"]:
                actions = build_actions(
                    frame,
                    frontier_by_dataset=frontier_by_dataset,
                    local_by_dataset=local_by_dataset,
                    mode=mode,
                    threshold=None,
                    budget_rate=None,
                )
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"qwen32_selector_{frontier_strategy}_{mode}",
                    lambda_cost=lambda_cost,
                )
                row["frontier_strategy"] = frontier_strategy
                rows.append(row)
            for threshold in np.linspace(0.15, 1.75, 17):
                actions = build_actions(
                    frame,
                    frontier_by_dataset=frontier_by_dataset,
                    local_by_dataset=local_by_dataset,
                    mode="threshold",
                    threshold=float(threshold),
                    budget_rate=None,
                )
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"qwen32_selector_{frontier_strategy}_threshold{threshold:.2f}",
                    lambda_cost=lambda_cost,
                )
                row["frontier_strategy"] = frontier_strategy
                rows.append(row)
            for budget_rate in [0.25, 0.30, 0.35, frontier_rate_target, 0.45, 0.50]:
                actions = build_actions(
                    frame,
                    frontier_by_dataset=frontier_by_dataset,
                    local_by_dataset=local_by_dataset,
                    mode="budget",
                    threshold=None,
                    budget_rate=float(budget_rate),
                )
                row = evaluate_actions(
                    frame,
                    actions,
                    method=f"qwen32_selector_{frontier_strategy}_budget{budget_rate:.2f}",
                    lambda_cost=lambda_cost,
                )
                row["frontier_strategy"] = frontier_strategy
                rows.append(row)
    return pd.DataFrame(rows)


def select_rows(results: pd.DataFrame, quality_gap_target: float, frontier_rate_target: float) -> pd.DataFrame:
    val = results[results["split"].eq("val")].copy()
    feasible = val[
        val["quality_gap_to_expanded_oracle"].le(quality_gap_target)
        & val["frontier_call_rate"].le(frontier_rate_target)
    ].copy()
    if feasible.empty:
        feasible = val[val["frontier_call_rate"].le(frontier_rate_target)].copy()
        feasible["selection_status"] = "no_validation_quality_feasible_under_frontier_cap"
    else:
        feasible["selection_status"] = "validation_feasible"
    if feasible.empty:
        feasible = val.copy()
        feasible["selection_status"] = "no_validation_frontier_feasible"
    selected = feasible.sort_values(["utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False).head(5)
    test = results[results["split"].eq("test")].copy()
    return selected.merge(test, on="method", how="left", suffixes=("_val", "_test"))


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
    raw_dir = output_dir / "raw_qwen32_selector"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.query_table)
    selector = selector_rows(table, args, raw_dir)
    merged = table.merge(selector, on=["query_id", "split", "dataset"], how="left")
    labeled_mask = merged["selector_confidence"].notna()
    eval_table = merged[merged["split"].eq("train") | labeled_mask].copy()
    results = evaluate(eval_table, args.lambda_cost, args.frontier_rate_target)
    selected = select_rows(results, args.quality_gap_target, args.frontier_rate_target)

    query_path = output_dir / "query_table_with_qwen32_selector.csv"
    selector_path = output_dir / "table_qwen32_selector_outputs.csv"
    results_path = output_dir / "table_qwen32_selector_gate.csv"
    selected_path = output_dir / "table_qwen32_selector_selected.csv"
    memo_path = output_dir / "QWEN32_LOCAL_SELECTOR_GATE_MEMO.md"
    merged.to_csv(query_path, index=False)
    selector.to_csv(selector_path, index=False)
    results.to_csv(results_path, index=False)
    selected.to_csv(selected_path, index=False)

    best_test = results[results["split"].eq("test")].sort_values(
        ["utility_ratio_to_expanded_cost_oracle", "mean_quality"], ascending=False
    ).head(10)
    selector_summary = selector.groupby(["split", "dataset"]).agg(
        n=("query_id", "size"),
        selector_local_quality=("selector_local_quality", "mean"),
        none_rate=("choice", lambda value: float(value.astype(str).str.upper().eq("NONE").mean())),
        need_frontier_rate=("selector_need_frontier", "mean"),
        mean_latency_s=("selector_latency_s", "mean"),
    )
    memo = [
        "# Qwen32 Local Selector Gate Memo",
        "",
        f"Source query table: `{args.query_table}`.",
        "This uses local Qwen3-32B-AWQ as a zero-shot candidate-answer selector over local model answers.",
        "The prompt includes query text, dataset, and four local candidate answers only. It does not include gold answers or frontier outputs.",
        "Thresholds and budgets are selected on validation; test is held out for reporting.",
        "",
        "## Selector Summary",
        "",
        markdown_table(selector_summary.reset_index()),
        "",
        "## Validation-Selected Policies",
        "",
        markdown_table(selected),
        "",
        "## Best Held-Out Test Diagnostics",
        "",
        markdown_table(best_test),
        "",
        "## Interpretation",
        "",
        "This is a local-probe attempt to close the local answer-selection gap. It is successful only if the validation-selected held-out row meets the quality, utility, cost, and frontier-rate targets.",
        "",
        "## Files",
        "",
        f"- `{query_path}`",
        f"- `{selector_path}`",
        f"- `{results_path}`",
        f"- `{selected_path}`",
    ]
    memo_path.write_text("\n".join(memo) + "\n", encoding="utf-8")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
