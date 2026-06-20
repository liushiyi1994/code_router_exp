from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from routecode.controlled.live_stage0 import normalize_answer, score_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect vLLM local logprob probe signals for broad100 rows.")
    parser.add_argument("--task-manifest", default="results/controlled/broad_target_manifest_100/broad_target_task_manifest.csv")
    parser.add_argument("--reference-outputs", default="results/controlled/live_broad100_stage0/model_outputs.parquet")
    parser.add_argument("--output-dir", default="results/controlled/broad100_qwen4_logprob_probe")
    parser.add_argument("--datasets", default="gpqa,mmlupro,math500")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-id", default="qwen3-4b-logprob-probe")
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw_logprob_probe"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args)
    rows = []
    for task in tasks.itertuples(index=False):
        raw_path = raw_dir / f"{str(task.query_id).replace(':', '_')}.json"
        cache_hit = raw_path.exists() and not args.force_rerun
        start = time.time()
        status = "success"
        error_type = ""
        payload: dict[str, Any] = {}
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status = str(payload.get("_status", "success"))
            error_type = str(payload.get("_error_type", ""))
        else:
            try:
                payload = call_vllm_logprob_probe(
                    base_url=str(args.base_url),
                    served_model_name=str(args.served_model_name),
                    prompt=str(task.query_text),
                    max_output_tokens=int(args.max_output_tokens),
                    top_logprobs=int(args.top_logprobs),
                    timeout_s=float(args.timeout_s),
                )
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error_type": error_type, "error": str(exc)[:1000]}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        raw_text = extract_text(payload)
        parsed, quality = score_output(raw_text, str(task.gold_answer), str(task.metric))
        if status != "success":
            parsed = ""
            quality = float("nan")
        features = extract_logprob_features(payload)
        rows.append(
            {
                "query_id": task.query_id,
                "query_text": task.query_text,
                "benchmark": task.dataset,
                "domain": task.domain,
                "split": task.split,
                "task_type": task.task_type,
                "metric": task.metric,
                "model_id": args.model_id,
                "status": status,
                "error_type": error_type,
                "cache_hit": cache_hit,
                "latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
                "raw_output_path": str(raw_path),
                "raw_text": raw_text,
                "parsed_answer": parsed,
                "normalized_answer": normalize_answer(parsed),
                "gold_answer": task.gold_answer,
                "quality_score": quality,
                **features,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_vllm_logprob_probe.csv", index=False)
    diagnostics = build_diagnostics(table)
    diagnostics.to_csv(out_dir / "table_vllm_logprob_probe_diagnostics.csv", index=False)
    write_memo(out_dir / "VLLM_LOGPROB_PROBE_MEMO.md", table, diagnostics, args)
    print(f"Wrote vLLM logprob probe artifacts to {out_dir}")


def load_tasks(args: argparse.Namespace) -> pd.DataFrame:
    manifest = pd.read_csv(args.task_manifest)
    ref = pd.read_parquet(args.reference_outputs).drop_duplicates("query_id")[["query_id", "benchmark"]]
    ref = add_broad_splits(ref)
    split_map = ref.set_index("query_id")["split"]
    tasks = manifest[manifest["query_id"].isin(split_map.index)].copy()
    tasks["split"] = tasks["query_id"].map(split_map)
    datasets = {item.strip() for item in str(args.datasets).split(",") if item.strip()}
    splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    if datasets:
        tasks = tasks[tasks["dataset"].astype(str).isin(datasets)]
    if splits:
        tasks = tasks[tasks["split"].astype(str).isin(splits)]
    tasks = tasks.sort_values(["dataset", "split", "query_id"]).drop_duplicates("query_id")
    if args.limit is not None:
        tasks = tasks.head(int(args.limit))
    tasks["metric"] = np.where(
        tasks["task_type"].astype(str).eq("multiple_choice"),
        "multiple_choice",
        np.where(tasks["task_type"].astype(str).eq("pass_at_1"), "pass_at_1", "exact_final_answer"),
    )
    return tasks


def add_broad_splits(query_order: pd.DataFrame) -> pd.DataFrame:
    query_order = query_order.sort_values(["benchmark", "query_id"]).copy()
    query_order["rank_in_benchmark"] = query_order.groupby("benchmark").cumcount()
    counts = query_order.groupby("benchmark")["query_id"].transform("count")
    train_cut = np.maximum(1, np.floor(counts * 0.60).astype(int))
    val_cut = np.maximum(train_cut + 1, np.floor(counts * 0.80).astype(int))
    query_order["split"] = np.where(
        query_order["rank_in_benchmark"] < train_cut,
        "train",
        np.where(query_order["rank_in_benchmark"] < val_cut, "val", "test"),
    )
    return query_order


def call_vllm_logprob_probe(
    *,
    base_url: str,
    served_model_name: str,
    prompt: str,
    max_output_tokens: int,
    top_logprobs: int,
    timeout_s: float,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": "Solve privately. Return only the final answer, final option letter, or short code result.",
        },
        {"role": "user", "content": prompt},
    ]
    payload = {
        "model": served_model_name,
        "messages": messages,
        "temperature": 0,
        "max_tokens": int(max_output_tokens),
        "logprobs": True,
        "top_logprobs": int(top_logprobs),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=data,
        headers={"Authorization": "Bearer local-routecode", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc


def extract_text(payload: dict[str, Any]) -> str:
    try:
        return str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    except Exception:
        return ""


def extract_logprob_features(payload: dict[str, Any]) -> dict[str, float | int]:
    content = payload.get("choices", [{}])[0].get("logprobs", {}).get("content", [])
    token_logprobs: list[float] = []
    margins: list[float] = []
    for item in content or []:
        if not isinstance(item, dict):
            continue
        logprob = item.get("logprob")
        if isinstance(logprob, int | float) and np.isfinite(logprob):
            token_logprobs.append(float(logprob))
        top = item.get("top_logprobs") or []
        top_values = []
        for candidate in top:
            if isinstance(candidate, dict):
                value = candidate.get("logprob")
                if isinstance(value, int | float) and np.isfinite(value):
                    top_values.append(float(value))
        top_values = sorted(top_values, reverse=True)
        if len(top_values) >= 2:
            margins.append(float(top_values[0] - top_values[1]))
    arr = np.asarray(token_logprobs, dtype=float)
    margin_arr = np.asarray(margins, dtype=float)
    return {
        "logprob_token_count": int(arr.size),
        "logprob_mean": float(arr.mean()) if arr.size else np.nan,
        "logprob_min": float(arr.min()) if arr.size else np.nan,
        "logprob_sum": float(arr.sum()) if arr.size else np.nan,
        "logprob_margin_mean": float(margin_arr.mean()) if margin_arr.size else np.nan,
        "logprob_margin_min": float(margin_arr.min()) if margin_arr.size else np.nan,
        "logprob_first_token_margin": float(margin_arr[0]) if margin_arr.size else np.nan,
    }


def build_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    successful = table[table["status"].eq("success")].copy()
    if successful.empty:
        return pd.DataFrame()
    successful["logprob_mean_bin"] = pd.qcut(
        successful["logprob_mean"].rank(method="first"),
        q=min(4, len(successful)),
        labels=False,
        duplicates="drop",
    )
    return (
        successful.groupby(["split", "benchmark", "logprob_mean_bin"], dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            mean_quality=("quality_score", "mean"),
            mean_logprob=("logprob_mean", "mean"),
            mean_margin=("logprob_margin_mean", "mean"),
            answer_token_count=("logprob_token_count", "mean"),
        )
        .reset_index()
        .sort_values(["split", "benchmark", "logprob_mean_bin"])
    )


def write_memo(path: Path, table: pd.DataFrame, diagnostics: pd.DataFrame, args: argparse.Namespace) -> None:
    status = table["status"].value_counts(dropna=False).sort_index().to_dict() if not table.empty else {}
    lines = [
        "# vLLM Logprob Probe Memo",
        "",
        f"Model: `{args.model_id}` served as `{args.served_model_name}` from `{args.base_url}`.",
        f"Rows: `{len(table)}`. Status counts: `{status}`.",
        f"Datasets: `{args.datasets}`. Splits: `{args.splits}`.",
        "",
        "This is a local-only probe collection. It makes no GPT, Gemini, or Claude calls.",
        "",
        "## Quality By Split And Benchmark",
        "",
        markdown_table(
            table.groupby(["split", "benchmark"], dropna=False)
            .agg(
                n_queries=("query_id", "size"),
                success_rate=("status", lambda s: float((s == "success").mean())),
                mean_quality=("quality_score", "mean"),
                mean_logprob=("logprob_mean", "mean"),
                mean_margin=("logprob_margin_mean", "mean"),
            )
            .reset_index()
        ),
        "",
        "## Logprob Bins",
        "",
        markdown_table(diagnostics),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
