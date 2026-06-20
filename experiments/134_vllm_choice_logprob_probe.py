from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CHOICES = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect vLLM first-choice logprob probes for multiple-choice rows.")
    parser.add_argument("--task-manifest", default="results/controlled/broad_target_manifest_100/broad_target_task_manifest.csv")
    parser.add_argument("--reference-outputs", default="results/controlled/live_broad100_stage0/model_outputs.parquet")
    parser.add_argument("--output-dir", default="results/controlled/broad100_qwen4_choice_logprob_probe")
    parser.add_argument("--datasets", default="gpqa,mmlupro")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-id", default="qwen3-4b-choice-logprob-probe")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-chat", action="store_true", help="Use /chat/completions instead of /completions.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="When --use-chat is set, request Qwen thinking mode. Defaults to no-thinking chat templates.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    raw_dir = out_dir / "raw_choice_logprob_probe"
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
                payload = call_choice_probe(
                    base_url=str(args.base_url),
                    served_model_name=str(args.served_model_name),
                    prompt=str(task.query_text),
                    top_logprobs=int(args.top_logprobs),
                    timeout_s=float(args.timeout_s),
                    use_chat=bool(args.use_chat),
                    enable_thinking=bool(args.enable_thinking),
                )
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error_type": error_type, "error": str(exc)[:1000]}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - start
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

        features = extract_choice_features(payload)
        pred_choice = features["choice_pred"]
        gold = str(task.gold_answer).strip().upper()
        quality = float(status == "success" and pred_choice == gold)
        rows.append(
            {
                "query_id": task.query_id,
                "query_text": task.query_text,
                "benchmark": task.dataset,
                "domain": task.domain,
                "split": task.split,
                "task_type": task.task_type,
                "metric": "multiple_choice",
                "model_id": args.model_id,
                "status": status,
                "error_type": error_type,
                "cache_hit": cache_hit,
                "latency_s": float(payload.get("_latency_s", time.time() - start) or 0.0),
                "raw_output_path": str(raw_path),
                "raw_text": extract_text(payload),
                "gold_answer": gold,
                "quality_score": quality,
                **features,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_vllm_choice_logprob_probe.csv", index=False)
    diagnostics = build_diagnostics(table)
    diagnostics.to_csv(out_dir / "table_vllm_choice_logprob_diagnostics.csv", index=False)
    write_memo(out_dir / "VLLM_CHOICE_LOGPROB_PROBE_MEMO.md", table, diagnostics, args)
    print(f"Wrote vLLM choice-logprob probe artifacts to {out_dir}")


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
    tasks = tasks[tasks["task_type"].astype(str).eq("multiple_choice")]
    tasks = tasks.sort_values(["dataset", "split", "query_id"]).drop_duplicates("query_id")
    if args.limit is not None:
        tasks = tasks.head(int(args.limit))
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


def call_choice_probe(
    *,
    base_url: str,
    served_model_name: str,
    prompt: str,
    top_logprobs: int,
    timeout_s: float,
    use_chat: bool,
    enable_thinking: bool,
) -> dict[str, Any]:
    if use_chat:
        payload = {
            "model": served_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": choice_prompt(prompt),
                }
            ],
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": int(top_logprobs),
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        endpoint = "chat/completions"
    else:
        payload = {
            "model": served_model_name,
            "prompt": choice_prompt(prompt),
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": int(top_logprobs),
        }
        endpoint = "completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/{endpoint}",
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


def choice_prompt(prompt: str) -> str:
    text = str(prompt)
    replacements = [
        "Think step by step before answering.",
        "The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.",
    ]
    for old in replacements:
        text = text.replace(old, "")
    return f"{text.strip()}\n\nChoose the best answer. Reply with exactly one letter: A, B, C, or D.\nAnswer:"


def extract_text(payload: dict[str, Any]) -> str:
    try:
        choice = payload.get("choices", [{}])[0]
        if "text" in choice:
            return str(choice.get("text", "") or "")
        return str(choice.get("message", {}).get("content", "") or "")
    except Exception:
        return ""


def extract_choice_features(payload: dict[str, Any]) -> dict[str, Any]:
    first = first_logprob_item(payload)
    generated = normalize_choice(first.get("token") or extract_text(payload))
    option_logprobs: dict[str, float] = {}
    if generated in CHOICES and isinstance(first.get("logprob"), int | float):
        option_logprobs[generated] = float(first["logprob"])
    for item in first.get("top_logprobs") or []:
        if not isinstance(item, dict):
            continue
        choice = normalize_choice(item.get("token", ""))
        value = item.get("logprob")
        if choice in CHOICES and isinstance(value, int | float) and np.isfinite(value):
            option_logprobs[choice] = max(float(value), option_logprobs.get(choice, -math.inf))

    pred = ""
    if option_logprobs:
        pred = sorted(option_logprobs.items(), key=lambda item: (-item[1], item[0]))[0][0]
    elif generated in CHOICES:
        pred = generated

    ordered = sorted(option_logprobs.items(), key=lambda item: item[1], reverse=True)
    margin = float(ordered[0][1] - ordered[1][1]) if len(ordered) >= 2 else np.nan
    entropy = option_entropy(option_logprobs)
    out: dict[str, Any] = {
        "choice_generated": generated,
        "choice_pred": pred,
        "choice_logprob_margin": margin,
        "choice_entropy": entropy,
        "choice_seen_count": int(len(option_logprobs)),
        "choice_missing_count": int(4 - len(option_logprobs)),
    }
    for choice in CHOICES:
        out[f"choice_logprob_{choice.lower()}"] = option_logprobs.get(choice, np.nan)
    return out


def first_logprob_item(payload: dict[str, Any]) -> dict[str, Any]:
    choice = payload.get("choices", [{}])[0]
    logprobs = choice.get("logprobs", {}) or {}
    if "tokens" in logprobs:
        tokens = logprobs.get("tokens") or []
        token_logprobs = logprobs.get("token_logprobs") or []
        top_logprobs = logprobs.get("top_logprobs") or []
        top = top_logprobs[0] if top_logprobs else {}
        normalized_top = [{"token": token, "logprob": value} for token, value in top.items()] if isinstance(top, dict) else []
        return {
            "token": tokens[0] if tokens else "",
            "logprob": token_logprobs[0] if token_logprobs else np.nan,
            "top_logprobs": normalized_top,
        }
    content = logprobs.get("content", [])
    if content and isinstance(content[0], dict):
        return content[0]
    return {}


def normalize_choice(value: object) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    text = text.replace("ANSWER:", "").strip()
    for char in text:
        if char in CHOICES:
            return char
        if char.isalpha():
            break
    return ""


def option_entropy(option_logprobs: dict[str, float]) -> float:
    if not option_logprobs:
        return np.nan
    values = np.asarray(list(option_logprobs.values()), dtype=float)
    values = values - float(np.max(values))
    probs = np.exp(values)
    probs = probs / max(float(probs.sum()), 1e-12)
    return float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())


def build_diagnostics(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    successful = table[table["status"].eq("success")].copy()
    return (
        successful.groupby(["split", "benchmark"], dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            mean_quality=("quality_score", "mean"),
            mean_margin=("choice_logprob_margin", "mean"),
            mean_entropy=("choice_entropy", "mean"),
            mean_seen_choices=("choice_seen_count", "mean"),
            pred_counts_json=("choice_pred", lambda s: s.value_counts().sort_index().to_json()),
        )
        .reset_index()
        .sort_values(["split", "benchmark"])
    )


def write_memo(path: Path, table: pd.DataFrame, diagnostics: pd.DataFrame, args: argparse.Namespace) -> None:
    status = table["status"].value_counts(dropna=False).sort_index().to_dict() if not table.empty else {}
    lines = [
        "# vLLM Choice Logprob Probe Memo",
        "",
        f"Model: `{args.model_id}` served as `{args.served_model_name}` from `{args.base_url}`.",
        f"Rows: `{len(table)}`. Status counts: `{status}`.",
        f"Datasets: `{args.datasets}`. Splits: `{args.splits}`.",
        "",
        "This is a local-only probe collection. It makes no GPT, Gemini, or Claude calls.",
        "",
        "## Diagnostics",
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
