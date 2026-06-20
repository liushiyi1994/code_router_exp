from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoTokenizer


YES_VARIANTS = (" yes", " YES", "Yes", "YES")
NO_VARIANTS = (" no", " NO", "No", "NO")
CAPS = (0.10, 0.20, 0.30, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Constrained YES/NO local vLLM probe for local-vs-large routing.")
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path(
            "results/controlled/broad100_slm_llm_early_signal_probe_pilot_qwen14_answerability/"
            "table_slm_llm_oracle_targets.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8006/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-14B-AWQ")
    parser.add_argument(
        "--tokenizer",
        default="/home/liush/.cache/huggingface/hub/models--Qwen--Qwen3-14B-AWQ/snapshots/31c69efc29464b6bb0aee1398b5a7b50a99340c3",
    )
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--prompt-modes", default="query_only,local_evidence")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--logit-bias", type=float, default=100.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166")
    target = pd.read_csv(args.target_table)
    yes_ids, no_ids = yes_no_token_ids(args.tokenizer)
    probe = collect_or_load_probe(target, args=args, yes_ids=yes_ids, no_ids=no_ids)
    target_with_signals = merge_probe_signals(target, probe)
    policy_table, selected = evaluate_policies(
        rc166,
        target_with_signals,
        lambda_cost=float(args.lambda_cost),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
    )
    cap_table = precision_at_caps(target_with_signals)

    target_with_signals.to_csv(args.output_dir / "table_constrained_yesno_targets.csv", index=False)
    probe.to_csv(args.output_dir / "table_constrained_yesno_probe.csv", index=False)
    policy_table.to_csv(args.output_dir / "table_constrained_yesno_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_constrained_yesno_policy_selected.csv", index=False)
    cap_table.to_csv(args.output_dir / "table_constrained_yesno_precision_at_caps.csv", index=False)
    write_figure(args.output_dir, policy_table)
    write_memo(
        args.output_dir / "CONSTRAINED_YESNO_PROBE_MEMO.md",
        args,
        yes_ids=yes_ids,
        no_ids=no_ids,
        probe=probe,
        policy_table=policy_table,
        selected=selected,
        cap_table=cap_table,
    )
    print(f"Wrote constrained YES/NO probe results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def yes_no_token_ids(tokenizer_path: str) -> tuple[dict[int, str], dict[int, str]]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    yes: dict[int, str] = {}
    no: dict[int, str] = {}
    for variant in YES_VARIANTS:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            yes[int(ids[0])] = variant
    for variant in NO_VARIANTS:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            no[int(ids[0])] = variant
    if not yes or not no:
        raise RuntimeError(f"Could not find single-token YES/NO variants: yes={yes}, no={no}")
    return yes, no


def collect_or_load_probe(
    target: pd.DataFrame,
    *,
    args: argparse.Namespace,
    yes_ids: dict[int, str],
    no_ids: dict[int, str],
) -> pd.DataFrame:
    path = args.output_dir / "table_constrained_yesno_probe.csv"
    existing = pd.read_csv(path) if path.exists() and not args.force_rerun else pd.DataFrame()
    done: set[tuple[str, str]] = set()
    if not existing.empty and {"query_id", "prompt_mode"}.issubset(existing.columns):
        done = set(zip(existing["query_id"].astype(str), existing["prompt_mode"].astype(str), strict=False))

    splits = set(parse_csv(args.splits))
    modes = parse_csv(args.prompt_modes)
    pending_rows: list[dict[str, Any]] = []
    for row in target[target["split"].isin(splits)].to_dict("records"):
        for mode in modes:
            key = (str(row["query_id"]), mode)
            if key in done:
                continue
            pending_rows.append({**row, "prompt_mode": mode})

    rows: list[dict[str, Any]] = []
    for start in range(0, len(pending_rows), int(args.batch_size)):
        batch = pending_rows[start : start + int(args.batch_size)]
        rows.extend(call_batch(batch, args=args, yes_ids=yes_ids, no_ids=no_ids))
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
        combined.to_csv(path, index=False)
        print(f"constrained yes/no rows {min(start + len(batch), len(pending_rows))}/{len(pending_rows)}", flush=True)
    return pd.read_csv(path) if path.exists() else pd.DataFrame(rows)


def call_batch(
    batch: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    yes_ids: dict[int, str],
    no_ids: dict[int, str],
) -> list[dict[str, Any]]:
    if not batch:
        return []
    prompts = [probe_prompt(row, str(row["prompt_mode"])) for row in batch]
    logit_bias = {str(token_id): float(args.logit_bias) for token_id in [*yes_ids.keys(), *no_ids.keys()]}
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{str(args.base_url).rstrip('/')}/completions",
            json={
                "model": args.served_model_name,
                "prompt": prompts,
                "temperature": 0.0,
                "max_tokens": 1,
                "logprobs": int(args.top_logprobs),
                "logit_bias": logit_bias,
            },
            timeout=float(args.timeout_s),
        )
        response.raise_for_status()
        payload = response.json()
        choices = sorted(payload.get("choices", []), key=lambda item: int(item.get("index", 0)))
    except Exception as exc:  # noqa: BLE001 - cache per-row errors so runs can resume.
        elapsed = time.perf_counter() - started
        return [
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "prompt_mode": str(row["prompt_mode"]),
                "status": "error",
                "error": repr(exc),
                "raw_text": "",
                "yes_probability": np.nan,
                "need_large_score": np.nan,
                "yes_logprob": np.nan,
                "no_logprob": np.nan,
                "yesno_margin": np.nan,
                "yesno_entropy": np.nan,
                "latency_s": elapsed / max(len(batch), 1),
            }
            for row in batch
        ]

    elapsed = time.perf_counter() - started
    out: list[dict[str, Any]] = []
    for row, choice in zip(batch, choices, strict=False):
        features = yesno_features(choice)
        out.append(
            {
                "query_id": str(row["query_id"]),
                "split": str(row["split"]),
                "benchmark": str(row["benchmark"]),
                "prompt_mode": str(row["prompt_mode"]),
                "status": "success",
                "error": "",
                "raw_text": str(choice.get("text", "")),
                "latency_s": elapsed / max(len(batch), 1),
                **features,
            }
        )
    return out


def probe_prompt(row: dict[str, Any], mode: str) -> str:
    query = re.sub(r"\s+", " ", str(row.get("query_text", ""))).strip()[:1800]
    benchmark = str(row.get("benchmark", ""))
    metric = str(row.get("metric", ""))
    base = (
        "Answer exactly YES or NO.\n"
        "YES means the cheap local route can be trusted to answer this item correctly without a larger model.\n"
        "NO means a stronger/local-large or frontier model is likely worth using for correctness.\n"
        f"Benchmark: {benchmark}\n"
        f"Metric: {metric}\n"
        f"Item: {query}\n"
    )
    if mode == "query_only":
        return base + "Can the cheap local route be trusted? Answer:"
    if mode == "local_evidence":
        return (
            base
            + f"Best cheap local action: {row.get('best_local_action', '')}\n"
            + f"Small-model answer: {empty_marker(row.get('slm_answer', ''))}\n"
            + f"Medium-14B answer: {empty_marker(row.get('medium14_answer', ''))}\n"
            + f"Medium-32B answer: {empty_marker(row.get('medium32_answer', ''))}\n"
            + f"Self-consistency answer: {empty_marker(row.get('self_majority_answer', ''))}\n"
            + f"Self-consistency vote_frac={as_float(row.get('self_vote_frac', 0.0)):.3f}, "
            + f"vote_margin={as_float(row.get('self_vote_margin', 0.0)):.3f}, "
            + f"entropy={as_float(row.get('self_vote_entropy', 0.0)):.3f}\n"
            + "Can the cheap local route be trusted? Answer:"
        )
    raise ValueError(f"unknown prompt mode: {mode}")


def empty_marker(value: object) -> str:
    text = str(value or "").strip()
    return text[:120] if text else "[empty]"


def yesno_features(choice: dict[str, Any]) -> dict[str, float | str]:
    raw_text = str(choice.get("text", ""))
    text_class = classify_yes_no(raw_text)
    logprobs = choice.get("logprobs") or {}
    top_logprobs = logprobs.get("top_logprobs") or []
    first = top_logprobs[0] if top_logprobs else {}
    yes_values: list[float] = []
    no_values: list[float] = []
    if isinstance(first, dict):
        for token, value in first.items():
            if not isinstance(value, int | float) or not np.isfinite(value):
                continue
            cls = classify_yes_no(str(token))
            if cls == "yes":
                yes_values.append(float(value))
            elif cls == "no":
                no_values.append(float(value))
    generated_logprob = np.nan
    token_logprobs = logprobs.get("token_logprobs") or []
    if token_logprobs:
        generated_logprob = as_float(token_logprobs[0], default=np.nan)

    yes_logprob = logsumexp(yes_values)
    no_logprob = logsumexp(no_values)
    if np.isfinite(yes_logprob) and np.isfinite(no_logprob):
        yes_prob = float(1.0 / (1.0 + math.exp(no_logprob - yes_logprob)))
    elif text_class == "yes":
        yes_prob = 1.0
    elif text_class == "no":
        yes_prob = 0.0
    else:
        yes_prob = 0.5
    yes_prob = float(np.clip(yes_prob, 0.0, 1.0))
    entropy = binary_entropy(yes_prob)
    return {
        "generated_class": text_class,
        "yes_probability": yes_prob,
        "need_large_score": float(1.0 - yes_prob),
        "yes_logprob": yes_logprob,
        "no_logprob": no_logprob,
        "generated_logprob": generated_logprob,
        "yesno_margin": abs(float(yes_logprob - no_logprob)) if np.isfinite(yes_logprob) and np.isfinite(no_logprob) else np.nan,
        "yesno_entropy": entropy,
    }


def classify_yes_no(text: str) -> str:
    norm = str(text or "").strip().lower()
    if norm.startswith("yes"):
        return "yes"
    if norm.startswith("no"):
        return "no"
    return "other"


def logsumexp(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return float("nan")
    high = float(np.max(finite))
    return float(high + np.log(np.exp(finite - high).sum()))


def binary_entropy(prob_yes: float) -> float:
    p = float(np.clip(prob_yes, 1e-12, 1.0 - 1e-12))
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def merge_probe_signals(target: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    successful = probe[probe["status"].eq("success")].copy() if not probe.empty else pd.DataFrame()
    for mode in sorted(successful["prompt_mode"].dropna().astype(str).unique()) if not successful.empty else []:
        prefix = f"signal_constrained_yesno_{safe_name(mode)}"
        rows = successful[successful["prompt_mode"].astype(str).eq(mode)].drop_duplicates("query_id").set_index("query_id")
        out[f"{prefix}_risk"] = out["query_id"].map(rows["need_large_score"]).astype(float)
        out[f"{prefix}_safe"] = out["query_id"].map(rows["yes_probability"]).astype(float)
        out[f"{prefix}_entropy"] = out["query_id"].map(rows["yesno_entropy"]).astype(float)
        margin = out["query_id"].map(rows["yesno_margin"]).astype(float)
        out[f"{prefix}_low_margin_risk"] = 1.0 - minmax(margin.fillna(margin.median()).to_numpy(dtype=float))
    risk_cols = [
        col
        for col in out.columns
        if col.startswith("signal_constrained_yesno_") and (col.endswith("_risk") or col.endswith("_entropy"))
    ]
    if risk_cols:
        existing_cols = [
            col
            for col in [
                "signal_early_rollout_instability",
                "signal_semantic_uncertainty",
                "signal_slm_medium_divergence",
                "signal_query_answerability_risk",
            ]
            if col in out.columns
        ]
        out["signal_constrained_yesno_max_risk"] = out[risk_cols].max(axis=1)
        out["signal_constrained_yesno_mean_risk"] = out[risk_cols].mean(axis=1)
        if existing_cols:
            out["signal_constrained_plus_cached_mean_risk"] = out[[*existing_cols, *risk_cols]].mean(axis=1)
            out["signal_constrained_plus_cached_max_risk"] = out[[*existing_cols, *risk_cols]].max(axis=1)
    return out


def evaluate_policies(
    rc166,
    target: pd.DataFrame,
    *,
    lambda_cost: float,
    bootstrap_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    signal_columns = [
        col
        for col in target.columns
        if col.startswith("signal_")
        and pd.api.types.is_numeric_dtype(target[col])
        and target[target["split"].eq("val")][col].notna().any()
    ]
    for split in ["val", "test"]:
        split_frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(split_frame, split=split, lambda_cost=lambda_cost))
    for signal in signal_columns:
        val_values = target[target["split"].eq("val")][signal].to_numpy(dtype=float)
        for direction in ["high", "low"]:
            for threshold in rc166.candidate_thresholds(val_values):
                for split in ["val", "test"]:
                    split_frame = target[target["split"].eq(split)].copy()
                    choose_large = rc166.threshold_decision(split_frame[signal].to_numpy(dtype=float), threshold, direction)
                    row = rc166.evaluate_decision(
                        split_frame,
                        choose_large,
                        split=split,
                        method=f"{signal}_{direction}_thr{threshold:.4g}",
                        family="threshold_signal",
                        lambda_cost=lambda_cost,
                    )
                    row.update({"signal": signal, "direction": direction, "threshold": float(threshold)})
                    rows.append(row)
    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    selected = rc166.validation_selected_rows(table, bootstrap_samples=bootstrap_samples, seed=seed)
    return table, selected


def precision_at_caps(target: pd.DataFrame) -> pd.DataFrame:
    signal_columns = [
        col
        for col in target.columns
        if col.startswith("signal_")
        and pd.api.types.is_numeric_dtype(target[col])
        and target[target["split"].eq("test")][col].notna().any()
    ]
    rows: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        labels = frame["need_large"].astype(bool).to_numpy()
        for signal in signal_columns:
            scores = frame[signal].to_numpy(dtype=float)
            auroc = safe_auroc(labels, scores)
            auprc = safe_auprc(labels, scores)
            order_scores = np.where(np.isfinite(scores), scores, -np.inf)
            order = np.argsort(order_scores)[::-1]
            for cap in CAPS:
                k = max(1, int(math.floor(cap * len(frame))))
                selected = np.zeros(len(frame), dtype=bool)
                selected[order[:k]] = True
                tp = int(np.sum(selected & labels))
                fp = int(np.sum(selected & ~labels))
                fn = int(np.sum(~selected & labels))
                rows.append(
                    {
                        "split": split,
                        "signal": signal,
                        "cap": float(cap),
                        "selected_count": int(k),
                        "precision": float(tp / max(tp + fp, 1)),
                        "recall": float(tp / max(tp + fn, 1)),
                        "auroc": auroc,
                        "auprc": auprc,
                    }
                )
    return pd.DataFrame(rows)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    keep = np.isfinite(scores)
    labels = labels[keep]
    scores = scores[keep]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#557a62")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Constrained YES/NO Probe Policy")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_constrained_yesno_probe_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    *,
    yes_ids: dict[int, str],
    no_ids: dict[int, str],
    probe: pd.DataFrame,
    policy_table: pd.DataFrame,
    selected: pd.DataFrame,
    cap_table: pd.DataFrame,
) -> None:
    selected_cols = [
        "method",
        "family",
        "split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "mean_utility_ci_low",
        "mean_utility_ci_high",
        "oracle_utility_ratio",
        "recovered_gap_vs_local",
        "large_call_rate",
        "frontier_call_rate",
        "need_large_precision",
        "need_large_recall",
        "signal",
        "direction",
        "threshold",
        "selection_rule",
    ]
    probe_summary = (
        probe.groupby(["split", "prompt_mode", "generated_class"], dropna=False)
        .agg(
            n=("query_id", "nunique"),
            success_rate=("status", lambda s: float((s == "success").mean())),
            mean_need_large_score=("need_large_score", "mean"),
            mean_entropy=("yesno_entropy", "mean"),
            mean_margin=("yesno_margin", "mean"),
        )
        .reset_index()
        .sort_values(["split", "prompt_mode", "generated_class"])
    )
    test_best = policy_table[policy_table["split"].eq("test")].sort_values(
        ["mean_utility", "mean_quality"], ascending=False
    )
    lines = [
        "# Constrained YES/NO Probe Policy",
        "",
        "## Commands Run",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/167_constrained_yesno_probe_policy.py",
        (
            "PYTHONPATH=src python experiments/167_constrained_yesno_probe_policy.py "
            f"--target-table {args.target_table} --output-dir {args.output_dir} --base-url {args.base_url} "
            f"--served-model-name {args.served_model_name} --splits {args.splits} --prompt-modes {args.prompt_modes}"
        ),
        "```",
        "",
        f"- Served model: `{args.served_model_name}` via `{args.base_url}`.",
        "- This uses local vLLM only. No GPT, Gemini, or Claude API calls are made.",
        f"- YES token ids: `{yes_ids}`.",
        f"- NO token ids: `{no_ids}`.",
        "- `logit_bias` constrains the next token to YES/NO variants, then the score is computed from their top-logprobs.",
        "",
        "## Probe Summary",
        "",
        "```csv",
        compact_csv(probe_summary, max_rows=80),
        "```",
        "",
        "## Validation-Selected And Diagnostics",
        "",
        "```csv",
        compact_csv(selected[[col for col in selected_cols if col in selected.columns]], max_rows=48),
        "```",
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        compact_csv(test_best[[col for col in selected_cols if col in test_best.columns]], max_rows=32),
        "```",
        "",
        "## Precision At 10 Percent Cap",
        "",
        "```csv",
        compact_csv(
            cap_table[cap_table["split"].eq("test") & np.isclose(cap_table["cap"].astype(float), 0.10)]
            .sort_values(["precision", "recall"], ascending=False),
            max_rows=40,
        ),
        "```",
        "",
        "## Interpretation",
        "",
        "- This tests whether a constrained local yes/no confidence score can predict when the large action has positive cost-aware value.",
        "- Validation-selected rows are deployable evidence; top held-out diagnostic rows are only headroom checks.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compact_csv(frame: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if frame.empty:
        return ""
    out = frame.head(max_rows).copy() if max_rows else frame.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.4f}")
    return out.to_csv(index=False).strip()


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=float)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if high <= low:
        return np.zeros_like(values, dtype=float)
    return (values - low) / (high - low)


def as_float(value: object, *, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").lower()


if __name__ == "__main__":
    main()
