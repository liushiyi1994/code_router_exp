from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRONG_MODEL_ID = "gemini-3.5-flash-strong-solve"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect a local answer-verifier probe and gate Gemini strong-solve.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("results/controlled/broad100_train_supervised_strong_gain_gate/model_outputs_with_gemini_strong_all_splits.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_qwen32_answer_verifier_strong_gate"),
    )
    parser.add_argument("--base-method", default="observable_local_state_v5_no_strong")
    parser.add_argument("--base-url", default="http://127.0.0.1:8007/v1")
    parser.add_argument("--served-model-name", default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--model-id", default="qwen3-32b-awq-answer-verifier-probe")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-query-chars", type=int, default=2200)
    parser.add_argument("--max-answer-chars", type=int, default=500)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    package = load_module("experiments/125_phase3_broad_target_method_package.py", "broad_target_package")
    outputs = load_outputs(args.outputs)
    outputs_no_strong = outputs[~outputs["model_id"].eq(STRONG_MODEL_ID)].copy()
    base = {
        split: base_selection(package, outputs_no_strong, base_name=str(args.base_method), split=split)
        for split in ["train", "val", "test"]
    }
    probe_inputs = build_probe_inputs(package, outputs_no_strong, base, splits=parse_csv_set(args.splits))
    if args.limit is not None:
        probe_inputs = probe_inputs.head(int(args.limit)).copy()
    probe = collect_verifier_probe(args, probe_inputs)
    probe.to_csv(args.output_dir / "table_vllm_answer_verifier_probe.csv", index=False)
    diagnostics = build_probe_diagnostics(outputs, probe)
    diagnostics.to_csv(args.output_dir / "table_vllm_answer_verifier_diagnostics.csv", index=False)
    eval_table, selected = evaluate_verifier_gates(package, outputs, base, probe, lambda_cost=float(args.lambda_cost))
    eval_table.to_csv(args.output_dir / "table_vllm_answer_verifier_gate_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_vllm_answer_verifier_gate_selected.csv", index=False)
    write_figure(args.output_dir, eval_table)
    write_memo(args.output_dir / "VLLM_ANSWER_VERIFIER_STRONG_GATE_MEMO.md", args, probe, diagnostics, eval_table, selected)
    print(f"Wrote vLLM answer-verifier strong gate results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_outputs(path: Path) -> pd.DataFrame:
    outputs = pd.read_parquet(path).copy()
    for column in ["quality_score", "utility", "cost_total_usd", "normalized_remote_cost", "latency_s"]:
        outputs[column] = pd.to_numeric(outputs[column], errors="coerce").fillna(0.0)
    outputs["query_id"] = outputs["query_id"].astype(str)
    outputs["model_id"] = outputs["model_id"].astype(str)
    outputs["split"] = outputs["split"].astype(str)
    return outputs


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in str(value).split(",") if item.strip()}


def base_selection(package, outputs_no_strong: pd.DataFrame, *, base_name: str, split: str) -> pd.Series:
    if base_name == "tool_probe_profile_v4_no_strong":
        return normalize_selection(package.profile_v4_selection_for_split(outputs_no_strong, split=split, exclude_models={STRONG_MODEL_ID}))
    if base_name == "observable_local_state_v5_no_strong":
        return normalize_selection(package.observable_local_state_selection(outputs_no_strong, split=split))
    raise ValueError(f"Unknown base method: {base_name}")


def build_probe_inputs(package, outputs_no_strong: pd.DataFrame, base: dict[str, pd.Series], *, splits: set[str]) -> pd.DataFrame:
    rows = []
    for split, selected in base.items():
        if split not in splits:
            continue
        selected_rows = package.selected_to_rows(outputs_no_strong, selected, split=split).copy()
        selected_rows["base_model_id"] = selected_rows["model_id"].astype(str)
        selected_rows["base_parsed_answer"] = selected_rows["parsed_answer"].fillna("").astype(str)
        rows.append(
            selected_rows[
                [
                    "query_id",
                    "query_text",
                    "benchmark",
                    "domain",
                    "metric",
                    "split",
                    "base_model_id",
                    "base_parsed_answer",
                    "quality_score",
                    "utility",
                ]
            ].rename(columns={"quality_score": "base_quality", "utility": "base_utility"})
        )
    return pd.concat(rows, ignore_index=True).sort_values(["split", "benchmark", "query_id"])


def collect_verifier_probe(args: argparse.Namespace, probe_inputs: pd.DataFrame) -> pd.DataFrame:
    raw_dir = args.output_dir / "raw_answer_verifier_probe" / safe_part(str(args.model_id)) / safe_part(str(args.base_method))
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, task in enumerate(probe_inputs.itertuples(index=False), start=1):
        raw_path = raw_dir / f"{safe_part(str(task.query_id))}_{cache_digest(str(task.query_id), str(task.base_model_id))}.json"
        cache_hit = raw_path.exists() and not args.force_rerun
        started = time.time()
        status = "success"
        error_type = ""
        if cache_hit:
            payload = json.loads(raw_path.read_text(encoding="utf-8"))
            status = str(payload.get("_status", "success"))
            error_type = str(payload.get("_error_type", ""))
        else:
            prompt = verifier_prompt(
                query_text=str(task.query_text),
                metric=str(task.metric),
                benchmark=str(task.benchmark),
                candidate_answer=str(task.base_parsed_answer),
                candidate_model=str(task.base_model_id),
                max_query_chars=int(args.max_query_chars),
                max_answer_chars=int(args.max_answer_chars),
            )
            try:
                payload = call_vllm_chat(
                    base_url=str(args.base_url),
                    served_model_name=str(args.served_model_name),
                    prompt=prompt,
                    max_tokens=int(args.max_tokens),
                    timeout_s=float(args.timeout_s),
                )
            except Exception as exc:
                status = "error"
                error_type = type(exc).__name__
                payload = {"error_type": error_type, "error": str(exc)[:1000]}
            payload["_status"] = status
            payload["_error_type"] = error_type
            payload["_latency_s"] = time.time() - started
            raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        raw_text = extract_text(payload)
        parsed = parse_verifier_output(raw_text)
        if status != "success":
            parsed = {"verdict": "unknown", "confidence": np.nan, "reason": ""}
        rows.append(
            {
                "query_id": task.query_id,
                "split": task.split,
                "benchmark": task.benchmark,
                "domain": task.domain,
                "metric": task.metric,
                "base_model_id": task.base_model_id,
                "base_quality": float(task.base_quality),
                "base_utility": float(task.base_utility),
                "model_id": args.model_id,
                "status": status,
                "error_type": error_type,
                "cache_hit": bool(cache_hit),
                "latency_s": float(payload.get("_latency_s", time.time() - started) or 0.0),
                "raw_output_path": str(raw_path),
                "raw_text": raw_text,
                "verdict": parsed["verdict"],
                "confidence": parsed["confidence"],
                "reason": parsed.get("reason", ""),
                "escalate_score": escalate_score(parsed["verdict"], parsed["confidence"]),
            }
        )
        if index % 50 == 0 or index == len(probe_inputs):
            print(f"answer verifier rows {index}/{len(probe_inputs)}")
    return pd.DataFrame(rows)


def verifier_prompt(
    *,
    query_text: str,
    metric: str,
    benchmark: str,
    candidate_answer: str,
    candidate_model: str,
    max_query_chars: int,
    max_answer_chars: int,
) -> str:
    query = truncate(query_text, max_query_chars)
    answer = truncate(candidate_answer, max_answer_chars) or "[empty]"
    return (
        "You are a local routing verifier. Your job is not to explain the task; "
        "decide whether the candidate answer is reliable enough to keep, or whether a stronger solver should be used.\n"
        "Use 'accept' only when the candidate is likely correct. Use 'escalate' when the candidate is wrong, malformed, empty, or uncertain.\n"
        'Return JSON only: {"verdict":"accept|escalate","confidence":0.0,"reason":"short"}\n\n'
        f"Benchmark: {benchmark}\n"
        f"Metric: {metric}\n"
        f"Candidate model: {candidate_model}\n\n"
        f"Task:\n{query}\n\n"
        f"Candidate answer:\n{answer}\n"
    )


def call_vllm_chat(*, base_url: str, served_model_name: str, prompt: str, max_tokens: int, timeout_s: float) -> dict[str, Any]:
    payload = {
        "model": served_model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
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


def parse_verifier_output(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    payload: dict[str, Any] = {}
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            maybe = json.loads(match.group(0))
            if isinstance(maybe, dict):
                payload = maybe
        except json.JSONDecodeError:
            payload = {}
    verdict = str(payload.get("verdict", "")).strip().lower()
    if verdict not in {"accept", "escalate"}:
        lowered = text.lower()
        if "escalate" in lowered or "stronger" in lowered:
            verdict = "escalate"
        elif "accept" in lowered:
            verdict = "accept"
        else:
            verdict = "unknown"
    try:
        confidence = float(payload.get("confidence", np.nan))
    except (TypeError, ValueError):
        confidence = np.nan
    if not np.isfinite(confidence):
        confidence = 0.5 if verdict != "unknown" else np.nan
    confidence = min(max(float(confidence), 0.0), 1.0) if np.isfinite(confidence) else np.nan
    return {"verdict": verdict, "confidence": confidence, "reason": str(payload.get("reason", ""))[:200]}


def escalate_score(verdict: str, confidence: float) -> float:
    if not np.isfinite(confidence):
        return 0.5
    if verdict == "escalate":
        return float(confidence)
    if verdict == "accept":
        return float(1.0 - confidence)
    return 0.5


def build_probe_diagnostics(outputs: pd.DataFrame, probe: pd.DataFrame) -> pd.DataFrame:
    gains = strong_gain_for_base(outputs, probe[["query_id", "base_model_id"]].drop_duplicates().set_index("query_id")["base_model_id"])
    frame = probe.join(gains.rename("strong_gain"), on="query_id")
    frame["strong_wins"] = frame["strong_gain"].astype(float) > 0.0
    return (
        frame.groupby(["split", "benchmark", "verdict"], dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            mean_confidence=("confidence", "mean"),
            mean_escalate_score=("escalate_score", "mean"),
            base_quality=("base_quality", "mean"),
            strong_win_rate=("strong_wins", "mean"),
            mean_strong_gain=("strong_gain", "mean"),
        )
        .reset_index()
        .sort_values(["split", "benchmark", "verdict"])
    )


def evaluate_verifier_gates(
    package,
    outputs: pd.DataFrame,
    base: dict[str, pd.Series],
    probe: pd.DataFrame,
    *,
    lambda_cost: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    score_by_split = {
        split: probe[probe["split"].eq(split)].set_index("query_id")["escalate_score"].astype(float)
        for split in ["val", "test"]
    }
    verdict_by_split = {
        split: probe[probe["split"].eq(split)].set_index("query_id")["verdict"].astype(str)
        for split in ["val", "test"]
    }
    for split in ["val", "test"]:
        rows.append(evaluate_selection(package, outputs, base[split], split=split, lambda_cost=lambda_cost, method="base_no_verifier", family="base"))
        rows.append(
            evaluate_selection(
                package,
                outputs,
                oracle_between_base_and_strong(outputs, base[split]),
                split=split,
                lambda_cost=lambda_cost,
                method="oracle_between_base_and_strong",
                family="diagnostic_oracle",
            )
        )
        selected = base[split].copy()
        for query_id, verdict in verdict_by_split[split].items():
            if verdict == "escalate":
                selected.loc[str(query_id)] = STRONG_MODEL_ID
        rows.append(
            evaluate_selection(
                package,
                outputs,
                selected,
                split=split,
                lambda_cost=lambda_cost,
                method="verdict_escalate_else_base",
                family="verifier_fixed_gate",
            )
        )

    val_candidates = []
    for threshold in candidate_thresholds(score_by_split["val"].to_numpy(dtype=float)):
        selected = apply_score_gate(base["val"], score_by_split["val"], threshold=threshold)
        row = evaluate_selection(
            package,
            outputs,
            selected,
            split="val",
            lambda_cost=lambda_cost,
            method=f"verifier_score_thr{threshold:.4f}",
            family="verifier_score_gate",
        )
        row["threshold"] = float(threshold)
        val_candidates.append(row)
    best = sorted(val_candidates, key=lambda row: (float(row["mean_utility"]), float(row["mean_quality"])), reverse=True)[0]
    rows.append(best)
    test_selected = apply_score_gate(base["test"], score_by_split["test"], threshold=float(best["threshold"]))
    test_row = evaluate_selection(
        package,
        outputs,
        test_selected,
        split="test",
        lambda_cost=lambda_cost,
        method=str(best["method"]),
        family="verifier_score_gate",
    )
    test_row["threshold"] = float(best["threshold"])
    rows.append(test_row)

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    return table, validation_selected_rows(table)


def apply_score_gate(base_selection_for_split: pd.Series, score: pd.Series, *, threshold: float) -> pd.Series:
    selected = base_selection_for_split.copy()
    selected.index = selected.index.astype(str)
    for query_id, value in score.items():
        if float(value) > float(threshold):
            selected.loc[str(query_id)] = STRONG_MODEL_ID
    return selected


def oracle_between_base_and_strong(outputs: pd.DataFrame, base_selection_for_split: pd.Series) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    selected = base_selection_for_split.copy()
    selected.index = selected.index.astype(str)
    for query_id, model_id in base_selection_for_split.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) not in by_query.index or (query_id, str(model_id)) not in by_query.index:
            continue
        if float(by_query.loc[(query_id, STRONG_MODEL_ID), "utility"]) > float(by_query.loc[(query_id, str(model_id)), "utility"]):
            selected.loc[query_id] = STRONG_MODEL_ID
    return selected


def strong_gain_for_base(outputs: pd.DataFrame, base_selection_for_split: pd.Series) -> pd.Series:
    by_query = outputs.set_index(["query_id", "model_id"])
    gains: dict[str, float] = {}
    for query_id, model_id in base_selection_for_split.items():
        query_id = str(query_id)
        if (query_id, STRONG_MODEL_ID) in by_query.index and (query_id, str(model_id)) in by_query.index:
            gains[query_id] = float(by_query.loc[(query_id, STRONG_MODEL_ID), "utility"]) - float(
                by_query.loc[(query_id, str(model_id)), "utility"]
            )
    return pd.Series(gains)


def evaluate_selection(
    package,
    outputs: pd.DataFrame,
    selected: pd.Series,
    *,
    split: str,
    lambda_cost: float,
    method: str,
    family: str,
) -> dict[str, Any]:
    target = outputs[outputs["split"].eq(split)]
    cost_oracle = target.loc[target.groupby("query_id")["utility"].idxmax()]
    quality_oracle = target.loc[target.groupby("query_id")["quality_score"].idxmax()]
    selected_rows = package.selected_to_rows(outputs, selected, split=split)
    row = package.evaluation_row(method, selected_rows, cost_oracle, quality_oracle, lambda_cost=lambda_cost)
    row["family"] = family
    row["strong_call_rate"] = float(selected_rows["model_id"].eq(STRONG_MODEL_ID).mean())
    row["verifier_call_rate"] = 1.0
    return row


def candidate_thresholds(values: np.ndarray) -> list[float]:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return [0.5]
    qs = np.quantile(finite, np.linspace(0.0, 0.98, 30)).tolist()
    fixed = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return sorted(set(float(value) for value in [*fixed, *qs]))


def validation_selected_rows(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for family, group in table.groupby("family"):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].sort_values(["mean_utility", "mean_quality"], ascending=False)
        if val.empty:
            continue
        best = val.iloc[0]
        rows.append(best)
        test = group[group["split"].eq("test") & group["method"].eq(best["method"])]
        if not test.empty:
            rows.append(test.iloc[0])
    return pd.DataFrame(rows)


def normalize_selection(selected: pd.Series) -> pd.Series:
    out = selected.copy()
    out.index = out.index.astype(str)
    return out.astype(str)


def truncate(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= int(max_chars):
        return text
    return text[: int(max_chars)] + "\n[truncated]"


def safe_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:90]


def cache_digest(query_id: str, model_id: str) -> str:
    return hashlib.sha1(f"{query_id}:{model_id}".encode("utf-8")).hexdigest()[:12]


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)
    labels = plot["family"].str.replace("_", " ", regex=False) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#587c7d")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Qwen32 Answer-Verifier Strong Gate")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_vllm_answer_verifier_gate_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    probe: pd.DataFrame,
    diagnostics: pd.DataFrame,
    eval_table: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    status = probe["status"].value_counts(dropna=False).sort_index().to_dict() if not probe.empty else {}
    verdict = probe["verdict"].value_counts(dropna=False).sort_index().to_dict() if not probe.empty else {}
    lines = [
        "# vLLM Answer-Verifier Strong Gate",
        "",
        f"Source outputs: `{args.outputs}`.",
        f"Local verifier model: `{args.model_id}` served as `{args.served_model_name}` from `{args.base_url}`.",
        f"Base method: `{args.base_method}`.",
        "This run uses local vLLM only for the verifier probe; it makes no GPT, Gemini, or Claude calls.",
        f"Rows: `{len(probe)}`. Status counts: `{status}`. Verdict counts: `{verdict}`.",
        "",
        "## Validation-Selected Rows",
        "",
        markdown_table(selected),
        "",
        "## Held-Out Evaluation",
        "",
        markdown_table(eval_table[eval_table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False)),
        "",
        "## Probe Diagnostics",
        "",
        markdown_table(diagnostics.head(40)),
        "",
        "## Interpretation",
        "",
        "- This tests answer-level uncertainty: the local model sees the current routed answer and decides whether to keep it or escalate.",
        "- The verifier is charged as a local probe signal in the action policy, not as a free oracle.",
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
