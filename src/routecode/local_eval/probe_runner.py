from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import re
import time
from typing import Any

import pandas as pd

from routecode.local_eval.generation_runner import GenerationResult
from routecode.probes.probe_features import PROBE_FEATURE_COLUMNS


PROBE_TEMPLATE_ID = "route_state_confidence_probe_v1"
PROBE_TYPE = "aligned_local_confidence_probe"
CONFIDENCE_RE = re.compile(
    r"\bconfidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b",
    flags=re.IGNORECASE,
)
ANSWER_RE = re.compile(r"^\s*answer\s*:\s*(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class LocalProbeTask:
    query_id: str
    query_text: str
    dataset: str
    domain: str


class DryRunProbeClient:
    """Deterministic no-server probe client for validating aligned logging."""

    def generate(
        self,
        *,
        model_id: str,
        prompt: str,
        generation_params: dict[str, Any],
        task: LocalProbeTask | None = None,
    ) -> GenerationResult:
        del model_id, generation_params
        started = time.perf_counter()
        query_id = task.query_id if task is not None else prompt
        digest = hashlib.sha256(str(query_id).encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 101
        confidence = 0.2 + 0.6 * (bucket / 100.0)
        raw_output = f"Answer: dry_run_route_signal\nConfidence: {confidence:.2f}"
        return GenerationResult(
            raw_output=raw_output,
            latency_sec=max(time.perf_counter() - started, 0.0),
            tokens_input=_count_tokens(prompt),
            tokens_output=_count_tokens(raw_output),
        )


def run_aligned_probe_matrix(
    *,
    tasks: list[LocalProbeTask],
    model_ids: list[str],
    client,
    generation_params: dict[str, Any],
    model_revision: str = "",
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    raw_logs: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for task in tasks:
        prompt = render_probe_prompt(task)
        for model_id in model_ids:
            created_at = _now()
            try:
                result = client.generate(
                    model_id=model_id,
                    prompt=prompt,
                    generation_params=generation_params,
                    task=task,
                )
                raw_output = str(result.raw_output)
                parsed_answer = parse_probe_answer(raw_output)
                confidence = parse_probe_confidence(raw_output)
                error_type = ""
                error_message = ""
                latency_sec = float(result.latency_sec)
                input_tokens = int(result.tokens_input)
                output_tokens = int(result.tokens_output)
            except Exception as exc:
                raw_output = ""
                parsed_answer = ""
                confidence = math.nan
                error_type = type(exc).__name__
                error_message = str(exc)
                latency_sec = 0.0
                input_tokens = _count_tokens(prompt)
                output_tokens = 0
                errors.append(
                    {
                        "query_id": task.query_id,
                        "model_id": model_id,
                        "error_type": error_type,
                        "error_message": error_message,
                        "created_at": created_at,
                    }
                )
            row = {
                "query_id": str(task.query_id),
                "probe_id": f"{PROBE_TYPE}:{model_id}:{PROBE_TEMPLATE_ID}",
                "probe_type": PROBE_TYPE,
                "probe_model_id": str(model_id),
                "prompt_template": PROBE_TEMPLATE_ID,
                "generation_params_json": json.dumps(generation_params, sort_keys=True),
                "raw_probe_output": raw_output,
                "parsed_probe_answer": parsed_answer,
                "self_confidence": confidence,
                "logprob_mean": math.nan,
                "entropy_proxy": _entropy_proxy(confidence),
                "agreement_score": math.nan,
                "knn_label_entropy": math.nan,
                "knn_winner_entropy": math.nan,
                "latency_sec": latency_sec,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "probe_cost_proxy": float(latency_sec + 0.001 * output_tokens),
                "error_type": error_type,
                "error_message": error_message,
                "created_at": created_at,
            }
            rows.append(row)
            raw_logs.append(
                {
                    **row,
                    "query_text": task.query_text,
                    "dataset": task.dataset,
                    "domain": task.domain,
                    "prompt": prompt,
                    "model_revision": model_revision,
                }
            )
    features = pd.DataFrame(rows, columns=PROBE_FEATURE_COLUMNS)
    if not features.empty:
        features["agreement_score"] = _agreement_scores(features)
    return features, raw_logs, errors


def render_probe_prompt(task: LocalProbeTask) -> str:
    return (
        "You are a cheap routing probe. Read the query and emit one short diagnostic answer "
        "plus a calibrated confidence in whether the query has a clear routing state.\n\n"
        "Return exactly these fields:\n"
        "Answer: <short diagnostic answer>\n"
        "Confidence: <number between 0 and 1>\n\n"
        f"Dataset: {task.dataset}\n"
        f"Domain: {task.domain}\n"
        f"Query: {task.query_text}\n"
    )


def parse_probe_confidence(raw_output: str) -> float:
    match = CONFIDENCE_RE.search(str(raw_output))
    if not match:
        return math.nan
    return min(max(float(match.group(1)), 0.0), 1.0)


def parse_probe_answer(raw_output: str) -> str:
    match = ANSWER_RE.search(str(raw_output))
    if match:
        return match.group(1).strip()
    return str(raw_output).strip().splitlines()[0].strip() if str(raw_output).strip() else ""


def _agreement_scores(features: pd.DataFrame) -> pd.Series:
    values = pd.Series(math.nan, index=features.index, dtype=float)
    for _, group in features.groupby("query_id", sort=False):
        answers = [str(value) for value in group["parsed_probe_answer"] if str(value)]
        counts = Counter(answers)
        denominator = len(answers)
        for row_idx, row in group.iterrows():
            answer = str(row["parsed_probe_answer"])
            values.loc[row_idx] = float(counts[answer] / denominator) if answer and denominator else math.nan
    return values


def _entropy_proxy(confidence: float) -> float:
    if math.isnan(confidence):
        return math.nan
    return float(1.0 - confidence)


def _count_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def _now() -> str:
    return datetime.now(UTC).isoformat()
