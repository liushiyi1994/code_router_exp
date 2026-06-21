from __future__ import annotations

import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COLUMNS = ["query_id", "query_text", "split", "benchmark", "domain"]


def build_local_behavior_probe_features(
    outputs: pd.DataFrame,
    *,
    local_models: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build deployable active-probe features from cached local model behavior.

    The features intentionally avoid `quality_score`, gold answers, utility, and
    frontier model outputs. They approximate what the router can observe after
    running cheap local probes: answer validity, answer agreement, output length,
    latency, and local-model disagreement patterns.
    """

    required = {"query_id", "query_text", "model_id", "parsed_answer", "status"}
    missing = required.difference(outputs.columns)
    if missing:
        raise ValueError(f"outputs missing required columns: {sorted(missing)}")
    frame = outputs.copy()
    frame["query_id"] = frame["query_id"].astype(str)
    frame["model_id"] = frame["model_id"].astype(str)
    if local_models is None:
        if "is_local" in frame.columns:
            local_models = sorted(frame[frame["is_local"].astype(bool)]["model_id"].astype(str).unique())
        else:
            local_models = sorted(model for model in frame["model_id"].unique() if "local" in str(model))
    local_models = tuple(str(model) for model in local_models)
    local = frame[frame["model_id"].isin(local_models)].copy()
    meta_cols = [col for col in ID_COLUMNS if col in frame.columns]
    meta = frame.sort_values(["query_id", "model_id"]).groupby("query_id", as_index=False).first()[meta_cols]

    rows: list[dict[str, Any]] = []
    for query_id, group in local.groupby("query_id", sort=False):
        answers = {
            str(row.model_id): normalize_probe_answer(getattr(row, "parsed_answer", ""))
            for row in group.itertuples(index=False)
        }
        by_model = {str(row.model_id): row for row in group.itertuples(index=False)}
        valid_answers = [answer for answer in answers.values() if answer]
        counts = Counter(valid_answers)
        top_counts = counts.most_common()
        top = top_counts[0][1] if top_counts else 0
        second = top_counts[1][1] if len(top_counts) > 1 else 0
        valid = len(valid_answers)
        output_tokens = np.asarray(
            [float(getattr(row, "output_tokens", 0.0) or 0.0) for row in by_model.values()],
            dtype=float,
        )
        latency = np.asarray(
            [float(getattr(row, "latency_s", 0.0) or 0.0) for row in by_model.values()],
            dtype=float,
        )
        raw_texts = {
            model_id: raw_probe_text(getattr(model_row, "raw_output_path", ""))
            for model_id, model_row in by_model.items()
        }
        raw_stats = [raw_text_features(text) for text in raw_texts.values()]
        raw_chars = np.asarray([stats["raw_chars"] for stats in raw_stats], dtype=float)
        raw_uncertainty = np.asarray([stats["raw_uncertainty_markers"] for stats in raw_stats], dtype=float)
        raw_refusal = np.asarray([stats["raw_refusal_markers"] for stats in raw_stats], dtype=float)
        raw_final = np.asarray([stats["raw_final_markers"] for stats in raw_stats], dtype=float)
        raw_numeric = np.asarray([stats["raw_numeric_tokens"] for stats in raw_stats], dtype=float)
        raw_code = np.asarray([stats["raw_code_markers"] for stats in raw_stats], dtype=float)
        row = {
            "query_id": str(query_id),
            "probe_local_model_count": float(len(local_models)),
            "probe_observed_model_count": float(len(by_model)),
            "probe_success_count": float(
                sum(str(getattr(model_row, "status", "")) == "success" for model_row in by_model.values())
            ),
            "probe_valid_answer_count": float(valid),
            "probe_missing_answer_count": float(max(0, len(local_models) - valid)),
            "probe_unique_answer_count": float(len(counts)),
            "probe_top_vote_count": float(top),
            "probe_second_vote_count": float(second),
            "probe_vote_frac": float(top / valid) if valid else 0.0,
            "probe_vote_margin": float((top - second) / valid) if valid else 0.0,
            "probe_vote_entropy": answer_entropy(counts),
            "probe_all_agree": float(valid > 0 and len(counts) == 1),
            "probe_any_disagree": float(valid > 1 and len(counts) > 1),
            "probe_output_tokens_mean": safe_stat(output_tokens, np.mean),
            "probe_output_tokens_std": safe_stat(output_tokens, np.std),
            "probe_output_tokens_max": safe_stat(output_tokens, np.max),
            "probe_latency_mean": safe_stat(latency, np.mean),
            "probe_latency_std": safe_stat(latency, np.std),
            "probe_latency_max": safe_stat(latency, np.max),
            "probe_raw_chars_mean": safe_stat(raw_chars, np.mean),
            "probe_raw_chars_std": safe_stat(raw_chars, np.std),
            "probe_raw_chars_max": safe_stat(raw_chars, np.max),
            "probe_raw_uncertainty_mean": safe_stat(raw_uncertainty, np.mean),
            "probe_raw_uncertainty_max": safe_stat(raw_uncertainty, np.max),
            "probe_raw_refusal_mean": safe_stat(raw_refusal, np.mean),
            "probe_raw_refusal_max": safe_stat(raw_refusal, np.max),
            "probe_raw_final_marker_mean": safe_stat(raw_final, np.mean),
            "probe_raw_numeric_mean": safe_stat(raw_numeric, np.mean),
            "probe_raw_code_marker_mean": safe_stat(raw_code, np.mean),
        }
        for model_id in local_models:
            short = short_model_name(model_id)
            model_row = by_model.get(model_id)
            answer = answers.get(model_id, "")
            raw_feature = raw_text_features(raw_texts.get(model_id, ""))
            row[f"probe_{short}_valid"] = float(bool(answer))
            row[f"probe_{short}_answer_chars"] = float(len(answer))
            row[f"probe_{short}_success"] = float(
                bool(model_row is not None and str(getattr(model_row, "status", "")) == "success")
            )
            row[f"probe_{short}_output_tokens"] = (
                float(getattr(model_row, "output_tokens", 0.0) or 0.0) if model_row is not None else 0.0
            )
            row[f"probe_{short}_latency_s"] = (
                float(getattr(model_row, "latency_s", 0.0) or 0.0) if model_row is not None else 0.0
            )
            for key, value in raw_feature.items():
                row[f"probe_{short}_{key}"] = float(value)
        for left, right in combinations(local_models, 2):
            left_answer = answers.get(left, "")
            right_answer = answers.get(right, "")
            key = f"probe_pair_agree::{short_model_name(left)}::{short_model_name(right)}"
            row[key] = float(bool(left_answer and right_answer and left_answer == right_answer))
        rows.append(row)

    features = meta.merge(pd.DataFrame(rows), on="query_id", how="left")
    numeric = [col for col in features.columns if col not in ID_COLUMNS]
    features[numeric] = features[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    features[numeric] = features[numeric].fillna(0.0)
    features = add_text_shape_features(features)
    return features.sort_values("query_id").reset_index(drop=True)


def add_text_shape_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    if "query_text" not in out.columns:
        return out
    text = out["query_text"].fillna("").astype(str)
    out["text_chars"] = text.str.len().astype(float)
    out["text_words"] = text.str.split().map(len).astype(float)
    out["text_digit_count"] = text.str.count(r"\d").astype(float)
    out["text_math_symbol_count"] = text.str.count(r"[=+\-*/^<>]|\\frac|\\sqrt|\\sum|\\int").astype(float)
    out["text_code_marker_count"] = text.str.count(
        r"\b(def|class|return|import|for|while|function|array|string|python|java|sql)\b",
        flags=re.IGNORECASE,
    ).astype(float)
    out["text_option_marker_count"] = text.str.count(r"(?m)^\s*[A-E][\).:]").astype(float)
    out["text_newline_count"] = text.str.count(r"\n").astype(float)
    return out


def normalize_probe_answer(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;")
    return text


def raw_probe_text(path_value: Any) -> str:
    """Return observable generated text from a cached local raw output file."""

    if path_value is None or (isinstance(path_value, float) and np.isnan(path_value)):
        return ""
    path = Path(str(path_value))
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ""
    parsed = payload.get("_parsed_text")
    if isinstance(parsed, str):
        return parsed
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return str(content or "")


def raw_text_features(text: str) -> dict[str, float]:
    value = str(text or "")
    lower = value.lower()
    return {
        "raw_chars": float(len(value)),
        "raw_lines": float(value.count("\n") + (1 if value else 0)),
        "raw_numeric_tokens": float(len(re.findall(r"[-+]?\d+(?:\.\d+)?", value))),
        "raw_uncertainty_markers": float(
            len(re.findall(r"\b(maybe|perhaps|likely|unsure|not sure|cannot determine|unclear)\b", lower))
        ),
        "raw_refusal_markers": float(
            len(re.findall(r"\b(cannot answer|can't answer|unable to|i cannot|i can't|insufficient information)\b", lower))
        ),
        "raw_final_markers": float(
            len(re.findall(r"\b(answer|final|therefore|thus)\b|\\boxed|\\textbf|\\mathrm", lower))
        ),
        "raw_code_markers": float(
            len(re.findall(r"```|\b(def|class|return|import|function|console\.log|public static)\b", lower))
        ),
    }


def answer_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.asarray([count / total for count in counts.values()], dtype=float)
    return float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())


def safe_stat(values: np.ndarray, fn) -> float:
    if values.size == 0:
        return 0.0
    return float(fn(values))


def short_model_name(model_id: str) -> str:
    text = str(model_id).replace("-local", "").replace("-awq", "")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return text or "model"


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    return [
        col
        for col in features.columns
        if col not in ID_COLUMNS and pd.api.types.is_numeric_dtype(features[col])
    ]
