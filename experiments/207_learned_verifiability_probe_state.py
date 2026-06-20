from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


K_VALUES = [2, 4, 8, 16]
PROBABILITY_QUANTILES = [0.75, 0.85, 0.90, 0.95]
TOOL_MODEL_ID = "deterministic_math_tool"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Learn a benchmark-agnostic verifiability state from cached broad local behavior, "
            "then use it for local-vs-large routing without direct benchmark lookup or direct tool flags."
        )
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path(
            "results/controlled/broad100_vllm_self_consistency_probe/"
            "model_outputs_with_self_consistency.parquet"
        ),
    )
    parser.add_argument(
        "--probe-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_learned_verifiability_probe_state"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166_for_207")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_composed_171_for_207")
    exp206 = load_module("experiments/206_probe_state_composed_yesno_policy.py", "probe_state_composed_206_for_207")

    outputs = pd.read_parquet(args.outputs)
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, outputs)
    target = merge_probe_features(target, pd.read_csv(args.probe_features))
    target = add_generic_text_features(target)
    feature_columns = generic_verifiability_features(target)

    scored, classifier_summary = fit_verifiability_models(target, feature_columns, seed=int(args.seed))
    table, score_table, assignments, state_cards = evaluate_learned_verifiability(
        scored,
        rc166,
        exp171,
        exp206,
        lambda_cost=float(args.lambda_cost),
        seed=int(args.seed),
    )
    selected = exp206.selected_rows(table, rc166, int(args.bootstrap_samples), int(args.seed))

    score_table.to_csv(args.output_dir / "table_learned_verifiability_scores.csv", index=False)
    classifier_summary.to_csv(args.output_dir / "table_learned_verifiability_classifier_summary.csv", index=False)
    table.to_csv(args.output_dir / "table_learned_verifiability_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_learned_verifiability_policy_selected.csv", index=False)
    assignments.to_csv(args.output_dir / "table_learned_verifiability_assignments.csv", index=False)
    state_cards.to_csv(args.output_dir / "table_learned_verifiability_code_cards.csv", index=False)
    write_code_cards_md(args.output_dir / "learned_verifiability_code_cards.md", state_cards)
    write_figure(args.output_dir, table)
    write_memo(
        args.output_dir / "LEARNED_VERIFIABILITY_PROBE_STATE_MEMO.md",
        args,
        feature_columns,
        classifier_summary,
        table,
        selected,
    )
    print(f"Wrote learned verifiability probe-state results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def merge_probe_features(target: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    extra_cols = []
    for col in features.columns:
        if col == "query_id" or col in target.columns:
            continue
        lower = col.lower()
        if lower.startswith("tool_") or "benchmark" in lower or lower in {"domain", "metric", "split"}:
            continue
        if pd.api.types.is_numeric_dtype(features[col]):
            extra_cols.append(col)
    return target.merge(features[["query_id", *extra_cols]], on="query_id", how="left")


def add_generic_text_features(target: pd.DataFrame) -> pd.DataFrame:
    out = target.copy()
    text = out["query_text"].fillna("").astype(str)
    out["text_query_chars"] = text.str.len().astype(float)
    out["text_query_words"] = text.str.split().map(len).astype(float)
    out["text_digit_count"] = text.str.count(r"\d").astype(float)
    out["text_digit_frac"] = out["text_digit_count"] / out["text_query_chars"].clip(lower=1.0)
    out["text_numeric_token_count"] = text.str.count(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?").astype(float)
    out["text_option_marker_count"] = text.str.count(r"(?m)^\s*[A-E][\).:]").astype(float)
    out["text_has_multiple_choice_form"] = (out["text_option_marker_count"] >= 4).astype(float)
    out["text_math_symbol_count"] = text.str.count(r"[=+\-*/^<>]|\\frac|\\sqrt|\\sum|\\int").astype(float)
    out["text_code_marker_count"] = text.str.count(
        r"\b(def|class|return|import|for|while|function|array|string|Python|Java|C\+\+|SQL)\b"
    ).astype(float)
    out["text_has_code_form"] = (out["text_code_marker_count"] >= 2).astype(float)
    out["text_answer_form_marker_count"] = text.str.count(
        r"\b(final answer|answer is|choose|which of|what is|compute|solve|prove|return)\b"
    ).astype(float)
    out["text_newline_count"] = text.str.count(r"\n").astype(float)
    return out


def generic_verifiability_features(target: pd.DataFrame) -> list[str]:
    blocked_exact = {
        "tool_available",
        "signal_query_train_prior_need_large",
        "signal_query_answerability_risk",
        "signal_query_answerability",
        "signal_vllm_answerability_score",
        "local_quality",
        "large_quality",
        "local_utility",
        "large_utility",
        "delta_large",
        "local_normalized_cost",
        "large_normalized_cost",
        "local_cost_usd",
        "large_cost_usd",
        "local_latency_s",
        "large_latency_s",
    }
    blocked_prefixes = ("tool_", "q32_choice_", "qwen32b_")
    allowed_prefixes = (
        "signal_constrained_",
        "signal_constrained_plus_cached_",
        "signal_early_rollout_",
        "signal_semantic_",
        "signal_slm_",
        "signal_medium_",
        "signal_vllm_answerability_score",
        "self_",
        "local_",
        "small_",
        "medium_",
        "q4_",
        "q8_",
        "q14_",
        "sc_",
        "answer_chars_",
        "output_tokens_",
        "is_multiple_choice_prompt",
        "is_exact_answer_prompt",
        "text_",
    )
    cols: list[str] = []
    for col in target.columns:
        lower = col.lower()
        if col in blocked_exact or lower.startswith(blocked_prefixes):
            continue
        if not pd.api.types.is_numeric_dtype(target[col]):
            continue
        if not target[col].notna().any():
            continue
        if any(col.startswith(prefix) for prefix in allowed_prefixes):
            cols.append(col)
    return sorted(dict.fromkeys(cols))


def fit_verifiability_models(
    target: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = target.copy()
    y_train = out.loc[out["split"].eq("train"), "tool_available"].astype(bool).to_numpy()
    x_train = out.loc[out["split"].eq("train"), feature_columns]
    x_all = out[feature_columns]
    specs = classifier_specs(seed)
    rows: list[dict[str, Any]] = []
    for name, model in specs.items():
        model.fit(x_train, y_train)
        score = model.predict_proba(x_all)[:, 1]
        out[f"pred_verifiability_score_{name}"] = score
        out[f"pred_verifiability_rank_{name}"] = pd.Series(score).rank(method="average", pct=True).to_numpy()
        for split in ["train", "val", "test"]:
            frame = out[out["split"].eq(split)].copy()
            labels = frame["tool_available"].astype(bool).to_numpy()
            scores = frame[f"pred_verifiability_score_{name}"].to_numpy(dtype=float)
            rows.append(
                {
                    "classifier": name,
                    "split": split,
                    "n_queries": int(len(frame)),
                    "positive_rate": float(np.mean(labels)) if len(labels) else np.nan,
                    "score_mean": float(np.mean(scores)) if len(scores) else np.nan,
                    "score_p90": float(np.quantile(scores, 0.90)) if len(scores) else np.nan,
                    "auroc": safe_auroc(labels, scores),
                    "auprc": safe_auprc(labels, scores),
                }
            )
    return out, pd.DataFrame(rows)


def classifier_specs(seed: int) -> dict[str, Pipeline]:
    return {
        "logreg_c0.3": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000, solver="liblinear", random_state=seed),
                ),
            ]
        ),
        "extratrees_d3_leaf8": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=3,
                        min_samples_leaf=8,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "gb_depth2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    GradientBoostingClassifier(
                        n_estimators=80,
                        learning_rate=0.05,
                        max_depth=2,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def evaluate_learned_verifiability(
    target: pd.DataFrame,
    rc166,
    exp171,
    exp206,
    *,
    lambda_cost: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_fns_all = exp171.candidate_policy_functions()
    policy_fns_no_tool = {name: fn for name, fn in policy_fns_all.items() if not str(name).startswith("tool_")}
    classifier_names = sorted(
        col.replace("pred_verifiability_score_", "")
        for col in target.columns
        if col.startswith("pred_verifiability_score_")
    )
    rows: list[dict[str, Any]] = []
    score_rows: list[pd.DataFrame] = []
    assignments: list[pd.DataFrame] = []
    cards: list[pd.DataFrame] = []

    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(frame, split=split, lambda_cost=lambda_cost))

    direct_tool = target.copy()
    for policy_name, policy_fn in policy_fns_all.items():
        if not policy_name.startswith("tool_"):
            continue
        for split in ["val", "test"]:
            frame = direct_tool[direct_tool["split"].eq(split)].copy()
            row = rc166.evaluate_decision(
                frame,
                policy_fn(frame),
                split=split,
                method=f"direct_tool_flag_{policy_name}",
                family="direct_tool_flag_positive_control",
                lambda_cost=lambda_cost,
            )
            row.update({"diagnostic": True})
            rows.append(row)

    for classifier in classifier_names:
        score_col = f"pred_verifiability_score_{classifier}"
        thresholds = candidate_score_thresholds(target[target["split"].eq("val")][score_col].to_numpy(dtype=float))
        for threshold in thresholds:
            scored = add_predicted_tool_flag(target, score_col, threshold)
            score_rows.append(
                scored[["query_id", "split", "benchmark", "tool_available", "pred_tool_available", score_col]].assign(
                    classifier=classifier,
                    threshold=float(threshold),
                )
            )
            for policy_name, policy_fn in policy_fns_all.items():
                for split in ["val", "test"]:
                    frame = scored[scored["split"].eq(split)].copy()
                    row = rc166.evaluate_decision(
                        frame,
                        policy_fn(frame),
                        split=split,
                        method=f"{classifier}_thr{threshold:.4f}_{policy_name}",
                        family="learned_verifiability_global",
                        lambda_cost=lambda_cost,
                    )
                    row.update(
                        {
                            "classifier": classifier,
                            "threshold": float(threshold),
                            "policy_name": policy_name,
                            "diagnostic": False,
                            "uses_predicted_verifiability": bool(policy_name.startswith("tool_")),
                        }
                    )
                    rows.append(row)

            state_feature_cols = state_features_for_classifier(scored, classifier)
            for k in K_VALUES:
                labels, model = exp206.fit_predict_states(scored, state_feature_cols, k=int(k), seed=seed)
                state_frame = scored.assign(probe_state=labels)
                state_policy = exp206.choose_policy_by_state(
                    state_frame[state_frame["split"].eq("train")],
                    policy_fns_all,
                    rc166,
                    lambda_cost=lambda_cost,
                )
                method = f"{classifier}_thr{threshold:.4f}_state_k{k}"
                for split in ["val", "test"]:
                    frame = state_frame[state_frame["split"].eq(split)].copy()
                    choose_large = exp206.compose_by_state(frame, state_policy, policy_fns_all)
                    row = rc166.evaluate_decision(
                        frame,
                        choose_large,
                        split=split,
                        method=method,
                        family="learned_verifiability_state",
                        lambda_cost=lambda_cost,
                    )
                    row.update(
                        {
                            "classifier": classifier,
                            "threshold": float(threshold),
                            "k": int(k),
                            "diagnostic": False,
                            "state_policy_json": json.dumps(state_policy, sort_keys=True),
                        }
                    )
                    rows.append(row)
                assignments.append(
                    state_frame[
                        [
                            "query_id",
                            "split",
                            "benchmark",
                            "probe_state",
                            "need_large",
                            "tool_available",
                            "pred_tool_available",
                            "local_utility",
                            "large_utility",
                        ]
                    ].assign(method=method, family="learned_verifiability_state", classifier=classifier, threshold=float(threshold), k=int(k))
                )
                cards.append(build_cards(state_frame, state_policy, method, classifier, threshold, state_feature_cols, k=int(k)))

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    score_table = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    assignment_frame = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()
    card_frame = pd.concat(cards, ignore_index=True) if cards else pd.DataFrame()
    return table, score_table, assignment_frame, card_frame


def add_predicted_tool_flag(target: pd.DataFrame, score_col: str, threshold: float) -> pd.DataFrame:
    out = target.copy()
    out["pred_verifiability_score"] = out[score_col].astype(float)
    out["pred_tool_available"] = out["pred_verifiability_score"] >= float(threshold)
    # Candidate tool policies read the tool_available column. Preserve the true
    # label separately and replace the route-time signal with the learned flag.
    out["true_tool_available"] = out["tool_available"].astype(bool)
    out["tool_available"] = out["pred_tool_available"].astype(bool)
    return out


def candidate_score_thresholds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([0.5])
    quantiles = np.quantile(values, PROBABILITY_QUANTILES)
    positives = np.quantile(values, [0.85, 0.90, 0.925, 0.95, 0.975, 0.99])
    return np.unique(np.concatenate([quantiles, positives]))


def state_features_for_classifier(target: pd.DataFrame, classifier: str) -> list[str]:
    base = [
        col
        for col in generic_verifiability_features(target)
        if col in target.columns and not col.startswith("pred_verifiability_")
    ]
    return sorted(dict.fromkeys([*base, f"pred_verifiability_score_{classifier}", f"pred_verifiability_rank_{classifier}", "pred_tool_available"]))


def build_cards(
    target: pd.DataFrame,
    state_policy: dict[str, str],
    method: str,
    classifier: str,
    threshold: float,
    features: list[str],
    *,
    k: int,
) -> pd.DataFrame:
    train = target[target["split"].eq("train")].copy()
    global_means = target[features].mean(numeric_only=True)
    rows: list[dict[str, Any]] = []
    for state, group in target.groupby("probe_state", sort=True):
        train_group = train[train["probe_state"].eq(state)].copy()
        means = group[features].mean(numeric_only=True)
        diffs = (means - global_means).abs().sort_values(ascending=False).head(8)
        rows.append(
            {
                "method": method,
                "family": "learned_verifiability_state",
                "classifier": classifier,
                "threshold": float(threshold),
                "k": int(k),
                "probe_state": int(state),
                "n_all": int(len(group)),
                "n_train": int(len(train_group)),
                "chosen_policy": state_policy.get(str(int(state)), ""),
                "train_need_large_rate": float(train_group["need_large"].mean()) if len(train_group) else np.nan,
                "train_true_tool_available_rate": float(train_group.get("true_tool_available", train_group["tool_available"]).mean()) if len(train_group) else np.nan,
                "train_pred_tool_available_rate": float(train_group["pred_tool_available"].mean()) if len(train_group) else np.nan,
                "train_mean_local_utility": float(train_group["local_utility"].mean()) if len(train_group) else np.nan,
                "train_mean_large_utility": float(train_group["large_utility"].mean()) if len(train_group) else np.nan,
                "top_feature_diffs_json": json.dumps({name: round(float(value), 4) for name, value in diffs.items()}, sort_keys=True),
                "benchmark_mix_json": json.dumps(group["benchmark"].value_counts().sort_index().to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def write_code_cards_md(path: Path, cards: pd.DataFrame) -> None:
    if cards.empty:
        path.write_text("# Learned Verifiability Probe-State Code Cards\n\n_No cards._\n", encoding="utf-8")
        return
    lines = ["# Learned Verifiability Probe-State Code Cards", ""]
    selected = cards.sort_values(["method", "probe_state"]).groupby("method", sort=False).head(32)
    for method, subset in selected.groupby("method", sort=False):
        lines.extend([f"## {method}", ""])
        for row in subset.sort_values("probe_state").to_dict("records"):
            lines.extend(
                [
                    f"### State {row['probe_state']}",
                    "",
                    f"- Chosen policy: `{row['chosen_policy']}`",
                    f"- Train rows: `{row['n_train']}`; all rows: `{row['n_all']}`",
                    f"- Train need-large rate: `{float(row['train_need_large_rate']):.4f}`",
                    f"- Train true verifiable rate: `{float(row['train_true_tool_available_rate']):.4f}`",
                    f"- Train predicted verifiable rate: `{float(row['train_pred_tool_available_rate']):.4f}`",
                    f"- Top feature shifts: `{row['top_feature_diffs_json']}`",
                    f"- Benchmark mix: `{row['benchmark_mix_json']}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(20)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str).str.slice(0, 60)
    fig, ax = plt.subplots(figsize=(11, 7.0))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#5b7f71")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Learned Verifiability Probe-State Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_learned_verifiability_probe_state_utility.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    feature_columns: list[str],
    classifier_summary: pd.DataFrame,
    table: pd.DataFrame,
    selected: pd.DataFrame,
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
        "selection_rule",
        "diagnostic",
        "classifier",
        "threshold",
        "k",
    ]
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    lines = [
        "# Learned Verifiability Probe-State Policy",
        "",
        "This cached experiment tries to convert the tool-aware positive control into a learned broad probe signal.",
        "",
        "```text",
        "query + cheap local behavior -> learned verifiability state -> local-vs-large policy",
        "```",
        "",
        "Main learned rows use train-only verifiability labels and do not expose direct benchmark ID, benchmark train priors, or direct tool availability at validation/test time.",
        "The direct tool-flag rows are positive-control diagnostics. No GPT, Gemini, Claude, local generation, or vLLM calls are made.",
        "",
        "Important caveat: this is still a local-vs-large abstraction using cached best-local and best-large actions; it is not yet a full concrete deployed router.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/207_learned_verifiability_probe_state.py",
        f"PYTHONPATH=src python experiments/207_learned_verifiability_probe_state.py --target-table {args.target_table} --outputs {args.outputs} --probe-features {args.probe_features} --output-dir {args.output_dir}",
        "```",
        "",
        "## Feature Policy",
        "",
        f"- Number of generic learned-verifiability features: `{len(feature_columns)}`",
        "- Blocked from main features: benchmark ID, domain, metric, train benchmark prior, outcome utility/quality/cost columns, direct tool flags, and direct tool output features.",
        "",
        "## Classifier Summary",
        "",
        "```csv",
        classifier_summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Validation-Selected Rows",
        "",
        "```csv",
        selected[[col for col in selected_cols if col in selected.columns]].to_csv(index=False).strip() if not selected.empty else "",
        "```",
        "",
        "## Target Gate Check",
        "",
        *target_gate_lines(selected),
        "",
        "## Best Held-Out Rows",
        "",
        "```csv",
        top_test[
            [
                "method",
                "family",
                "split",
                "n_queries",
                "mean_quality",
                "mean_utility",
                "oracle_utility_ratio",
                "large_call_rate",
                "frontier_call_rate",
                "need_large_precision",
                "need_large_recall",
                "diagnostic",
            ]
        ].to_csv(index=False).strip(),
        "```",
        "",
        "## Interpretation",
        "",
        "- If learned rows approach the direct tool-flag positive control, the verifiability signal is observable from broad local behavior.",
        "- If learned rows miss while direct tool rows pass, the useful positive-control signal is still not captured by benchmark-agnostic cached features.",
        "- Direct tool-flag rows must remain diagnostics because they expose the exact tool availability signal at route time.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def target_gate_lines(selected: pd.DataFrame) -> list[str]:
    if selected.empty:
        return ["- No selected rows."]
    rows = selected[
        selected["split"].eq("test")
        & selected["selection_rule"].astype(str).isin(["val_target_gate_test", "val_best_utility_test"])
    ].copy()
    if rows.empty:
        rows = selected[selected["split"].eq("test")].copy()
    if rows.empty:
        return ["- No held-out rows."]
    out: list[str] = []
    for row in rows.to_dict("records"):
        utility_target = 0.95 * float(row["local_large_oracle_mean_utility"])
        quality_target = float(row["local_large_oracle_mean_quality"]) - 0.03
        out.append(
            (
                f"- `{row['method']}` ({row['family']}, {row['selection_rule']}): "
                f"utility `{float(row['mean_utility']):.4f}` vs target `{utility_target:.4f}`; "
                f"quality `{float(row['mean_quality']):.4f}` vs target `{quality_target:.4f}`; "
                f"frontier rate `{float(row['frontier_call_rate']):.4f}`."
            )
        )
    return out


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


if __name__ == "__main__":
    main()
