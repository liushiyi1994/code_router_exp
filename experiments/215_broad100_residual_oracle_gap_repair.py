from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cached Broad100 residual oracle-gap repair over the learned-verifiability "
            "target method. This makes no provider, vLLM, or local generation calls."
        )
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
        "--target-table",
        type=Path,
        default=Path("results/controlled/broad100_constrained_yesno_probe_qwen14b/table_constrained_yesno_targets.csv"),
    )
    parser.add_argument(
        "--probe-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--learned-scores",
        type=Path,
        default=Path(
            "results/controlled/broad100_learned_verifiability_probe_state/"
            "table_learned_verifiability_scores.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_residual_oracle_gap_repair"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--frontier-cap", type=float, default=0.40)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "exp171_for_215")
    exp207 = load_module("experiments/207_learned_verifiability_probe_state.py", "exp207_for_215")
    exp213 = load_module("experiments/213_broad100_target_method_package.py", "exp213_for_215")

    target, feature_columns = build_dataset(exp171, exp207, exp213, args)
    all_rows, detail_rows = run_residual_sweep(exp213, target, feature_columns, args)
    selected = select_rows(all_rows, float(args.frontier_cap))
    selected_details = selected_query_details(detail_rows, selected)

    all_rows.to_csv(args.output_dir / "table_residual_oracle_gap_repair_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_residual_oracle_gap_repair_selected.csv", index=False)
    selected_details.to_csv(args.output_dir / "table_residual_oracle_gap_repair_query_choices.csv", index=False)
    write_figure(args.output_dir, all_rows, selected)
    write_memo(args.output_dir / "RESIDUAL_ORACLE_GAP_REPAIR_MEMO.md", args, all_rows, selected, feature_columns)
    print(f"Wrote residual oracle-gap repair results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_dataset(exp171: Any, exp207: Any, exp213: Any, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    base_target = pd.read_csv(args.target_table)
    full_target = exp213.rebuild_target_pool(
        base_target,
        outputs,
        exp213.FULL_LOCAL_ACTIONS,
        exp213.LARGE_ACTIONS,
        float(args.lambda_cost),
    )

    features = pd.read_csv(args.probe_features)
    feature_extra = features.drop(
        columns=[col for col in ["split", "benchmark", "domain", "metric", "query_text"] if col in features.columns],
        errors="ignore",
    )
    scores = pd.read_csv(args.learned_scores)
    score_cols = [col for col in scores.columns if col.startswith("pred_verifiability_score_")]
    score_wide = scores[["query_id", *score_cols]].groupby("query_id", as_index=False).max()

    target = full_target.merge(feature_extra, on="query_id", how="left").merge(score_wide, on="query_id", how="left")
    target = exp207.add_generic_text_features(target)
    target = add_base_decision(exp171, exp213, target, scores)
    target["oracle_choose_large"] = target["large_utility"].astype(float) >= target["local_utility"].astype(float)
    target["base_utility"] = np.where(target["base_choose_large"], target["large_utility"], target["local_utility"])
    target["alt_utility"] = np.where(target["base_choose_large"], target["local_utility"], target["large_utility"])
    target["flip_gain"] = target["alt_utility"].astype(float) - target["base_utility"].astype(float)
    target["delta_large"] = target["large_utility"].astype(float) - target["local_utility"].astype(float)
    return target, residual_feature_columns(target)


def add_base_decision(exp171: Any, exp213: Any, target: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    policy_fns = exp171.candidate_policy_functions()
    spec = exp213.parse_global_method(exp213.SELECTED_GLOBAL_METHOD)
    flags = exp213.predicted_verifiability_flags(scores, spec["classifier"], spec["threshold"])
    work = exp213.merge_route_signal(target, flags)
    base_by_index: dict[int, bool] = {}
    for _split, frame in work.groupby("split", sort=False):
        decisions = np.asarray(policy_fns[spec["policy_name"]](frame.copy()), dtype=bool)
        base_by_index.update({int(idx): bool(value) for idx, value in zip(frame.index, decisions)})
    work["base_choose_large"] = work.index.map(base_by_index).astype(bool)
    return work


def residual_feature_columns(target: pd.DataFrame) -> list[str]:
    blocked_exact = {
        "tool_available",
        "need_large",
        "need_large_positive_gain",
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
        "base_utility",
        "alt_utility",
        "flip_gain",
        "oracle_choose_large",
    }
    blocked_names = {
        "query_id",
        "query_text",
        "gold_answer",
        "best_local_action",
        "best_large_action",
        "split",
        "benchmark",
        "domain",
        "metric",
        "slm_answer",
        "medium14_answer",
        "medium32_answer",
        "self_majority_answer",
        "self_answer_norms_json",
    }
    allowed_prefixes = (
        "signal_",
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
        "is_",
        "text_",
        "pred_verifiability_",
    )
    cols: list[str] = []
    for col in target.columns:
        if col in blocked_exact or col in blocked_names:
            continue
        if col == "base_choose_large":
            cols.append(col)
            continue
        if not (pd.api.types.is_numeric_dtype(target[col]) or pd.api.types.is_bool_dtype(target[col])):
            continue
        if any(col.startswith(prefix) for prefix in allowed_prefixes):
            if target[col].notna().any():
                cols.append(col)
    return sorted(dict.fromkeys(cols))


def run_residual_sweep(
    exp213: Any,
    target: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = target[target["split"].eq("train")].copy()
    val = target[target["split"].eq("val")].copy()
    test = target[target["split"].eq("test")].copy()
    rows: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []

    for split, frame in [("val", val), ("test", test)]:
        row, detail = exp213.evaluate_policy(
            frame,
            frame["base_choose_large"].to_numpy(dtype=bool),
            oracle_reference=frame,
            split=split,
            method="base_learned_verifiability_global",
            family="base_reference",
            action_pool_variant="full_action_pool",
            lambda_cost=float(args.lambda_cost),
        )
        rows.append(row | {"model": "base", "target_kind": "base", "threshold": np.nan, "cap": np.nan})
        details.append(detail)
        oracle_decision = frame["oracle_choose_large"].to_numpy(dtype=bool)
        row, detail = exp213.evaluate_policy(
            frame,
            oracle_decision,
            oracle_reference=frame,
            split=split,
            method="oracle_local_vs_large_gate",
            family="diagnostic_oracle",
            action_pool_variant="full_action_pool",
            lambda_cost=float(args.lambda_cost),
        )
        rows.append(row | {"model": "oracle", "target_kind": "oracle", "threshold": np.nan, "cap": np.nan})
        details.append(detail)

    x_train = train[feature_columns]
    x_val = val[feature_columns]
    x_test = test[feature_columns]
    for name, target_kind, model in model_specs(int(args.seed)):
        if target_kind == "flip":
            y_train = train["flip_gain"].astype(float).gt(1e-12).astype(int)
            model.fit(x_train, y_train)
            val_scores = model.predict_proba(x_val)[:, 1]
            test_scores = model.predict_proba(x_test)[:, 1]
            thresholds = quantile_thresholds(val_scores)
            caps: list[float | None] = [None, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.0]
            for threshold in thresholds:
                for cap in caps:
                    for split, frame, scores in [("val", val, val_scores), ("test", test, test_scores)]:
                        flip = scores >= float(threshold)
                        if cap is not None:
                            flip &= top_cap_mask(scores, float(cap))
                        choose = np.where(
                            flip,
                            ~frame["base_choose_large"].to_numpy(dtype=bool),
                            frame["base_choose_large"].to_numpy(dtype=bool),
                        )
                        method = f"{name}_thr{float(threshold):.4f}_cap{cap}"
                        row, detail = exp213.evaluate_policy(
                            frame,
                            choose,
                            oracle_reference=frame,
                            split=split,
                            method=method,
                            family="residual_flip",
                            action_pool_variant="full_action_pool",
                            lambda_cost=float(args.lambda_cost),
                        )
                        row.update({"model": name, "target_kind": target_kind, "threshold": float(threshold), "cap": cap})
                        rows.append(row)
                        details.append(detail)
        elif target_kind == "oracle_gate":
            y_train = train["oracle_choose_large"].astype(bool).astype(int)
            model.fit(x_train, y_train)
            val_scores = model.predict_proba(x_val)[:, 1]
            test_scores = model.predict_proba(x_test)[:, 1]
            for threshold in quantile_thresholds(val_scores):
                for split, frame, scores in [("val", val, val_scores), ("test", test, test_scores)]:
                    choose = scores >= float(threshold)
                    method = f"{name}_thr{float(threshold):.4f}"
                    row, detail = exp213.evaluate_policy(
                        frame,
                        choose,
                        oracle_reference=frame,
                        split=split,
                        method=method,
                        family="direct_oracle_gate_model",
                        action_pool_variant="full_action_pool",
                        lambda_cost=float(args.lambda_cost),
                    )
                    row.update({"model": name, "target_kind": target_kind, "threshold": float(threshold), "cap": np.nan})
                    rows.append(row)
                    details.append(detail)
        elif target_kind == "delta":
            y_train = train["delta_large"].astype(float)
            model.fit(x_train, y_train)
            val_scores = model.predict(x_val)
            test_scores = model.predict(x_test)
            for threshold in quantile_thresholds(val_scores):
                for split, frame, scores in [("val", val, val_scores), ("test", test, test_scores)]:
                    choose = scores >= float(threshold)
                    method = f"{name}_thr{float(threshold):.4f}"
                    row, detail = exp213.evaluate_policy(
                        frame,
                        choose,
                        oracle_reference=frame,
                        split=split,
                        method=method,
                        family="predicted_delta_gate_model",
                        action_pool_variant="full_action_pool",
                        lambda_cost=float(args.lambda_cost),
                    )
                    row.update({"model": name, "target_kind": target_kind, "threshold": float(threshold), "cap": np.nan})
                    rows.append(row)
                    details.append(detail)
        else:
            raise ValueError(target_kind)

    table = exp213.add_target_gates(pd.DataFrame(rows))
    return table, pd.concat(details, ignore_index=True)


def model_specs(seed: int) -> list[tuple[str, str, Pipeline]]:
    return [
        (
            "logit_flip_c0.1",
            "flip",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.1,
                            class_weight="balanced",
                            max_iter=2000,
                            solver="liblinear",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "et_flip_leaf4",
            "flip",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            min_samples_leaf=4,
                            class_weight="balanced",
                            random_state=seed + 1,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "logit_oracle_c0.1",
            "oracle_gate",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.1,
                            class_weight="balanced",
                            max_iter=2000,
                            solver="liblinear",
                            random_state=seed + 2,
                        ),
                    ),
                ]
            ),
        ),
        (
            "et_oracle_leaf4",
            "oracle_gate",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=400,
                            min_samples_leaf=4,
                            class_weight="balanced",
                            random_state=seed + 3,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "ridge_delta",
            "delta",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
        ),
        (
            "et_delta_leaf8",
            "delta",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=400,
                            min_samples_leaf=8,
                            random_state=seed + 4,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "gb_delta_leaf8",
            "delta",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            random_state=seed + 5,
                            max_depth=2,
                            min_samples_leaf=8,
                            n_estimators=100,
                        ),
                    ),
                ]
            ),
        ),
    ]


def quantile_thresholds(scores: np.ndarray) -> list[float]:
    clean = np.asarray(scores, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return [0.0]
    return sorted(set(float(value) for value in np.quantile(clean, np.linspace(0.0, 1.0, 61))))


def top_cap_mask(scores: np.ndarray, cap: float) -> np.ndarray:
    mask = np.zeros(len(scores), dtype=bool)
    if len(scores) == 0:
        return mask
    order = np.argsort(np.where(np.isfinite(scores), scores, -np.inf))[::-1]
    mask[order[: max(1, int(float(cap) * len(scores)))]] = True
    return mask


def select_rows(table: pd.DataFrame, frontier_cap: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    oracle_rows = table[table["method"].eq("oracle_local_vs_large_gate")].copy()
    if not oracle_rows.empty:
        for split in ["val", "test"]:
            row = oracle_rows[oracle_rows["split"].eq(split)].head(1).copy()
            if not row.empty:
                rows.append(row.assign(selection_rule=f"diagnostic_oracle_{split}"))

    selectable = table[
        ~table["family"].astype(str).str.contains("oracle", case=False, na=False)
        & ~table["method"].astype(str).str.contains("oracle_local_vs_large_gate", na=False)
    ].copy()
    val = selectable[selectable["split"].eq("val")].copy()
    base_val = val[val["method"].eq("base_learned_verifiability_global")].head(1)
    base_recall = float(base_val.iloc[0]["need_large_recall"]) if not base_val.empty else 0.0
    if not base_val.empty:
        base = base_val.iloc[0]
        tethered = val[
            val["family"].astype(str).eq("residual_flip")
            & (val["mean_utility"].astype(float) >= float(base["mean_utility"]) - 1e-12)
            & (val["mean_quality"].astype(float) >= float(base["mean_quality"]) - 1e-12)
            & (val["frontier_call_rate"].astype(float) < float(base["frontier_call_rate"]) - 1e-12)
            & (val["large_call_rate"].astype(float) <= float(base["large_call_rate"]) + 1e-12)
            & (val["need_large_recall"].astype(float) >= float(base["need_large_recall"]) - 0.01)
        ].copy()
    else:
        tethered = val.iloc[0:0].copy()
    rules = [
        ("base_reference", base_val),
        ("val_base_tethered_residual_flip", tethered),
        ("val_primary_gate_best_utility", val[val["meets_primary_numeric_target"]]),
        ("val_frontier_cap_best_utility", val[val["frontier_call_rate"] <= frontier_cap]),
        (
            "val_large_recall_guard_best_utility",
            val[(val["frontier_call_rate"] <= frontier_cap) & (val["need_large_recall"] >= max(0.0, base_recall - 0.02))],
        ),
        ("val_best_utility", val),
    ]
    seen: set[tuple[str, str, str]] = set()
    for rule, candidates in rules:
        if candidates.empty:
            continue
        if rule == "val_base_tethered_residual_flip":
            best = candidates.sort_values(
                ["mean_utility", "mean_quality", "frontier_call_rate"],
                ascending=[False, False, True],
            ).head(1).copy()
        else:
            best = candidates.sort_values(["mean_utility", "mean_quality"], ascending=False).head(1).copy()
        method = str(best.iloc[0]["method"])
        key = (rule, method, "val")
        if key not in seen:
            rows.append(best.assign(selection_rule=rule))
            seen.add(key)
        test = table[table["split"].eq("test") & table["method"].eq(method)].head(1).copy()
        if not test.empty:
            key = (rule, method, "test")
            if key not in seen:
                rows.append(test.assign(selection_rule=f"{rule}_test"))
                seen.add(key)

    top_test = (
        selectable[selectable["split"].eq("test")]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(20)
    )
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def selected_query_details(details: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty or details.empty:
        return pd.DataFrame()
    wanted = selected[["method", "split", "selection_rule"]].drop_duplicates()
    return details.merge(wanted, on=["method", "split"], how="inner")


def write_figure(output_dir: Path, table: pd.DataFrame, selected: pd.DataFrame) -> None:
    test = table[table["split"].eq("test")].copy()
    test = test[~test["family"].astype(str).str.contains("oracle", case=False, na=False)]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.scatter(
        test["frontier_call_rate"],
        test["mean_utility"],
        s=20,
        alpha=0.35,
        label="candidate residual policies",
    )
    marked = selected[selected["split"].eq("test")].copy()
    if not marked.empty:
        ax.scatter(
            marked["frontier_call_rate"],
            marked["mean_utility"],
            s=60,
            marker="x",
            color="black",
            label="selected / diagnostic rows",
        )
    oracle = table[table["method"].eq("oracle_local_vs_large_gate") & table["split"].eq("test")]
    if not oracle.empty:
        ax.axhline(float(oracle.iloc[0]["mean_utility"]), color="tab:green", linestyle="--", linewidth=1.2, label="oracle")
    ax.set_xlabel("frontier call rate")
    ax.set_ylabel("held-out mean utility")
    ax.set_title("Broad100 residual oracle-gap repair")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_residual_oracle_gap_repair.pdf")
    plt.close(fig)


def write_memo(
    path: Path,
    args: argparse.Namespace,
    table: pd.DataFrame,
    selected: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    selected_show = selected[
        [
            "method",
            "selection_rule",
            "split",
            "mean_quality",
            "mean_utility",
            "oracle_utility_ratio",
            "frontier_call_rate",
            "large_call_rate",
            "need_large_recall",
            "meets_primary_numeric_target",
        ]
    ].copy()
    for col in [
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "frontier_call_rate",
        "large_call_rate",
        "need_large_recall",
    ]:
        selected_show[col] = pd.to_numeric(selected_show[col], errors="coerce").map(lambda value: f"{value:.4f}")
    best_test = (
        table[
            table["split"].eq("test")
            & ~table["family"].astype(str).str.contains("oracle", case=False, na=False)
            & ~table["method"].astype(str).str.contains("oracle_local_vs_large_gate", na=False)
        ]
        .sort_values(["mean_utility", "mean_quality"], ascending=False)
        .head(10)
    )
    best_test_show = best_test[
        [
            "method",
            "mean_quality",
            "mean_utility",
            "oracle_utility_ratio",
            "frontier_call_rate",
            "large_call_rate",
            "meets_primary_numeric_target",
        ]
    ].copy()
    for col in ["mean_quality", "mean_utility", "oracle_utility_ratio", "frontier_call_rate", "large_call_rate"]:
        best_test_show[col] = pd.to_numeric(best_test_show[col], errors="coerce").map(lambda value: f"{value:.4f}")

    path.write_text(
        "\n".join(
            [
                "# Broad100 Residual Oracle-Gap Repair",
                "",
                "This cached-only experiment tests whether a small residual reliability layer can move the",
                "current learned-verifiability target method closer to the full cost-aware oracle.",
                "It makes no provider calls, no vLLM calls, and no local generation calls.",
                "",
                "## Command",
                "",
                "```bash",
                "PYTHONPATH=src python experiments/215_broad100_residual_oracle_gap_repair.py",
                "```",
                "",
                "## Setup",
                "",
                f"- Outputs: `{args.outputs}`",
                f"- Target table: `{args.target_table}`",
                f"- Probe features: `{args.probe_features}`",
                f"- Learned scores: `{args.learned_scores}`",
                f"- Lambda cost: `{float(args.lambda_cost):.2f}`",
                f"- Feature count: `{len(feature_columns)}`",
                "",
                "The residual models are fit on train only. Thresholds are selected on validation.",
                "The `top_test_diagnostic` rows are reported only to measure unused headroom.",
                "",
                "## Selected Rows",
                "",
                "```csv",
                selected_show.to_csv(index=False).strip(),
                "```",
                "",
                "## Best Held-Out Diagnostics",
                "",
                "```csv",
                best_test_show.to_csv(index=False).strip(),
                "```",
                "",
                "## Interpretation",
                "",
                "- The current learned-verifiability base already satisfies the primary Broad100 numeric target.",
                "- A conservative base-tethered residual-flip selector is validation-selected only if it preserves validation quality, utility, recall, and reduces large/frontier use relative to the base.",
                "- The base-tethered selector provides only a tiny held-out improvement, while the validation-best residual still overfits the validation split.",
                "- This supports keeping the current verifiability/action-pool bridge as the strongest target-level method while treating residual reliability as only partially resolved.",
                "",
                "## Artifacts",
                "",
                "- `table_residual_oracle_gap_repair_all.csv`",
                "- `table_residual_oracle_gap_repair_selected.csv`",
                "- `table_residual_oracle_gap_repair_query_choices.csv`",
                "- `fig_residual_oracle_gap_repair.pdf`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
