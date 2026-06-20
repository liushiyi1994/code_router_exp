from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


DEPTHS: list[int | None] = [3, 5, None]
LEAF_SIZES = [4, 8, 16, 32]
STATE_K = [4, 8, 16, 32]
N_ESTIMATORS = [100]
KMEANS_N_INIT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decision-aware benchmark-agnostic ProbeCode states over cached Broad100 outputs. "
            "No provider, vLLM, local generation, or benchmark-specific checker calls are made."
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
        "--probe-features",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_features.csv"),
    )
    parser.add_argument(
        "--probe-state-reference",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_routecode/table_probe_state_policy_selected.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_decision_aware_probe_state_routecode"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--frontier-cap", type=float, default=0.40)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    exp201 = load_module("experiments/201_benchmark_agnostic_probe_state_routecode.py", "probe_state_201_for_209")

    outputs = pd.read_parquet(args.outputs).copy()
    outputs["utility"] = (
        outputs["quality_score"].astype(float)
        - float(args.lambda_cost) * outputs["normalized_remote_cost"].astype(float)
    )
    features = pd.read_csv(args.probe_features)
    matrix = exp201.build_utility_matrix(outputs)
    cols = exp201.feature_columns(features)

    standard_all = run_standard_eval(exp201, features, matrix, cols, args)
    standard_selected = select_by_validation(standard_all, float(args.frontier_cap))
    heldout_all = run_benchmark_heldout_eval(exp201, features, matrix, cols, args)
    heldout_selected = select_by_validation(heldout_all, float(args.frontier_cap)) if not heldout_all.empty else heldout_all
    cards = build_selected_cards(exp201, features, matrix, cols, standard_selected, args)
    reference = pd.read_csv(args.probe_state_reference) if args.probe_state_reference.exists() else pd.DataFrame()

    standard_all.to_csv(args.output_dir / "table_decision_aware_probe_state_all.csv", index=False)
    standard_selected.to_csv(args.output_dir / "table_decision_aware_probe_state_selected.csv", index=False)
    heldout_all.to_csv(args.output_dir / "table_decision_aware_probe_state_benchmark_heldout_all.csv", index=False)
    heldout_selected.to_csv(
        args.output_dir / "table_decision_aware_probe_state_benchmark_heldout_selected.csv", index=False
    )
    cards.to_csv(args.output_dir / "table_decision_aware_probe_state_code_cards.csv", index=False)
    write_code_cards(args.output_dir / "decision_aware_probe_state_code_cards.md", cards)
    write_memo(
        args.output_dir / "DECISION_AWARE_PROBE_STATE_MEMO.md",
        args,
        features,
        standard_selected,
        heldout_selected,
        reference,
        cards,
    )
    print(f"Wrote decision-aware probe-state results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_standard_eval(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    train_ids = features.loc[features["split"].eq("train"), "query_id"].astype(str).tolist()
    val_ids = features.loc[features["split"].eq("val"), "query_id"].astype(str).tolist()
    test_ids = features.loc[features["split"].eq("test"), "query_id"].astype(str).tolist()
    return run_candidates(exp201, features, matrix, cols, train_ids, val_ids, test_ids, args, "standard", "")


def run_benchmark_heldout_eval(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for heldout in sorted(features["benchmark"].dropna().astype(str).unique()):
        train_ids = features.loc[
            features["split"].eq("train") & features["benchmark"].astype(str).ne(heldout), "query_id"
        ].astype(str).tolist()
        val_ids = features.loc[
            features["split"].eq("val") & features["benchmark"].astype(str).ne(heldout), "query_id"
        ].astype(str).tolist()
        test_ids = features.loc[
            features["split"].eq("test") & features["benchmark"].astype(str).eq(heldout), "query_id"
        ].astype(str).tolist()
        if train_ids and val_ids and test_ids:
            rows.append(
                run_candidates(
                    exp201,
                    features,
                    matrix,
                    cols,
                    train_ids,
                    val_ids,
                    test_ids,
                    args,
                    "benchmark_heldout",
                    heldout,
                )
            )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_candidates(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(reference_rows(exp201, matrix, train_ids, val_ids, test_ids, args, scenario, heldout))
    rows.extend(extra_trees_action_probability_states(exp201, features, matrix, cols, train_ids, val_ids, test_ids, args, scenario, heldout))
    rows.extend(extra_trees_utility_prediction_states(exp201, features, matrix, cols, train_ids, val_ids, test_ids, args, scenario, heldout))
    rows.extend(utility_tree_leaf_states(exp201, features, matrix, cols, train_ids, val_ids, test_ids, args, scenario, heldout))
    return pd.DataFrame(rows)


def reference_rows(
    exp201: Any,
    matrix: dict[str, Any],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    global_action = exp201.best_action_for_ids(matrix, train_ids)
    for split, ids in [("val", val_ids), ("test", test_ids)]:
        selected = pd.Series(global_action, index=ids)
        rows.append(
            exp201.metric_row(
                matrix,
                ids,
                selected,
                "global_best_single",
                "global_best_single",
                split,
                scenario,
                heldout,
                args,
            )
        )
    return rows


def extra_trees_action_probability_states(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    index = features.set_index("query_id")
    x_train = index.loc[train_ids, cols].to_numpy()
    y_train = matrix["utility"].loc[train_ids].to_numpy().argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for depth in DEPTHS:
        for leaf in LEAF_SIZES:
            clf = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=200,
                            max_depth=depth,
                            min_samples_leaf=int(leaf),
                            class_weight="balanced",
                            random_state=int(args.seed),
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
            clf.fit(x_train, y_train)
            train_prob = clf.predict_proba(x_train)
            split_prob = {
                "val": clf.predict_proba(index.loc[val_ids, cols].to_numpy()),
                "test": clf.predict_proba(index.loc[test_ids, cols].to_numpy()),
            }
            for split, ids, prob in [("val", val_ids, split_prob["val"]), ("test", test_ids, split_prob["test"])]:
                selected = actions_from_indices(prob.argmax(axis=1), matrix["model_ids"], ids)
                rows.append(
                    exp201.metric_row(
                        matrix,
                        ids,
                        selected,
                        f"et_actionprob_direct_depth{depth_name(depth)}_leaf{leaf}",
                        "decision_aware_direct_probe_router",
                        split,
                        scenario,
                        heldout,
                        args,
                        depth=depth_name(depth),
                        leaf=int(leaf),
                    )
                )
            for k in STATE_K:
                kmeans = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
                train_state = kmeans.fit_predict(train_prob)
                action_by_state = exp201.best_action_by_label(matrix, train_ids, train_state)
                fallback = exp201.best_action_for_ids(matrix, train_ids)
                for split, ids, prob in [("val", val_ids, split_prob["val"]), ("test", test_ids, split_prob["test"])]:
                    state = kmeans.predict(prob)
                    selected = pd.Series([action_by_state.get(int(z), fallback) for z in state], index=ids)
                    rows.append(
                        exp201.metric_row(
                            matrix,
                            ids,
                            selected,
                            f"et_actionprob_state_depth{depth_name(depth)}_leaf{leaf}_k{k}",
                            "decision_aware_actionprob_state",
                            split,
                            scenario,
                            heldout,
                            args,
                            depth=depth_name(depth),
                            leaf=int(leaf),
                            k=int(k),
                        )
                    )
    return rows


def extra_trees_utility_prediction_states(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    index = features.set_index("query_id")
    x_train = index.loc[train_ids, cols].to_numpy()
    y_train = matrix["utility"].loc[train_ids].to_numpy()
    rows: list[dict[str, Any]] = []
    for n_estimators in N_ESTIMATORS:
        for depth in [5, 6, None]:
            for leaf in [4, 8, 16]:
                reg = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            ExtraTreesRegressor(
                                n_estimators=int(n_estimators),
                                max_depth=depth,
                                min_samples_leaf=int(leaf),
                                random_state=int(args.seed),
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )
                reg.fit(x_train, y_train)
                train_pred = reg.predict(x_train)
                split_pred = {
                    "val": reg.predict(index.loc[val_ids, cols].to_numpy()),
                    "test": reg.predict(index.loc[test_ids, cols].to_numpy()),
                }
                for split, ids, pred in [("val", val_ids, split_pred["val"]), ("test", test_ids, split_pred["test"])]:
                    selected = actions_from_indices(pred.argmax(axis=1), matrix["model_ids"], ids)
                    rows.append(
                        exp201.metric_row(
                            matrix,
                            ids,
                            selected,
                            f"et_utility_direct_est{n_estimators}_depth{depth_name(depth)}_leaf{leaf}",
                            "decision_aware_utility_direct",
                            split,
                            scenario,
                            heldout,
                            args,
                            n_estimators=int(n_estimators),
                            depth=depth_name(depth),
                            leaf=int(leaf),
                        )
                    )
                for k in STATE_K:
                    kmeans = KMeans(n_clusters=int(k), random_state=int(args.seed), n_init=KMEANS_N_INIT)
                    train_state = kmeans.fit_predict(train_pred)
                    action_by_state = exp201.best_action_by_label(matrix, train_ids, train_state)
                    fallback = exp201.best_action_for_ids(matrix, train_ids)
                    for split, ids, pred in [("val", val_ids, split_pred["val"]), ("test", test_ids, split_pred["test"])]:
                        state = kmeans.predict(pred)
                        selected = pd.Series([action_by_state.get(int(z), fallback) for z in state], index=ids)
                        rows.append(
                            exp201.metric_row(
                                matrix,
                                ids,
                                selected,
                                f"et_utility_state_est{n_estimators}_depth{depth_name(depth)}_leaf{leaf}_k{k}",
                                "decision_aware_utility_state",
                                split,
                                scenario,
                                heldout,
                                args,
                                n_estimators=int(n_estimators),
                                depth=depth_name(depth),
                                leaf=int(leaf),
                                k=int(k),
                            )
                        )
    return rows


def utility_tree_leaf_states(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    args: argparse.Namespace,
    scenario: str,
    heldout: str,
) -> list[dict[str, Any]]:
    index = features.set_index("query_id")
    x_train = index.loc[train_ids, cols].to_numpy()
    y_train = matrix["utility"].loc[train_ids].to_numpy()
    rows: list[dict[str, Any]] = []
    for depth in [4, 5, 6, 8, None]:
        for leaf in LEAF_SIZES:
            reg = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", DecisionTreeRegressor(max_depth=depth, min_samples_leaf=int(leaf), random_state=int(args.seed))),
                ]
            )
            reg.fit(x_train, y_train)
            model = reg.named_steps["model"]
            train_scaled = reg.named_steps["scaler"].transform(reg.named_steps["imputer"].transform(x_train))
            train_state = model.apply(train_scaled)
            action_by_state = exp201.best_action_by_label(matrix, train_ids, train_state)
            fallback = exp201.best_action_for_ids(matrix, train_ids)
            for split, ids in [("val", val_ids), ("test", test_ids)]:
                x_eval = index.loc[ids, cols].to_numpy()
                x_scaled = reg.named_steps["scaler"].transform(reg.named_steps["imputer"].transform(x_eval))
                state = model.apply(x_scaled)
                selected = pd.Series([action_by_state.get(int(z), fallback) for z in state], index=ids)
                rows.append(
                    exp201.metric_row(
                        matrix,
                        ids,
                        selected,
                        f"utility_tree_leaf_depth{depth_name(depth)}_leaf{leaf}",
                        "decision_aware_utility_tree_leaf_state",
                        split,
                        scenario,
                        heldout,
                        args,
                        depth=depth_name(depth),
                        leaf=int(leaf),
                        n_states=int(len(set(train_state))),
                    )
                )
    return rows


def select_by_validation(candidates: pd.DataFrame, frontier_cap: float) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for (scenario, heldout, family), group in candidates.groupby(["scenario", "heldout_benchmark", "family"], dropna=False):
        val = group[group["eval_split"].eq("val")].copy()
        test = group[group["eval_split"].eq("test")].copy()
        if val.empty:
            continue
        rows.extend(selection_pair(val, test, "val_best_mean_utility", require_cap=None))
        capped = val[val["frontier_call_rate"].astype(float).le(float(frontier_cap))]
        if not capped.empty:
            rows.extend(selection_pair(capped, test, f"val_best_mean_utility_frontier_cap_{frontier_cap:g}", require_cap=frontier_cap))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["scenario", "heldout_benchmark", "family", "selection_rule", "eval_split"])


def selection_pair(
    val: pd.DataFrame,
    test: pd.DataFrame,
    selection_rule: str,
    require_cap: float | None,
) -> list[pd.Series]:
    best = val.sort_values(["mean_utility", "frontier_call_rate"], ascending=[False, True]).iloc[0].copy()
    best["selection_rule"] = selection_rule
    best["selection_frontier_cap"] = require_cap if require_cap is not None else np.nan
    rows = [best]
    match = test[test["method"].astype(str).eq(str(best["method"]))]
    if not match.empty:
        test_row = match.iloc[0].copy()
        test_row["selection_rule"] = f"{selection_rule}_test"
        test_row["selection_frontier_cap"] = require_cap if require_cap is not None else np.nan
        rows.append(test_row)
    return rows


def build_selected_cards(
    exp201: Any,
    features: pd.DataFrame,
    matrix: dict[str, Any],
    cols: list[str],
    selected: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    test = selected[
        selected["scenario"].eq("standard")
        & selected["eval_split"].eq("test")
        & selected["selection_rule"].astype(str).eq("val_best_mean_utility_test")
        & selected["family"].astype(str).str.endswith("_state")
    ].copy()
    if test.empty:
        return pd.DataFrame()
    row = test.sort_values("mean_utility", ascending=False).iloc[0]
    method = str(row["method"])
    if not method.startswith("et_actionprob_state_"):
        return pd.DataFrame()

    params = parse_method_params(method)
    if not params:
        return pd.DataFrame()
    index = features.set_index("query_id")
    train_ids = features.loc[features["split"].eq("train"), "query_id"].astype(str).tolist()
    x_train = index.loc[train_ids, cols].to_numpy()
    y_train = matrix["utility"].loc[train_ids].to_numpy().argmax(axis=1)
    clf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=params["depth"],
                    min_samples_leaf=int(params["leaf"]),
                    class_weight="balanced",
                    random_state=int(args.seed),
                    n_jobs=-1,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)
    train_prob = clf.predict_proba(x_train)
    kmeans = KMeans(n_clusters=int(params["k"]), random_state=int(args.seed), n_init=KMEANS_N_INIT)
    train_state = kmeans.fit_predict(train_prob)
    action_by_state = exp201.best_action_by_label(matrix, train_ids, train_state)
    scaled_train = clf.named_steps["scaler"].transform(clf.named_steps["imputer"].transform(x_train))
    global_feature_mean = pd.Series(scaled_train.mean(axis=0), index=cols)

    rows: list[dict[str, Any]] = []
    utility = matrix["utility"].loc[train_ids].copy()
    for state_id in range(int(params["k"])):
        member_positions = np.flatnonzero(train_state == state_id)
        member_ids = [train_ids[int(pos)] for pos in member_positions]
        if len(member_positions) == 0:
            continue
        state_features = pd.DataFrame(scaled_train[member_positions], columns=cols)
        top_features = (state_features.mean(axis=0) - global_feature_mean).abs().sort_values(ascending=False).head(8)
        state_probs = train_prob[member_positions].mean(axis=0)
        prob_actions = {
            matrix["model_ids"][int(i)]: float(state_probs[int(i)])
            for i in np.argsort(state_probs)[::-1][:5]
        }
        action = action_by_state.get(state_id, exp201.best_action_for_ids(matrix, train_ids))
        rows.append(
            {
                "method": method,
                "probe_state": int(state_id),
                "n_train_queries": int(len(member_ids)),
                "selected_action": action,
                "train_mean_selected_utility": float(utility.loc[member_ids, action].mean()),
                "frontier_if_selected": bool(matrix["frontier"].loc[member_ids, action].astype(bool).any()),
                "top_features_json": json.dumps(top_features.round(3).to_dict(), sort_keys=True),
                "top_action_probabilities_json": json.dumps(prob_actions, sort_keys=True),
                "benchmark_mix_json": json.dumps(
                    index.loc[member_ids, "benchmark"].astype(str).value_counts().sort_index().to_dict(),
                    sort_keys=True,
                ),
            }
        )
    return pd.DataFrame(rows)


def parse_method_params(method: str) -> dict[str, Any] | None:
    prefix = "et_actionprob_state_depth"
    if not method.startswith(prefix):
        return None
    rest = method[len(prefix) :]
    depth_text, rest = rest.split("_leaf", 1)
    leaf_text, k_text = rest.split("_k", 1)
    return {
        "depth": None if depth_text == "none" else int(depth_text),
        "leaf": int(leaf_text),
        "k": int(k_text),
    }


def actions_from_indices(indices: np.ndarray, model_ids: list[str], ids: list[str]) -> pd.Series:
    return pd.Series([model_ids[int(index)] for index in indices], index=ids)


def depth_name(depth: int | None | str) -> str:
    if depth is None or str(depth) == "None":
        return "none"
    return str(int(depth))


def write_code_cards(path: Path, cards: pd.DataFrame) -> None:
    if cards.empty:
        path.write_text("# Decision-Aware Probe-State Code Cards\n\nNo selected action-probability state cards were produced.\n", encoding="utf-8")
        return
    lines = ["# Decision-Aware Probe-State Code Cards", ""]
    for row in cards.itertuples(index=False):
        lines.extend(
            [
                f"## Probe State {row.probe_state}",
                "",
                f"- Method: `{row.method}`",
                f"- Selected action: `{row.selected_action}`",
                f"- Train queries: `{row.n_train_queries}`",
                f"- Train selected utility: `{row.train_mean_selected_utility:.4f}`",
                f"- Top feature deviations: `{row.top_features_json}`",
                f"- Mean action probabilities: `{row.top_action_probabilities_json}`",
                f"- Benchmark mix: `{row.benchmark_mix_json}`",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_memo(
    path: Path,
    args: argparse.Namespace,
    features: pd.DataFrame,
    selected: pd.DataFrame,
    heldout_selected: pd.DataFrame,
    reference: pd.DataFrame,
    cards: pd.DataFrame,
) -> None:
    cols = [
        "family",
        "method",
        "eval_split",
        "n_queries",
        "mean_quality",
        "mean_utility",
        "oracle_utility_ratio",
        "mean_normalized_cost",
        "frontier_call_rate",
        "selection_rule",
    ]
    standard_test = selected[
        selected["scenario"].eq("standard")
        & selected["eval_split"].eq("test")
        & selected["selection_rule"].astype(str).str.endswith("_test")
    ].copy()
    heldout_test = heldout_selected[
        heldout_selected["eval_split"].eq("test")
        & heldout_selected["selection_rule"].astype(str).str.endswith("_test")
    ].copy() if not heldout_selected.empty else heldout_selected
    heldout_summary = (
        heldout_test.groupby(["family", "selection_rule"], as_index=False)
        .agg(
            mean_heldout_quality=("mean_quality", "mean"),
            mean_heldout_utility=("mean_utility", "mean"),
            mean_heldout_oracle_ratio=("oracle_utility_ratio", "mean"),
            mean_frontier_call_rate=("frontier_call_rate", "mean"),
        )
        .sort_values("mean_heldout_utility", ascending=False)
        if not heldout_test.empty
        else pd.DataFrame()
    )
    reference_test = reference[
        reference.get("eval_split", pd.Series(dtype=str)).astype(str).eq("test")
        & reference.get("selection_rule", pd.Series(dtype=str)).astype(str).str.endswith("_test")
    ].copy() if not reference.empty else reference
    lines = [
        "# Decision-Aware Probe-State RouteCode",
        "",
        "This cached experiment tries a decision-aware version of benchmark-agnostic ProbeCode.",
        "It learns discrete probe states from train-only utility/action targets, then maps each state to a cost-aware action using train utilities.",
        "",
        "No GPT, Gemini, Claude, vLLM, local generation, or benchmark-specific checker calls are made.",
        "",
        "## Command",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/209_decision_aware_probe_state_routecode.py",
        f"PYTHONPATH=src python experiments/209_decision_aware_probe_state_routecode.py --output-dir {args.output_dir}",
        "```",
        "",
        "## Data",
        "",
        f"- Queries: `{len(features)}`",
        f"- Splits: `{features['split'].value_counts().to_dict()}`",
        f"- Benchmarks: `{sorted(features['benchmark'].dropna().astype(str).unique().tolist())}`",
        "",
        "## Reference Probe-State Rows From Experiment 201",
        "",
        markdown_table(reference_test[cols]) if not reference_test.empty else "No reference rows found.",
        "",
        "## Selected Standard Rows",
        "",
        markdown_table(standard_test[cols]) if not standard_test.empty else "No selected rows.",
        "",
        "## Benchmark-Heldout Summary",
        "",
        markdown_table(heldout_summary) if not heldout_summary.empty else "No heldout rows.",
        "",
        "## Interpretation",
        "",
        "- Decision-aware probe states are still benchmark-agnostic: the main features exclude benchmark ID and no task-specific checker is used.",
        "- If these rows improve over KMeans probe states but remain below oracle RouteCode labels, the bottleneck is not simple feature clustering; it is still cheap observability of action identity.",
        "- Frontier-cap selection rows are included to test the controlled-experiment frontier-call constraint without tuning on test.",
        "",
        "## Artifacts",
        "",
        f"- All rows: `{args.output_dir / 'table_decision_aware_probe_state_all.csv'}`",
        f"- Selected rows: `{args.output_dir / 'table_decision_aware_probe_state_selected.csv'}`",
        f"- Heldout selected rows: `{args.output_dir / 'table_decision_aware_probe_state_benchmark_heldout_selected.csv'}`",
        f"- Code cards: `{args.output_dir / 'decision_aware_probe_state_code_cards.md'}`",
        f"- Code-card table: `{args.output_dir / 'table_decision_aware_probe_state_code_cards.csv'}`",
        "",
        f"Code cards produced: `{len(cards)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
