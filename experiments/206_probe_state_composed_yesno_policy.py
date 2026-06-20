from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


K_VALUES = [2, 4, 8, 16, 32]
KMEANS_N_INIT = 20
TOOL_MODEL_ID = "deterministic_math_tool"
MAIN_FEATURES = [
    "signal_query_answerability_risk",
    "signal_query_answerability",
    "signal_early_rollout_instability",
    "signal_semantic_uncertainty",
    "signal_slm_medium_divergence",
    "signal_medium_consensus_disagrees_with_slm",
    "signal_combined_mean_risk",
    "signal_combined_max_risk",
    "signal_constrained_yesno_local_evidence_risk",
    "signal_constrained_yesno_local_evidence_safe",
    "signal_constrained_yesno_local_evidence_entropy",
    "signal_constrained_yesno_local_evidence_low_margin_risk",
    "signal_constrained_yesno_query_only_risk",
    "signal_constrained_yesno_query_only_safe",
    "signal_constrained_yesno_query_only_entropy",
    "signal_constrained_yesno_query_only_low_margin_risk",
    "signal_constrained_yesno_max_risk",
    "signal_constrained_yesno_mean_risk",
    "signal_constrained_plus_cached_mean_risk",
    "signal_constrained_plus_cached_max_risk",
]
PRIOR_FEATURES = ["signal_query_train_prior_need_large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-agnostic probe-state composition of simple local-vs-large policies. "
            "This tests observable probe states rather than per-benchmark policy lookup."
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
        "--output-dir",
        type=Path,
        default=Path("results/controlled/broad100_probe_state_composed_yesno_policy"),
    )
    parser.add_argument("--lambda-cost", type=float, default=0.35)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rc166 = load_module("experiments/166_slm_llm_early_signal_probe_pilot.py", "slm_llm_pilot_166_for_206")
    exp171 = load_module("experiments/171_tool_aware_benchmark_composed_policy.py", "tool_composed_171_for_206")
    target = pd.read_csv(args.target_table)
    target = exp171.add_tool_availability(target, pd.read_parquet(args.outputs))
    table, assignments, state_cards = run_experiment(target, rc166, exp171, args)
    selected = selected_rows(table, rc166, int(args.bootstrap_samples), int(args.seed))

    table.to_csv(args.output_dir / "table_probe_state_composed_policy_all.csv", index=False)
    selected.to_csv(args.output_dir / "table_probe_state_composed_policy_selected.csv", index=False)
    assignments.to_csv(args.output_dir / "table_probe_state_composed_assignments.csv", index=False)
    state_cards.to_csv(args.output_dir / "table_probe_state_composed_code_cards.csv", index=False)
    write_code_cards_md(args.output_dir / "probe_state_composed_code_cards.md", state_cards)
    write_figure(args.output_dir, table)
    write_memo(
        args.output_dir / "PROBE_STATE_COMPOSED_POLICY_MEMO.md",
        args,
        table,
        selected,
        state_cards,
    )
    print(f"Wrote probe-state composed YES/NO policy results to {args.output_dir}")


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_experiment(target: pd.DataFrame, rc166, exp171, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_fns_all = exp171.candidate_policy_functions()
    policy_fns_no_tool = {
        name: fn for name, fn in policy_fns_all.items() if not str(name).startswith("tool_")
    }
    specs = [
        {
            "family": "probe_state_composed",
            "feature_set": "main_no_benchmark_no_tool",
            "features": MAIN_FEATURES,
            "policy_fns": policy_fns_no_tool,
            "diagnostic": False,
        },
        {
            "family": "probe_state_with_train_prior_diagnostic",
            "feature_set": "main_plus_train_benchmark_prior",
            "features": MAIN_FEATURES + PRIOR_FEATURES,
            "policy_fns": policy_fns_no_tool,
            "diagnostic": True,
        },
        {
            "family": "probe_state_tool_aware_diagnostic",
            "feature_set": "main_plus_tool_available",
            "features": MAIN_FEATURES + ["tool_available"],
            "policy_fns": policy_fns_all,
            "diagnostic": True,
        },
    ]
    rows: list[dict[str, Any]] = []
    assignments: list[pd.DataFrame] = []
    cards: list[pd.DataFrame] = []
    for split in ["val", "test"]:
        frame = target[target["split"].eq(split)].copy()
        rows.extend(rc166.reference_rows(frame, split=split, lambda_cost=float(args.lambda_cost)))

    for spec in specs:
        features = [col for col in spec["features"] if col in target.columns]
        for k in K_VALUES:
            labels, model = fit_predict_states(target, features, k=int(k), seed=int(args.seed))
            target_with_labels = target.assign(probe_state=labels)
            state_policy = choose_policy_by_state(
                target_with_labels[target_with_labels["split"].eq("train")],
                spec["policy_fns"],
                rc166,
                lambda_cost=float(args.lambda_cost),
            )
            method = f"{spec['feature_set']}_k{k}"
            for split in ["val", "test"]:
                frame = target_with_labels[target_with_labels["split"].eq(split)].copy()
                choose_large = compose_by_state(frame, state_policy, spec["policy_fns"])
                row = rc166.evaluate_decision(
                    frame,
                    choose_large,
                    split=split,
                    method=method,
                    family=spec["family"],
                    lambda_cost=float(args.lambda_cost),
                )
                row.update(
                    {
                        "k": int(k),
                        "feature_set": str(spec["feature_set"]),
                        "diagnostic": bool(spec["diagnostic"]),
                        "state_policy_json": json.dumps(state_policy, sort_keys=True),
                    }
                )
                rows.append(row)
            assignments.append(
                target_with_labels[
                    ["query_id", "split", "benchmark", "probe_state", "need_large", "local_utility", "large_utility"]
                ].assign(method=method, family=spec["family"], k=int(k), feature_set=str(spec["feature_set"]))
            )
            cards.append(build_cards(target_with_labels, state_policy, method, spec["family"], features, k=int(k), model=model))

    table = pd.DataFrame(rows).sort_values(["split", "mean_utility", "mean_quality"], ascending=[True, False, False])
    assignment_frame = pd.concat(assignments, ignore_index=True) if assignments else pd.DataFrame()
    card_frame = pd.concat(cards, ignore_index=True) if cards else pd.DataFrame()
    return table, assignment_frame, card_frame


def fit_predict_states(target: pd.DataFrame, features: list[str], *, k: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    train = target[target["split"].eq("train")]
    x_train = train[features].copy()
    x_all = target[features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(imputer.fit_transform(x_train))
    x_all_scaled = scaler.transform(imputer.transform(x_all))
    kmeans = KMeans(n_clusters=int(k), random_state=int(seed), n_init=KMEANS_N_INIT)
    kmeans.fit(x_train_scaled)
    return kmeans.predict(x_all_scaled), {"imputer": imputer, "scaler": scaler, "kmeans": kmeans}


def choose_policy_by_state(
    train: pd.DataFrame,
    policy_fns: dict[str, Callable[[pd.DataFrame], np.ndarray]],
    rc166,
    *,
    lambda_cost: float,
) -> dict[str, str]:
    state_policy: dict[str, str] = {}
    global_policy = best_policy(train, policy_fns, rc166, lambda_cost=lambda_cost)
    for state, group in train.groupby("probe_state", sort=True):
        if len(group) < 4:
            state_policy[str(int(state))] = global_policy
        else:
            state_policy[str(int(state))] = best_policy(group, policy_fns, rc166, lambda_cost=lambda_cost)
    return state_policy


def best_policy(
    frame: pd.DataFrame,
    policy_fns: dict[str, Callable[[pd.DataFrame], np.ndarray]],
    rc166,
    *,
    lambda_cost: float,
) -> str:
    scored: list[dict[str, Any]] = []
    for name, fn in policy_fns.items():
        row = rc166.evaluate_decision(
            frame,
            fn(frame),
            split="train_state",
            method=name,
            family="state_candidate",
            lambda_cost=lambda_cost,
        )
        scored.append(row)
    scores = pd.DataFrame(scored)
    best = scores.sort_values(
        ["mean_utility", "mean_quality", "frontier_call_rate", "large_call_rate"],
        ascending=[False, False, True, True],
    ).iloc[0]
    return str(best["method"])


def compose_by_state(
    frame: pd.DataFrame,
    state_policy: dict[str, str],
    policy_fns: dict[str, Callable[[pd.DataFrame], np.ndarray]],
) -> np.ndarray:
    choose = np.zeros(len(frame), dtype=bool)
    states = frame["probe_state"].to_numpy()
    fallback = next(iter(policy_fns))
    for state in sorted(set(states)):
        positions = np.where(states == state)[0]
        policy_name = state_policy.get(str(int(state)), fallback)
        choose[positions] = policy_fns[policy_name](frame.iloc[positions].copy())
    return choose


def selected_rows(table: pd.DataFrame, rc166, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for family, group in table.groupby("family", sort=False):
        if family == "diagnostic_oracle":
            continue
        val = group[group["split"].eq("val")].copy()
        if val.empty:
            continue
        best = val.sort_values(
            ["mean_utility", "frontier_call_rate", "large_call_rate"],
            ascending=[False, True, True],
        ).head(1)
        method = str(best.iloc[0]["method"])
        rows.append(best.assign(selection_rule="val_best_utility"))
        test = group[group["split"].eq("test") & group["method"].eq(method)].copy()
        if not test.empty:
            rows.append(test.assign(selection_rule="val_best_utility_test"))

        val = val.copy()
        val["quality_gap_to_local_large_oracle"] = val["local_large_oracle_mean_quality"] - val["mean_quality"]
        target_candidates = val[
            (val["oracle_utility_ratio"] >= 0.95)
            & (val["quality_gap_to_local_large_oracle"] <= 0.03)
            & (val["frontier_call_rate"] <= 0.40)
        ].copy()
        if not target_candidates.empty:
            chosen = target_candidates.sort_values(
                ["mean_quality", "mean_utility", "frontier_call_rate"],
                ascending=[False, False, True],
            ).head(1)
            method = str(chosen.iloc[0]["method"])
            rows.append(chosen.drop(columns=["quality_gap_to_local_large_oracle"]).assign(selection_rule="val_target_gate"))
            test = group[group["split"].eq("test") & group["method"].eq(method)].copy()
            if not test.empty:
                rows.append(test.assign(selection_rule="val_target_gate_test"))

    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(24)
    if not top_test.empty:
        rows.append(top_test.assign(selection_rule="top_test_diagnostic"))
    selected = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not selected.empty:
        selected = rc166.add_bootstrap_ci(selected, bootstrap_samples=bootstrap_samples, seed=seed)
        selected = selected.drop(columns=["_utility_values"], errors="ignore")
    return selected


def build_cards(
    target: pd.DataFrame,
    state_policy: dict[str, str],
    method: str,
    family: str,
    features: list[str],
    *,
    k: int,
    model: dict[str, Any],
) -> pd.DataFrame:
    train = target[target["split"].eq("train")].copy()
    all_rows = target.copy()
    rows: list[dict[str, Any]] = []
    for state, group in all_rows.groupby("probe_state", sort=True):
        train_group = train[train["probe_state"].eq(state)].copy()
        means = group[features].mean(numeric_only=True)
        global_means = all_rows[features].mean(numeric_only=True)
        diffs = (means - global_means).abs().sort_values(ascending=False).head(6)
        rows.append(
            {
                "method": method,
                "family": family,
                "k": int(k),
                "probe_state": int(state),
                "n_all": int(len(group)),
                "n_train": int(len(train_group)),
                "chosen_policy": state_policy.get(str(int(state)), ""),
                "train_need_large_rate": float(train_group["need_large"].mean()) if len(train_group) else np.nan,
                "all_need_large_rate": float(group["need_large"].mean()) if len(group) else np.nan,
                "train_mean_local_utility": float(train_group["local_utility"].mean()) if len(train_group) else np.nan,
                "train_mean_large_utility": float(train_group["large_utility"].mean()) if len(train_group) else np.nan,
                "top_feature_diffs_json": json.dumps({name: round(float(value), 4) for name, value in diffs.items()}, sort_keys=True),
                "benchmark_mix_json": json.dumps(group["benchmark"].value_counts().sort_index().to_dict(), sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def write_code_cards_md(path: Path, cards: pd.DataFrame) -> None:
    if cards.empty:
        path.write_text("# Probe-State Composed Code Cards\n\n_No cards._\n", encoding="utf-8")
        return
    lines = ["# Probe-State Composed Code Cards", ""]
    selected_methods = set(cards["method"].drop_duplicates().head(6).tolist())
    for method in selected_methods:
        subset = cards[cards["method"].eq(method)].sort_values("probe_state")
        lines.extend([f"## {method}", ""])
        for row in subset.to_dict("records"):
            lines.extend(
                [
                    f"### State {row['probe_state']}",
                    "",
                    f"- Chosen policy: `{row['chosen_policy']}`",
                    f"- Train rows: `{row['n_train']}`; all rows: `{row['n_all']}`",
                    f"- Train need-large rate: `{float(row['train_need_large_rate']):.4f}`",
                    f"- Top feature shifts: `{row['top_feature_diffs_json']}`",
                    f"- Benchmark mix: `{row['benchmark_mix_json']}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figure(out_dir: Path, table: pd.DataFrame) -> None:
    plot = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    labels = plot["family"].astype(str) + " / " + plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(labels.iloc[::-1], plot["mean_utility"].iloc[::-1], color="#546f89")
    ax.set_xlabel("Held-out test mean utility")
    ax.set_title("Probe-State Composed Local-vs-Large Policies")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_probe_state_composed_policy_utility.pdf")
    plt.close(fig)


def write_memo(path: Path, args: argparse.Namespace, table: pd.DataFrame, selected: pd.DataFrame, cards: pd.DataFrame) -> None:
    cols = [
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
    ]
    target_lines = target_gate_lines(selected)
    top_test = table[table["split"].eq("test")].sort_values(["mean_utility", "mean_quality"], ascending=False).head(18)
    lines = [
        "# Probe-State Composed YES/NO Policy",
        "",
        "This cached experiment replaces per-benchmark policy lookup with train-fit probe states.",
        "",
        "```text",
        "query + cheap broad local signals -> probe_state -> local-vs-large policy",
        "```",
        "",
        "Main rows exclude benchmark ID, benchmark train-prior, and deterministic-tool suppression.",
        "Prior/tool rows are diagnostic. No GPT, Gemini, Claude, local generation, or vLLM calls are made.",
        "",
        "Important caveat: this evaluates the local-vs-large abstraction using cached best local and best large actions.",
        "It is not yet a full concrete multi-action deployed router.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src python -m py_compile experiments/206_probe_state_composed_yesno_policy.py",
        f"PYTHONPATH=src python experiments/206_probe_state_composed_yesno_policy.py --target-table {args.target_table} --outputs {args.outputs} --output-dir {args.output_dir}",
        "```",
        "",
        "## Validation-Selected Rows",
        "",
        "```csv",
        selected[[col for col in cols if col in selected.columns]].to_csv(index=False).strip() if not selected.empty else "",
        "```",
        "",
        "## Target Gate Check",
        "",
        *target_lines,
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
                "recovered_gap_vs_local",
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
        "- If main rows pass the target, the evidence supports benchmark-agnostic probe-state observability for the coarse local-vs-large decision.",
        "- If only diagnostic prior/tool rows pass, then the target still depends on benchmark artifacts or tool-specific evidence.",
        "- Concrete action selection remains separate; this experiment does not solve exact model identity inside the local or large family.",
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
    out = []
    for row in rows.to_dict("records"):
        oracle_u = float(row["local_large_oracle_mean_utility"])
        oracle_q = float(row["local_large_oracle_mean_quality"])
        utility_target = 0.95 * oracle_u
        quality_target = oracle_q - 0.03
        out.append(
            (
                f"- `{row['method']}` ({row['family']}, {row['selection_rule']}): "
                f"utility `{float(row['mean_utility']):.4f}` vs target `{utility_target:.4f}`; "
                f"quality `{float(row['mean_quality']):.4f}` vs target `{quality_target:.4f}`; "
                f"frontier rate `{float(row['frontier_call_rate']):.4f}`."
            )
        )
    return out


if __name__ == "__main__":
    main()
