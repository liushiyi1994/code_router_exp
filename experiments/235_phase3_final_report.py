from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml


DEFAULT_CONFIG = Path("configs/probecode_final_eval.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble the current Phase 3 final-evaluation report.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    root = Path(config["outputs"]["root"])
    root.mkdir(parents=True, exist_ok=True)

    main_eval = read_csv(root / "main_eval/table_main_routing_eval.csv")
    baseline_status = read_csv(root / "main_eval/table_literature_baseline_status.csv")
    calibration = read_csv(root / "calibration_strata/table_state_variance.csv")
    onboarding = read_csv(root / "new_model_onboarding/table_new_model_onboarding.csv")
    frozen = read_csv(root / "frozen_state_vs_retrain/table_frozen_state_vs_retrain.csv")
    sensitivity = read_csv(root / "sensitivity/table_price_sensitivity.csv")
    ablation = read_csv(root / "ablation/table_final_ablation.csv")
    real_calibration = read_csv(root / "real_new_model_calibration/table_real_new_model_calibration.csv")
    predicted_utility_claims = read_csv(root / "predicted_utility_states/table_predicted_state_claims.csv")
    predicted_utility_onboarding = read_csv(root / "predicted_utility_states/table_predicted_state_onboarding.csv")
    predicted_utility_variance = read_csv(root / "predicted_utility_states/table_predicted_state_variance.csv")
    live_predicted_claims = read_csv(root / "live_predicted_utility_states/table_live_predicted_state_claims.csv")
    live_predicted_onboarding = read_csv(root / "live_predicted_utility_states/table_live_predicted_state_onboarding.csv")
    live_predicted_variance = read_csv(root / "live_predicted_utility_states/table_live_predicted_state_variance.csv")
    live_frontier_validation = read_csv(root / "live_predicted_utility_states/table_live_frontier_onboarding_validation.csv")
    live_frontier_test = read_csv(root / "live_predicted_utility_states/table_live_frontier_onboarding_test.csv")
    live_frontier_budget_efficiency = read_csv(root / "live_predicted_utility_states/table_live_frontier_budget_efficiency.csv")
    final_claims = read_csv("results/controlled/phase3_final_claim_package/table_phase3_final_claims.csv")

    claims = build_claim_table(
        main_eval,
        baseline_status,
        calibration,
        onboarding,
        frozen,
        sensitivity,
        ablation,
        real_calibration,
        predicted_utility_claims,
        live_predicted_claims,
        final_claims,
    )
    claims.to_csv(root / "table_final_claims.csv", index=False)
    if not main_eval.empty:
        main_eval.to_csv(root / "table_final_main_eval.csv", index=False)
        baseline_roles = ["random", "all_gpt", "all_gemini", "local_model", "local_reference", "literature_baseline"]
        main_eval[main_eval["method_role"].isin(baseline_roles)].to_csv(root / "table_final_baselines.csv", index=False)
    if not calibration.empty:
        calibration.to_csv(root / "table_final_calibration.csv", index=False)
    if not onboarding.empty:
        onboarding.to_csv(root / "table_final_onboarding.csv", index=False)
    if not sensitivity.empty:
        sensitivity.to_csv(root / "table_final_sensitivity.csv", index=False)
    if not ablation.empty:
        ablation.to_csv(root / "table_final_ablation.csv", index=False)
    if not real_calibration.empty:
        real_calibration.to_csv(root / "table_final_real_calibration.csv", index=False)
    if not predicted_utility_claims.empty:
        predicted_utility_claims.to_csv(root / "table_final_predicted_utility_state_claims.csv", index=False)
    if not predicted_utility_onboarding.empty:
        predicted_utility_onboarding.to_csv(root / "table_final_predicted_utility_state_onboarding.csv", index=False)
    if not predicted_utility_variance.empty:
        predicted_utility_variance.to_csv(root / "table_final_predicted_utility_state_variance.csv", index=False)
    if not live_predicted_claims.empty:
        live_predicted_claims.to_csv(root / "table_final_live_predicted_utility_state_claims.csv", index=False)
    if not live_predicted_onboarding.empty:
        live_predicted_onboarding.to_csv(root / "table_final_live_predicted_utility_state_onboarding.csv", index=False)
    if not live_predicted_variance.empty:
        live_predicted_variance.to_csv(root / "table_final_live_predicted_utility_state_variance.csv", index=False)
    if not live_frontier_validation.empty:
        live_frontier_validation.to_csv(root / "table_final_live_frontier_onboarding_validation.csv", index=False)
    if not live_frontier_test.empty:
        live_frontier_test.to_csv(root / "table_final_live_frontier_onboarding_test.csv", index=False)
    if not live_frontier_budget_efficiency.empty:
        live_frontier_budget_efficiency.to_csv(root / "table_final_live_frontier_budget_efficiency.csv", index=False)

    write_report(
        root / "FINAL_EVALUATION_REPORT.md",
        main_eval,
        baseline_status,
        calibration,
        onboarding,
        frozen,
        sensitivity,
        ablation,
        real_calibration,
        predicted_utility_claims,
        predicted_utility_onboarding,
        predicted_utility_variance,
        live_predicted_claims,
        live_predicted_onboarding,
        live_predicted_variance,
        live_frontier_validation,
        live_frontier_test,
        live_frontier_budget_efficiency,
        claims,
        config,
    )
    print(f"Wrote final report scaffold to {root}")


def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def build_claim_table(
    main_eval: pd.DataFrame,
    baseline_status: pd.DataFrame,
    calibration: pd.DataFrame,
    onboarding: pd.DataFrame,
    frozen: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
    real_calibration: pd.DataFrame,
    predicted_utility_claims: pd.DataFrame,
    live_predicted_claims: pd.DataFrame,
    final_claims: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    if not main_eval.empty:
        current = main_eval[main_eval["method_role"].eq("probecode_statecal")]
        if not current.empty:
            row = current.iloc[0]
            rows.append(
                {
                    "claim_id": "final_main_broad100_oracle_level",
                    "status": "supported_on_cached_broad100" if bool(row["meets_primary_gate"]) else "not_supported",
                    "evidence": (
                        f"quality_gap={float(row['quality_gap_to_oracle']):.4f};"
                        f"utility_ratio={float(row['oracle_utility_ratio']):.4f};"
                        f"frontier_rate={float(row['frontier_call_rate']):.4f}"
                    ),
                    "caveat": "Cache-backed Broad100 final-eval pass; no live provider calls.",
                }
            )
    if not baseline_status.empty:
        pending = baseline_status[~baseline_status["broad100_final_eval_included"].astype(bool)]
        rows.append(
            {
                "claim_id": "final_literature_baseline_coverage",
                "status": "incomplete" if len(pending) else "complete",
                "evidence": f"pending_or_not_included={len(pending)};total={len(baseline_status)}",
                "caveat": "RouteLLM and Avengers-Pro use cached Broad100 adapters; GraphRouter slot uses the documented LLMRouter kNN fallback.",
            }
        )
    if not calibration.empty:
        test = calibration[calibration["split"].astype(str).eq("test")].copy()
        route = test[test["group_method"].astype(str).str.contains("routecode", case=False, na=False)].sort_values(
            "weighted_utility_variance"
        )
        simple = test[test["group_method"].astype(str).isin(["benchmark_label"]) | test["group_method"].astype(str).str.contains("text_cluster")]
        if not route.empty and not simple.empty:
            route_var = float(route.iloc[0]["weighted_utility_variance"])
            simple_var = float(simple["weighted_utility_variance"].min())
            status = "supported_on_cached_broad100" if route_var < simple_var else "not_supported_on_cached_broad100"
            evidence = f"best_routecode_test_variance={route_var:.4f};best_label_or_text_variance={simple_var:.4f}"
        else:
            status = "table_generated"
            evidence = f"rows={len(calibration)}"
        rows.append(
            {
                "claim_id": "states_as_calibration_strata",
                "status": status,
                "evidence": evidence,
                "caveat": "Uses cached Broad100 outcomes; utility-cluster rows are diagnostic and not deployable.",
            }
        )
    if not onboarding.empty:
        active = onboarding[onboarding["method"].astype(str).eq("active_route_state_calibration")]
        random = onboarding[onboarding["method"].astype(str).eq("random_query_route_state_calibration")]
        if not active.empty and not random.empty:
            max_budget = int(min(active["budget"].max(), random["budget"].max()))
            active_u = float(active[active["budget"].eq(max_budget)]["mean_utility"].mean())
            random_u = float(random[random["budget"].eq(max_budget)]["mean_utility"].mean())
            status = "supported_on_cached_broad100" if active_u > random_u else "not_supported_on_cached_broad100"
            evidence = f"budget={max_budget};active_utility={active_u:.4f};random_utility={random_u:.4f}"
        else:
            status = "table_generated"
            evidence = f"rows={len(onboarding)}"
        rows.append(
            {
                "claim_id": "state_calibration_new_model_onboarding",
                "status": status,
                "evidence": evidence,
                "caveat": "Simulated held-out model/action onboarding from cached outcomes, not live deployment.",
            }
        )
    if not frozen.empty:
        rows.append(
            {
                "claim_id": "frozen_state_vs_direct_retrain_proxy",
                "status": "table_generated",
                "evidence": f"rows={len(frozen)}",
                "caveat": "Direct retraining baseline is a probe-feature utility-regressor proxy.",
            }
        )
    if not sensitivity.empty:
        rows.append(
            {
                "claim_id": "state_action_table_price_adaptation",
                "status": "table_generated",
                "evidence": f"rows={len(sensitivity)}",
                "caveat": "Price changes are simulated by scaling cached frontier costs.",
            }
        )
    if not ablation.empty:
        rows.append(
            {
                "claim_id": "final_ablation_coverage",
                "status": "table_generated",
                "evidence": f"rows={len(ablation)}",
                "caveat": "Ablation table consolidates cached Broad100 result rows and simulated onboarding rows.",
            }
        )
    if not real_calibration.empty:
        local_success = real_calibration[
            real_calibration["provider"].astype(str).eq("local")
            & real_calibration["status"].astype(str).eq("success")
        ]
        frontier_blocked = real_calibration[
            real_calibration["provider"].astype(str).isin(["openai", "google"])
            & real_calibration["status"].astype(str).str.contains("blocked", case=False, na=False)
        ]
        best_quality = float(local_success["mean_quality"].max()) if not local_success.empty else float("nan")
        rows.append(
            {
                "claim_id": "real_local_frontier_new_model_calibration",
                "status": "partial_local_live_only" if not local_success.empty else "not_run_or_blocked",
                "evidence": (
                    f"local_success_rows={len(local_success)};"
                    f"frontier_blocked_rows={len(frontier_blocked)};"
                    f"best_live_local_quality={best_quality:.4f}"
                ),
                "caveat": "Local vLLM smoke/live artifacts exist; GPT/Gemini live calls were not made without provider keys and budget approval.",
            }
        )
    if not predicted_utility_claims.empty:
        for row in predicted_utility_claims.to_dict("records"):
            rows.append(
                {
                    "claim_id": str(row.get("claim_id", "")),
                    "status": str(row.get("status", "")),
                    "evidence": str(row.get("evidence", "")),
                    "caveat": str(row.get("caveat", "")),
                }
            )
    if not live_predicted_claims.empty:
        for row in live_predicted_claims.to_dict("records"):
            rows.append(
                {
                    "claim_id": str(row.get("claim_id", "")),
                    "status": str(row.get("status", "")),
                    "evidence": str(row.get("evidence", "")),
                    "caveat": str(row.get("caveat", "")),
                }
            )
    if not final_claims.empty:
        for row in final_claims.to_dict("records"):
            rows.append(
                {
                    "claim_id": str(row.get("claim_id", "")),
                    "status": str(row.get("status", "")),
                    "evidence": str(row.get("evidence", "")),
                    "caveat": str(row.get("caveat", "")),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    path: Path,
    main_eval: pd.DataFrame,
    baseline_status: pd.DataFrame,
    calibration: pd.DataFrame,
    onboarding: pd.DataFrame,
    frozen: pd.DataFrame,
    sensitivity: pd.DataFrame,
    ablation: pd.DataFrame,
    real_calibration: pd.DataFrame,
    predicted_utility_claims: pd.DataFrame,
    predicted_utility_onboarding: pd.DataFrame,
    predicted_utility_variance: pd.DataFrame,
    live_predicted_claims: pd.DataFrame,
    live_predicted_onboarding: pd.DataFrame,
    live_predicted_variance: pd.DataFrame,
    live_frontier_validation: pd.DataFrame,
    live_frontier_test: pd.DataFrame,
    live_frontier_budget_efficiency: pd.DataFrame,
    claims: pd.DataFrame,
    config: dict,
) -> None:
    lines = [
        "# Phase 3 Final Evaluation Report",
        "",
        "This report is the current Phase 3 final evaluation package. The main routing, strata, onboarding, sensitivity, and ablation sections are cache-backed; the live-calibration section records local vLLM smoke calls and provider-readiness checks.",
        "",
    ]
    if not main_eval.empty:
        current = main_eval[main_eval["method_role"].eq("probecode_statecal")]
        if not current.empty:
            row = current.iloc[0]
            lines.extend(
                [
                    "## Main Broad100 Result",
                    "",
                    f"- Method: `{row['method']}`",
                    f"- Mean quality: `{float(row['mean_quality']):.4f}`",
                    f"- Mean utility: `{float(row['mean_utility']):.4f}`",
                    f"- Quality gap to oracle: `{float(row['quality_gap_to_oracle']):.4f}`",
                    f"- Oracle utility ratio: `{float(row['oracle_utility_ratio']):.4f}`",
                    f"- Frontier-call rate: `{float(row['frontier_call_rate']):.4f}`",
                    "",
                ]
            )
    lines.extend(["## Literature Baseline Status", ""])
    if baseline_status.empty:
        lines.append("No literature baseline status table found.")
    else:
        for row in baseline_status.to_dict("records"):
            lines.append(f"- `{row['baseline']}`: `{row['status']}`")
    if not calibration.empty:
        lines.extend(["", "## Calibration Strata", ""])
        test = calibration[calibration["split"].astype(str).eq("test")].sort_values("weighted_utility_variance")
        for row in test.head(8).to_dict("records"):
            lines.append(
                f"- `{row['group_method']}`: test utility variance `{float(row['weighted_utility_variance']):.4f}`"
            )
    if not onboarding.empty:
        lines.extend(["", "## New-Model Onboarding", ""])
        max_budget = int(onboarding[onboarding["budget"].ge(0)]["budget"].max())
        summary = (
            onboarding[onboarding["budget"].eq(max_budget)]
            .groupby("method", as_index=False)["mean_utility"]
            .mean()
            .sort_values("mean_utility", ascending=False)
        )
        for row in summary.head(10).to_dict("records"):
            lines.append(f"- `{row['method']}` at budget `{max_budget}`: utility `{float(row['mean_utility']):.4f}`")
    if not frozen.empty:
        lines.extend(["", "## Frozen State vs Retrain", ""])
        best = frozen.sort_values("mean_utility", ascending=False).groupby("comparison_family").head(1)
        for row in best.to_dict("records"):
            lines.append(
                f"- `{row['comparison_family']}`: best utility `{float(row['mean_utility']):.4f}` at budget `{int(row['budget'])}`"
            )
    if not sensitivity.empty:
        lines.extend(["", "## Cost Sensitivity", ""])
        state = sensitivity[sensitivity["method"].astype(str).eq("frozen_routecode_state_action_table")]
        for row in state.head(12).to_dict("records"):
            lines.append(
                f"- lambda `{float(row['lambda_cost']):.2f}`, price x`{float(row['frontier_price_multiplier']):.1f}`: "
                f"utility `{float(row['mean_utility']):.4f}`, frontier rate `{float(row['frontier_call_rate']):.4f}`"
            )
    if not ablation.empty:
        lines.extend(["", "## Ablation", ""])
        for row in ablation.head(12).to_dict("records"):
            lines.append(
                f"- `{row['ablation']}`: utility `{float(row['mean_utility']):.4f}`, "
                f"delta vs full `{float(row['utility_delta_vs_full']):.4f}`"
            )
    if not real_calibration.empty:
        lines.extend(["", "## Real Local/Frontier Calibration", ""])
        for row in real_calibration.head(12).to_dict("records"):
            quality = row.get("mean_quality", float("nan"))
            lines.append(
                f"- `{row['experiment']}` / `{row['model_id']}`: status `{row['status']}`, "
                f"calls `{int(row.get('n_calls', 0))}`, mean quality `{float(quality):.4f}`"
            )
    if not predicted_utility_claims.empty:
        lines.extend(["", "## Predicted Utility-State Calibration Update", ""])
        lines.append(
            "This update learns utility states on train and predicts them from observable cached probe features. "
            "It is the current best evidence for calibration strata and state-based new-model onboarding."
        )
        lines.append("")
        for row in predicted_utility_claims.to_dict("records"):
            lines.append(f"- `{row['claim_id']}`: `{row['status']}`; {row['evidence']}")
        if not predicted_utility_variance.empty:
            test = predicted_utility_variance[predicted_utility_variance["split"].astype(str).eq("test")]
            key_rows = test[
                test["group_method"].astype(str).isin(
                    ["predicted_utility_state_rf_probe_only_k24", "benchmark_label", "text_cluster_k8"]
                )
            ].sort_values("weighted_utility_variance")
            lines.extend(["", "Key held-out test strata variance rows:", ""])
            for row in key_rows.to_dict("records"):
                lines.append(f"- `{row['group_method']}`: `{float(row['weighted_utility_variance']):.4f}`")
        if not predicted_utility_onboarding.empty:
            budgeted = predicted_utility_onboarding[predicted_utility_onboarding["budget"].ge(0)]
            if not budgeted.empty:
                max_budget = int(budgeted["budget"].max())
                budget_rows = (
                    predicted_utility_onboarding[predicted_utility_onboarding["budget"].eq(max_budget)]
                    .groupby("method", as_index=False)
                    .agg(mean_utility=("mean_utility", "mean"), mean_quality=("mean_quality", "mean"))
                    .sort_values("mean_utility", ascending=False)
                )
                lines.extend(["", f"Key onboarding rows at budget {max_budget}:", ""])
                for row in budget_rows.to_dict("records"):
                    lines.append(
                        f"- `{row['method']}`: utility `{float(row['mean_utility']):.4f}`, "
                        f"quality `{float(row['mean_quality']):.4f}`"
                    )
    if not live_predicted_claims.empty:
        lines.extend(["", "## Live Broad100 Predicted Utility-State Update", ""])
        lines.append(
            "This update runs the same predicted utility-state logic on the live Stage0 matrix with GPT, Gemini, and local vLLM rows."
        )
        lines.append("")
        for row in live_predicted_claims.to_dict("records"):
            lines.append(f"- `{row['claim_id']}`: `{row['status']}`; {row['evidence']}")
        if not live_predicted_variance.empty:
            test = live_predicted_variance[live_predicted_variance["split"].astype(str).eq("test")]
            key_rows = test[
                test["group_method"].astype(str).isin(
                    [
                        "predicted_utility_state_rf_probe_plus_benchmark_k16",
                        "benchmark_label",
                        "text_cluster_k8",
                    ]
                )
            ].sort_values("weighted_utility_variance")
            lines.extend(["", "Live held-out test strata variance rows:", ""])
            for row in key_rows.to_dict("records"):
                lines.append(f"- `{row['group_method']}`: `{float(row['weighted_utility_variance']):.4f}`")
        if not live_predicted_onboarding.empty:
            budgeted = live_predicted_onboarding[live_predicted_onboarding["budget"].ge(0)]
            if not budgeted.empty:
                max_budget = int(budgeted["budget"].max())
                budget_rows = (
                    live_predicted_onboarding[live_predicted_onboarding["budget"].eq(max_budget)]
                    .groupby(["method", "acquisition"], as_index=False)
                    .agg(mean_utility=("mean_utility", "mean"), mean_quality=("mean_quality", "mean"))
                    .sort_values("mean_utility", ascending=False)
                )
                lines.extend(["", f"Live onboarding rows at budget {max_budget}:", ""])
                for row in budget_rows.to_dict("records"):
                    lines.append(
                        f"- `{row['method']}` / `{row['acquisition']}`: utility `{float(row['mean_utility']):.4f}`, "
                        f"quality `{float(row['mean_quality']):.4f}`"
                    )
        if not live_frontier_validation.empty and not live_frontier_test.empty:
            selected = live_frontier_validation.sort_values(
                ["active_minus_best_competitor", "active_mean_utility", "budget"],
                ascending=[False, False, True],
            ).head(1)
            if not selected.empty:
                budget = int(selected.iloc[0]["budget"])
                test_row = live_frontier_test[live_frontier_test["budget"].eq(budget)]
                lines.extend(["", "Live frontier onboarding slice:", ""])
                lines.append(
                    f"- Validation-selected budget `{budget}`: active validation margin "
                    f"`{float(selected.iloc[0]['active_minus_best_competitor']):.4f}`"
                )
                if not test_row.empty:
                    row = test_row.iloc[0]
                    lines.append(
                        f"- Test at budget `{budget}`: active `{float(row['active_mean_utility']):.4f}`, "
                        f"best competitor `{float(row['best_competitor_mean_utility']):.4f}`, "
                        f"margin `{float(row['active_minus_best_competitor']):.4f}`"
                    )
        if not live_frontier_budget_efficiency.empty:
            lines.extend(["", "Live frontier budget-to-match:", ""])
            for row in live_frontier_budget_efficiency.to_dict("records"):
                if bool(row["matched_active_utility"]):
                    match = f"matches at `{int(row['competitor_match_budget'])}` evals"
                else:
                    match = f"does not match by `{int(row['competitor_best_tested_budget'])}` evals"
                lines.append(
                    f"- `{row['competitor']}` {match}; active uses `{int(row['active_budget'])}` evals; "
                    f"reduction lower bound `{float(row['eval_reduction_lower_bound']):.1f}x`"
                )
    lines.extend(["", "## Claim Table", ""])
    if claims.empty:
        lines.append("No claims assembled.")
    else:
        for row in claims.to_dict("records"):
            lines.append(f"- `{row['claim_id']}`: `{row['status']}`; {row['evidence']}")
    remaining = []
    if baseline_status.empty or not bool(baseline_status["broad100_final_eval_included"].astype(bool).all()):
        remaining.append("Final split-aligned RouteLLM, GraphRouter, and Avengers-Pro comparisons on the same Broad100/action matrix.")
    if real_calibration.empty:
        remaining.append("Real local/frontier new-model calibration if new calls are approved.")
    else:
        frontier_ready = real_calibration[
            real_calibration["provider"].astype(str).isin(["openai", "google"])
            & real_calibration["status"].astype(str).eq("ready")
        ]
        if frontier_ready.empty:
            remaining.append("Approved GPT/Gemini live calibration calls with API keys, budget, token logging, and refreshed pricing.")
    lines.extend(["", "## Remaining Optional Or Follow-Up Work", ""])
    for item in remaining:
        lines.append(f"- {item}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    write_final_figures(path.parent)
    write_completion_audit(path.parent, baseline_status, calibration, onboarding, frozen, sensitivity, claims, real_calibration, config)


def write_final_figures(root: Path) -> None:
    copies = {
        root / "main_eval/fig_quality_cost_frontier.pdf": root / "fig_final_quality_cost_frontier.pdf",
        root / "new_model_onboarding/fig_utility_vs_calibration_budget.pdf": root / "fig_final_calibration_efficiency.pdf",
        root / "main_eval/fig_oracle_gap.pdf": root / "fig_final_oracle_gap.pdf",
    }
    for source, target in copies.items():
        if source.exists():
            shutil.copyfile(source, target)


def write_completion_audit(
    root: Path,
    baseline_status: pd.DataFrame,
    calibration: pd.DataFrame,
    onboarding: pd.DataFrame,
    frozen: pd.DataFrame,
    sensitivity: pd.DataFrame,
    claims: pd.DataFrame,
    real_calibration: pd.DataFrame,
    config: dict,
) -> None:
    required_paths = {
        "final_method": [
            root / "final_method/METHOD_CARD.md",
            root / "final_method/table_final_state_cards.csv",
            root / "final_method/code_cards.md",
        ],
        "main_routing": [root / "main_eval/table_main_routing_eval.csv"],
        "baseline_table": [root / "table_final_baselines.csv"],
        "calibration_strata": [
            root / "calibration_strata/table_state_variance.csv",
            root / "calibration_strata/table_state_estimation_error.csv",
            root / "calibration_strata/table_state_best_model_accuracy.csv",
        ],
        "new_model_onboarding": [root / "new_model_onboarding/table_new_model_onboarding.csv"],
        "frozen_state_vs_retrain": [root / "frozen_state_vs_retrain/table_frozen_state_vs_retrain.csv"],
        "cost_price_sensitivity": [root / "sensitivity/table_price_sensitivity.csv"],
        "final_report": [root / "FINAL_EVALUATION_REPORT.md", root / "table_final_claims.csv"],
        "final_root_figures": [
            root / "fig_final_quality_cost_frontier.pdf",
            root / "fig_final_calibration_efficiency.pdf",
            root / "fig_final_oracle_gap.pdf",
        ],
    }
    rows = []
    for item, paths in required_paths.items():
        rows.append(
            {
                "criterion": item,
                "status": "present" if all(path.exists() and path.stat().st_size > 0 for path in paths) else "missing",
                "evidence": ", ".join(str(path.relative_to(root)) for path in paths),
            }
        )
    output_matrix_path = Path(config.get("inputs", {}).get("broad100_outputs", ""))
    rows.append(
        {
            "criterion": "model_action_outputs_cached",
            "status": "present" if output_matrix_path.exists() and output_matrix_path.stat().st_size > 0 else "missing",
            "evidence": str(output_matrix_path),
        }
    )
    per_benchmark_path = root / "main_eval/table_per_benchmark_eval.csv"
    if per_benchmark_path.exists():
        per_benchmark = pd.read_csv(per_benchmark_path)
        n_benchmarks = int(per_benchmark["benchmark"].nunique()) if "benchmark" in per_benchmark else 0
        rows.append(
            {
                "criterion": "minimum_benchmark_scope",
                "status": "complete" if n_benchmarks >= 8 else "incomplete",
                "evidence": f"benchmark_families={n_benchmarks};path={per_benchmark_path.relative_to(root)}",
            }
        )
    if output_matrix_path.exists():
        try:
            matrix = pd.read_parquet(output_matrix_path, columns=["query_id", "model_id", "benchmark", "split"])
            rows.append(
                {
                    "criterion": "final_matrix_shape",
                    "status": "complete" if matrix["benchmark"].nunique() >= 8 and matrix["split"].astype(str).eq("test").any() else "incomplete",
                    "evidence": (
                        f"queries={matrix['query_id'].nunique()};"
                        f"actions={matrix['model_id'].nunique()};"
                        f"benchmarks={matrix['benchmark'].nunique()};"
                        f"splits={','.join(sorted(matrix['split'].astype(str).unique()))}"
                    ),
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit should record unexpected read failures.
            rows.append(
                {
                    "criterion": "final_matrix_shape",
                    "status": "unverified",
                    "evidence": f"{type(exc).__name__}: {exc}",
                }
            )
    if not baseline_status.empty:
        pending = baseline_status[~baseline_status["broad100_final_eval_included"].astype(bool)]
        rows.append(
            {
                "criterion": "three_literature_baselines",
                "status": "complete" if len(baseline_status) == 3 and pending.empty else "incomplete",
                "evidence": f"rows={len(baseline_status)};pending={len(pending)}",
            }
        )
    if not calibration.empty:
        claim = claim_status(claims, "states_as_calibration_strata")
        rows.append(
            {
                "criterion": "calibration_strata_claim_result",
                "status": claim or "table_generated",
                "evidence": "supported/unsupported claim is recorded conservatively in table_final_claims.csv",
            }
        )
    if not onboarding.empty:
        claim = claim_status(claims, "state_calibration_new_model_onboarding")
        rows.append(
            {
                "criterion": "active_onboarding_claim_result",
                "status": claim or "table_generated",
                "evidence": "supported/unsupported claim is recorded conservatively in table_final_claims.csv",
            }
        )
    predicted_strata = claim_status(claims, "predicted_states_as_calibration_strata")
    if predicted_strata:
        rows.append(
            {
                "criterion": "predicted_utility_state_calibration_strata_claim",
                "status": predicted_strata,
                "evidence": "new predicted-state claim is recorded in table_final_claims.csv",
            }
        )
    predicted_onboarding = claim_status(claims, "predicted_state_new_model_onboarding")
    if predicted_onboarding:
        rows.append(
            {
                "criterion": "predicted_utility_state_onboarding_claim",
                "status": predicted_onboarding,
                "evidence": "new predicted-state onboarding claim is recorded in table_final_claims.csv",
            }
        )
    active_advantage = claim_status(claims, "active_acquisition_advantage")
    if active_advantage:
        rows.append(
            {
                "criterion": "active_acquisition_advantage_claim",
                "status": active_advantage,
                "evidence": "active acquisition advantage is recorded separately from state-based onboarding",
            }
        )
    live_strata = claim_status(claims, "live_predicted_states_as_calibration_strata")
    if live_strata:
        rows.append(
            {
                "criterion": "live_predicted_utility_state_calibration_strata_claim",
                "status": live_strata,
                "evidence": "live Stage0 predicted-state claim is recorded in table_final_claims.csv",
            }
        )
    live_onboarding = claim_status(claims, "live_predicted_state_new_model_onboarding")
    if live_onboarding:
        rows.append(
            {
                "criterion": "live_predicted_utility_state_onboarding_claim",
                "status": live_onboarding,
                "evidence": "live Stage0 predicted-state onboarding claim is recorded in table_final_claims.csv",
            }
        )
    live_active = claim_status(claims, "live_active_acquisition_advantage")
    if live_active:
        rows.append(
            {
                "criterion": "live_active_acquisition_advantage_claim",
                "status": live_active,
                "evidence": "live Stage0 active acquisition advantage is recorded separately",
            }
        )
    live_frontier = claim_status(claims, "live_frontier_active_onboarding_low_budget")
    if live_frontier:
        rows.append(
            {
                "criterion": "live_frontier_active_onboarding_low_budget_claim",
                "status": live_frontier,
                "evidence": "GPT/Gemini frontier onboarding slice is recorded in table_final_claims.csv",
            }
        )
    live_frontier_efficiency = claim_status(claims, "live_frontier_budget_efficiency")
    if live_frontier_efficiency:
        rows.append(
            {
                "criterion": "live_frontier_budget_efficiency_claim",
                "status": live_frontier_efficiency,
                "evidence": "frontier budget-to-match table is recorded in table_final_claims.csv",
            }
        )
    if not sensitivity.empty:
        claim = claim_status(claims, "state_action_table_price_adaptation")
        rows.append(
            {
                "criterion": "price_adaptation_claim_result",
                "status": claim or "table_generated",
                "evidence": f"rows={len(sensitivity)}",
            }
        )
    if not real_calibration.empty:
        claim = claim_status(claims, "real_local_frontier_new_model_calibration")
        rows.append(
            {
                "criterion": "optional_live_call_status",
                "status": claim or "table_generated",
                "evidence": "local vLLM calls plus GPT/Gemini readiness status are recorded",
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(root / "table_final_completion_audit.csv", index=False)
    lines = [
        "# Phase 3 Final Completion Audit",
        "",
        "This audit maps the completion criteria in `phase3/PHASE3_FINAL_EVALUATION_GOAL.md` to concrete artifacts in this result folder.",
        "",
    ]
    for row in rows:
        lines.append(f"- `{row['criterion']}`: `{row['status']}`; {row['evidence']}")
    lines.append("")
    (root / "COMPLETION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def claim_status(claims: pd.DataFrame, claim_id: str) -> str:
    if claims.empty:
        return ""
    match = claims[claims["claim_id"].astype(str).eq(claim_id)]
    if match.empty:
        return ""
    return str(match.iloc[0]["status"])


if __name__ == "__main__":
    main()
