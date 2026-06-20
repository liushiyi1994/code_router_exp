from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from routecode.probes.policies import select_models_from_belief
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-beliefs", required=True)
    parser.add_argument("--after-beliefs", required=True)
    parser.add_argument("--state-model-utility", required=True)
    parser.add_argument("--query-model-utility", required=True)
    parser.add_argument("--output-dir", default="results/phase2/probe_action_observability")
    args = parser.parse_args()
    run(
        before_beliefs_path=args.before_beliefs,
        after_beliefs_path=args.after_beliefs,
        state_model_utility_path=args.state_model_utility,
        query_model_utility_path=args.query_model_utility,
        output_dir=args.output_dir,
    )


def run(
    *,
    before_beliefs_path: str,
    after_beliefs_path: str,
    state_model_utility_path: str,
    query_model_utility_path: str,
    output_dir: str,
) -> dict[str, str]:
    before_beliefs = _read_matrix(before_beliefs_path)
    after_beliefs = _read_matrix(after_beliefs_path)
    state_model_utility = _read_matrix(state_model_utility_path)
    query_model_utility = _read_matrix(query_model_utility_path)
    by_query = build_action_observability_table(
        before_beliefs=before_beliefs,
        after_beliefs=after_beliefs,
        state_model_utility=state_model_utility,
        query_model_utility=query_model_utility,
    )
    summary = summarize_action_observability(by_query)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": str(out_dir / "table_probe_action_observability.csv"),
        "by_query": str(out_dir / "table_probe_action_observability_by_query.csv"),
    }
    summary.to_csv(paths["summary"], index=False)
    by_query.to_csv(paths["by_query"], index=False)
    write_memo(out_dir, summary, by_query, paths)
    append_readme(out_dir, summary, paths)
    print(f"Wrote probe action-observability summary to {paths['summary']}")
    print(f"Wrote probe action-observability by-query table to {paths['by_query']}")
    return paths


def build_action_observability_table(
    *,
    before_beliefs: pd.DataFrame,
    after_beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    query_model_utility: pd.DataFrame,
) -> pd.DataFrame:
    common_index = before_beliefs.index.intersection(after_beliefs.index).intersection(query_model_utility.index)
    if common_index.empty:
        raise ValueError("No overlapping query IDs across beliefs and query-model utility")
    before = before_beliefs.loc[common_index]
    after = after_beliefs.loc[common_index]
    query_utility = query_model_utility.loc[common_index]
    before_state = before.idxmax(axis=1)
    after_state = after.idxmax(axis=1)
    before_state_action = state_model_utility.loc[before_state].idxmax(axis=1)
    after_state_action = state_model_utility.loc[after_state].idxmax(axis=1)
    before_selected = select_models_from_belief(before, state_model_utility)
    after_selected = select_models_from_belief(after, state_model_utility)
    oracle_selected = query_utility.idxmax(axis=1)
    before_utility = _selected_values(query_utility, before_selected)
    after_utility = _selected_values(query_utility, after_selected)
    oracle_utility = query_utility.max(axis=1)
    belief_l1 = (before - after).abs().sum(axis=1)
    belief_top_prob_shift = after.max(axis=1) - before.max(axis=1)
    frame = pd.DataFrame(
        {
            "query_id": common_index.astype(str),
            "before_top_state": before_state.to_numpy(),
            "after_top_state": after_state.to_numpy(),
            "before_top_state_action": before_state_action.to_numpy(),
            "after_top_state_action": after_state_action.to_numpy(),
            "before_selected_model": before_selected.to_numpy(),
            "after_selected_model": after_selected.to_numpy(),
            "oracle_selected_model": oracle_selected.to_numpy(),
            "before_utility": before_utility.to_numpy(dtype=float),
            "after_utility": after_utility.to_numpy(dtype=float),
            "oracle_utility": oracle_utility.to_numpy(dtype=float),
            "belief_l1_shift": belief_l1.to_numpy(dtype=float),
            "belief_top_prob_shift": belief_top_prob_shift.to_numpy(dtype=float),
        }
    )
    frame["top_state_changed"] = frame["before_top_state"] != frame["after_top_state"]
    frame["top_state_action_changed"] = frame["before_top_state_action"] != frame["after_top_state_action"]
    frame["selected_model_changed"] = frame["before_selected_model"] != frame["after_selected_model"]
    frame["action_equivalent_top_state_change"] = frame["top_state_changed"] & ~frame["selected_model_changed"]
    frame["before_matches_oracle_model"] = frame["before_selected_model"] == frame["oracle_selected_model"]
    frame["after_matches_oracle_model"] = frame["after_selected_model"] == frame["oracle_selected_model"]
    frame["before_regret"] = frame["oracle_utility"] - frame["before_utility"]
    frame["after_regret"] = frame["oracle_utility"] - frame["after_utility"]
    frame["utility_delta"] = frame["after_utility"] - frame["before_utility"]
    frame["regret_delta"] = frame["after_regret"] - frame["before_regret"]
    columns = [
        "query_id",
        "before_top_state",
        "after_top_state",
        "top_state_changed",
        "before_top_state_action",
        "after_top_state_action",
        "top_state_action_changed",
        "before_selected_model",
        "after_selected_model",
        "selected_model_changed",
        "action_equivalent_top_state_change",
        "oracle_selected_model",
        "before_matches_oracle_model",
        "after_matches_oracle_model",
        "before_utility",
        "after_utility",
        "oracle_utility",
        "utility_delta",
        "before_regret",
        "after_regret",
        "regret_delta",
        "belief_l1_shift",
        "belief_top_prob_shift",
    ]
    return frame[columns]


def summarize_action_observability(by_query: pd.DataFrame) -> pd.DataFrame:
    n_queries = int(len(by_query))
    return pd.DataFrame(
        [
            {
                "n_queries": n_queries,
                "top_state_changes": int(by_query["top_state_changed"].sum()),
                "top_state_change_rate": _mean_bool(by_query["top_state_changed"]),
                "top_state_action_changes": int(by_query["top_state_action_changed"].sum()),
                "selected_model_changes": int(by_query["selected_model_changed"].sum()),
                "selected_model_change_rate": _mean_bool(by_query["selected_model_changed"]),
                "action_equivalent_top_state_changes": int(by_query["action_equivalent_top_state_change"].sum()),
                "before_oracle_model_match_rate": _mean_bool(by_query["before_matches_oracle_model"]),
                "after_oracle_model_match_rate": _mean_bool(by_query["after_matches_oracle_model"]),
                "mean_before_utility": float(by_query["before_utility"].mean()) if n_queries else 0.0,
                "mean_after_utility": float(by_query["after_utility"].mean()) if n_queries else 0.0,
                "mean_utility_delta": float(by_query["utility_delta"].mean()) if n_queries else 0.0,
                "mean_before_regret": float(by_query["before_regret"].mean()) if n_queries else 0.0,
                "mean_after_regret": float(by_query["after_regret"].mean()) if n_queries else 0.0,
                "mean_regret_delta": float(by_query["regret_delta"].mean()) if n_queries else 0.0,
                "mean_belief_l1_shift": float(by_query["belief_l1_shift"].mean()) if n_queries else 0.0,
                "mean_belief_top_prob_shift": float(by_query["belief_top_prob_shift"].mean()) if n_queries else 0.0,
            }
        ]
    )


def write_memo(out_dir: Path, summary: pd.DataFrame, by_query: pd.DataFrame, paths: dict[str, str]) -> None:
    row = summary.iloc[0]
    lines = [
        "# Probe Action Observability",
        "",
        "This diagnostic checks whether a probe changes latent-state beliefs in ways that change the final selected model.",
        "",
        _markdown_table(summary),
        "",
        (
            f"Top-state changes: `{int(row['top_state_changes'])}/{int(row['n_queries'])}`; "
            f"selected-model changes: `{int(row['selected_model_changes'])}/{int(row['n_queries'])}`."
        ),
        (
            f"Action-equivalent top-state changes: `{int(row['action_equivalent_top_state_changes'])}`."
        ),
        "",
        "Outputs:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
        "Rows with selected-model changes:",
        "",
        _markdown_table(by_query[by_query["selected_model_changed"]].head(10)),
        "",
    ]
    (out_dir / "m14_probe_action_observability_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, summary: pd.DataFrame, paths: dict[str, str]) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## Probe Action Observability"
    row = summary.iloc[0]
    lines = [
        marker,
        "",
        (
            f"Top-state changes: `{int(row['top_state_changes'])}/{int(row['n_queries'])}`; "
            f"selected-model changes: `{int(row['selected_model_changes'])}/{int(row['n_queries'])}`."
        ),
        f"Action-equivalent top-state changes: `{int(row['action_equivalent_top_state_changes'])}`.",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _read_matrix(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path) if Path(path).suffix == ".parquet" else pd.read_csv(path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    if "state_label" in frame.columns:
        return frame.set_index("state_label")
    return frame


def _selected_values(query_model_utility: pd.DataFrame, selected_models: pd.Series) -> pd.Series:
    return pd.Series(
        [float(query_model_utility.loc[query_id, model_id]) for query_id, model_id in selected_models.items()],
        index=selected_models.index,
        name="selected_utility",
    )


def _mean_bool(values: pd.Series) -> float:
    return float(values.astype(bool).mean()) if len(values) else 0.0


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                value = "" if pd.isna(value) else f"{float(value):.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
