from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from routecode.probes.policies import select_models_from_belief
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-beliefs", required=True)
    parser.add_argument("--after-beliefs", required=True)
    parser.add_argument("--state-model-utility", required=True)
    parser.add_argument("--query-model-utility", required=True)
    parser.add_argument("--output-dir", default="results/phase2/true_probe_decision_value")
    parser.add_argument("--predicted-gain", default="")
    parser.add_argument("--probe-cost", default="")
    args = parser.parse_args()
    run(
        before_beliefs_path=args.before_beliefs,
        after_beliefs_path=args.after_beliefs,
        state_model_utility_path=args.state_model_utility,
        query_model_utility_path=args.query_model_utility,
        output_dir=args.output_dir,
        predicted_gain_path=args.predicted_gain or None,
        probe_cost_path=args.probe_cost or None,
    )


def run(
    *,
    before_beliefs_path: str,
    after_beliefs_path: str,
    state_model_utility_path: str,
    query_model_utility_path: str,
    output_dir: str,
    predicted_gain_path: str | None = None,
    probe_cost_path: str | None = None,
) -> dict[str, str]:
    before_beliefs = _read_matrix(before_beliefs_path)
    after_beliefs = _read_matrix(after_beliefs_path)
    state_model_utility = _read_matrix(state_model_utility_path)
    query_model_utility = _read_matrix(query_model_utility_path)
    predicted_gain = _read_optional_series(predicted_gain_path)
    probe_cost = _read_optional_series(probe_cost_path)

    by_query = build_decision_value_table(
        before_beliefs=before_beliefs,
        after_beliefs=after_beliefs,
        state_model_utility=state_model_utility,
        query_model_utility=query_model_utility,
        predicted_gain=predicted_gain,
        probe_cost=probe_cost,
    )
    summary = summarize_decision_value(by_query)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": str(out_dir / "table_true_probe_decision_value.csv"),
        "by_query": str(out_dir / "table_true_probe_decision_value_by_query.csv"),
    }
    summary.to_csv(paths["summary"], index=False)
    by_query.to_csv(paths["by_query"], index=False)
    write_memo(out_dir, summary, by_query, paths)
    append_readme(out_dir, summary, paths)
    print(f"Wrote true-probe decision-value summary to {paths['summary']}")
    print(f"Wrote true-probe decision-value by-query table to {paths['by_query']}")
    return paths


def build_decision_value_table(
    *,
    before_beliefs: pd.DataFrame,
    after_beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    query_model_utility: pd.DataFrame,
    predicted_gain: pd.Series | None = None,
    probe_cost: pd.Series | None = None,
) -> pd.DataFrame:
    common_index = before_beliefs.index.intersection(after_beliefs.index).intersection(query_model_utility.index)
    if common_index.empty:
        raise ValueError("No overlapping query IDs across beliefs and query-model utility")
    before = before_beliefs.loc[common_index]
    after = after_beliefs.loc[common_index]
    utility = query_model_utility.loc[common_index]
    before_selected = select_models_from_belief(before, state_model_utility)
    after_selected = select_models_from_belief(after, state_model_utility)
    before_utility = _selected_values(utility, before_selected)
    after_utility = _selected_values(utility, after_selected)
    frame = pd.DataFrame(
        {
            "query_id": common_index.astype(str),
            "before_selected_model": before_selected.reindex(common_index).astype(str).to_numpy(),
            "after_selected_model": after_selected.reindex(common_index).astype(str).to_numpy(),
            "before_utility": before_utility.reindex(common_index).to_numpy(dtype=float),
            "after_utility": after_utility.reindex(common_index).to_numpy(dtype=float),
        }
    )
    frame["selected_changed"] = frame["before_selected_model"] != frame["after_selected_model"]
    frame["utility_delta"] = frame["after_utility"] - frame["before_utility"]
    frame["predicted_gain"] = _align_optional(predicted_gain, common_index)
    frame["probe_cost"] = _align_optional(probe_cost, common_index)
    return frame[
        [
            "query_id",
            "before_selected_model",
            "after_selected_model",
            "selected_changed",
            "before_utility",
            "after_utility",
            "utility_delta",
            "predicted_gain",
            "probe_cost",
        ]
    ]


def summarize_decision_value(by_query: pd.DataFrame) -> pd.DataFrame:
    n_queries = int(len(by_query))
    selected_changes = int(by_query["selected_changed"].sum())
    predicted_gain = pd.to_numeric(by_query["predicted_gain"], errors="coerce").fillna(0.0)
    probe_cost = pd.to_numeric(by_query["probe_cost"], errors="coerce").fillna(0.0)
    return pd.DataFrame(
        [
            {
                "n_queries": n_queries,
                "selected_model_changes": selected_changes,
                "selected_model_change_rate": selected_changes / n_queries if n_queries else 0.0,
                "mean_before_utility": float(by_query["before_utility"].mean()) if n_queries else 0.0,
                "mean_after_utility": float(by_query["after_utility"].mean()) if n_queries else 0.0,
                "mean_utility_delta": float(by_query["utility_delta"].mean()) if n_queries else 0.0,
                "positive_utility_delta_rows": int((by_query["utility_delta"] > 0.0).sum()),
                "negative_utility_delta_rows": int((by_query["utility_delta"] < 0.0).sum()),
                "mean_predicted_gain": float(predicted_gain.mean()) if n_queries else 0.0,
                "nonzero_predicted_gain_rows": int((predicted_gain.abs() > 1e-12).sum()),
                "mean_probe_cost": float(probe_cost.mean()) if n_queries else 0.0,
            }
        ]
    )


def write_memo(out_dir: Path, summary: pd.DataFrame, by_query: pd.DataFrame, paths: dict[str, str]) -> None:
    row = summary.iloc[0]
    lines = [
        "# True Local Probe Decision Value",
        "",
        "This decision value diagnostic measures whether true local probe features change the latent-state routing decision before probe-cost accounting.",
        "",
        "Summary:",
        "",
        _markdown_table(summary),
        "",
        (
            f"The selected model changed on `{int(row['selected_model_changes'])}/{int(row['n_queries'])}` "
            "held-out utility rows."
        ),
        (
            f"Mean realized utility moved from `{float(row['mean_before_utility']):.4f}` to "
            f"`{float(row['mean_after_utility']):.4f}`."
        ),
        (
            f"Nonzero predicted-gain rows: `{int(row['nonzero_predicted_gain_rows'])}/{int(row['n_queries'])}`."
        ),
        "",
        "Outputs:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
        "Changed-decision rows:",
        "",
        _markdown_table(by_query[by_query["selected_changed"]].head(10)),
        "",
    ]
    (out_dir / "m13_true_probe_decision_value_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, summary: pd.DataFrame, paths: dict[str, str]) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## True Local Probe Decision Value"
    row = summary.iloc[0]
    lines = [
        marker,
        "",
        "Measures whether true local probe features produce decision value after the query/probe -> latent-state belief update.",
        "",
        (
            f"- Selected model changed on `{int(row['selected_model_changes'])}/{int(row['n_queries'])}` "
            "held-out utility rows."
        ),
        (
            f"- Mean realized utility before/after: `{float(row['mean_before_utility']):.4f}` / "
            f"`{float(row['mean_after_utility']):.4f}`."
        ),
        f"- Nonzero predicted-gain rows: `{int(row['nonzero_predicted_gain_rows'])}/{int(row['n_queries'])}`.",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _read_matrix(path: str) -> pd.DataFrame:
    frame = _read_table(path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    if "state_label" in frame.columns:
        return frame.set_index("state_label")
    return frame


def _read_optional_series(path: str | None) -> pd.Series | None:
    if not path:
        return None
    frame = _read_table(path)
    if "query_id" in frame.columns:
        value_columns = [column for column in frame.columns if column != "query_id"]
        if len(value_columns) != 1:
            raise ValueError(f"Expected one value column in {path}")
        return frame.set_index("query_id")[value_columns[0]]
    if frame.shape[1] != 1:
        raise ValueError(f"Expected one value column in {path}")
    return frame.iloc[:, 0]


def _read_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if table_path.suffix == ".parquet":
        return pd.read_parquet(table_path)
    return pd.read_csv(table_path)


def _selected_values(query_model_utility: pd.DataFrame, selected_models: pd.Series) -> pd.Series:
    return pd.Series(
        [float(query_model_utility.loc[query_id, model_id]) for query_id, model_id in selected_models.items()],
        index=selected_models.index,
        name="selected_utility",
    )


def _align_optional(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(0.0, index=index, dtype=float).to_numpy(dtype=float)
    return pd.to_numeric(series.reindex(index), errors="coerce").fillna(0.0).to_numpy(dtype=float)


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
            if isinstance(value, float):
                value = "" if pd.isna(value) else f"{value:.4f}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
