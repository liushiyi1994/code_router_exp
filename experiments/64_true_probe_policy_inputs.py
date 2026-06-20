from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from routecode.probes.policies import select_models_from_belief
from routecode.reporting import upsert_markdown_section


PROBE_NUMERIC_COLUMNS = [
    "self_confidence",
    "agreement_score",
    "knn_label_entropy",
    "knn_winner_entropy",
    "latency_sec",
    "input_tokens",
    "output_tokens",
    "probe_cost_proxy",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-features", required=True)
    parser.add_argument("--state-targets", required=True)
    parser.add_argument("--query-features", required=True)
    parser.add_argument("--state-model-utility", required=True)
    parser.add_argument("--query-model-utility", required=True)
    parser.add_argument("--output-dir", default="results/phase2/true_probe_policy_inputs")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    run(
        probe_features_path=args.probe_features,
        state_targets_path=args.state_targets,
        query_features_path=args.query_features,
        state_model_utility_path=args.state_model_utility,
        query_model_utility_path=args.query_model_utility,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


def run(
    *,
    probe_features_path: str,
    state_targets_path: str,
    query_features_path: str,
    state_model_utility_path: str,
    query_model_utility_path: str,
    output_dir: str,
    random_state: int = 0,
) -> dict[str, str]:
    probe_features = pd.read_parquet(probe_features_path)
    state_targets = pd.read_csv(state_targets_path)
    query_features = pd.read_csv(query_features_path)
    state_model_utility = _read_policy_matrix(state_model_utility_path, index_column="state_label")
    query_model_utility = _read_policy_matrix(query_model_utility_path, index_column="query_id")
    bundle = build_true_probe_policy_inputs(
        probe_features=probe_features,
        state_targets=state_targets,
        query_features=query_features,
        state_model_utility=state_model_utility,
        query_model_utility=query_model_utility,
        random_state=random_state,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = write_outputs(out_dir, bundle)
    write_memo(out_dir, paths, bundle, probe_features_path, state_targets_path, query_features_path)
    append_readme(out_dir, paths, bundle)
    print(f"Wrote true-probe policy inputs to {out_dir}")
    return paths


def build_true_probe_policy_inputs(
    *,
    probe_features: pd.DataFrame,
    state_targets: pd.DataFrame,
    query_features: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    query_model_utility: pd.DataFrame,
    random_state: int = 0,
) -> dict[str, pd.DataFrame]:
    _require_columns(state_targets, ["query_id", "state_label", "split"], "state_targets")
    _require_columns(query_features, ["query_id"], "query_features")
    _require_columns(probe_features, ["query_id"], "probe_features")
    probe_by_query = _aggregate_probe_features(probe_features)
    query_by_query = _normalise_query_features(query_features)
    frame = (
        state_targets[["query_id", "state_label", "split"]]
        .drop_duplicates("query_id")
        .merge(query_by_query, on="query_id", how="inner")
        .merge(probe_by_query, on="query_id", how="inner")
    )
    policy_query_ids = list(query_model_utility.index.intersection(frame["query_id"]))
    if not policy_query_ids:
        raise ValueError("No policy query IDs overlap state targets, query features, probes, and query-model utility")
    train = frame[frame["split"].astype(str).eq("train")].copy()
    policy_frame = frame[frame["query_id"].isin(policy_query_ids)].copy()
    if train["state_label"].nunique() < 2:
        raise ValueError("Training rows must contain at least two route-state classes")
    state_columns = list(state_model_utility.index.astype(str))
    query_columns = [column for column in query_by_query.columns if column != "query_id"]
    probe_columns = [column for column in PROBE_NUMERIC_COLUMNS if column in frame.columns]
    before_model = _fit_state_model(train, query_columns, random_state=random_state)
    after_model = _fit_state_model(train, query_columns + probe_columns, random_state=random_state)
    before = _predict_beliefs(before_model, policy_frame, query_columns, state_columns)
    after = _predict_beliefs(after_model, policy_frame, query_columns + probe_columns, state_columns)
    before_value = _selected_value(before, state_model_utility, query_model_utility)
    after_value = _selected_value(after, state_model_utility, query_model_utility)
    predicted_gain = pd.DataFrame(
        {
            "query_id": policy_query_ids,
            "predicted_gain": (after_value - before_value).reindex(policy_query_ids).to_numpy(dtype=float),
        }
    )
    probe_cost = (
        policy_frame[["query_id", "probe_cost_proxy"]]
        .drop_duplicates("query_id")
        .set_index("query_id")
        .reindex(policy_query_ids)
        .fillna(0.0)
        .rename(columns={"probe_cost_proxy": "probe_cost"})
        .reset_index()
    )
    return {
        "before_beliefs": before.reset_index().rename(columns={"index": "query_id"}),
        "after_beliefs": after.reset_index().rename(columns={"index": "query_id"}),
        "state_model_utility": state_model_utility.reset_index().rename(columns={state_model_utility.index.name or "index": "state_label"}),
        "query_model_utility": query_model_utility.loc[policy_query_ids].reset_index().rename(
            columns={query_model_utility.index.name or "index": "query_id"}
        ),
        "probe_cost": probe_cost,
        "predicted_gain": predicted_gain,
        "metadata": pd.DataFrame(
            [
                {
                    "train_rows": int(len(train)),
                    "policy_rows": int(len(policy_query_ids)),
                    "state_count": int(len(state_columns)),
                    "query_feature_count": int(len(query_columns)),
                    "probe_feature_count": int(len(probe_columns)),
                }
            ]
        ),
    }


def write_outputs(out_dir: Path, bundle: dict[str, pd.DataFrame]) -> dict[str, str]:
    paths = {
        "before_beliefs": str(out_dir / "true_probe_before_beliefs.csv"),
        "after_beliefs": str(out_dir / "true_probe_after_beliefs.csv"),
        "state_model_utility": str(out_dir / "true_probe_state_model_utility.csv"),
        "query_model_utility": str(out_dir / "true_probe_query_model_utility.csv"),
        "probe_cost": str(out_dir / "true_probe_cost.csv"),
        "predicted_gain": str(out_dir / "true_probe_predicted_gain.csv"),
        "metadata": str(out_dir / "true_probe_policy_input_metadata.json"),
    }
    for key, path in paths.items():
        if key == "metadata":
            Path(path).write_text(
                json.dumps(bundle[key].iloc[0].to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        else:
            bundle[key].to_csv(path, index=False)
    return paths


def write_memo(
    out_dir: Path,
    paths: dict[str, str],
    bundle: dict[str, pd.DataFrame],
    probe_features_path: str,
    state_targets_path: str,
    query_features_path: str,
) -> None:
    metadata = bundle["metadata"].iloc[0].to_dict()
    lines = [
        "# True Local Probe Policy Inputs",
        "",
        "This step turns true local probe features into latent route-state beliefs for M5 policy evaluation. It is not direct probe-to-model routing.",
        "",
        "Inputs:",
        "",
        f"- Probe features: `{probe_features_path}`",
        f"- State targets: `{state_targets_path}`",
        f"- Query features: `{query_features_path}`",
        "",
        "Summary:",
        "",
        f"- Train rows for belief models: `{int(metadata['train_rows'])}`.",
        f"- Policy query rows: `{int(metadata['policy_rows'])}`.",
        f"- Route states: `{int(metadata['state_count'])}`.",
        "",
        "Outputs:",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    (out_dir / "m12_true_probe_policy_inputs_memo.md").write_text("\n".join(lines), encoding="utf-8")


def append_readme(out_dir: Path, paths: dict[str, str], bundle: dict[str, pd.DataFrame]) -> None:
    readme_path = out_dir / "README.md"
    existing = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "# RouteCode Phase 2 Results\n"
    marker = "## True Local Probe Policy Inputs"
    metadata = bundle["metadata"].iloc[0].to_dict()
    lines = [
        marker,
        "",
        "Creates before/after latent route-state beliefs from true local probe features for M5. This preserves the Phase 2 invariant: query/probe -> belief over latent route states -> selected model.",
        "",
        f"- Train rows: `{int(metadata['train_rows'])}`.",
        f"- Policy rows: `{int(metadata['policy_rows'])}`.",
        f"- Route states: `{int(metadata['state_count'])}`.",
        "",
        _markdown_table(pd.DataFrame({"artifact": list(paths), "path": list(paths.values())})),
        "",
    ]
    readme_path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


def _fit_state_model(frame: pd.DataFrame, feature_columns: list[str], *, random_state: int) -> Pipeline:
    feature_columns = _dedupe(feature_columns)
    if not feature_columns:
        raise ValueError("At least one feature column is required")
    model = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                                    ("scale", StandardScaler()),
                                ]
                            ),
                            feature_columns,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    model.fit(frame[feature_columns], _state_names(frame["state_label"]))
    return model


def _predict_beliefs(model: Pipeline, frame: pd.DataFrame, feature_columns: list[str], state_columns: list[str]) -> pd.DataFrame:
    probabilities = model.predict_proba(frame[feature_columns])
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    beliefs = pd.DataFrame(0.0, index=frame["query_id"].astype(str), columns=state_columns)
    for class_index, state_name in enumerate(classes):
        if state_name in beliefs.columns:
            beliefs[state_name] = probabilities[:, class_index]
    row_sums = beliefs.sum(axis=1).replace(0.0, np.nan)
    beliefs = beliefs.div(row_sums, axis=0).fillna(1.0 / max(len(state_columns), 1))
    beliefs.index.name = "query_id"
    return beliefs


def _selected_value(
    beliefs: pd.DataFrame,
    state_model_utility: pd.DataFrame,
    query_model_utility: pd.DataFrame,
) -> pd.Series:
    selected = select_models_from_belief(beliefs, state_model_utility)
    return pd.Series(
        [float(query_model_utility.loc[query_id, model_id]) for query_id, model_id in selected.items()],
        index=selected.index,
    )


def _aggregate_probe_features(probe_features: pd.DataFrame) -> pd.DataFrame:
    columns = ["query_id"] + [column for column in PROBE_NUMERIC_COLUMNS if column in probe_features.columns]
    frame = probe_features[columns].copy()
    for column in columns:
        if column != "query_id":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.groupby("query_id", as_index=False).mean(numeric_only=True)


def _normalise_query_features(query_features: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column for column in query_features.select_dtypes(include=[np.number]).columns if column != "query_id"
    ]
    return query_features[["query_id"] + numeric_columns].drop_duplicates("query_id")


def _read_policy_matrix(path: str, *, index_column: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if index_column not in frame.columns:
        raise ValueError(f"{path} missing required column {index_column}")
    return frame.set_index(index_column)


def _state_names(labels: pd.Series) -> pd.Series:
    return labels.astype(int).map(lambda value: f"z{value}")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


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
