from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config
from routecode.pipeline import prepare_from_config
from routecode.probes.policies import select_models_from_belief
from routecode.reporting import upsert_markdown_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/phase2/oracle_gap_gate")
    parser.add_argument("--threshold", type=float, default=0.03)
    parser.add_argument("--config", default="")
    parser.add_argument("--query-model-utility", default="")
    parser.add_argument("--policy-table", action="append", default=[], help="name=path to table_proberoute_policy.csv")
    parser.add_argument("--policy-input-dir", action="append", default=[], help="name=directory with true_probe_* inputs")
    parser.add_argument(
        "--routecode-candidate",
        action="append",
        default=[],
        help="K:alpha[:selection_basis] for train-fit RouteCode embedding-centroid candidates",
    )
    parser.add_argument(
        "--dataset-model-candidate",
        action="append",
        default=[],
        help="name;dataset=model,dataset=model;selection_basis for benchmark-label route rules",
    )
    args = parser.parse_args()
    run(
        output_dir=args.output_dir,
        threshold=args.threshold,
        config_path=args.config or None,
        query_model_utility_path=args.query_model_utility or None,
        policy_table_specs=args.policy_table,
        policy_input_dir_specs=args.policy_input_dir,
        routecode_candidate_specs=args.routecode_candidate,
        dataset_model_candidate_specs=args.dataset_model_candidate,
    )


def run(
    *,
    output_dir: str,
    threshold: float = 0.03,
    config_path: str | None = None,
    query_model_utility_path: str | None = None,
    policy_table_specs: list[str] | None = None,
    policy_input_dir_specs: list[str] | None = None,
    routecode_candidate_specs: list[str] | None = None,
    dataset_model_candidate_specs: list[str] | None = None,
) -> dict[str, str]:
    rows: list[dict[str, object]] = []
    for name, path in _parse_named_paths(policy_table_specs or []):
        rows.extend(_rows_from_policy_table(name, path, threshold))
    for name, path in _parse_named_paths(policy_input_dir_specs or []):
        rows.extend(_rows_from_policy_input_dir(name, Path(path), threshold))
    if routecode_candidate_specs:
        if not config_path or not query_model_utility_path:
            raise ValueError("--config and --query-model-utility are required for --routecode-candidate")
        rows.extend(
            _rows_from_routecode_candidates(
                config_path=config_path,
                query_model_utility_path=query_model_utility_path,
                candidate_specs=routecode_candidate_specs,
                threshold=threshold,
            )
        )
    if dataset_model_candidate_specs:
        if not config_path or not query_model_utility_path:
            raise ValueError("--config and --query-model-utility are required for --dataset-model-candidate")
        rows.extend(
            _rows_from_dataset_model_candidates(
                config_path=config_path,
                query_model_utility_path=query_model_utility_path,
                candidate_specs=dataset_model_candidate_specs,
                threshold=threshold,
            )
        )
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No oracle-gap rows were produced")
    table = table.sort_values(
        ["within_threshold", "relative_gap_to_oracle", "candidate"],
        ascending=[False, True, True],
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "table_oracle_gap_gate.csv"
    readme_path = out_dir / "README.md"
    table.to_csv(table_path, index=False)
    write_readme(readme_path, table, threshold)
    print(f"Wrote oracle-gap gate table to {table_path}")
    return {"table": str(table_path), "readme": str(readme_path)}


def _rows_from_policy_table(name: str, path: str, threshold: float) -> list[dict[str, object]]:
    table = pd.read_csv(path)
    rows = []
    for _, row in table[table["status"].astype(str).eq("executed")].iterrows():
        mean = float(row["mean_net_utility"])
        oracle = float(row["mean_net_utility"] + row["mean_oracle_regret"])
        rows.append(
            _gate_row(
                candidate=f"{name}:{row['policy']}",
                candidate_type="proberoute_policy",
                selection_basis="current_phase2_policy_table",
                deployable=True,
                n_queries=int(row["n_queries"]),
                mean_utility=mean,
                oracle_mean_utility=oracle,
                threshold=threshold,
                regret_count=np.nan,
                notes="Existing M5 policy row with probe-cost accounting.",
            )
        )
    return rows


def _rows_from_policy_input_dir(name: str, path: Path, threshold: float) -> list[dict[str, object]]:
    before = _read_matrix(path / "true_probe_before_beliefs.csv")
    after = _read_matrix(path / "true_probe_after_beliefs.csv")
    state_utility = _read_matrix(path / "true_probe_state_model_utility.csv")
    query_utility = _read_matrix(path / "true_probe_query_model_utility.csv")
    rows = []
    for label, beliefs in [("before_belief_expected", before), ("after_belief_expected", after)]:
        selected = select_models_from_belief(beliefs, state_utility)
        rows.append(
            _selection_gate_row(
                candidate=f"{name}:{label}",
                candidate_type="belief_expected_action",
                selection_basis="current_phase2_policy_inputs",
                deployable=True,
                selected=selected,
                query_utility=query_utility,
                threshold=threshold,
                notes="Expected model utility under predicted state belief.",
            )
        )
    for label, beliefs in [("before_hard_top_state", before), ("after_hard_top_state", after)]:
        top_state = beliefs.idxmax(axis=1)
        selected = pd.Series(
            state_utility.loc[top_state].idxmax(axis=1).to_numpy(),
            index=beliefs.index,
            name="selected_model",
        )
        rows.append(
            _selection_gate_row(
                candidate=f"{name}:{label}",
                candidate_type="hard_top_state_action",
                selection_basis="current_phase2_policy_inputs",
                deployable=True,
                selected=selected,
                query_utility=query_utility,
                threshold=threshold,
                notes="Best action for top predicted state only; no belief averaging.",
            )
        )
    return rows


def _rows_from_dataset_model_candidates(
    *,
    config_path: str,
    query_model_utility_path: str,
    candidate_specs: list[str],
    threshold: float,
) -> list[dict[str, object]]:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    eval_utility = _read_matrix(query_model_utility_path)
    query_info = prepared.outcomes.drop_duplicates("query_id").set_index("query_id")
    rows = []
    for spec in candidate_specs:
        name, mapping, selection_basis = _parse_dataset_model_candidate(spec)
        selected = _select_by_dataset_model_map(
            query_info=query_info.loc[eval_utility.index],
            mapping=mapping,
            fallback_model=None,
        )
        row = _selection_gate_row(
            candidate=f"dataset_model_rule:{name}",
            candidate_type="benchmark_label_route_rule",
            selection_basis=selection_basis,
            deployable=True,
            selected=selected,
            query_utility=eval_utility,
            threshold=threshold,
            notes="Benchmark-label route rule: query metadata -> route label -> selected model.",
        )
        row.update(_dataset_model_val_test_gap_columns(prepared, mapping))
        rows.append(row)
    return rows


def _dataset_model_val_test_gap_columns(prepared, mapping: dict[str, str]) -> dict[str, float]:
    columns: dict[str, float] = {}
    query_info = prepared.outcomes.drop_duplicates("query_id").set_index("query_id")
    for split_name in ["val", "test"]:
        matrix = prepared.matrices[split_name]
        query_ids = matrix.utility.index[matrix.query_info["dataset"].astype(str).isin(mapping)]
        if len(query_ids) == 0:
            columns[f"{split_name}_relative_gap_to_oracle"] = np.nan
            continue
        selected = _select_by_dataset_model_map(
            query_info=query_info.loc[query_ids],
            mapping=mapping,
            fallback_model=None,
        )
        values = _selected_values(matrix.utility.loc[query_ids], selected)
        oracle = matrix.utility.loc[query_ids].max(axis=1)
        columns[f"{split_name}_relative_gap_to_oracle"] = _relative_gap(values, oracle)
    return columns


def _select_by_dataset_model_map(
    *,
    query_info: pd.DataFrame,
    mapping: dict[str, str],
    fallback_model: str | None,
) -> pd.Series:
    selected = query_info["dataset"].astype(str).map(mapping)
    if selected.isna().any():
        if fallback_model is None:
            missing = sorted(query_info.loc[selected.isna(), "dataset"].astype(str).unique())
            raise ValueError(f"Dataset-model rule missing datasets: {missing}")
        selected = selected.fillna(fallback_model)
    return selected.rename("selected_model")


def _rows_from_routecode_candidates(
    *,
    config_path: str,
    query_model_utility_path: str,
    candidate_specs: list[str],
    threshold: float,
) -> list[dict[str, object]]:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    val = prepared.matrices["val"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    eval_utility = _read_matrix(query_model_utility_path)
    rows = []
    for spec in candidate_specs:
        k, alpha, selection_basis = _parse_routecode_candidate(spec)
        codebook = PredictabilityConstrainedRouteCode(
            k,
            alpha=alpha,
            beta=float(config.get("predictability_constrained", {}).get("beta", 0.0)),
            random_state=int(config.get("run", {}).get("random_seed", 0)),
            max_iter=int(config.get("predictability_constrained", {}).get("max_iter", 25)),
            refinement_iter=int(config.get("predictability_constrained", {}).get("refinement_iter", 10)),
        ).fit(train.query_info, train.utility, embeddings)
        labels = codebook.predict_labels(embeddings.loc[eval_utility.index])
        selected = codebook.predict_from_labels(labels)
        row = _selection_gate_row(
            candidate=f"routecode_embedding_predicted:k{k}:alpha{alpha:g}",
            candidate_type="routecode_embedding_centroid",
            selection_basis=selection_basis,
            deployable=True,
            selected=selected,
            query_utility=eval_utility,
            threshold=threshold,
            notes="Train-fit RouteCode labels predicted from query embeddings.",
        )
        row.update(_val_test_gap_columns(codebook, val.utility, test.utility, embeddings))
        rows.append(row)
        joint_labels = codebook.predict_joint_labels(eval_utility, embeddings.loc[eval_utility.index])
        joint_selected = codebook.predict_from_labels(joint_labels)
        upper = _selection_gate_row(
            candidate=f"routecode_state_oracle_upper:k{k}:alpha{alpha:g}",
            candidate_type="routecode_state_oracle_upper",
            selection_basis=f"{selection_basis}; diagnostic_uses_eval_utility_for_label_assignment",
            deployable=False,
            selected=joint_selected,
            query_utility=eval_utility,
            threshold=threshold,
            notes="Diagnostic upper bound: assigns labels using eval utility, not deployable.",
        )
        upper.update(_val_test_gap_columns(codebook, val.utility, test.utility, embeddings))
        rows.append(upper)
    return rows


def _val_test_gap_columns(
    codebook: PredictabilityConstrainedRouteCode,
    val_utility: pd.DataFrame,
    test_utility: pd.DataFrame,
    embeddings: pd.DataFrame,
) -> dict[str, float]:
    val_selected = codebook.predict_from_labels(codebook.predict_labels(embeddings.loc[val_utility.index]))
    test_selected = codebook.predict_from_labels(codebook.predict_labels(embeddings.loc[test_utility.index]))
    return {
        "val_relative_gap_to_oracle": _relative_gap(_selected_values(val_utility, val_selected), val_utility.max(axis=1)),
        "test_relative_gap_to_oracle": _relative_gap(
            _selected_values(test_utility, test_selected),
            test_utility.max(axis=1),
        ),
    }


def _selection_gate_row(
    *,
    candidate: str,
    candidate_type: str,
    selection_basis: str,
    deployable: bool,
    selected: pd.Series,
    query_utility: pd.DataFrame,
    threshold: float,
    notes: str,
) -> dict[str, object]:
    selected = selected.reindex(query_utility.index)
    values = _selected_values(query_utility, selected)
    oracle = query_utility.max(axis=1)
    return _gate_row(
        candidate=candidate,
        candidate_type=candidate_type,
        selection_basis=selection_basis,
        deployable=deployable,
        n_queries=len(query_utility),
        mean_utility=float(values.mean()),
        oracle_mean_utility=float(oracle.mean()),
        threshold=threshold,
        regret_count=int((oracle - values > 1e-12).sum()),
        notes=notes,
    )


def _gate_row(
    *,
    candidate: str,
    candidate_type: str,
    selection_basis: str,
    deployable: bool,
    n_queries: int,
    mean_utility: float,
    oracle_mean_utility: float,
    threshold: float,
    regret_count: int | float,
    notes: str,
) -> dict[str, object]:
    abs_gap = float(oracle_mean_utility - mean_utility)
    rel_gap = abs_gap / float(oracle_mean_utility) if oracle_mean_utility else np.nan
    return {
        "candidate": candidate,
        "candidate_type": candidate_type,
        "selection_basis": selection_basis,
        "deployable": bool(deployable),
        "n_queries": int(n_queries),
        "mean_utility": float(mean_utility),
        "oracle_mean_utility": float(oracle_mean_utility),
        "abs_gap_to_oracle": abs_gap,
        "relative_gap_to_oracle": rel_gap,
        "threshold": float(threshold),
        "within_threshold": bool(rel_gap <= threshold),
        "regret_count": regret_count,
        "notes": notes,
    }


def _selected_values(query_utility: pd.DataFrame, selected: pd.Series) -> pd.Series:
    return pd.Series(
        [float(query_utility.loc[query_id, model_id]) for query_id, model_id in selected.items()],
        index=selected.index,
        name="selected_utility",
    )


def _relative_gap(values: pd.Series, oracle: pd.Series) -> float:
    oracle_mean = float(oracle.mean())
    return float((oracle_mean - float(values.mean())) / oracle_mean) if oracle_mean else np.nan


def _read_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    if "state_label" in frame.columns:
        return frame.set_index("state_label")
    return frame


def _parse_named_paths(specs: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Expected name=path, got {spec}")
        name, path = spec.split("=", 1)
        parsed.append((name.strip(), path.strip()))
    return parsed


def _parse_routecode_candidate(spec: str) -> tuple[int, float, str]:
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise ValueError(f"Expected K:alpha[:selection_basis], got {spec}")
    basis = parts[2] if len(parts) == 3 else "manual_candidate"
    return int(parts[0]), float(parts[1]), basis


def _parse_dataset_model_candidate(spec: str) -> tuple[str, dict[str, str], str]:
    parts = spec.split(";", 2)
    if len(parts) != 3:
        raise ValueError(f"Expected name;dataset=model,dataset=model;selection_basis, got {spec}")
    name, raw_mapping, basis = [part.strip() for part in parts]
    mapping = {}
    for item in raw_mapping.split(","):
        if "=" not in item:
            raise ValueError(f"Expected dataset=model item in {spec}")
        dataset, model = item.split("=", 1)
        mapping[dataset.strip()] = model.strip()
    if not name or not mapping:
        raise ValueError(f"Invalid dataset-model candidate: {spec}")
    return name, mapping, basis


def write_readme(path: Path, table: pd.DataFrame, threshold: float) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else "# RouteCode Phase 2 Oracle-Gap Gate\n"
    marker = "## Oracle-Gap Gate"
    best = table.sort_values("relative_gap_to_oracle").iloc[0]
    deployable = table[table["deployable"].astype(bool)]
    best_deployable = deployable.sort_values("relative_gap_to_oracle").iloc[0] if not deployable.empty else None
    current = table[table["selection_basis"].eq("current_phase2_policy_table")]
    current_best = current.sort_values("relative_gap_to_oracle").iloc[0] if not current.empty else None
    lines = [
        marker,
        "",
        f"Threshold: relative gap to query oracle <= `{threshold:.4f}`.",
        "",
        (
            f"Best row: `{best['candidate']}` with mean utility `{best['mean_utility']:.4f}` "
            f"versus oracle `{best['oracle_mean_utility']:.4f}` "
            f"(relative gap `{best['relative_gap_to_oracle']:.4f}`)."
        ),
    ]
    if best_deployable is not None:
        lines.append(
            f"Best deployable row: `{best_deployable['candidate']}` with mean utility "
            f"`{best_deployable['mean_utility']:.4f}` versus oracle "
            f"`{best_deployable['oracle_mean_utility']:.4f}` "
            f"(relative gap `{best_deployable['relative_gap_to_oracle']:.4f}`)."
        )
    if current_best is not None:
        lines.append(
            f"Best current Phase 2 policy row: `{current_best['candidate']}` with relative gap "
            f"`{current_best['relative_gap_to_oracle']:.4f}`."
        )
    lines.extend(
        [
            "",
            "Important: rows marked `deployable = False` are diagnostic upper bounds. Rows whose `selection_basis` says `policy_slice` are useful as candidates, but should be validated on a held-out selection protocol before being reported as final.",
            "",
            _markdown_table(table),
            "",
        ]
    )
    path.write_text(upsert_markdown_section(existing, marker, lines), encoding="utf-8")


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
