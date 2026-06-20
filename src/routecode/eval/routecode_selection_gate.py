from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from routecode.codes.predictability_constrained import PredictabilityConstrainedRouteCode
from routecode.config import load_config
from routecode.pipeline import prepare_from_config


@dataclass(frozen=True)
class RouteCodeSelectionConfig:
    k_values: tuple[int, ...]
    alpha_values: tuple[float, ...]
    training_datasets: tuple[str, ...] | None = None
    validation_datasets: tuple[str, ...] = ("aime", "math500")
    threshold: float = 0.03
    target_k: int | None = None


def evaluate_routecode_selection_gate(
    *,
    config_path: str,
    query_model_utility_path: str,
    output_dir: str,
    selection: RouteCodeSelectionConfig,
) -> dict[str, str]:
    config = load_config(config_path)
    prepared = prepare_from_config(config)
    train = prepared.matrices["train"]
    val = prepared.matrices["val"]
    test = prepared.matrices["test"]
    embeddings = prepared.embeddings
    policy_utility = _read_matrix(query_model_utility_path)

    train_ids = (
        _dataset_query_ids(train.query_info, selection.training_datasets)
        if selection.training_datasets
        else train.utility.index
    )
    val_ids = _dataset_query_ids(val.query_info, selection.validation_datasets)
    test_ids = _dataset_query_ids(test.query_info, selection.validation_datasets)
    rows: list[dict[str, object]] = []
    for k in selection.k_values:
        for alpha in selection.alpha_values:
            codebook = PredictabilityConstrainedRouteCode(
                k,
                alpha=alpha,
                beta=float(config.get("predictability_constrained", {}).get("beta", 0.0)),
                random_state=int(config.get("run", {}).get("random_seed", 0)),
                max_iter=int(config.get("predictability_constrained", {}).get("max_iter", 25)),
                refinement_iter=int(config.get("predictability_constrained", {}).get("refinement_iter", 10)),
            ).fit(train.query_info.loc[train_ids], train.utility.loc[train_ids], embeddings)
            val_selected = codebook.predict_from_labels(codebook.predict_labels(embeddings.loc[val_ids]))
            test_selected = codebook.predict_from_labels(codebook.predict_labels(embeddings.loc[test_ids]))
            policy_selected = codebook.predict_from_labels(codebook.predict_labels(embeddings.loc[policy_utility.index]))
            rows.append(
                {
                    "candidate": f"routecode_embedding_predicted:k{k}:alpha{alpha:g}",
                    "k": k,
                    "alpha": alpha,
                    "selection_basis": "exact_math_validation_grid",
                    "training_datasets": ",".join(selection.training_datasets or ("all",)),
                    "validation_datasets": ",".join(selection.validation_datasets),
                    "n_train_queries": len(train_ids),
                    "n_validation_queries": len(val_ids),
                    "n_test_queries": len(test_ids),
                    "n_policy_slice_queries": len(policy_utility),
                    "val_relative_gap_to_oracle": _relative_gap(
                        _selected_values(val.utility.loc[val_ids], val_selected),
                        val.utility.loc[val_ids],
                    ),
                    "test_relative_gap_to_oracle": _relative_gap(
                        _selected_values(test.utility.loc[test_ids], test_selected),
                        test.utility.loc[test_ids],
                    ),
                    "policy_slice_relative_gap_to_oracle": _relative_gap(
                        _selected_values(policy_utility, policy_selected),
                        policy_utility,
                    ),
                    "policy_slice_mean_utility": float(_selected_values(policy_utility, policy_selected).mean()),
                    "policy_slice_oracle_mean_utility": float(policy_utility.max(axis=1).mean()),
                    "policy_slice_regret_count": int(
                        (policy_utility.max(axis=1) - _selected_values(policy_utility, policy_selected) > 1e-12).sum()
                    ),
                }
            )

    table = rank_selection_candidates(pd.DataFrame(rows), threshold=selection.threshold, target_k=selection.target_k)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "table_routecode_exact_math_selection.csv"
    memo_path = out_dir / "m_routecode_exact_math_selection.md"
    table.to_csv(table_path, index=False)
    memo_path.write_text(_memo(table, selection, config_path, query_model_utility_path, output_dir), encoding="utf-8")
    return {"table": str(table_path), "memo": str(memo_path)}


def rank_selection_candidates(table: pd.DataFrame, *, threshold: float, target_k: int | None = None) -> pd.DataFrame:
    ranked = table.copy()
    ranked = ranked.sort_values(["val_relative_gap_to_oracle", "policy_slice_relative_gap_to_oracle", "k", "alpha"])
    ranked["val_selection_rank"] = np.arange(1, len(ranked) + 1)
    ranked["selected_by_val"] = ranked["val_selection_rank"].eq(1)
    ranked["policy_slice_within_threshold"] = ranked["policy_slice_relative_gap_to_oracle"].le(threshold)
    ranked["val_selected_policy_slice_within_threshold"] = (
        ranked["selected_by_val"] & ranked["policy_slice_within_threshold"]
    )
    ranked["selected_by_target_rate"] = False
    ranked["target_rate_selection_rank"] = np.nan
    ranked["target_rate_policy_slice_within_threshold"] = False
    if target_k is not None:
        target_candidates = ranked[ranked["k"].eq(int(target_k))].copy()
        if not target_candidates.empty:
            target_candidates = target_candidates.sort_values(["val_relative_gap_to_oracle", "alpha"])
            target_ranks = pd.Series(
                np.arange(1, len(target_candidates) + 1),
                index=target_candidates.index,
                dtype=float,
            )
            ranked.loc[target_ranks.index, "target_rate_selection_rank"] = target_ranks
            selected_index = target_candidates.index[0]
            ranked.loc[selected_index, "selected_by_target_rate"] = True
            ranked["target_rate_policy_slice_within_threshold"] = (
                ranked["selected_by_target_rate"] & ranked["policy_slice_within_threshold"]
            )
    return ranked


def _dataset_query_ids(query_info: pd.DataFrame, datasets: tuple[str, ...]) -> pd.Index:
    mask = query_info["dataset"].astype(str).isin(datasets)
    ids = query_info.index[mask]
    if len(ids) == 0:
        raise ValueError(f"No validation queries found for datasets={datasets}")
    return ids


def _selected_values(query_utility: pd.DataFrame, selected: pd.Series) -> pd.Series:
    selected = selected.reindex(query_utility.index)
    return pd.Series(
        [float(query_utility.loc[query_id, model_id]) for query_id, model_id in selected.items()],
        index=query_utility.index,
        name="selected_utility",
    )


def _relative_gap(values: pd.Series, query_utility: pd.DataFrame) -> float:
    oracle = query_utility.max(axis=1)
    oracle_mean = float(oracle.mean())
    return float((oracle_mean - float(values.mean())) / oracle_mean) if oracle_mean else np.nan


def _read_matrix(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if "query_id" in frame.columns:
        return frame.set_index("query_id")
    return frame


def _memo(
    table: pd.DataFrame,
    selection: RouteCodeSelectionConfig,
    config_path: str,
    query_model_utility_path: str,
    output_dir: str,
) -> str:
    selected = table[table["selected_by_val"]].iloc[0]
    target_selected = table[table["selected_by_target_rate"]].iloc[0] if table["selected_by_target_rate"].any() else None
    within = table[table["policy_slice_within_threshold"]].copy()
    lines = [
        "# RouteCode Exact-Math Selection Gate",
        "",
        "Command:",
        "",
        "```bash",
        (
            "PYTHONPATH=src python experiments/72_routecode_exact_math_selection.py "
            f"--config {config_path} "
            f"--query-model-utility {query_model_utility_path} "
            f"--output-dir {output_dir} "
            f"--k-values {','.join(str(value) for value in selection.k_values)} "
            f"--alpha-values {','.join(f'{value:g}' for value in selection.alpha_values)} "
            f"--validation-datasets {','.join(selection.validation_datasets)} "
            f"--threshold {selection.threshold:g}"
            + (f" --target-k {selection.target_k}" if selection.target_k is not None else "")
            + (
                f" --training-datasets {','.join(selection.training_datasets)}"
                if selection.training_datasets
                else ""
            )
        ),
        "```",
        "",
        f"Training datasets: `{','.join(selection.training_datasets or ('all',))}`.",
        f"Validation datasets: `{','.join(selection.validation_datasets)}`.",
        f"Policy-slice threshold: `{selection.threshold:.4f}` relative gap to oracle.",
        f"Target K selector: `{selection.target_k if selection.target_k is not None else 'disabled'}`.",
        "",
        "Validation-selected candidate:",
        "",
        (
            f"- `{selected['candidate']}`: validation gap `{selected['val_relative_gap_to_oracle']:.4f}`, "
            f"test gap `{selected['test_relative_gap_to_oracle']:.4f}`, "
            f"policy-slice gap `{selected['policy_slice_relative_gap_to_oracle']:.4f}`, "
            f"within threshold `{bool(selected['policy_slice_within_threshold'])}`."
        ),
        "",
    ]
    if target_selected is not None:
        lines.extend(
            [
                "Target-rate validation candidate:",
                "",
                (
                    f"- `{target_selected['candidate']}`: validation gap "
                    f"`{target_selected['val_relative_gap_to_oracle']:.4f}`, "
                    f"test gap `{target_selected['test_relative_gap_to_oracle']:.4f}`, "
                    f"policy-slice gap `{target_selected['policy_slice_relative_gap_to_oracle']:.4f}`, "
                    f"within threshold `{bool(target_selected['policy_slice_within_threshold'])}`."
                ),
                "",
            ]
        )
    if within.empty:
        lines.append("No candidate in this grid reaches the policy-slice threshold.")
    else:
        lines.extend(
            [
                "Candidates within the policy-slice threshold:",
                "",
                _markdown_table(
                    within[
                        [
                            "candidate",
                            "val_selection_rank",
                            "val_relative_gap_to_oracle",
                            "test_relative_gap_to_oracle",
                            "policy_slice_relative_gap_to_oracle",
                            "policy_slice_regret_count",
                        ]
                    ]
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Interpretation: a candidate that is within 3% on the held-out policy slice is not enough by itself. "
            "It must also be selected by a pre-declared validation protocol before it can replace the current core policy.",
            "",
        ]
    )
    return "\n".join(lines)


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
