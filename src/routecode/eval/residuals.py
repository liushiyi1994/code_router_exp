from __future__ import annotations

import numpy as np
import pandas as pd


def residual_concentration_table(
    oracle_utility: pd.Series,
    selected_utility: pd.Series,
    fractions: list[float] | None = None,
) -> pd.DataFrame:
    if fractions is None:
        fractions = [0.05, 0.10, 0.20]
    regret = (oracle_utility - selected_utility).clip(lower=0).sort_values(ascending=False)
    total_regret = float(regret.sum())
    rows = []
    for fraction in fractions:
        n_top = max(1, int(np.ceil(float(fraction) * len(regret))))
        top_regret = float(regret.iloc[:n_top].sum())
        rows.append(
            {
                "top_fraction": float(fraction),
                "n_queries": n_top,
                "top_regret": top_regret,
                "total_regret": total_regret,
                "regret_mass_fraction": 0.0 if total_regret == 0 else top_regret / total_regret,
            }
        )
    return pd.DataFrame(rows)


def residual_query_table(
    utility: pd.DataFrame,
    selected_models: pd.Series,
    labels: pd.Series | None = None,
    embeddings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    selected_models = selected_models.reindex(utility.index)
    oracle_models = utility.idxmax(axis=1)
    oracle_utility = utility.max(axis=1)
    selected_utility = pd.Series(
        [float(utility.loc[query_id, model_id]) for query_id, model_id in selected_models.items()],
        index=utility.index,
    )
    sorted_utility = np.sort(utility.to_numpy(), axis=1)
    margins = sorted_utility[:, -1] - sorted_utility[:, -2] if utility.shape[1] > 1 else np.zeros(len(utility))

    table = pd.DataFrame(
        {
            "selected_model": selected_models,
            "oracle_model": oracle_models,
            "selected_utility": selected_utility,
            "oracle_utility": oracle_utility,
            "regret": (oracle_utility - selected_utility).clip(lower=0),
            "oracle_margin": margins,
        },
        index=utility.index,
    )
    if labels is not None:
        labels = labels.reindex(utility.index)
        table["route_label"] = labels
    if labels is not None and embeddings is not None:
        aligned_embeddings = embeddings.loc[utility.index]
        table["distance_to_label_centroid"] = _distance_to_label_centroid(aligned_embeddings, labels)
    return table


def residual_risk_coverage_table(
    residuals: pd.DataFrame,
    score_columns: list[str],
    top_fractions: list[float] | None = None,
    regret_column: str = "regret",
) -> pd.DataFrame:
    if top_fractions is None:
        top_fractions = [0.05, 0.10, 0.20]
    if regret_column not in residuals.columns:
        raise ValueError(f"residuals must include {regret_column}")

    regret = residuals[regret_column].astype(float).clip(lower=0)
    positive = regret > 0
    total_regret = float(regret.sum())
    total_positive = int(positive.sum())
    rows = []
    for score_column in score_columns:
        if score_column not in residuals.columns:
            continue
        score = residuals[score_column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(float("-inf"))
        ordered = score.sort_values(ascending=False).index
        auc = _binary_auc(positive, score)
        for fraction in top_fractions:
            n_flagged = max(1, int(np.ceil(float(fraction) * len(residuals)))) if len(residuals) else 0
            flagged = ordered[:n_flagged]
            flagged_regret = float(regret.loc[flagged].sum()) if n_flagged else 0.0
            flagged_positive = int(positive.loc[flagged].sum()) if n_flagged else 0
            unflagged = regret.index.difference(flagged)
            rows.append(
                {
                    "score": str(score_column),
                    "top_fraction": float(fraction),
                    "n_flagged": int(n_flagged),
                    "flagged_regret": flagged_regret,
                    "total_regret": total_regret,
                    "regret_mass_fraction": 0.0 if total_regret == 0 else flagged_regret / total_regret,
                    "positive_regret_recall": 0.0 if total_positive == 0 else flagged_positive / total_positive,
                    "mean_regret_flagged": float(regret.loc[flagged].mean()) if n_flagged else 0.0,
                    "mean_regret_unflagged": float(regret.loc[unflagged].mean()) if len(unflagged) else 0.0,
                    "auc_regret_positive": auc,
                }
            )
    return pd.DataFrame(rows)


def label_residual_summary(query_residuals: pd.DataFrame) -> pd.DataFrame:
    if "route_label" not in query_residuals.columns:
        raise ValueError("query_residuals must include route_label")
    grouped = query_residuals.groupby("route_label")
    rows = []
    for label, group in grouped:
        rows.append(
            {
                "route_label": label,
                "n_queries": len(group),
                "mean_regret": float(group["regret"].mean()),
                "p90_regret": float(group["regret"].quantile(0.90)),
                "max_regret": float(group["regret"].max()),
                "winner_entropy": _entropy(group["oracle_model"].astype(str)),
                "mean_oracle_margin": float(group["oracle_margin"].mean()),
                "mean_distance_to_centroid": float(group.get("distance_to_label_centroid", pd.Series(0)).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_regret", ascending=False)


def _distance_to_label_centroid(embeddings: pd.DataFrame, labels: pd.Series) -> pd.Series:
    distances = pd.Series(0.0, index=embeddings.index)
    for label, query_ids in labels.groupby(labels).groups.items():
        ids = list(query_ids)
        centroid = embeddings.loc[ids].mean(axis=0).to_numpy(dtype=float)
        values = embeddings.loc[ids].to_numpy(dtype=float)
        distances.loc[ids] = np.sqrt(((values - centroid) ** 2).sum(axis=1))
    return distances


def _entropy(labels: pd.Series) -> float:
    counts = labels.value_counts()
    probs = counts / counts.sum()
    value = float(-(probs * np.log2(probs)).sum())
    return 0.0 if abs(value) < 1e-12 else value


def _binary_auc(labels: pd.Series, scores: pd.Series) -> float:
    positives = scores[labels.astype(bool)]
    negatives = scores[~labels.astype(bool)]
    if positives.empty or negatives.empty:
        return float("nan")
    greater = 0.0
    for value in positives:
        greater += float((value > negatives).sum())
        greater += 0.5 * float((value == negatives).sum())
    return greater / float(len(positives) * len(negatives))
