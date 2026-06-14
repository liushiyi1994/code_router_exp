from __future__ import annotations

import pandas as pd


class DatasetLabelRouter:
    def __init__(self, label_column: str = "dataset") -> None:
        self.label_column = label_column
        self.label_to_model: dict[str, str] = {}
        self.fallback_model: str | None = None

    def fit(self, query_info: pd.DataFrame, utility: pd.DataFrame) -> "DatasetLabelRouter":
        if self.label_column not in query_info.columns:
            raise ValueError(f"Missing label column: {self.label_column}")

        self.fallback_model = str(utility.mean(axis=0).idxmax())
        self.label_to_model = {}
        labels = query_info[self.label_column].astype(str)
        for label, query_ids in labels.groupby(labels).groups.items():
            label_utility = utility.loc[list(query_ids)]
            self.label_to_model[str(label)] = str(label_utility.mean(axis=0).idxmax())
        return self

    def predict(self, query_info: pd.DataFrame) -> pd.Series:
        if self.fallback_model is None:
            raise RuntimeError("DatasetLabelRouter must be fit before predict")
        if self.label_column not in query_info.columns:
            raise ValueError(f"Missing label column: {self.label_column}")
        if "query_id" in query_info.columns:
            query_info = query_info.set_index("query_id", drop=False)
        selected = [
            self.label_to_model.get(str(label), self.fallback_model)
            for label in query_info[self.label_column].astype(str)
        ]
        return pd.Series(selected, index=query_info.index, name="selected_model")


class DatasetOracleRouter(DatasetLabelRouter):
    """Dataset-level oracle diagnostic fit on the matrix being evaluated.

    This is not a deployable train-only router. It is a named upper-bound
    diagnostic for how much dataset identity alone can explain routing.
    """
