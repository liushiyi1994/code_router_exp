from __future__ import annotations

import pandas as pd


class BestSingleRouter:
    def __init__(self) -> None:
        self.model_id: str | None = None

    def fit(self, query_info: pd.DataFrame, utility: pd.DataFrame) -> "BestSingleRouter":
        del query_info
        self.model_id = str(utility.mean(axis=0).idxmax())
        return self

    def predict(self, query_info: pd.DataFrame) -> pd.Series:
        if self.model_id is None:
            raise RuntimeError("BestSingleRouter must be fit before predict")
        return pd.Series(self.model_id, index=query_info.index, name="selected_model")


class CheapestRouter:
    def __init__(self) -> None:
        self.model_id: str | None = None

    def fit(self, query_info: pd.DataFrame, cost: pd.DataFrame) -> "CheapestRouter":
        del query_info
        self.model_id = str(cost.mean(axis=0).idxmin())
        return self

    def predict(self, query_info: pd.DataFrame) -> pd.Series:
        if self.model_id is None:
            raise RuntimeError("CheapestRouter must be fit before predict")
        return pd.Series(self.model_id, index=query_info.index, name="selected_model")
