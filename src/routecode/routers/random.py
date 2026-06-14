from __future__ import annotations

import numpy as np
import pandas as pd


class RandomRouter:
    """Seeded random model selector over the train model pool."""

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = int(random_state)
        self.model_ids: list[str] = []

    def fit(self, query_info: pd.DataFrame, utility: pd.DataFrame) -> "RandomRouter":
        del query_info
        self.model_ids = [str(model_id) for model_id in utility.columns]
        if not self.model_ids:
            raise ValueError("RandomRouter requires at least one model")
        return self

    def predict(self, query_info: pd.DataFrame) -> pd.Series:
        if not self.model_ids:
            raise RuntimeError("RandomRouter must be fit before predict")
        rng = np.random.default_rng(self.random_state)
        selected = rng.choice(self.model_ids, size=len(query_info), replace=True)
        return pd.Series(selected, index=query_info.index, name="selected_model").astype(str)
