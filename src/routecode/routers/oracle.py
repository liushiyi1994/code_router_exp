from __future__ import annotations

import pandas as pd


class OracleRouter:
    """Query oracle upper bound; it uses the evaluation utility matrix directly."""

    def predict(self, utility: pd.DataFrame) -> pd.Series:
        return utility.idxmax(axis=1).rename("selected_model")
