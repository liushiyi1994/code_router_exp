from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_query(
    outcomes: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    query_ids = outcomes["query_id"].drop_duplicates().to_numpy()
    if len(query_ids) < 3:
        raise ValueError("At least 3 queries are required for train/val/test splitting")

    rng = np.random.default_rng(seed)
    shuffled = query_ids.copy()
    rng.shuffle(shuffled)

    n_queries = len(shuffled)
    n_train = max(1, int(round(train_frac * n_queries)))
    n_val = max(1, int(round(val_frac * n_queries)))
    if n_train + n_val >= n_queries:
        n_train = max(1, n_queries - 2)
        n_val = 1

    train_ids = set(shuffled[:n_train])
    val_ids = set(shuffled[n_train : n_train + n_val])

    def assign(query_id: str) -> str:
        if query_id in train_ids:
            return "train"
        if query_id in val_ids:
            return "val"
        return "test"

    split = outcomes.copy()
    split["split"] = split["query_id"].map(assign)
    return split
