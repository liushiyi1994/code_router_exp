from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def configured_sweep_values(
    config: dict,
    section: str,
    key: str,
    base_value: T,
    cast: Callable[[object], T] = float,
) -> list[T]:
    values = config.get(section, {}).get(key, [])
    casted = [cast(value) for value in values]
    base = cast(base_value)
    if base not in casted:
        casted.append(base)
    unique: list[T] = []
    for value in casted:
        if value not in unique:
            unique.append(value)
    return unique


def sample_train_query_ids(index: pd.Index, fraction: float, seed: int) -> pd.Index:
    if len(index) == 0:
        return index
    bounded = min(max(float(fraction), 0.0), 1.0)
    n_take = max(1, int(round(bounded * len(index))))
    n_take = min(n_take, len(index))
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(index.to_numpy(), size=n_take, replace=False)
    return pd.Index(selected, name=index.name)
