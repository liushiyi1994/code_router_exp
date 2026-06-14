from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer


def build_hashing_embeddings(query_info: pd.DataFrame, n_features: int = 128) -> pd.DataFrame:
    """Create deterministic local text features without fitting on train+test."""
    if "query_text" not in query_info.columns:
        raise ValueError("query_info must contain query_text for hashing embeddings")
    vectorizer = HashingVectorizer(
        n_features=int(n_features),
        alternate_sign=False,
        norm="l2",
        lowercase=True,
    )
    matrix = vectorizer.transform(query_info["query_text"].fillna("").astype(str).tolist())
    dense = matrix.toarray()
    return pd.DataFrame(
        dense,
        index=query_info.index,
        columns=[f"hash_{idx}" for idx in range(dense.shape[1])],
    )
