from __future__ import annotations

import pandas as pd

from routecode.routers.knn import KNNRouter


def test_knn_router_predicts_from_neighbor_positions_with_nontrivial_index_order():
    utility = pd.DataFrame(
        {
            "m0": [0.1, 1.0, 0.2],
            "m1": [1.0, 0.1, 0.9],
        },
        index=["q10", "q2", "q7"],
    )
    query_info = pd.DataFrame({"dataset": ["a", "a", "b"]}, index=utility.index)
    train_embeddings = pd.DataFrame(
        [[10.0, 0.0], [0.0, 0.0], [9.0, 0.0]],
        index=utility.index,
    )
    test_info = pd.DataFrame({"dataset": ["x", "y"]}, index=["t0", "t1"])
    all_embeddings = pd.concat(
        [
            train_embeddings,
            pd.DataFrame([[0.1, 0.0], [9.5, 0.0]], index=test_info.index),
        ],
        axis=0,
    )

    selected = KNNRouter(n_neighbors=1).fit(query_info, utility, all_embeddings).predict(test_info, all_embeddings)

    assert selected.to_dict() == {"t0": "m0", "t1": "m1"}
