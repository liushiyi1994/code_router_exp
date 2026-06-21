import numpy as np
import pandas as pd

from routecode.states.observable_tree_states import fit_observable_tree_state_model, selected_utility


def test_observable_tree_state_model_predicts_feature_defined_states():
    rng = np.random.default_rng(7)
    xs = np.repeat(np.arange(4), 8)
    features = pd.DataFrame(
        {
            "x": xs.astype(float),
            "noise": rng.normal(0, 0.01, size=len(xs)),
        },
        index=[f"q{i}" for i in range(len(xs))],
    )
    utility = pd.DataFrame(
        {
            "m0": (xs == 0).astype(float),
            "m1": (xs == 1).astype(float),
            "m2": (xs == 2).astype(float),
            "m3": (xs == 3).astype(float),
        },
        index=features.index,
    )

    model = fit_observable_tree_state_model(features, utility, n_states=4, min_samples_leaf=2, random_state=7)
    labels = model.predict_states(features)
    selected = model.select_models(labels)

    assert model.n_states == 4
    assert labels.index.equals(features.index)
    assert set(selected) == {"m0", "m1", "m2", "m3"}
    assert selected_utility(utility, selected).mean() == 1.0
