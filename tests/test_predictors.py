import pandas as pd

from routecode.codes.routecode import RouteCodeCodebook
from routecode.predictors.classifiers import (
    LogisticModelRouter,
    MLPModelRouter,
    MLPRouteCodeLabelClassifier,
    PredictedLabelLookupRouter,
    RouteCodeLabelClassifier,
)


def separable_training_case():
    utility = pd.DataFrame(
        {
            "cheap": [0.9, 0.88, 0.86, 0.2, 0.25, 0.22],
            "strong": [0.2, 0.22, 0.24, 0.92, 0.9, 0.88],
        },
        index=["q0", "q1", "q2", "q3", "q4", "q5"],
    )
    query_info = pd.DataFrame(
        {
            "query_id": utility.index,
            "dataset": ["easy", "easy", "easy", "hard", "hard", "hard"],
            "domain": ["easy", "easy", "easy", "hard", "hard", "hard"],
        }
    ).set_index("query_id")
    embeddings = pd.DataFrame(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ],
        index=utility.index,
    )
    return query_info, utility, embeddings


def test_logistic_model_router_predicts_best_model_from_embeddings():
    query_info, utility, embeddings = separable_training_case()
    router = LogisticModelRouter(random_state=0).fit(query_info, utility, embeddings)
    selected = router.predict(query_info, embeddings)
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q5"] == "strong"


def test_routecode_label_classifier_predicts_labels_then_models():
    query_info, utility, embeddings = separable_training_case()
    codebook = RouteCodeCodebook(n_labels=2, random_state=0).fit(query_info, utility, embeddings)
    classifier = RouteCodeLabelClassifier(random_state=0).fit(codebook, embeddings)
    selected = classifier.predict(query_info, embeddings)
    labels = classifier.predict_labels(embeddings)
    confidence = classifier.predict_confidence(embeddings)
    assert set(labels.tolist()).issubset({0, 1})
    assert confidence.index.equals(embeddings.index)
    assert confidence.between(0.0, 1.0).all()
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q5"] == "strong"


def test_mlp_model_router_predicts_best_model_from_embeddings():
    query_info, utility, embeddings = separable_training_case()
    router = MLPModelRouter(random_state=0, hidden_layer_sizes=(8,), max_iter=500).fit(
        query_info,
        utility,
        embeddings,
    )
    selected = router.predict(query_info, embeddings)
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q5"] == "strong"


def test_mlp_routecode_label_classifier_predicts_labels_then_models():
    query_info, utility, embeddings = separable_training_case()
    codebook = RouteCodeCodebook(n_labels=2, random_state=0).fit(query_info, utility, embeddings)
    classifier = MLPRouteCodeLabelClassifier(random_state=0, hidden_layer_sizes=(8,), max_iter=500).fit(
        codebook,
        embeddings,
    )
    selected = classifier.predict(query_info, embeddings)
    labels = classifier.predict_labels(embeddings)
    confidence = classifier.predict_confidence(embeddings)
    assert set(labels.tolist()).issubset({0, 1})
    assert confidence.index.equals(embeddings.index)
    assert confidence.between(0.0, 1.0).all()
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q5"] == "strong"


def test_predicted_label_lookup_router_trains_label_predictor_then_routes():
    query_info, utility, embeddings = separable_training_case()
    router = PredictedLabelLookupRouter(label_column="dataset", random_state=0).fit(
        query_info,
        utility,
        embeddings,
    )
    labels = router.predict_labels(embeddings)
    selected = router.predict(query_info, embeddings)
    assert set(labels.tolist()).issubset({"easy", "hard"})
    assert selected.loc["q0"] == "cheap"
    assert selected.loc["q5"] == "strong"
