import numpy as np
import pytest
# Для ручки
# вручную
# from assistant_lab2.src.analysis.clusterization_methods
from src.analysis.classification_methods import ClassificationSuite


@pytest.fixture
def suite():
    return ClassificationSuite(test_size=0.25, random_state=42)


@pytest.fixture
def simple_dataset():
    # Две линейно разделимые группы
    rng = np.random.RandomState(0)
    X1 = rng.normal(loc=-2.0, scale=0.5, size=(50, 2))
    X2 = rng.normal(loc=+2.0, scale=0.5, size=(50, 2))
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    return X, y


def test_determine_linearity_linearly_separable(suite, simple_dataset):
    X, y = simple_dataset
    res = suite.determine_linearity(X, y, threshold=0.7)
    assert res in ("Linearly separable", "Linearly inseparable")
    assert res == "Linearly separable"


@pytest.mark.parametrize("algo", [
    "naive_bayes", "knn", "svm", "logreg",
    "decision_tree", "random_forest", "gradboost"
])
def test_train_and_predict_returns_tuple(suite, simple_dataset, algo):
    X, y = simple_dataset
    method = getattr(suite, algo)
    y_pred, model, y_test = method(X=X, y=y)
    assert y_pred.shape == y_test.shape
    assert hasattr(model, "predict")


def test_predict_loaded_logreg(suite, simple_dataset):
    X, y = simple_dataset
    # сначала обучим и получим модель
    _, model, _ = suite.logreg(X=X, y=y)
    # сериализуем и проверим путь predict_loaded
    loaded = suite.dump_model(model)
    y_hat = suite.logreg(loaded_model=loaded, data=X[:10])
    assert y_hat.shape[0] == 10
