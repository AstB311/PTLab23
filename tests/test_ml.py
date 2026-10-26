"""Реальное обучение на синтетическом воспроизводимом наборе."""
import pickle
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from src.analysis.base_ml_suite import BaseMLSuite
from src.analysis.classification_methods import ClassificationSuite
from src.analysis.clusterization_methods import Clusterizer


@pytest.mark.unit
class TestBaseMLSuite:
    @pytest.mark.parametrize("seed", [0, 1, 42, 2026, 65535, 2**32 - 1])
    def test_seed(self, seed):
        assert BaseMLSuite(seed).random_state == seed

    @pytest.mark.parametrize("seed", [0, 42, 2026])
    def test_read_only(self, seed):
        obj = BaseMLSuite(seed)
        with pytest.raises(AttributeError):
            obj.random_state = 99
        assert obj.random_state == seed

    def test_inherited(self):
        assert isinstance(ClassificationSuite(), BaseMLSuite)
        assert isinstance(Clusterizer(), BaseMLSuite)


METHODS = [
    "naive_bayes",
    "knn",
    "svm",
    "logreg",
    "decision_tree",
    "random_forest",
    "gradboost"]


@pytest.mark.unit
class TestClassificationSuite:
    @pytest.mark.parametrize("method", METHODS)
    def test_train(self, suite, ml_data, method):
        X, y = ml_data
        predicted, model, expected = getattr(suite, method)(X=X, y=y)
        assert len(predicted) == 12
        assert np.mean(predicted == expected) >= .9
        assert hasattr(model, "predict")

    @pytest.mark.parametrize("method", METHODS)
    def test_loaded_predict(self, suite, ml_data, method):
        X, y = ml_data
        _, model, _ = getattr(suite, method)(X=X, y=y)
        expected = model.predict(X[:3])
        result = getattr(suite, method)(suite.dump_model(model), X[:3])
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("value", [-.1, 0, 1, 1.1, float('nan')])
    def test_test_size_boundary(self, value):
        with pytest.raises(ValueError):
            ClassificationSuite(test_size=value)

    @pytest.mark.parametrize("value", [None, "0.2", True])
    def test_test_size_type(self, value):
        with pytest.raises(TypeError):
            ClassificationSuite(test_size=value)

    def test_setter_keeps_previous_value(self, suite):
        with pytest.raises(ValueError):
            suite.test_size = 2
        assert suite.test_size == .2

    def test_split_reproducible(self, suite, ml_data):
        X, y = ml_data
        for a, b in zip(suite._split(X, y), suite._split(X, y)):
            np.testing.assert_array_equal(a, b)

    def test_scaler_fit_only_training_data(self, suite, ml_data, monkeypatch):
        X, y = ml_data
        calls = []
        original = StandardScaler.fit

        def tracked(self, data, *args, **kwargs):
            calls.append(len(data))
            return original(self, data, *args, **kwargs)

        monkeypatch.setattr(StandardScaler, "fit", tracked)
        assert suite.determine_linearity(X, y) == "Linearly separable"
        assert calls == [48]

    def test_model_dump(self):
        assert pickle.loads(
            ClassificationSuite.dump_model({"a": 1})) == {"a": 1}


CLUSTERS = [
    ("kmeans", {"n_clusters": 3, "init": "k-means++", "max_iter": 100}),
    ("agglomerative", {"n_clusters": 3, "linkage": "ward"}),
    ("spectral", {"n_clusters": 3, "affinity": "rbf", "gamma": 1.0}),
    ("dbscan", {"eps": 1.0, "min_samples": 2}),
    ("affinity", {"damping": .8, "preference": -50})]


@pytest.mark.unit
class TestClusterizer:
    @pytest.mark.parametrize("method,params", CLUSTERS)
    def test_train(self, clusterizer, ml_data, method, params):
        X, _ = ml_data
        labels, model = getattr(clusterizer, method)(X, params)
        assert len(labels) == len(X)
        assert len(np.unique(labels)) >= 2
        assert model is not None

    @pytest.mark.parametrize("method,params", [CLUSTERS[0], CLUSTERS[4]])
    def test_loaded_uses_predict(self, clusterizer, ml_data, method, params):
        X, _ = ml_data
        _, model = getattr(clusterizer, method)(X, params)
        expected = model.predict(X[:4])
        actual = getattr(clusterizer, method)(
            X[:4], loaded_model=pickle.dumps(model))
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("method,params", CLUSTERS[1:4])
    def test_loaded_transductive_rejected(
            self, clusterizer, ml_data, method, params):
        X, _ = ml_data
        _, model = getattr(clusterizer, method)(X, params)
        with pytest.raises(ValueError, match="не поддерживает predict"):
            getattr(clusterizer, method)(
                X[:4], loaded_model=pickle.dumps(model))

    def test_missing_model_and_parameters(self, clusterizer, ml_data):
        with pytest.raises(ValueError, match="params"):
            clusterizer.kmeans(ml_data[0])

    def test_zero_clusters(self, clusterizer, ml_data):
        with pytest.raises(ValueError):
            clusterizer.kmeans(
                ml_data[0], {
                    "n_clusters": 0, "init": "random", "max_iter": 10})
