import numpy as np
import pytest
import optuna
from sklearn.cluster import KMeans, DBSCAN
from sklearn.naive_bayes import GaussianNB
from src.analysis import two_methods_included as utils


@pytest.mark.parametrize("side", ["front", "behind"])
def test_concatenate_side(side):
    result = utils.concatenate_data_with_labels(
        np.array([[1, 2], [3, 4]]), np.array([5, 6]), side)
    assert result.shape == (2, 3)
    assert result[0, 0 if side == "front" else -1] == 5


def test_concatenate_two_time_columns():
    result = utils.concatenate_data_with_labels(
        np.ones((2, 3)), np.array([[1, 2], [3, 4]]), "front")
    assert result.shape == (2, 5)


@pytest.mark.parametrize("data,labels,side", [
    (np.ones((2, 2)), np.array([1]), "front"),
    (np.ones((2, 2)), np.array([1, 2]), "wrong"),
    (np.ones(2), np.array([1, 2]), "front")])
def test_concatenate_invalid(data, labels, side):
    with pytest.raises(ValueError):
        utils.concatenate_data_with_labels(data, labels, side)


def test_format_integer_columns():
    result = utils.data_formater(
        np.array([[1, 2], [3, 4]]), "LEARN", "classification")
    assert result.columns.tolist() == [0, "claster_coloumn"]
    X, count, ids, labels = utils.data_formater(
        np.array([[1, 2, 0], [2, 3, 1]]), "LEARN", "clasterization")
    assert X.shape == (2, 1)
    assert count == 2


def test_format_learning(memory_db):
    X, clusters, ids, labels = utils.data_formater(
        memory_db[0]["learn"], "LEARN", "clasterization")
    assert X.shape == (60, 2)
    assert clusters == 3
    assert ids[0] == 1
    assert len(labels) == 60


def test_format_prediction(memory_db):
    X, clusters, ids, timestamps = utils.data_formater(
        memory_db[0]["equipment"], "PREDICT", "clasterization")
    assert X.shape == (6, 2)
    assert timestamps.shape == (6, 1)
    assert clusters == -1


@pytest.mark.parametrize("task,method",
                         [("OTHER", "classification"), ("LEARN", "other")])
def test_format_unknown(task, method):
    with pytest.raises(ValueError):
        utils.data_formater([[1, 2]], task, method)


def test_format_empty():
    with pytest.raises(ValueError, match="пуст"):
        utils.data_formater([], "LEARN", "classification")


@pytest.mark.parametrize("value,expected",
                         [("1", (1, 1)),
                          ("5", (1, 5)),
                          ("2:5", (2, 5)),
                          (" 1:3 ", (1, 3))])
def test_row_limits(value, expected):
    assert utils.parse_row_limit(value) == expected


@pytest.mark.parametrize("value",
                         ["0", "-1", "3:1", "1:2:3", "abc", "", "0:2"])
def test_invalid_limits(value):
    with pytest.raises(ValueError):
        utils.parse_row_limit(value)


async def test_critical_row_preserved():
    data = [[1, "Норма"], [2, "Перегрузка"], [99, "Предел"]]
    assert await utils.processing_limit_str(data, "1") == [data[0], data[2]]


async def test_empty_filter_stays_empty():
    assert await utils.processing_limit_str([[5, "Норма"]], "1") == []
    result = await utils.processing_limit_label([[1, "Норма"]], "Перегрузка")
    assert result == []


@pytest.mark.parametrize("label,count",
                         [("Все", 3),
                          ("Норма", 2),
                          ("Перегрузка", 2),
                          ("Предел", 1)])
async def test_labels(label, count):
    data = [[1, "Норма"], [2, "Перегрузка"], [3, "Предел"]]
    assert len(await utils.processing_limit_label(data, label)) == count


async def test_unknown_label():
    with pytest.raises(ValueError):
        await utils.processing_limit_label([], "wrong")


def test_rule_selection():
    rules = {"a": {"x": ["KNN"]}, "defaults": {"algorithm": "SVM"}}
    assert utils.method_selector_by_analysis({"a": "x"}, rules) == "KNN"
    assert utils.method_selector_by_analysis({}, rules) == "SVM"


def test_clustering_optimization(ml_data):
    params, score = utils.optimize_hyperparameters_claster(
        KMeans, {"n_clusters": [3], "random_state": [42]},
        ml_data[0], n_trials=2)
    assert params["n_clusters"] == 3
    assert .8 < score <= 1


def test_single_cluster_trial(ml_data):
    trial = optuna.trial.FixedTrial({"eps": 100.0, "min_samples": 2})
    assert utils.objective_claster(
        trial, DBSCAN, {
            "eps": [100.0], "min_samples": [2]}, ml_data[0]) == -1


def test_classification_optimization(ml_data):
    X, y = ml_data
    params, score = utils.optimize_hyperparameters_classif(
        GaussianNB, {"var_smoothing": [1e-9]}, X, y, n_trials=2)
    assert params["var_smoothing"] == 1e-9
    assert score > .9


def test_insufficient_classes(ml_data):
    with pytest.raises(ValueError, match="два класса"):
        utils.objective_classif(
            optuna.trial.FixedTrial(
                {}),
            GaussianNB,
            {},
            ml_data[0],
            np.zeros(60))
