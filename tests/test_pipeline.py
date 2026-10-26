"""Связь API, оркестрации, реальных ML моделей и памяти вместо PostgreSQL."""
import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException
from unittest.mock import AsyncMock
from main_scripts import main


@pytest.fixture
def fast_search(monkeypatch):
    # Проверяется pipeline, а поиск гиперпараметров тестируется отдельно.
    # Обучение, сериализация и предсказание здесь настоящие.
    def select(analysis, rules):
        return "KMeans" if "data_size" in analysis else "NaiveBayes"
    monkeypatch.setattr(
        main.two_methods_included,
        "method_selector_by_analysis",
        select)
    monkeypatch.setattr(
        main.two_methods_included, "optimize_hyperparameters_claster",
        lambda *a, **k: (dict(n_clusters=3, init="k-means++",
                              max_iter=100), .5))
    monkeypatch.setattr(
        main.two_methods_included, "optimize_hyperparameters_classif",
        lambda *a, **k: ({"var_smoothing": 1e-9}, .5))


@pytest.mark.integration
def test_full_lifecycle(client, memory_db, payload, fast_search):
    tables, connections = memory_db
    result = client.post(
        "/task/train_and_prediction",
        json={
            **payload,
            "task_manager": "LEARN AND PREDICT"})
    assert result.status_code == 200, result.text
    assert len(result.json()["data"]) == 6
    assert len(tables["data_classif"]) == len(tables["data_claster"]) == 1
    assert tables["data_claster"][0]["accuracy"] == .5
    assert tables["data_classif"][0]["accuracy"] == 1.0
    result = client.post(
        "/task/prediction",
        json={
            **payload,
            "task_manager": "PREDICT"})
    assert result.status_code == 200
    deleted = client.request(
        "DELETE",
        "/task/delete",
        json={
            **payload,
            "task_manager": "DELETE"})
    assert deleted.status_code == 200
    assert tables["data_classif"] == tables["data_claster"] == []
    assert all(c.closed for c in connections)


def test_learn_without_predict_table(client, memory_db, payload, fast_search):
    request = {**payload, "name_table_for_predict": None}
    assert client.post("/task/train", json=request).status_code == 200


def test_missing_models_409(client, memory_db, payload):
    result = client.post(
        "/task/prediction",
        json={
            **payload,
            "task_manager": "PREDICT"})
    assert result.status_code == 409
    assert all(c.closed for c in memory_db[1])


def test_missing_table_404(client, memory_db, payload):
    result = client.post(
        "/task/train",
        json={
            **payload,
            "name_table_for_learn": "missing"})
    assert result.status_code == 404
    assert all(c.closed for c in memory_db[1])


def test_empty_saved_models_409(client, memory_db, payload):
    memory_db[0].update({"data_classif": [], "data_claster": []})
    result = client.post(
        "/task/prediction",
        json={
            **payload,
            "task_manager": "PREDICT"})
    assert result.status_code == 409
    assert all(c.closed for c in memory_db[1])


def test_cleanup_on_training_error(client, memory_db, payload, monkeypatch):
    monkeypatch.setattr(
        main, "data_learn_claster_classif_distribution", AsyncMock(
            side_effect=RuntimeError("forced")))
    assert client.post("/task/train", json=payload).status_code == 500
    assert all(c.closed for c in memory_db[1])


async def test_empty_output(memory_db):
    obj = main.DatabaseConnector("h", 1, "d", "u", "p", "learn", "target")
    result = await main.processing_result_by_task(obj, [], "PREDICT")
    assert result == {"equipment": "target", "data": []}


CLASSIFIER_PARAMS = [
    ("NaiveBayes", {"var_smoothing": 1e-9}),
    ("KNN", {"n_neighbors": 3}),
    ("SVM", {"C": 1, "kernel": "linear"}),
    ("LogisticRegression", {"C": 1, "max_iter": 1000}),
    ("DecisionTree", {"max_depth": 3}),
    ("RandomForest", {"n_estimators": 10}),
    ("GradientBoosting", {"n_estimators": 10})]


@pytest.mark.parametrize("algorithm,params", CLASSIFIER_PARAMS)
def test_training_dispatch(ml_data, monkeypatch, algorithm, params):
    X, y = ml_data
    monkeypatch.setattr(
        main.two_methods_included,
        "method_selector_by_analysis",
        lambda *a: algorithm)
    monkeypatch.setattr(main.two_methods_included,
                        "optimize_hyperparameters_classif", lambda * a, **
                        k: (params, .5))
    name, model, score = main.data_classification(
        pd.DataFrame(X), pd.Series(y))
    assert name == algorithm
    assert score > .9
    assert hasattr(model, "predict")


def test_small_clustering_input(monkeypatch):
    with pytest.raises(ValueError, match="минимум 3"):
        main.data_clasterization(np.ones((2, 2)), 2)


def test_bad_rules_path(ml_data):
    X, y = ml_data
    with pytest.raises(FileNotFoundError):
        main.data_classification(pd.DataFrame(X), pd.Series(y), "missing.json")


def test_bad_rules_json(ml_data, tmp_path):
    file = tmp_path / "bad.json"
    file.write_text("{")
    X, y = ml_data
    with pytest.raises(ValueError):
        main.data_classification(pd.DataFrame(X), pd.Series(y), str(file))
