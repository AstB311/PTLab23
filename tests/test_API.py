import pytest
import requests
from unittest.mock import patch


BASE_URL = "http://127.0.0.1:8000/task"


@pytest.fixture
def base_data():
    return {
        "server": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "5552225",
        "name_database_agent": "bake_agent",
    }


@pytest.fixture
def mock_post_response():
    class MockResponse:

        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json = json_data or {"data": "mocked"}

        def json(self):
            return self._json

    return MockResponse


@pytest.fixture
def mock_get_response():
    class MockResponse:

        def __init__(self, status_code=200, json_data=None):
            self.status_code = status_code
            self._json = json_data or {"data": "mocked"}

        def json(self):
            return self._json

    return MockResponse


@patch("requests.post")
def test_train_and_prediction(mock_post, base_data, mock_post_response):
    mock_post.return_value = mock_post_response()
    data = base_data.copy()
    data.update({
        "name_database_data": "bake_data",
        "name_table_for_learn": "bake_cooling_system_learn",
        "name_table_for_predict": "bake_cooling_system",
        "label_limit": "Все",
        "str_limit": "1:21",
        "task_manager": "LEARN AND PREDICT",
    })
    response = requests.post(f"{BASE_URL}/train_and_prediction", json=data)
    assert response.status_code == 200
    assert "data" in response.json()

    # Проверяем на верный адрес запроса
    mock_post.assert_called_once_with(
        f"{BASE_URL}/train_and_prediction",
        json=data,
    )


@patch("requests.get")
def test_train(mock_get, base_data, mock_get_response):
    mock_get.return_value = mock_get_response()
    data = base_data.copy()
    data.update({
        "name_database_data": "bake_data",
        "name_table_for_learn": "bake_cooling_system_learn",
        "task_manager": "LEARN",
    })
    response = requests.get(f"{BASE_URL}/train", json=data)
    assert response.status_code == 200
    assert "data" in response.json()

    # Проверяем на верный адрес запроса
    mock_get.assert_called_once_with(
        f"{BASE_URL}/train",
        json=data,
    )


@patch("requests.post")
def test_prediction(mock_post, base_data, mock_post_response):
    mock_post.return_value = mock_post_response()
    data = base_data.copy()
    data.update({
        "name_database_data": "bake_data",
        "name_table_for_learn": "bake_cooling_system",
        "label_limit": "Норма",
        "str_limit": "1:20",
        "task_manager": "PREDICT",
    })
    response = requests.post(f"{BASE_URL}/prediction", json=data)
    assert response.status_code == 200
    assert "data" in response.json()

    # Проверяем на верный адрес запроса
    mock_post.assert_called_once_with(
        f"{BASE_URL}/prediction",
        json=data,
    )


@patch("requests.delete")
def test_delete(mock_delete, base_data, mock_post_response):
    mock_delete.return_value = mock_post_response()
    data = base_data.copy()
    data.update({"task_manager": "DELETE"})
    response = requests.delete(f"{BASE_URL}/delete", json=data)
    assert response.status_code == 200
    assert "data" in response.json()

    # Проверяем на верный адрес запроса
    mock_delete.assert_called_once_with(
        f"{BASE_URL}/delete",
        json=data,
    )
