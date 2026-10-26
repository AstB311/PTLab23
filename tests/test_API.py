"""Запросы проходят через настоящее ASGI приложение."""
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from main_scripts import main


@pytest.mark.functional
@pytest.mark.parametrize("method,path,task", [
    ("POST", "/task/train", "LEARN"), ("GET", "/task/train", "LEARN"),
    ("POST", "/task/prediction", "PREDICT"),
    ("POST", "/task/train_and_prediction", "LEARN AND PREDICT"),
    ("DELETE", "/task/delete", "DELETE")])
def test_routes_execute_service(
        client, payload, monkeypatch, method, path, task):
    service = AsyncMock(return_value={"data": "executed"})
    monkeypatch.setattr(main, "task_processing", service)
    response = client.request(
        method, path, json={
            **payload, "task_manager": task})
    assert response.status_code == 200
    assert response.json() == {"data": "executed"}
    assert service.await_count == 1
    assert service.await_args.args[0]["task_manager"] == task


@pytest.mark.parametrize("code", [404, 409, 422, 503])
def test_http_errors_preserved(client, payload, monkeypatch, code):
    monkeypatch.setattr(
        main, "task_processing", AsyncMock(
            side_effect=HTTPException(
                code, "expected")))
    response = client.post("/task/train", json=payload)
    assert response.status_code == code
    assert response.json()["detail"] == "expected"


@pytest.mark.parametrize("exc,code",
                         [(ValueError("invalid"),
                           422),
                          (RuntimeError("private"),
                           500)])
def test_service_failures(client, payload, monkeypatch, exc, code):
    monkeypatch.setattr(main, "task_processing", AsyncMock(side_effect=exc))
    response = client.post("/task/train", json=payload)
    assert response.status_code == code
    assert "private" not in response.text


@pytest.mark.parametrize("field,value", [
    ("port", True), ("port", "5432"), ("port", 0), ("port", 65536),
    ("task_manager", "OTHER"), ("server", ""),
    ("name_database_data", ""), ("name_table_for_learn", ""),
    ("label_limit", "wrong"), ("str_limit", "0:3"), ("str_limit", "3:1")])
def test_invalid_payload(client, payload, field, value):
    assert client.post(
        "/task/train",
        json={
            **payload,
            field: value}).status_code == 422


def test_wrong_route_task(client, payload):
    assert client.post("/task/prediction", json=payload).status_code == 422


def test_delete_needs_no_dataset(client, payload, monkeypatch):
    monkeypatch.setattr(
        main, "task_processing", AsyncMock(
            return_value={
                "data": "ok"}))
    keys = ("server", "port", "user", "password", "name_database_agent")
    request = {k: payload[k] for k in keys}
    response = client.request(
        "DELETE",
        "/task/delete",
        json={
            **request,
            "task_manager": "DELETE"})
    assert response.status_code == 200


@pytest.mark.parametrize("task",
                         ["LEARN", "PREDICT", "LEARN AND PREDICT", "DELETE"])
def test_web_tasks(client, payload, monkeypatch, task):
    service = AsyncMock(return_value={"data": "Модель обучена"})
    monkeypatch.setattr(main, "task_processing", service)
    response = client.post("/web/run", data={**payload, "task_manager": task})
    assert response.status_code == 200
    assert "TEST_SECRET" not in response.text
    assert service.await_args.args[0]["task_manager"] == task


def test_web_warning(client, payload, monkeypatch):
    result = {"equipment": "demo", "data": [
        {"ID": 1, "Timestamp": "12:00", "Label": "Предел"}]}
    monkeypatch.setattr(
        main, "task_processing", AsyncMock(
            return_value=result))
    response = client.post(
        "/web/run",
        data={
            **payload,
            "task_manager": "PREDICT"})
    assert "Обнаружены строки" in response.text
    assert "TEST_SECRET" not in response.text


def test_web_failure(client, payload, monkeypatch):
    monkeypatch.setattr(
        main, "task_processing", AsyncMock(
            side_effect=ValueError("bad")))
    response = client.post("/web/run", data=payload)
    assert "Ошибка" in response.text
    assert "TEST_SECRET" not in response.text


@pytest.mark.parametrize("path,expected",
                         [("/", "Форма менеджера"),
                          ("/static/style.css", "container")])
def test_public_pages(client, path, expected):
    response = client.get(path)
    assert response.status_code == 200
    assert expected in response.text
