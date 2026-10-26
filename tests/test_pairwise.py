"""Генерация набора и проверка полного покрытия пар факторов."""
from itertools import combinations, product
from unittest.mock import AsyncMock
import pytest
from main_scripts import main


FACTORS = [("LEARN", "PREDICT", "LEARN AND PREDICT", "DELETE"),
           (None, "Все", "Норма"), (None, "1", "1:3")]


def pairs(row):
    return {(i, row[i], j, row[j]) for i, j in combinations(range(3), 2)}


def generate():
    candidates = list(product(*FACTORS))
    remaining = set().union(*(pairs(row) for row in candidates))
    selected = []
    while remaining:
        row = max(candidates, key=lambda item: len(pairs(item) & remaining))
        selected.append(row)
        remaining -= pairs(row)
        candidates.remove(row)
    return selected


CASES = generate()


def test_pair_coverage():
    expected = set().union(*(pairs(row) for row in product(*FACTORS)))
    assert set().union(*(pairs(row) for row in CASES)) == expected
    assert len(CASES) < 36


@pytest.mark.parametrize("task,label,limit", CASES)
def test_pairwise_api(client, payload, monkeypatch, task, label, limit):
    routes = {"LEARN": ("POST", "/task/train"),
              "PREDICT": ("POST", "/task/prediction"),
              "LEARN AND PREDICT": ("POST", "/task/train_and_prediction"),
              "DELETE": ("DELETE", "/task/delete")}
    method, path = routes[task]
    service = AsyncMock(return_value={"data": "pairwise"})
    monkeypatch.setattr(main, "task_processing", service)
    result = client.request(
        method,
        path,
        json={
            **payload,
            "task_manager": task,
            "label_limit": label,
            "str_limit": limit})
    assert result.status_code == 200
    assert service.await_args.args[0]["label_limit"] == label
    assert service.await_args.args[0]["str_limit"] == limit
