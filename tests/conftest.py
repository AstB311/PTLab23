"""Общие воспроизводимые данные и подмена внешней БД."""
import pickle
from unittest.mock import AsyncMock, Mock
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from fastapi.testclient import TestClient
from main_scripts import main
from src.analysis.classification_methods import ClassificationSuite
from src.analysis.clusterization_methods import Clusterizer
from src.analysis.connector import DatabaseConnector


@pytest.fixture
def ml_data():
    return make_blobs(n_samples=60, centers=[[-8, -8], [0, 8], [8, -8]],
                      cluster_std=0.25, random_state=42)


@pytest.fixture
def suite():
    return ClassificationSuite()


@pytest.fixture
def clusterizer():
    return Clusterizer()


@pytest.fixture
def connection():
    connection = Mock()
    connection.is_closed.return_value = False
    for method in ("fetch", "fetchval", "execute", "close"):
        setattr(connection, method, AsyncMock())
    return connection


@pytest.fixture
def connector():
    return DatabaseConnector("localhost", 5432, "test", "tester", "test-only")


@pytest.fixture
def connected(connector, connection, monkeypatch):
    monkeypatch.setattr("src.analysis.connector.asyncpg.connect",
                        AsyncMock(return_value=connection))
    return connector, connection


@pytest.fixture
def payload():
    return {
        "server": "localhost",
        "port": 5432,
        "user": "tester",
        "password": "TEST_SECRET",
        "name_database_agent": "models",
        "name_database_data": "measurements",
        "name_table_for_learn": "learn",
        "name_table_for_predict": "equipment",
        "task_manager": "LEARN"}


@pytest.fixture
def client():
    with TestClient(main.app) as client:
        yield client


@pytest.fixture
def memory_db(monkeypatch, ml_data):
    """Реальные алгоритмы; только хранилище и сеть заменены памятью."""
    X, y = ml_data
    names = np.array(["Норма", "Перегрузка", "Предел"])
    tables = {
        "learn": [{"id": i + 1, "timestamp": "2026-10-01T00:00:00Z",
                   "x1": float(x[0]), "x2": float(x[1]), "label": names[label]}
                  for i, (x, label) in enumerate(zip(X, y))],
        "equipment": [{"id": i + 1, "timestamp": "2026-10-02T00:00:00Z",
                       "x1": float(x[0]), "x2": float(x[1])}
                      for i, x in enumerate(X[:6])],
    }
    instances = []

    class MemoryConnector:
        def __init__(self, server, port, database, user, password,
                     equipment=None, equipment_predict=None):
            self.equipment = equipment
            self.equipment_predict = equipment_predict
            self.closed = False
            instances.append(self)

        async def connect(self):
            return True

        async def close(self):
            self.closed = True

        async def check_table_exists(self, table):
            return table in tables

        async def create_model_table(self, table, column):
            tables.setdefault(table, [])

        async def get_data_table(self, table):
            return tables[table]

        async def insert_data(self, table, values):
            tables[table].append(
                {**values, "model": pickle.dumps(values["model"]),
                 "id": len(tables[table]) + 1})

        async def get_data_table_in_coloumn(self, table, column, machine):
            return [r for r in tables[table] if r[column] == machine]

        async def delete_table_agent(self, table):
            tables[table].clear()

    monkeypatch.setattr(main, "DatabaseConnector", MemoryConnector)
    return tables, instances
