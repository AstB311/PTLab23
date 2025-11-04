import numpy as np
import pickle
import pytest
# Для ручки
# вручную
# from assistant_lab2.src.analysis.clusterization_methods import Clusterizer
from src.analysis.clusterization_methods import Clusterizer


@pytest.fixture
def data_blobs():
    rng = np.random.RandomState(0)
    a = rng.normal(loc=[-2, -2], scale=0.4, size=(50, 2))
    b = rng.normal(loc=[+2, +2], scale=0.4, size=(50, 2))
    X = np.vstack([a, b])
    return X


def test_kmeans_train_and_then_use_loaded_model(data_blobs):
    cl = Clusterizer(random_state=42)
    # шаг 1: обучаем (params задан, loaded_model=None)
    labels1, model = cl.kmeans(
        data_blobs,
        params={"n_clusters": 2, "init": "k-means++", "max_iter": 100}
    )
    assert labels1.shape[0] == data_blobs.shape[0]
    # шаг 2: используем сериализованную модель
    raw = pickle.dumps(model)
    labels2 = cl.kmeans(data_blobs, params=None, loaded_model=raw)
    assert isinstance(labels2, np.ndarray)
    assert labels2.shape[0] == data_blobs.shape[0]


def test_agglomerative_minimal_params(data_blobs):
    cl = Clusterizer(random_state=42)
    labels, model = cl.agglomerative(
        data_blobs,
        params={"n_clusters": 2, "linkage": "ward"}
    )
    assert labels.shape[0] == data_blobs.shape[0]
    # повторно через loaded_model (как и предусмотрено реализацией)
    raw = pickle.dumps(model)
    labels2 = cl.agglomerative(data_blobs, params=None, loaded_model=raw)
    assert isinstance(labels2, np.ndarray)
    assert labels2.shape[0] == data_blobs.shape[0]


def test_dbscan_minimal_params(data_blobs):
    cl = Clusterizer(random_state=42)
    labels, model = cl.dbscan(
        data_blobs,
        params={"eps": 0.5, "min_samples": 5}
    )
    assert labels.shape[0] == data_blobs.shape[0]
    raw = pickle.dumps(model)
    labels2 = cl.dbscan(data_blobs, params=None, loaded_model=raw)
    assert labels2.shape[0] == data_blobs.shape[0]
