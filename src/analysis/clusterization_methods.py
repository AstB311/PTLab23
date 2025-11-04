import pickle
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    SpectralClustering,
)

ReturnType = Union[np.ndarray, Tuple[np.ndarray, Any]]


class Clusterizer:
    """Единый класс со всеми методами кластеризации."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    # =Если модель была загружена
    def _fit_with(
        self,
        data: np.ndarray,
        params: Optional[Dict[str, Any]],
        loaded_model: Optional[bytes],
        build_model_fn,
    ) -> ReturnType:
        if loaded_model is None and params is not None:
            model = build_model_fn(params)
            labels = model.fit_predict(data)
            return labels, model
        # loaded_model передан: используем его
        model = pickle.loads(loaded_model)
        labels = model.fit_predict(data)
        return labels

    # Алгоритмы
    def kmeans(
        self,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        loaded_model: Optional[bytes] = None,
    ) -> ReturnType:
        """K-средних (KMeans)."""
        def build(p: Dict[str, Any]) -> KMeans:
            return KMeans(
                n_clusters=p["n_clusters"],
                init=p["init"],
                max_iter=p["max_iter"],
                random_state=self.random_state,
            )
        return self._fit_with(data, params, loaded_model, build)

    def agglomerative(
        self,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        loaded_model: Optional[bytes] = None,
    ) -> ReturnType:
        """Агломеративная кластеризация."""
        def build(p: Dict[str, Any]) -> AgglomerativeClustering:
            return AgglomerativeClustering(
                n_clusters=p["n_clusters"],
                linkage=p["linkage"],
            )
        return self._fit_with(data, params, loaded_model, build)

    def spectral(
        self,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        loaded_model: Optional[bytes] = None,
    ) -> ReturnType:
        """Спектральная кластеризация."""
        def build(p: Dict[str, Any]) -> SpectralClustering:
            return SpectralClustering(
                n_clusters=p["n_clusters"],
                affinity=p["affinity"],
                gamma=p["gamma"],
                random_state=self.random_state,
            )
        return self._fit_with(data, params, loaded_model, build)

    def dbscan(
        self,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        loaded_model: Optional[bytes] = None,
    ) -> ReturnType:
        """DBSCAN."""
        def build(p: Dict[str, Any]) -> DBSCAN:
            return DBSCAN(
                eps=p["eps"],
                min_samples=p["min_samples"],
            )
        return self._fit_with(data, params, loaded_model, build)

    def affinity(
        self,
        data: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
        loaded_model: Optional[bytes] = None,
    ) -> ReturnType:
        """Affinity Propagation."""
        def build(p: Dict[str, Any]) -> AffinityPropagation:
            return AffinityPropagation(
                damping=p["damping"],
                preference=p["preference"],
                random_state=self.random_state,
            )
        return self._fit_with(data, params, loaded_model, build)
