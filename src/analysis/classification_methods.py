import pickle
from typing import Any, Optional, Tuple
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ClassificationSuite:
    """
    Класс объединяет процедуры обучения/предсказания
    и функцию для оценки линейной разделимости.
    """

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = 42
                 ):
        self.test_size = test_size
        self.random_state = random_state

    # Деление
    def _split(self, X: np.ndarray, y: np.ndarray):
        return train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state
        )

    def _train_and_predict(
        self,
        estimator_cls,
        X: np.ndarray,
        y: np.ndarray,
        best_params: Optional[dict] = None,
    ) -> Tuple[np.ndarray, Any, np.ndarray]:
        params = best_params or {}
        X_train, X_test, \
            y_train, y_test = self._split(X, y)
        model = estimator_cls(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return y_pred, model, y_test

    def _predict_loaded(self,
                        loaded_model: Any,
                        data: Any
                        ) -> Any:
        model = pickle.loads(loaded_model)
        return model.predict(data)

    # Линейная разделимость
    def determine_linearity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: float = 0.85,
    ) -> str:
        """
        Определяет, являются ли данные линейно
        разделимыми с помощью
        LogisticRegression и линейного SVM.
        """
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=self.test_size,
            random_state=self.random_state
        )

        # Логистическая регрессия
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)
        acc_log = accuracy_score(y_test, log_reg.predict(X_test))

        # SVM с линейным ядром
        svm_linear = SVC(kernel="linear")
        svm_linear.fit(X_train, y_train)
        acc_svm = accuracy_score(y_test, svm_linear.predict(X_test))

        return (
            "Linearly separable" if (acc_log > threshold
                                     and acc_svm > threshold)
            else "Linearly inseparable"
        )

    # Методы классификации
    def naive_bayes(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """GaussianNB"""
        if loaded_model is None:
            return self._train_and_predict(
                GaussianNB, X, y,
                best_params)
        return self._predict_loaded(loaded_model, data)

    def knn(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """KNeighborsClassifier"""
        if loaded_model is None:
            return self._train_and_predict(
                KNeighborsClassifier, X,
                y, best_params)
        return self._predict_loaded(loaded_model, data)

    def svm(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """SVC"""
        if loaded_model is None:
            return self._train_and_predict(
                SVC, X, y, best_params)
        return self._predict_loaded(loaded_model, data)

    def logreg(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """LogisticRegression"""
        if loaded_model is None:
            return self._train_and_predict(
                LogisticRegression,
                X, y, best_params)
        return self._predict_loaded(loaded_model, data)

    def decision_tree(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """DecisionTreeClassifier"""
        if loaded_model is None:
            return self._train_and_predict(
                DecisionTreeClassifier,
                X, y, best_params)
        return self._predict_loaded(loaded_model, data)

    def random_forest(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """RandomForestClassifier"""
        if loaded_model is None:
            return self._train_and_predict(
                RandomForestClassifier,
                X, y, best_params)
        return self._predict_loaded(loaded_model, data)

    def gradboost(
        self,
        loaded_model: Optional[Any] = None,
        data: Optional[Any] = None,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
    ) -> Any:
        """GradientBoostingClassifier"""
        if loaded_model is None:
            return self._train_and_predict(
                GradientBoostingClassifier,
                X, y, best_params)
        return self._predict_loaded(loaded_model, data)

    # Сериализация
    @staticmethod
    def dump_model(model: Any) -> bytes:
        return pickle.dumps(model)
