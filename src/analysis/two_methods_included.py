import logging
import warnings
from collections import Counter
from typing import List, Dict, Any, Type, Tuple

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Склеивает список словарей data и столбец labels
def concatenate_data_with_labels(
        data: np.ndarray,
        labels_data: np.ndarray,
        side: str
) -> np.ndarray:
    """Склеивает массив данных с метками кластеров,
    добавляя столбец меток с указанной стороны."""
    if len(data) != len(labels_data):
        raise ValueError("Длина массива data и столбца должна совпадать.")

    data = np.asarray(data)
    labels_data = np.asarray(labels_data)
    if data.ndim != 2:
        raise ValueError("data должен быть двумерным массивом")
    if labels_data.ndim == 1:
        labels_data = labels_data.reshape(-1, 1)
    if labels_data.ndim != 2:
        raise ValueError("Метки должны быть одномерными или двумерными")
    if side == "front":
        result_data = np.concatenate((labels_data, data), axis=1)
    elif side == "behind":
        result_data = np.concatenate((data, labels_data), axis=1)
    else:
        raise ValueError(
            "Значение параметра side должно быть 'front' или 'behind'."
        )
    return result_data


# Приведение данных в нужный формат
def data_formater(data: List[Dict], task_manager: str, method: str) -> Any:
    """Форматирует данные в зависимости от задачи (LEARN/PREDICT) и метода."""
    try:
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("Набор данных пуст")
        num_clusters = -1
        id_column = []

        if task_manager == "LEARN":
            if method == "clasterization":
                id_column = df.iloc[:, 0].to_numpy()  # ID
                columns_to_exclude = [
                    col for col in df.columns if 'time' in str(col).lower()
                ]
                if len(df.columns) > 0:
                    columns_to_exclude.append(df.columns[0])  # ID

                label_column = df.iloc[:, -1]
                unique_labels = label_column.unique()
                num_clusters = len(unique_labels)

                columns_to_use = [
                    col for col in df.columns if col not in columns_to_exclude
                ]
                df_features = df[columns_to_use]
                data_for_learning = df_features.iloc[:, :-1].to_numpy()
                return data_for_learning, num_clusters, id_column, label_column

            if method == "classification":
                last_col = df.columns[-1]
                data_for_learning = df.rename(
                    columns={last_col: "claster_coloumn"}
                )
                return data_for_learning

        if task_manager == "PREDICT":
            if method == "clasterization":
                id_column = df.iloc[:, 0].to_numpy()
                columns_time = [
                    col for col in df.columns if 'time' in str(col).lower()
                ]
                df_time = df[columns_time].to_numpy()
                columns_to_use = [
                    col for col in df.columns
                    if col not in columns_time and col != df.columns[0]
                ]
                df_features = df[columns_to_use]
                data_for_prediction = df_features.to_numpy()
                return data_for_prediction, num_clusters, id_column, df_time

            if method == "classification":
                return data

        raise ValueError("Неизвестная задача или метод форматирования")
    except (KeyError, IndexError) as exc:
        raise ValueError("Некорректная структура данных") from exc


# Общая функция для выбора метода по анализу
def method_selector_by_analysis(
        analysis_results: Dict[str, str],
        rules: Dict
) -> str:
    """Выбирает метод анализа на основе результатов анализа и правил."""
    method_counts = Counter()

    for criterion, value in analysis_results.items():
        if criterion in rules:
            possible_methods = rules[criterion].get(value)
            if possible_methods:
                method_counts.update(possible_methods)

    if not method_counts:
        return rules['defaults']['algorithm']

    most_common_method, _ = method_counts.most_common(1)[0]
    return most_common_method


# Objective для кластеризации
def objective_claster(
        trial: optuna.Trial,
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray
) -> float:
    """Objective-функция для подбора гиперпараметров кластеризации."""
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    params = {}

    for param_name, param_range in param_grid.items():
        if isinstance(param_range, tuple):
            params[param_name] = trial.suggest_int(
                param_name, param_range[0], param_range[1]
            )
        else:
            params[param_name] = trial.suggest_categorical(
                param_name, param_range
            )

    model = model_class(**params)
    model.fit(X)
    labels = model.labels_
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(X):
        return -1.0
    score = silhouette_score(X, labels)
    if np.isnan(score):
        return -1
    return score


# Оптимизация гиперпараметров кластеризации
def optimize_hyperparameters_claster(
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray,
        n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Оптимизация гиперпараметров для кластеризации."""
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    study = optuna.create_study(
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=42)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        study.optimize(
            lambda trial: objective_claster(trial, model_class, param_grid, X),
            n_trials=n_trials,
            catch=(ValueError,),
        )
    return study.best_params, study.best_value


# Objective для классификации
def objective_classif(
        trial: optuna.Trial,
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray
) -> float:
    """Objective-функция для подбора гиперпараметров классификации."""
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    params = {}

    for param_name, param_range in param_grid.items():
        if isinstance(param_range, tuple):
            params[param_name] = trial.suggest_int(
                param_name, param_range[0], param_range[1]
            )
        else:
            params[param_name] = trial.suggest_categorical(
                param_name, param_range
            )

    model = model_class(**params)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError("Нужно минимум два класса и два объекта в классе")
    cv = StratifiedKFold(
        n_splits=min(5, int(counts.min())), shuffle=True, random_state=42
    )
    score = cross_val_score(model, X, y, scoring='accuracy', cv=cv).mean()
    if np.isnan(score):
        return -1
    return score


# Оптимизация гиперпараметров классификации
def optimize_hyperparameters_classif(
        model_class: Type[Any],
        param_grid: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 50
) -> Tuple[Dict[str, Any], float]:
    """Оптимизация гиперпараметров для классификации."""
    optuna.logging.get_logger("optuna").setLevel(logging.WARNING)
    study = optuna.create_study(
        direction='maximize', sampler=optuna.samplers.TPESampler(seed=42)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        study.optimize(
            lambda trial: objective_classif(
                trial, model_class, param_grid, X, y
            ),
            n_trials=n_trials,
            catch=(ValueError,),
        )
    return study.best_params, study.best_value


def parse_row_limit(value: str) -> Tuple[int, int]:
    """Положительное N означает 1:N; A:B задаёт включительные границы ID."""
    parts = str(value).strip().split(":")
    if len(parts) not in (1, 2) or any(not p.strip().isdigit() for p in parts):
        raise ValueError("Ограничение строк должно иметь вид N или A:B")
    start, end = (1, int(parts[0])) if len(parts) == 1 else map(int, parts)
    if start < 1 or end < start:
        raise ValueError("Нужны границы 1 <= A <= B")
    return start, end


async def processing_limit_str(data, str_limit):
    """Сохраняет критические строки даже вне заданных границ ID."""
    start, end = parse_row_limit(str_limit)
    return [row for row in data
            if start <= row[0] <= end or row[-1] == "Предел"]


async def processing_limit_label(data, label_limit):
    """Фильтрует метки; пустой результат остаётся пустым."""
    allowed = {"Все", "Норма", "Перегрузка", "Предел"}
    if label_limit not in allowed:
        raise ValueError("Неизвестная метка")
    labels = allowed - {"Все"} if label_limit == "Все" else {
        label_limit, "Предел"
    }
    return [row for row in data if row[-1] in labels]
