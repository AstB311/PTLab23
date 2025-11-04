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

    labels_data = labels_data.reshape(-1, 1)
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
        num_clusters = -1
        id_column = []

        if task_manager == "LEARN":
            if method == "clasterization":
                id_column = df.iloc[:, 0].to_numpy()  # ID
                columns_to_exclude = [
                    col for col in df.columns if 'time' in col.lower()
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
                    col for col in df.columns if 'time' in col.lower()
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

        print(
            f"Неизвестный task_manager: {task_manager}. "
            f"Возвращаем пустой массив."
        )
        return np.array([])

    except Exception:
        return np.array([])


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
    study = optuna.create_study(direction='maximize')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        study.optimize(
            lambda trial: objective_claster(trial, model_class, param_grid, X),
            n_trials=n_trials,
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
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
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
    study = optuna.create_study(direction='maximize')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        study.optimize(
            lambda trial: objective_classif(
                trial, model_class, param_grid, X, y
            ),
            n_trials=n_trials,
        )
    return study.best_params, study.best_value


# Обработка ограничений строк
async def processing_limit_str(
        data: List[List[Any]],
        str_limit: str
) -> List[List[Any]]:
    """Фильтрует список строк на основе диапазона и наличия строки 'Предел'."""
    print(str_limit)
    start_index = 1
    end_index = int(str_limit)
    print(start_index, end_index)
    filtered_data = [
        row for row in data
        if start_index <= row[0] <= end_index or row[-1] == "Предел"
    ]
    if filtered_data:
        print("\tУказанный диапазон найден. Возвращаются нужные данные.")
        return filtered_data

    print("\tДиапазон не найден. Возвращаются все данные.")
    return data


# Обработка ограничений меток
async def processing_limit_label(
        data: List[List[Any]],
        label_limit: str
) -> List[List[Any]]:
    """Фильтрует список строк на основе меток."""
    if label_limit == "Все":
        filtered_data = [
            row for row in data
            if row[-1] in ("Норма", "Перегрузка", "Предел")
        ]
    else:
        filtered_data = [
            row for row in data
            if row[-1] in (label_limit, "Предел")
        ]

    if filtered_data:
        print("\tУказанные метки найдены. Возвращаются нужные данные.")
        return filtered_data

    print("\tМетки не найдены. Возвращаются все данные.")
    return data
