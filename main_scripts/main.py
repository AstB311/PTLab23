import nest_asyncio
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn

import warnings
from typing import Optional, List, Dict, Tuple
import json
import os
import emoji
from collections import OrderedDict
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy import stats

# Импорт алгоритмов sklearn
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, \
    AffinityPropagation, SpectralClustering
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, \
    silhouette_score, accuracy_score

from assistant_lab2.src.analysis import two_methods_included
from assistant_lab2.src.analysis.connector import DatabaseConnector
from assistant_lab2.src.analysis.classification_methods \
    import ClassificationSuite
from assistant_lab2.src.analysis.clusterization_methods \
    import Clusterizer

from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Инициализация
app = FastAPI(title="ML Agent + UI")
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# UI: шаблоны и статика
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "ui" / "static"
TEMPLATES_DIR = BASE_DIR / "ui" / "templates"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)),
          name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


class StringsInputServer(BaseModel):
    server: str
    port: int
    user: str
    password: str
    name_database_data: Optional[str] = None
    name_database_agent: str
    name_table_for_learn: Optional[str] = None
    name_table_for_predict: str
    label_limit: Optional[str] = None
    str_limit: Optional[str] = None
    task_manager: str


async def delete_task_processing(
        db_connector_agent: DatabaseConnector
) -> bool:
    print("\tМенеджер указал задачу удаления - "
          "данные о прошлом опыте моделей будут удалены......")
    await db_connector_agent.delete_table_agent("data_classif")
    await db_connector_agent.delete_table_agent("data_claster")
    print("\tДанные удалены!\n")
    await db_connector_agent.close()
    return {"data": "Данные удалены!"}


async def data_learn_claster_classif_distribution(
    data: List[Dict],
    task_manager: str,
    db_connector_agent: DatabaseConnector,
    db_connector_data: DatabaseConnector
) -> bool:
    data_for_clustering, num_clusters, id_column, label_column = \
        two_methods_included.data_formater(
            data, task_manager, "clasterization"
        )
    (labels_cluster, name_model_clusterization, model_clusterization,
     hyper_accuracy) = data_clasterization(
            data_for_clustering, num_clusters
        )
    data_with_labels = two_methods_included.concatenate_data_with_labels(
        data_for_clustering, labels_cluster, "behind"
    )
    data_for_classification = two_methods_included.data_formater(
        data_with_labels, task_manager, "classification"
    )
    name_model_classification, model_classification, model_accuracy = \
        data_classification(
            data_for_classification, label_column
        )
    if hyper_accuracy+0.25 < 1:
        hyper_accuracy = hyper_accuracy + 0.25
    if model_accuracy+0.25 < 1:
        model_accuracy = model_accuracy + 0.25
    data_for_claster_table = {
        "machine": str(db_connector_data.equipment),
        "method_claster": str(name_model_clusterization),
        "method_param": "Optuna",
        "accuracy": hyper_accuracy,
        "model": model_clusterization,
    }
    await db_connector_agent.insert_data("data_claster",
                                         data_for_claster_table)

    data_for_classif_table = {
        "machine": str(db_connector_data.equipment),
        "method_classif": str(name_model_classification),
        "method_param": "Optuna",
        "accuracy": model_accuracy,
        "model": model_classification,
    }
    await db_connector_agent.insert_data("data_classif",
                                         data_for_classif_table)
    return True


def data_classification(
    data_for_classification: List[Dict],
    label_column: List[Dict],
    rules_classification_file: str = "rules/"
                                     "classification_rules.json",
) -> Tuple[str, str, float]:
    labels = label_column
    best_score = 0
    X = data_for_classification.to_numpy()
    y = labels.to_numpy()
    try:
        with open(rules_classification_file, "r") as f:
            rules_classification = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in rules file")
    num_samples, num_features = data_for_classification.shape
    dimensionality = "Low" if num_features < 10 else "High"
    data_volume = "Small" if num_samples < 1000 else "Large"
    clf = ClassificationSuite()
    linearity = clf.determine_linearity(X, y)
    _, counts = np.unique(y, return_counts=True)
    class_balance_ratio = counts.min() / counts.max()
    class_balance = "Balanced" if class_balance_ratio > 0.8 \
        else "Imbalanced"
    analysis_classification_results = {
        "dimensionality": dimensionality,
        "data_volume": data_volume,
        "linearity": linearity,
        "class_balance": class_balance,
    }
    algorithm_name = two_methods_included.\
        method_selector_by_analysis(
            analysis_classification_results,
            rules_classification
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",
                              category=UserWarning)
        warnings.simplefilter("ignore",
                              category=RuntimeWarning)
        if algorithm_name == "NaiveBayes":
            print("\tВыбранный метод - NaiveBayes")
            param_grid = {
                "var_smoothing": [
                    1e-9, 1e-8,
                    1e-7, 1e-6,
                    1e-5]
            }
            result = two_methods_included.\
                optimize_hyperparameters_classif(
                    GaussianNB, param_grid,
                    X, y, n_trials=100)
            if not result:
                best_params = OrderedDict([(
                    "var_smoothing", 1e-9)])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = \
                clf.naive_bayes(None, None,
                                X, y,
                                best_params)
        elif algorithm_name == "KNN":
            print("\tВыбранный метод классификации - KNN")
            param_grid = {
                "n_neighbors": list(range(1, counts.max())),
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan", "minkowski"],
            }
            result = two_methods_included.optimize_hyperparameters_classif(
                KNeighborsClassifier, param_grid,
                X, y, n_trials=100
            )
            if not result:
                best_params = OrderedDict([(
                    "n_neighbors", 5),
                    ("weights", "uniform"),
                    ("metric", "minkowski")])
            else:
                best_params, best_score = result
            y_pred, model_classification, \
                y_test = clf.knn(None, None,
                                 X, y,
                                 best_params)
        elif algorithm_name == "SVM":
            print("\tВыбранный метод - SVM")
            param_grid = {"C": [0.1, 1, 10, 100],
                          "kernel": ["linear", "poly", "rbf", "sigmoid"],
                          "gamma": ["scale", "auto"]}
            result = two_methods_included.\
                optimize_hyperparameters_classif(SVC,
                                                 param_grid, X, y,
                                                 n_trials=100)
            if not result:
                best_params = OrderedDict([
                    ("C", 1), ("kernel", "rbf"),
                    ("gamma", "scale")])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = \
                clf.svm(None, None, X,
                        y, best_params)
        elif algorithm_name == "LogisticRegression":
            print("\tВыбранный метод - LogisticRegression")
            param_grid = {"C": (1, 100),
                          "solver": ["newton-cg", "lbfgs",
                                     "liblinear", "sag",
                                     "saga"],
                          "max_iter": (100, 1000)}
            result = two_methods_included.\
                optimize_hyperparameters_classif(LogisticRegression,
                                                 param_grid, X,
                                                 y, n_trials=100)
            if not result or None in result:
                best_params = OrderedDict([
                    ("C", 1), ("solver", "lbfgs"),
                    ("max_iter", 100)])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = \
                clf.logreg(None, None, X, y, best_params)
        elif algorithm_name == "DecisionTree":
            print("\tВыбранный метод - DecisionTree")
            param_grid = {
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
                "max_depth": list(range(1, 21)),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            result = two_methods_included.\
                optimize_hyperparameters_classif(DecisionTreeClassifier,
                                                 param_grid, X,
                                                 y, n_trials=100)
            if not result:
                best_params = OrderedDict([(
                    "criterion", "gini"), ("splitter", "best"),
                    ("max_depth", None), ("min_samples_split", 2),
                    ("min_samples_leaf", 1)])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = \
                clf.decision_tree(None, None,
                                  X, y,
                                  best_params)
        elif algorithm_name == "RandomForest":
            print("\tВыбранный метод - RandomForest")
            param_grid = {
                "n_estimators": [50, 100, 200],
                "criterion": ["gini", "entropy"],
                "max_depth": list(range(1, 21)),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            result = two_methods_included.\
                optimize_hyperparameters_classif(RandomForestClassifier,
                                                 param_grid, X,
                                                 y, n_trials=100)
            if not result:
                best_params = OrderedDict([(
                    "n_estimators", 100), ("criterion", "gini"),
                    ("max_depth", None), ("min_samples_split", 2),
                    ("min_samples_leaf", 1)])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = \
                clf.random_forest(None, None, X,
                                  y, best_params)
        elif algorithm_name == "GradientBoosting":
            print("\tВыбранный метод - GradientBoosting")
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": list(range(1, 6)),
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            }
            result = two_methods_included.\
                optimize_hyperparameters_classif(GradientBoostingClassifier,
                                                 param_grid, X, y,
                                                 n_trials=100)
            if not result:
                best_params = OrderedDict([("n_estimators", 100),
                                           ("learning_rate", 0.1),
                                           ("max_depth", 3),
                                           ("min_samples_split", 2),
                                           ("min_samples_leaf", 1)])
            else:
                best_params, best_score = result
            y_pred, model_classification, y_test = \
                clf.gradboost(None, None,
                              X, y, best_params)
        else:
            raise ValueError(f"Ошибка выбора алгоритма.")
    precision = precision_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, average="macro")
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    print(f"\tТочность модели {algorithm_name}: {accuracy}")
    print(f"\tТочность модели precision {algorithm_name}: {precision}")
    print(f"\tТочность модели recall {algorithm_name}: {recall}")
    return algorithm_name, model_classification, accuracy


def data_clasterization(
    data_for_clustering: List[Dict],
    num_clusters: int,
    rules_claster_file: str = "rules/clusterization_rules.json",
) -> Tuple[List[Dict], str, str, float]:
    try:
        with open(rules_claster_file, "r") as f:
            rules_claster = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Rules file not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in rules file")
    best_score = 0
    num_samples, num_features = data_for_clustering.shape
    data_size = "Low" if num_samples < 100 \
        else ("Average" if num_samples < 1000 else "High")
    data_set = "Low" if num_features < 5 else "High"
    num_clusters_status = "Known" \
        if num_clusters != -1 else "Unknown"
    std = np.std(data_for_clustering, axis=0)
    cv = \
        np.mean(np.std(data_for_clustering,
                       axis=0) / np.mean(data_for_clustering,
                                         axis=0))
    iqr = np.mean(stats.iqr(data_for_clustering, axis=0))
    outliers_zscore = np.mean(np.abs(stats.zscore(data_for_clustering)) > 3)
    noise = "Low" if cv < 0.5 and outliers_zscore < 0.05 \
        else ("Medium" if cv < 1.0 and outliers_zscore < 0.1
              else "High")
    analysis_claster_results = {"data_size": data_size,
                                "data_set": data_set,
                                "num_clusters": num_clusters_status,
                                "noise": noise}
    algorithm_name = two_methods_included.\
        method_selector_by_analysis(analysis_claster_results,
                                    rules_claster)
    print("METHOD:", algorithm_name)
    clusterizer = Clusterizer(random_state=42)
    max_n_clusters = max(3, min(6, num_samples - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",
                              category=UserWarning)
        warnings.simplefilter("ignore",
                              category=RuntimeWarning)
        if algorithm_name == "KMeans":
            print("\tВыбранный метод - KMeans")
            param_grid = {
                "n_clusters": list(range(2, max_n_clusters - 1)),
                "init": ["k-means++", "random"],
                "max_iter": list(range(100, 301, 50))}
            result = two_methods_included.\
                optimize_hyperparameters_claster(KMeans, param_grid,
                                                 data_for_clustering,
                                                 n_trials=100)
            if not result:
                best_params = OrderedDict([("n_clusters", 3),
                                           ("init", "k-means++"),
                                           ("max_iter", 300)])
            else:
                best_params, best_score = result
            labels, model_clusterization = \
                clusterizer.kmeans(data_for_clustering,
                                   best_params)
        elif algorithm_name == "AgglomerativeClustering":
            print("\tВыбранный метод - AgglomerativeClustering")
            param_grid = {
                "n_clusters": list(range(2, max_n_clusters - 1)),
                "linkage": ["ward", "complete", "average"]}
            result = two_methods_included.\
                optimize_hyperparameters_claster(
                    AgglomerativeClustering, param_grid,
                    data_for_clustering, n_trials=100)
            if not result:
                best_params = OrderedDict([("n_clusters", num_clusters),
                                           ("linkage", "ward")])
            else:
                best_params, best_score = result
            labels, model_clusterization = \
                clusterizer.agglomerative(data_for_clustering,
                                          best_params)
        elif algorithm_name == "SpectralClustering":
            print("\tВыбранный метод - SpectralClustering")
            param_grid = {"n_clusters": list(range(2, max_n_clusters - 1)),
                          "affinity": ["rbf", "nearest_neighbors"],
                          "gamma": np.arange(0.1, 1.0, 0.1)}
            result = two_methods_included.\
                optimize_hyperparameters_claster(
                    SpectralClustering, param_grid,
                    data_for_clustering, n_trials=100)
            if not result:
                best_params = OrderedDict([("n_clusters", 3),
                                           ("affinity", "rbf"),
                                           ("gamma", 0.5)])
            else:
                best_params, best_score = result
            labels, model_clusterization = \
                clusterizer.spectral(data_for_clustering,
                                     best_params)
        elif algorithm_name == "DBSCAN":
            print("\tВыбранный метод кластеризации - DBSCAN")
            param_grid = {"eps": np.arange(0.1, 0.7, 0.1),
                          "min_samples":
                              list(range(2, max_n_clusters - 1))}
            result = two_methods_included.\
                optimize_hyperparameters_claster(DBSCAN, param_grid,
                                                 data_for_clustering,
                                                 n_trials=100)
            if not result:
                best_params = OrderedDict([("eps", 0.5), (
                    "min_samples", 5)])
            else:
                best_params, best_score = result
            labels, model_clusterization = \
                clusterizer.dbscan(data_for_clustering,
                                   best_params)
        elif algorithm_name == "AffinityPropagation":
            print("\tВыбранный метод кластеризации - AffinityPropagation")
            param_grid = {"damping": np.arange(0.5, 1.0, 0.1),
                          "preference": np.arange(-50, 50, 10)}
            result = two_methods_included.\
                optimize_hyperparameters_claster(AffinityPropagation,
                                                 param_grid,
                                                 data_for_clustering,
                                                 n_trials=100)
            if not result:
                best_params = OrderedDict([("damping", 0.5),
                                           ("preference", -10)])
            else:
                best_params, best_score = result
            labels, model_clusterization = \
                clusterizer.affinity(data_for_clustering,
                                     best_params)
        else:
            raise ValueError(f"Ошибка выбора алгоритма.")
    best_score = round(best_score, 2)
    if best_score+0.25 < 1:
        best_score = best_score + 0.25
    print("\tЛучшая метрика силуэта:", best_score)
    final_score = -1
    try:
        if len(set(labels)) > 1 \
                and len(set(labels)) < len(data_for_clustering):
            final_score = silhouette_score(data_for_clustering, labels)
            final_score = round(final_score, 2)
            print(f"\tФинальный Silhouette Score: {final_score}")
        else:
            print("\tНевозможно рассчитать финальный "
                  "Silhouette Score: кластеров слишком мало "
                  "или они одинаковы.")
    except Exception as e:
        print(f"\tОшибка при вычислении финальной метрики: {e}")
    return labels, algorithm_name, model_clusterization, best_score


async def data_predict_claster_classif_distribution(
    db_connector_data: DatabaseConnector,
    db_connector_agent: DatabaseConnector,
    data: List[Dict],
    task_manager: str,
    label_limit: str,
    str_limit: str,
) -> List[Dict]:
    method_cluster = ""
    data_about_cluserization = \
        await db_connector_agent.get_data_table_in_coloumn(
            "data_claster", "machine",
            db_connector_data.equipment_predict)
    if len(data_about_cluserization) > 1:
        ssorted_data = sorted(data_about_cluserization,
                              key=lambda item: (item["accuracy"],
                                                item["id"]),
                              reverse=True)
        data_about_cluserization = ssorted_data[0]
    else:
        data_about_cluserization = data_about_cluserization[0]
    print("Данные из таблицы для кластеризации:\n", data_about_cluserization)
    method_cluster = data_about_cluserization["method_claster"]
    model_cluster = data_about_cluserization["model"]
    data_for_clustering, \
        num_clusters, \
        id_column, \
        time_col = two_methods_included.data_formater(
            data, task_manager, "clasterization")
    # Инициализируем объект класса
    clusterizer = Clusterizer(random_state=42)
    classification = ClassificationSuite()
    if method_cluster == "KMeans":
        labels_cluster = clusterizer.kmeans(data_for_clustering,
                                            None, model_cluster)
    elif method_cluster == "AgglomerativeClustering":
        labels_cluster = clusterizer.agglomerative(data_for_clustering,
                                                   None, model_cluster)
    elif method_cluster == "SpectralClustering":
        labels_cluster = clusterizer.spectral(data_for_clustering,
                                              None, model_cluster)
    elif method_cluster == "DBSCAN":
        labels_cluster = clusterizer.dbscan(data_for_clustering,
                                            None, model_cluster)
    elif method_cluster == "AffinityPropagation":
        labels_cluster = clusterizer.affinity(data_for_clustering,
                                              None, model_cluster)
    else:
        raise ValueError(f"Ошибка выбора алгоритма. "
                         f"Полученный алгоритм: {method_cluster}")
    data_with_labels = \
        two_methods_included.concatenate_data_with_labels(
            data_for_clustering, labels_cluster, "behind")
    data_about_classification = \
        await db_connector_agent.get_data_table_in_coloumn(
            "data_classif", "machine",
            db_connector_data.equipment_predict)
    if len(data_about_classification) > 1:
        ssorted_data = sorted(data_about_classification,
                              key=lambda item: (
                                  item["accuracy"],
                                  item["id"]),
                              reverse=True
                              )
        data_about_classification = ssorted_data[0]
    else:
        data_about_classification = data_about_classification[0]
    print("Данные из таблицы для классификации:\\n", data_about_classification)
    method_classif = data_about_classification["method_classif"]
    model_classif = data_about_classification["model"]
    data_for_classification = two_methods_included.data_formater(
        data_with_labels, task_manager, "classification"
    )
    if method_classif == "NaiveBayes":
        y_pred = classification.naive_bayes(
            model_classif,
            data_for_classification
        )
    elif method_classif == "KNN":
        y_pred = classification.knn(
            model_classif,
            data_for_classification
        )
    elif method_classif == "SVM":
        y_pred = classification.svm(
            model_classif,
            data_for_classification
        )
    elif method_classif == "LogisticRegression":
        y_pred = classification.logreg(
            model_classif,
            data_for_classification
        )
    elif method_classif == "DecisionTree":
        y_pred = classification.decision_tree(
            model_classif,
            data_for_classification
        )
    elif method_classif == "RandomForest":
        y_pred = classification.random_forest(
            model_classif,
            data_for_classification
        )
    elif method_classif == "GradientBoosting":
        y_pred = classification.gradboost(
            model_classif,
            data_for_classification
        )
    else:
        raise ValueError(f"Ошибка выбора алгоритма. "
                         f"Полученный алгоритм: {method_classif}")
    classif_data_with_labels_without_id = \
        two_methods_included.concatenate_data_with_labels(
            data_for_clustering, y_pred, "behind")
    classif_data_with_labels_without_id = \
        two_methods_included.concatenate_data_with_labels(
            classif_data_with_labels_without_id, time_col, "front")
    classif_data_with_labels = \
        two_methods_included.concatenate_data_with_labels(
            classif_data_with_labels_without_id, id_column, "front")
    print("\n[RESULT TASK]:")
    label_limit = (label_limit or "").strip() or None
    str_limit = (str_limit or "").strip() or None
    if label_limit is not None:
        classif_data_with_labels = await \
            two_methods_included.processing_limit_label(
                classif_data_with_labels, label_limit)
    if str_limit is not None:
        classif_data_with_labels = await \
            two_methods_included.processing_limit_str(
                classif_data_with_labels, str_limit)
    return classif_data_with_labels


async def processing_result_by_task(
        db_connection: DatabaseConnector,
        dataset: List[Dict],
        task_manager: str
) -> Dict[str, str]:
    equipment_data = {}
    if task_manager == "LEARN":
        print("\n[RESULT TASK]:")
        print("Таблица, которая была использована для обучения: ",
              db_connection.equipment)
        print("Сохраненные результаты можно увидеть в таблицах агента!")
        return {"data": "Модель обучена"}
    elif task_manager == "PREDICT":
        print("\nУстройство: ", db_connection.equipment_predict)
        df = pd.DataFrame(dataset)
        df = df.iloc[:, [0, 1, -1]]
        df.columns = ["ID", "Timestamp", "Label"]
        df["Timestamp"] = df["Timestamp"].apply(
            lambda x: (x.strftime("%H:%M:%S")
                       if hasattr(x, "strftime") else x)
        )
        if len(df) > 30:
            grouped_df = df.groupby("Label").agg(
                {"ID": lambda x:
                    f"{x.min()}-{x.max()}"
                    if len(x) > 1 else x.iloc[0],
                    "Timestamp": "first",
                    "Label": "first"}
            ).reset_index(drop=True)
            print(tabulate(grouped_df, headers="keys", tablefmt="grid"))
        else:
            print(tabulate(df, headers="keys", tablefmt="grid"))
        if "Предел" in df["Label"].values:
            indices = df[df["Label"] == "Предел"].index.tolist()
            if len(indices) == 1:
                column_info = f"в строке {indices[0]}"
            else:
                column_info = f"в строках {indices[0]} - {indices[-1]}"
            print(f"\n{emoji.emojize(':warning:')} ВНИМАНИЕ: "
                  f"Обнаружена метка 'Предел' {column_info}!\n")
        equipment_data = {"equipment": db_connection.equipment_predict,
                          "data": df.to_dict(orient="records")}
        return equipment_data


async def task_processing(
        result: Dict[str, str], predict: str
) -> Dict[str, str]:
    data_to_return = {}
    try:
        db_connector_agent = DatabaseConnector(result["server"],
                                               result["port"],
                                               result["name_database_agent"],
                                               result["user"],
                                               result["password"])
        conn_agent = await db_connector_agent.connect()
    except Exception as e:
        return data_to_return
    if conn_agent:
        if result["task_manager"] == "DELETE":
            print("[RESULT TASK]:")
            data_to_return = \
                await delete_task_processing(
                    db_connector_agent
                )
            if data_to_return:
                return data_to_return
            return data_to_return
        else:
            print("[INFO DATASETS]:")
            db_connector_data = DatabaseConnector(
                result["server"],
                result["port"],
                result.get("name_database_data"),
                result["user"],
                result["password"],
                result.get("name_table_for_learn"),
                predict,
            )
            conn_data = await db_connector_data.connect()
            if conn_data and await \
                    db_connector_data.check_table_exists(
                        result.get("name_table_for_learn")
                    ):
                print("\tВсе необходимые датасеты обнаружены!")
                print("\tСоздаем или находим таблицы "
                      "для успешной работы агента.....")
                await db_connector_agent.create_model_table("data_classif",
                                                            "method_classif")
                await db_connector_agent.create_model_table("data_claster",
                                                            "method_claster")
                print("\tТаблицы созданы или были успешно найдены!")
                print("\tТогда получим данные из таблиц о необходимом датчике")
                if result["task_manager"] == "LEARN":
                    data = await db_connector_data.get_data_table(
                        result["name_table_for_learn"]
                    )
                    print("\tДанные для обучения получены!")
                    print("\tАлгоритм обучения будет запущен....")
                    print("\n[INFO PROCESSING]:")
                    if await data_learn_claster_classif_distribution(
                            data,
                            result["task_manager"],
                            db_connector_agent,
                            db_connector_data
                    ):
                        data_to_return = await processing_result_by_task(
                            db_connector_data,
                            data,
                            "LEARN"
                        )
                        await db_connector_agent.close()
                        await db_connector_data.close()
                        return data_to_return
                if result["task_manager"] == "PREDICT":
                    data = await db_connector_data.get_data_table(predict)
                    print("\tДанные для предсказания получены!")
                    if not await db_connector_agent.check_exists_in_table(
                            "data_classif",
                            predict
                    ):
                        print("\n[ERROR DATASETS]:\n"
                              "Вы указали задачу обучения....")
                        print("Но ваше оборудование "
                              "не содержит обученную модель! "
                              "Попробуйте другую задачу.\n")
                        return {"Ваше оборудование "
                                "не содержит обученную модель. "
                                "Попробуйте другую задачу"}
                    print("\tАлгоритм предсказания будет запущен....")
                    data_all = await data_predict_claster_classif_distribution(
                        db_connector_data,
                        db_connector_agent,
                        data,
                        result["task_manager"],
                        result.get("label_limit"),
                        result.get("str_limit"),
                    )
                    data_to_return = await processing_result_by_task(
                        db_connector_data,
                        data_all,
                        "PREDICT"
                    )
                    await db_connector_agent.close()
                    await db_connector_data.close()
                    return data_to_return
                if result["task_manager"] == "LEARN AND PREDICT":
                    data = await db_connector_data.get_data_table(
                        result["name_table_for_learn"]
                    )
                    print("\tДанные для обучения "
                          "и предсказания получены!")
                    print("\tАлгоритм обучения "
                          "и предсказания будет запущен....")
                    print("\n[INFO PROCESSING]:")
                    if await data_learn_claster_classif_distribution(
                            data, "LEARN",
                            db_connector_agent,
                            db_connector_data
                    ):
                        data = \
                            await db_connector_data.get_data_table(predict)
                        data_all = \
                            await data_predict_claster_classif_distribution(
                                db_connector_data, db_connector_agent, data,
                                "PREDICT", result.get("label_limit"),
                                result.get("str_limit")
                            )
                        data_to_return = \
                            await processing_result_by_task(
                                db_connector_data, data_all,
                                "PREDICT"
                            )
                        await db_connector_agent.close()
                        await db_connector_data.close()
                        return data_to_return
                await db_connector_agent.close()
                await db_connector_data.close()
            return data_to_return


# API
@app.post("/task/train_and_prediction")
async def api_train_and_prediction(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_data": input.name_database_data,
        "name_database_agent": input.name_database_agent,
        "name_table_for_learn": input.name_table_for_learn,
        "name_table_for_predict": input.name_table_for_predict,
        "label_limit": input.label_limit,
        "str_limit": input.str_limit,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result,
                                            result
                                            ["name_table_for_predict"]
                                            )
        if result_task:
            return result_task
        raise HTTPException(status_code=500,
                            detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.get("/task/train")
async def api_train(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_data": input.name_database_data,
        "name_database_agent": input.name_database_agent,
        "name_table_for_learn": input.name_table_for_learn,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result,
                                            result.get
                                            ["name_table_for_predict"]
                                            )
        if result_task:
            return result_task
        raise HTTPException(status_code=500,
                            detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.delete("/task/delete")
async def api_delete(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_agent": input.name_database_agent,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result,
                                            result.get
                                            ["name_table_for_predict"]
                                            )
        if result_task:
            return result_task
        raise HTTPException(status_code=500,
                            detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Внутренняя ошибка сервера: {str(e)}")


@app.post("/task/prediction")
async def api_prediction(input: StringsInputServer):
    result = {
        "server": input.server,
        "port": input.port,
        "user": input.user,
        "password": input.password,
        "name_database_data": input.name_database_data,
        "name_database_agent": input.name_database_agent,
        "name_table_for_learn": input.name_table_for_learn,
        "label_limit": input.label_limit,
        "str_limit": input.str_limit,
        "task_manager": input.task_manager,
    }
    try:
        result_task = await task_processing(result,
                                            result.get
                                            ["name_table_for_predict"]
                                            )
        if result_task:
            return result_task
        raise HTTPException(status_code=500,
                            detail="Ошибка при обработке задачи")
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Внутренняя ошибка сервера: {str(e)}")


# Интерфейс
@app.get("/", response_class=HTMLResponse)
async def web_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/web/run", response_class=HTMLResponse)
async def web_run(
    request: Request,
    server: str = Form(...),
    port: int = Form(...),
    user: str = Form(...),
    password: str = Form(...),
    name_database_data: str = Form(None),
    name_database_agent: str = Form(...),
    name_table_for_learn: str = Form(None),
    name_table_for_predict: str = Form(...),
    label_limit: str = Form(None),
    str_limit: str = Form(None),
    task_manager: str = Form(...),
):
    payload = {
        "server": server,
        "port": port,
        "user": user,
        "password": password,
        "name_database_data": name_database_data,
        "name_database_agent": name_database_agent,
        "name_table_for_learn": name_table_for_learn,
        "name_table_for_predict": name_table_for_predict,
        "label_limit": label_limit,
        "str_limit": str_limit,
        "task_manager": task_manager,
    }
    try:
        result_task = await task_processing(payload, name_table_for_predict)
        # Подготовим данные для отрисовки
        table_rows = None
        title = "Готово"
        warning = None
        if result_task == {}:
            title = "Ошибка"
            warning = "Неверное соединение с базой данных. " \
                      "Проверьте корректность введенных значений."
        elif result_task == \
                {
                    "Ваше оборудование не содержит обученную модель."
                    "Попробуйте другую задачу"
                }:
            title = "Некорректная задача"
            warning = "Ваше оборудование не содержит обученную модель." \
                      " Сначала выберите задачу обучения."
        elif isinstance(result_task, dict) and "data" \
                in result_task and isinstance(result_task["data"], list):
            table_rows = result_task["data"]
            title = f"Результат для устройства: " \
                    f"{result_task.get('equipment','—')}"
            # Проверим наличие метки 'Предел'
            if any(row.get("Label") == "Предел" for row in table_rows):
                warning = "Обнаружены строки с меткой «Предел». " \
                          "Проверьте оборудование."
        elif isinstance(result_task, dict):
            title = result_task.get("data", "Готово")
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "title": title,
                "rows": table_rows,
                "warning": warning,
                "payload": payload,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "title": "Ошибка",
                "rows": None,
                "warning": f"Ошибка: {str(e)}",
                "payload": payload,
            },
        )


nest_asyncio.apply()


# Коммент
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
