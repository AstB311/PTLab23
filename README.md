# Лабораторная 2-3 по дисциплине "Технологии программирования"

---

## Цель работы
1. Освоить основы работы с системой контроля версий **Git**.  
2. Научиться использовать **GitHub** для хранения проектов.  
3. Освоить механизм автоматического тестирования и проверки стиля кода через **GitHub Actions (CI/CD)**.  
4. Реализовать задание с использованием языка **Python** и модульного тестирования (**pytest**).  
5. Реализовать **интерфейс** для взаимодействия с проектом.
6. Освоить подключение к базе данных **PostgreSQL**.

---

## Постановка задачи
Проект: система классификации на основе кластеризации данных мониторинга оборудования. 
Выполняется передача параметров и доступ к данным на основе API. 
Интерфейс представлен.

---

## Структура проекта
```
assistant/
 ├── .github/workflows/github-actions-testing.yml   # CI/CD
 ├── main_scripts/
 │    ├── rules/
 │    │    ├── classification_rules.json
 │    │    └── clusterization_rules.json
 │    └── main.py
 ├── src/
 │    ├── rules/
 │    │    ├── classification_methods.py
 │    │    ├── clusterization_methods.py
 │    │    ├── connector.py
 │    │    ├── base_ML_Suite.py
 │    │    └── two_methods_included.py
 ├── ui/
 │    ├── static/
 │    │    └── style.css
 │    ├── templates/
 │    │    ├── index.html
 │    │    └── result.html
 ├── test/
 │    ├── test_API.py
 │    └── test_database.py
 ├── requirements.txt
 ├── .gitignore
 └── README.md
```

---

## Используемые технологии
1. Язык программирования: **Python 3.10+**  
2. Управление зависимостями: **requirements.txt**  
3. Модульное тестирование: **pytest**   
4. CI/CD: **GitHub Actions**  
5. Проверка стиля кода: **pycodestyle (PEP8)**
6. Интерфейс: **jinja2**

---

## Инструкция по запуску

### Клонирование репозитория
```bash
git clone https://github.com/<ваш_логин>/PTLab2.git
```

### Создание баз данных PostgresSQL (обязательно для полной работы создать два экзепляра)
```bash
CREATE DATABASE <ваше_название_базы_данных> OWNER postgres;
```

### Установка переменной окружения с паролем PostgreSQL (или настройте PostgreSQL вручную)
```bash
$env:DATABASE_PASSWORD = "ps_password"
```

### Установка зависимостей
```bash
pip install -r requirements.txt
```

### Запуск тестов
```bash
pytest tests
```

### Проверка стиля кода (PEP8)
```bash
pycodestyle src tests main_scripts
```

### Запуск программы
```bash
python main_scripts/main.py
```

Перейти в браузере по адресу: http://127.0.0.1:8000
---

## Диаграмма классов
```mermaid
classDiagram
direction TB

%% ======================= Базовый класс =======================
class BaseMLSuite {
  - __random_state: int
  + random_state: int
  + __init__(random_state: int = 42)
}

%% ======================= Классификация =======================
class ClassificationSuite {
  - test_size: float
  + __init__(test_size: float = 0.2, random_state: int = 42)
  - _split(X, y) Tuple
  - _train_and_predict(estimator_cls, X, y, best_params) Tuple
  - _predict_loaded(loaded_model, data) Any
  + determine_linearity(X, y, threshold) str
  + naive_bayes(loaded_model, data, X, y, best_params) Any
  + knn(loaded_model, data, X, y, best_params) Any
  + svm(loaded_model, data, X, y, best_params) Any
  + logreg(loaded_model, data, X, y, best_params) Any
  + decision_tree(loaded_model, data, X, y, best_params) Any
  + random_forest(loaded_model, data, X, y, best_params) Any
  + gradboost(loaded_model, data, X, y, best_params) Any
  + dump_model(model) bytes
}

%% ======================= Кластеризация =======================
class Clusterizer {
  + __init__(random_state: int = 42)
  - _fit_with(data, params, loaded_model, build_model_fn) ReturnType
  + kmeans(data, params, loaded_model) ReturnType
  + agglomerative(data, params, loaded_model) ReturnType
  + spectral(data, params, loaded_model) ReturnType
  + dbscan(data, params, loaded_model) ReturnType
  + affinity(data, params, loaded_model) ReturnType
}

%% ======================= Подключение к БД =======================
class DatabaseConnector {
  - __server: str
  - __port: int
  - __database: str
  - __user: str
  - __password: str
  - __conn: asyncpg.Connection
  + equipment: str
  + equipment_predict: str
  + server: str
  + port: int
  + database: str
  + user: str
  + is_connected: bool
  - __require_connection() Connection
  + __init__(server, port, database, user, password, equipment, equipment_predict)
  + connect() bool
  + check_table_exists(table_name, schema) bool
  + check_exists_in_table(table_name, machine_name, schema) bool
  + create_model_table(table_name, table_column_name, schema) None
  + insert_data(table_name, data, schema) None
  + get_data_table(table_name, schema) List
  + get_data_table_in_coloumn(table_name, coloumn_name, machine_name, schema) List
  + delete_table_agent(table_name, schema) None
  + close() None
}

%% ======================= Модель входных данных API =======================
class StringsInputServer {
  + server: str
  + port: int
  + user: str
  + password: str
  + name_database_data: Optional~str~
  + name_database_agent: str
  + name_table_for_learn: Optional~str~
  + name_table_for_predict: str
  + label_limit: Optional~str~
  + str_limit: Optional~str~
  + task_manager: str
}

%% ======================= Вспомогательный модуль =======================
class two_methods_included {
  <<module>>
  + concatenate_data_with_labels(data, labels_data, side) ndarray
  + data_formater(data, task_manager, method) Any
  + method_selector_by_analysis(analysis_results, rules) str
  + objective_claster(trial, model_class, param_grid, X) float
  + optimize_hyperparameters_claster(model_class, param_grid, X, n_trials) Tuple
  + objective_classif(trial, model_class, param_grid, X, y) float
  + optimize_hyperparameters_classif(model_class, param_grid, X, y, n_trials) Tuple
  + processing_limit_str(data, str_limit) List
  + processing_limit_label(data, label_limit) List
}

%% ======================= FastAPI-приложение =======================
class FastAPIApp {
  <<main.py>>
  + task_processing(result, predict) Dict
  + data_learn_claster_classif_distribution(...) bool
  + data_predict_claster_classif_distribution(...) List
  + data_classification(...) Tuple
  + data_clasterization(...) Tuple
  + delete_task_processing(...) bool
  + processing_result_by_task(...) Dict
  + api_train_and_prediction(input) Dict
  + api_train(input) Dict
  + api_delete(input) Dict
  + api_prediction(input) Dict
  + web_index(request) HTMLResponse
  + web_run(...) HTMLResponse
}

%% ======================= Связи =======================
BaseMLSuite <|-- ClassificationSuite : наследование
BaseMLSuite <|-- Clusterizer : наследование

FastAPIApp ..> StringsInputServer : валидация входа
FastAPIApp ..> DatabaseConnector : создаёт и использует
FastAPIApp ..> ClassificationSuite : создаёт и использует
FastAPIApp ..> Clusterizer : создаёт и использует
FastAPIApp ..> two_methods_included : вызывает функции

two_methods_included ..> ClassificationSuite : оптимизация гиперпараметров
two_methods_included ..> Clusterizer : оптимизация гиперпараметров

ClassificationSuite ..> DatabaseConnector : сохраняет/загружает модель через pickle
Clusterizer ..> DatabaseConnector : сохраняет/загружает модель через pickle

ClassificationSuite ..> Clusterizer : использует метки кластеров
```

---

## Выводы
В ходе выполнения лабораторных работы были освоены:  
1. Базовые команды **Git** и принципы работы с репозиторием на GitHub.  
2. Настройка CI/CD через **GitHub Actions**.
3. Применение **pytest** для тестирования.
4. Контроль качества кода с помощью **pycodestyle**.
5. Интерфейс с помощью **jinja2**, обращение к API.
6. Работа с подключением к базе данных **PostgreSQL**.

Проект успешно протестирован и соответствует требованиям.  