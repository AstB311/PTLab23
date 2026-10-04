# Система классификации на основе кластеризации данных мониторинга оборудования

## Описание
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

%% =================== Базовый класс ===================
class BaseMLSuite {
  - __random_state : int
  + random_state : int
  + __init__(random_state: int = 42)
}

%% =================== Классификация ===================
class ClassificationSuite {
  - test_size : float
  + __init__(test_size: float = 0.2, random_state: int = 42)
  - _split(X, y)
  - _train_and_predict(estimator_cls, X, y, best_params)
  - _predict_loaded(loaded_model, data)
  + determine_linearity(X, y, threshold) str
  + naive_bayes(loaded_model, data, X, y, best_params)
  + knn(loaded_model, data, X, y, best_params)
  + svm(loaded_model, data, X, y, best_params)
  + logreg(loaded_model, data, X, y, best_params)
  + decision_tree(loaded_model, data, X, y, best_params)
  + random_forest(loaded_model, data, X, y, best_params)
  + gradboost(loaded_model, data, X, y, best_params)
  + dump_model(model) bytes
}

%% =================== Кластеризация ===================
class Clusterizer {
  + __init__(random_state: int = 42)
  - _fit_with(data, params, loaded_model, build_model_fn)
  + kmeans(data, params, loaded_model)
  + agglomerative(data, params, loaded_model)
  + spectral(data, params, loaded_model)
  + dbscan(data, params, loaded_model)
  + affinity(data, params, loaded_model)
}

%% =================== Подключение к БД ===================
class DatabaseConnector {
  - __server : str
  - __port : int
  - __database : str
  - __user : str
  - __password : str
  - __conn : Connection
  + equipment : str
  + equipment_predict : str
  + server : str
  + port : int
  + database : str
  + user : str
  + is_connected : bool
  - __require_connection() Connection
  + __init__(server, port, database, user, password, equipment, equipment_predict)
  + connect() bool
  + check_table_exists(table_name, schema) bool
  + check_exists_in_table(table_name, machine_name, schema) bool
  + create_model_table(table_name, table_column_name, schema)
  + insert_data(table_name, data, schema)
  + get_data_table(table_name, schema) List
  + get_data_table_in_coloumn(table_name, coloumn_name, machine_name, schema) List
  + delete_table_agent(table_name, schema)
  + close()
}

%% =================== Pydantic-модель ===================
class StringsInputServer {
  + server : str
  + port : int
  + user : str
  + password : str
  + name_database_data : Optional~str~
  + name_database_agent : str
  + name_table_for_learn : Optional~str~
  + name_table_for_predict : str
  + label_limit : Optional~str~
  + str_limit : Optional~str~
  + task_manager : str
}

%% =================== Вспомогательный модуль ===================
class two_methods_included {
  <<module>>
  + concatenate_data_with_labels(data, labels_data, side)
  + data_formater(data, task_manager, method)
  + method_selector_by_analysis(analysis_results, rules) str
  + objective_claster(trial, model_class, param_grid, X) float
  + optimize_hyperparameters_claster(model_class, param_grid, X, n_trials)
  + objective_classif(trial, model_class, param_grid, X, y) float
  + optimize_hyperparameters_classif(model_class, param_grid, X, y, n_trials)
  + processing_limit_str(data, str_limit) List
  + processing_limit_label(data, label_limit) List
}

%% =================== FastAPI-приложение ===================
class FastAPIApp {
  <<main.py>>
  + task_processing(result, predict)
  + data_learn_claster_classif_distribution(...)
  + data_predict_claster_classif_distribution(...)
  + data_classification(...)
  + data_clasterization(...)
  + delete_task_processing(...)
  + processing_result_by_task(...)
  + api_train_and_prediction(input)
  + api_train(input)
  + api_delete(input)
  + api_prediction(input)
  + web_index(request)
  + web_run(...)
}

%% =================== Наследование (обобщение) ===================
BaseMLSuite <|-- ClassificationSuite : extends
BaseMLSuite <|-- Clusterizer : extends

%% =================== Зависимости (dependency) ===================
FastAPIApp ..> StringsInputServer       : использует как входной DTO
FastAPIApp ..> DatabaseConnector        : создаёт и вызывает
FastAPIApp ..> ClassificationSuite      : создаёт и вызывает
FastAPIApp ..> Clusterizer              : создаёт и вызывает
FastAPIApp ..> two_methods_included     : вызывает функции

ClassificationSuite ..> two_methods_included : использует утилиты
Clusterizer         ..> two_methods_included : использует утилиты

ClassificationSuite ..> DatabaseConnector : pickle-модели в БД
Clusterizer         ..> DatabaseConnector : pickle-модели в БД

%% =================== Внутренняя связь алгоритмов ===================
ClassificationSuite ..> Clusterizer : использует метки кластеров
```