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

class DatabaseConnector {
  - server: str
  - port: int
  - database: str
  - user: str
  - password: str
  - equipment: str
  - equipment_predict: str
  - conn: Connection
  + connect(timeout)
  + check_table_exists(table_name, schema)
  + check_exists_in_table(table_name, machine_name, schema)
  + create_model_table(table_name, column_name, schema)
  + insert_data(table_name, data, schema)
  + get_data_table(table_name, schema)
  + get_data_table_in_coloumn(table_name, column_name, machine_name, schema)
  + delete_table_agent(table_name, schema)
  + close()
}

class ClassificationSuite {
  - test_size: float
  - random_state: int
  + determine_linearity(X, y, threshold)
  + naive_bayes(...)
  + knn(...)
  + svm(...)
  + logreg(...)
  + decision_tree(...)
  + random_forest(...)
  + gradboost(...)
}

class Clusterizer {
  - random_state: int
  + kmeans(...)
  + agglomerative(...)
  + spectral(...)
  + dbscan(...)
  + affinity(...)
}

%% связи по твоему описанию
DatabaseConnector ..> ClassificationSuite : использует анализ
ClassificationSuite ..> DatabaseConnector : использует подключение к БД
Clusterizer ..> DatabaseConnector : использует подключение к БД
ClassificationSuite ..> Clusterizer : использует данные кластеризации
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