# PTLab23 Тестирование Python проекта

Исходный проект классификации на основе кластеризации дополнен лабораторной работой 2.
Методичка описывает C++ Библиотека. Использование этого Python проекта требует согласования замены с преподавателем. pytest используется как аналог Google Test. Предметная область не заменена библиотекой книг.

## Быстрый запуск в Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pycodestyle src tests main_scripts
.\.venv\Scripts\python.exe -m pytest tests --cov=src --cov=main_scripts --cov-branch --cov-report=html:artifacts/htmlcov --cov-report=json:artifacts/coverage.json --junitxml=artifacts/tests.xml
```

Автоматические тесты не требуют PostgreSQL и работающего внешнего HTTP сервера. Данные синтетические. В модульных тестах БД заменены вызовы asyncpg, в интеграционном сценарии — хранилище. В API тестах запрос проходит через реальное ASGI приложение; HTTP ответ не подделывается.

## Приложение

```powershell
python -m uvicorn main_scripts.main:app --host 127.0.0.1 --port 8000
python -m main_scripts.cli --config config.local.json
```

Для приложения требуется настоящий PostgreSQL. Подготовка описана в docs/manual_postgresql.md. DELETE очищает все модели из двух таблиц базы агента. Пароль повторного web запуска вводится заново.

## Состав

- main_scripts/main.py — API, web форма и оркестрация.
- main_scripts/cli.py — меню четырёх операций и выхода.
- src/analysis/ — исходные классы и утилиты с исправлениями.
- tests/ — модульные, функциональные и компонентные проверки.
- docs/test_plan.md, test_cases.csv, pairwise.csv — план и сценарии.
- docs/bug_reports.md — 16 дефектов исходной версии и регрессия.
- docs/control_questions.md — ответы на вопросы методички.
- artifacts/tests.txt, tests.xml, coverage.json, htmlcov/ — фактический прогон.
- examples/demo_data.sql — синтетические данные для отдельной ручной проверки БД.

## Фактические результаты

217 passed; пропущенных и упавших тестов нет. Операторы 88.93%, ветви 70.99%, совокупное покрытие 84.86%. Это результат Python 3.12.14 в Linux; локальный Windows и GitHub Actions необходимо прогнать самостоятельно.

Большинство ветвей библиотечных классов покрыто, но часть ветвей оркестрации и резервных параметров не исполнена. Промышленная точность ML, безопасность развёртывания и настоящие транзакции PostgreSQL не подтверждаются этим набором.

Модели pickle загружать только из доверенного собственного хранилища. Для новых данных кластерный сервис выбирает KMeans или AffinityPropagation; трансдуктивные алгоритмы не переобучаются скрыто.
