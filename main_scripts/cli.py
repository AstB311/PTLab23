"""Консоль вызывает те же операции, что API."""
import argparse
import asyncio
import json
from pathlib import Path

from fastapi import HTTPException
from main_scripts.main import StringsInputServer, task_processing


MENU = {"1": "LEARN", "2": "PREDICT", "3": "LEARN AND PREDICT", "4": "DELETE"}


def run_menu(config, input_fn=input, output_fn=print, runner=None):
    """В тестах заменяются ввод и внешние зависимости."""
    runner = runner or task_processing
    while True:
        output_fn("1 Обучение | 2 Предсказание | 3 Обучение и предсказание | "
                  "4 Удалить ВСЕ модели | 0 Выход")
        try:
            choice = input_fn("Выберите пункт: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if choice == "0":
            output_fn("Работа завершена")
            return 0
        if choice not in MENU:
            output_fn("Ошибка: неизвестный пункт меню")
            continue
        if choice == "4":
            try:
                confirm = input_fn(
                    "Для удаления всех моделей введите DELETE: ")
            except (EOFError, KeyboardInterrupt):
                return 0
            if confirm != "DELETE":
                output_fn("Удаление отменено")
                continue
        try:
            request = StringsInputServer(
                **{**config, "task_manager": MENU[choice]})
            result = asyncio.run(
                runner(
                    request.model_dump(),
                    request.name_table_for_predict))
            output_fn(json.dumps(result, ensure_ascii=False, default=str))
        except (ValueError, HTTPException) as exc:
            detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
            output_fn(f"Ошибка: {detail}")
        except Exception:
            output_fn("Ошибка выполнения задачи; проверьте подключение")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Консоль менеджера моделей")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Конфигурация должна быть JSON объектом")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Ошибка конфигурации: {exc}\n")
    return run_menu(config)


if __name__ == "__main__":
    raise SystemExit(main())
