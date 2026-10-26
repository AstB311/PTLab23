import json
import subprocess
import sys
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from main_scripts import cli


@pytest.mark.functional
@pytest.mark.parametrize("option,task", list(cli.MENU.items()))
def test_all_menu_operations(payload, option, task):
    values = iter([option] + (["DELETE"] if option == "4" else []) + ["0"])
    output = []
    service = AsyncMock(return_value={"data": "ok"})
    assert cli.run_menu(
        payload,
        lambda _: next(values),
        output.append,
        service) == 0
    assert service.await_args.args[0]["task_manager"] == task
    assert any('"data": "ok"' in line for line in output)


def test_cancel_delete(payload):
    values = iter(["4", "no", "0"])
    service = AsyncMock()
    output = []
    cli.run_menu(payload, lambda _: next(values), output.append, service)
    service.assert_not_awaited()
    assert "Удаление отменено" in output


@pytest.mark.parametrize("exc",
                         [ValueError("bad"),
                          HTTPException(409, "no model"),
                          RuntimeError("private")])
def test_service_error(payload, exc):
    values = iter(["1", "0"])
    output = []
    cli.run_menu(
        payload,
        lambda _: next(values),
        output.append,
        AsyncMock(
            side_effect=exc))
    assert any(line.startswith("Ошибка") for line in output)
    assert not any("private" in line for line in output)


def test_invalid_config_data(payload):
    values = iter(["1", "0"])
    output = []
    service = AsyncMock()
    cli.run_menu({**payload, "port": 0},
                 lambda _: next(values), output.append, service)
    service.assert_not_awaited()


def test_eof(payload):
    def stop(_):
        raise EOFError
    assert cli.run_menu(payload, stop, lambda _: None) == 0


def test_console_real_process(tmp_path, payload):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    process = subprocess.run([sys.executable,
                              "-m",
                              "main_scripts.cli",
                              "--config",
                              str(config)],
                             input="wrong\n0\n",
                             text=True,
                             encoding="utf-8",
                             capture_output=True,
                             timeout=30)
    assert process.returncode == 0
    assert "неизвестный пункт" in process.stdout
    assert "Работа завершена" in process.stdout


@pytest.mark.parametrize("text", ["not-json", "[]"])
def test_config_file_errors(tmp_path, text):
    config = tmp_path / "bad.json"
    config.write_text(text)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(config)])
    assert exc.value.code == 2
