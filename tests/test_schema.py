import pytest
from pydantic import ValidationError
from main_scripts.main import StringsInputServer


@pytest.mark.unit
class TestStringsInputServer:
    @pytest.mark.parametrize("task", ["LEARN",
                             "PREDICT", "LEARN AND PREDICT", "DELETE"])
    def test_valid_tasks(self, payload, task):
        assert StringsInputServer(
            **{**payload, "task_manager": task}).task_manager == task

    @pytest.mark.parametrize("port", [1, 65535])
    def test_valid_port_bounds(self, payload, port):
        assert StringsInputServer(**{**payload, "port": port}).port == port

    @pytest.mark.parametrize("field", ["server", "user", "name_database_agent",
                             "name_database_data", "name_table_for_learn"])
    def test_required_fields(self, payload, field):
        with pytest.raises(ValidationError):
            StringsInputServer(**{**payload, field: " "})

    def test_predict_requires_target(self, payload):
        with pytest.raises(ValidationError):
            StringsInputServer(**{**payload,
                                  "task_manager": "PREDICT",
                                  "name_table_for_predict": None})

    def test_delete_minimal_fields(self, payload):
        keys = ("server", "port", "user", "password", "name_database_agent")
        request = {k: payload[k] for k in keys}
        assert StringsInputServer(
            **request, task_manager="DELETE").name_database_data is None
