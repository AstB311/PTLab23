import pickle
from unittest.mock import AsyncMock
import asyncpg
import pytest
from fastapi import HTTPException
from src.analysis.connector import DatabaseConnector


@pytest.mark.unit
class TestDatabaseConnector:
    @pytest.mark.parametrize("port", [1, 5432, 65535])
    def test_valid_ports(self, port):
        assert DatabaseConnector("host", port, "db", "user", "pw").port == port

    @pytest.mark.parametrize("port", [0, -1, 65536])
    def test_port_bounds(self, port):
        with pytest.raises(ValueError):
            DatabaseConnector("host", port, "db", "user", "pw")

    @pytest.mark.parametrize("port", [True, "5432", 5432.0, None])
    def test_port_type(self, port):
        with pytest.raises(TypeError):
            DatabaseConnector("host", port, "db", "user", "pw")

    def test_properties(self, connector):
        assert (
            connector.server,
            connector.database,
            connector.user) == (
            "localhost",
            "test",
            "tester")
        assert not hasattr(connector, "password")
        assert not connector.is_connected

    @pytest.mark.parametrize("field", ["server", "port", "database", "user"])
    def test_read_only(self, connector, field):
        with pytest.raises(AttributeError):
            setattr(connector, field, "other")

    async def test_connect_idempotent(self, connected):
        obj, conn = connected
        await obj.connect()
        await obj.connect()
        assert obj.is_connected

    async def test_connect_error(self, connector, monkeypatch):
        monkeypatch.setattr(
            "src.analysis.connector.asyncpg.connect", AsyncMock(
                side_effect=asyncpg.PostgresError("fail")))
        with pytest.raises(HTTPException) as exc:
            await connector.connect()
        assert exc.value.status_code == 500
        assert not connector.is_connected

    @pytest.mark.parametrize("method,args", [
        ("check_table_exists", ("models",)),
        ("get_data_table", ("models",)),
        ("delete_table_agent", ("models",)),
        ("create_model_table", ("models", "algorithm")),
        ("insert_data", ("models", {"id": 1})),
        ("get_data_table_in_coloumn", ("models", "machine", "A"))])
    async def test_operation_without_connection(self, connector, method, args):
        with pytest.raises(HTTPException) as exc:
            await getattr(connector, method)(*args)
        assert exc.value.status_code == 500

    async def test_insert_serializes_without_mutation(self, connected):
        obj, conn = connected
        await obj.connect()
        data = {"machine": "A", "model": {"a": 1}}
        await obj.insert_data("models", data)
        assert data["model"] == {"a": 1}
        query, machine, blob = conn.execute.await_args.args
        assert "VALUES ($1, $2)" in query
        assert machine == "A"
        assert pickle.loads(blob) == {"a": 1}

    async def test_empty_insert(self, connected):
        obj, conn = connected
        await obj.connect()
        with pytest.raises(ValueError):
            await obj.insert_data("models", {})
        conn.execute.assert_not_awaited()

    @pytest.mark.parametrize("name", ["x;DROP TABLE models", "", "a b", "a.b"])
    async def test_invalid_sql_identifier(self, connected, name):
        obj, conn = connected
        await obj.connect()
        with pytest.raises(ValueError):
            await obj.get_data_table(name)
        conn.fetch.assert_not_awaited()

    @pytest.mark.parametrize("method,args,operation", [
        ("check_table_exists", ("models",), "fetchval"),
        ("check_exists_in_table", ("models", "A"), "fetchval"),
        ("get_data_table", ("models",), "fetch"),
        ("get_data_table_in_coloumn", ("models", "machine", "A"), "fetch"),
        ("delete_table_agent", ("models",), "execute"),
        ("create_model_table", ("models", "algorithm"), "execute"),
        ("insert_data", ("models", {"id": 1}), "execute")])
    async def test_query_errors(self, connected, method, args, operation):
        obj, conn = connected
        await obj.connect()
        getattr(conn, operation).side_effect = asyncpg.PostgresError(
            "test error")
        with pytest.raises(HTTPException) as exc:
            await getattr(obj, method)(*args)
        assert exc.value.status_code == 500

    async def test_close_twice(self, connected):
        obj, conn = connected
        await obj.connect()
        await obj.close()
        await obj.close()
        assert not obj.is_connected
        conn.close.assert_awaited_once()

    async def test_close_already_closed(self, connected):
        obj, conn = connected
        await obj.connect()
        conn.is_closed.return_value = True
        await obj.close()
        conn.close.assert_not_awaited()
