import pytest
import pickle
from unittest.mock import AsyncMock, patch, ANY
from src.analysis.connector import DatabaseConnector
from fastapi import HTTPException
import asyncpg


@pytest.fixture
def connector():
    return DatabaseConnector("localhost", 5432, "db", "user", "pass")


# CONNECT
@pytest.mark.asyncio
@patch("src.analysis.connector.asyncpg.connect", new_callable=AsyncMock)
async def test_connect_passes_required_params(mock_connect, connector):
    mock_conn = AsyncMock()
    mock_connect.return_value = mock_conn

    conn = await connector.connect()

    assert conn is mock_conn
    mock_connect.assert_awaited_once_with(
        user="user",
        password="pass",
        database="db",
        host="localhost",
        port=5432,
    )


@pytest.mark.asyncio
@patch("src.analysis.connector.asyncpg.connect", new_callable=AsyncMock)
async def test_connect_error(mock_connect, connector):
    mock_connect.side_effect = asyncpg.PostgresError("boom")
    with pytest.raises(HTTPException) as ei:
        await connector.connect()
    assert "Database connection error" in ei.value.detail


# check_table_exists
@pytest.mark.asyncio
async def test_check_table_exists(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    mock_conn.fetchval.return_value = True

    ok = await connector.check_table_exists(table_name="MyTable",
                                            schema="myschema")
    assert ok is True

    # args: (sql, schema, table_name_lower)
    args = mock_conn.fetchval.call_args[0]
    assert args[1] is not None


# check_exists_in_table
@pytest.mark.asyncio
async def test_check_exists_in_table(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    mock_conn.fetchval.return_value = True

    ok = await connector.check_exists_in_table("models",
                                               "machX",
                                               schema="custom")
    assert ok is True

    # args: (sql, like_param)
    args = mock_conn.fetchval.call_args[0]
    assert args[0] is not None  # sql, игнорируем
    assert args[1] is not None


# create_model_table
@pytest.mark.asyncio
async def test_create_model_table(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    await connector.create_model_table("models", "model_name", schema="public")
    mock_conn.execute.assert_awaited_once()


# get_data_table
@pytest.mark.asyncio
async def test_get_data_table(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    mock_conn.fetch.return_value = [{"id": 1}, {"id": 2}]

    rows = await connector.get_data_table("models", schema="public")
    assert isinstance(rows, list)
    mock_conn.fetch.assert_awaited_once()


# get_data_table_in_coloumn
@pytest.mark.asyncio
async def test_get_data_table_in_coloumn(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn
    mock_conn.fetch.return_value = [{"machine": "A"}]

    res = await connector.get_data_table_in_coloumn(
        "models", "machine", "mach", schema="public"
    )
    assert res == [{"machine": "A"}]

    # args: (sql, like_param)
    args = mock_conn.fetch.call_args[0]
    assert args[0] is not None  # sql, игнорируем
    assert args[1] == "%mach%"


# delete_table_agent
@pytest.mark.asyncio
async def test_delete_table_agent(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    await connector.delete_table_agent("models", schema="public")
    mock_conn.execute.assert_awaited_once()


# close
@pytest.mark.asyncio
async def test_close_calls(connector):
    mock_conn = AsyncMock()
    connector.conn = mock_conn

    await connector.close()
    mock_conn.close.assert_awaited_once()
