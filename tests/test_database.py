import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.analysis.connector import DatabaseConnector


@pytest.fixture
def connector():
    return DatabaseConnector(
        server="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_password",
    )


def make_mock_connection():
    """Создаёт mock соединения с поведением asyncpg.Connection."""
    mock_conn = Mock()

    # В asyncpg этот метод синхронный.
    mock_conn.is_closed.return_value = False

    # Эти методы асинхронные.
    mock_conn.fetchval = AsyncMock()
    mock_conn.fetch = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_conn.close = AsyncMock()

    return mock_conn


@pytest.mark.asyncio
@patch(
    "src.analysis.connector.asyncpg.connect",
    new_callable=AsyncMock,
)
async def test_connect_passes_required_params(mock_connect, connector):
    mock_conn = make_mock_connection()
    mock_connect.return_value = mock_conn

    result = await connector.connect()

    assert result is True

    mock_connect.assert_awaited_once_with(
        user="test_user",
        password="test_password",
        database="test_db",
        host="localhost",
        port=5432,
    )


@pytest.mark.asyncio
async def test_check_table_exists(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    mock_conn.fetchval.return_value = True

    ok = await connector.check_table_exists(
        table_name="MyTable",
        schema="myschema",
    )

    assert ok is True

    mock_conn.fetchval.assert_awaited_once_with(
        (
            "SELECT EXISTS ("
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = $1 AND table_name = $2)"
        ),
        "myschema",
        "mytable",
    )


@pytest.mark.asyncio
async def test_check_exists_in_table(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    mock_conn.fetchval.return_value = True

    ok = await connector.check_exists_in_table(
        "models",
        "machX",
        schema="custom",
    )

    assert ok is True

    mock_conn.fetchval.assert_awaited_once_with(
        (
            "SELECT EXISTS (SELECT 1 FROM custom.models "
            "WHERE machine LIKE $1);"
        ),
        "%machX%",
    )


@pytest.mark.asyncio
async def test_create_model_table(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    await connector.create_model_table(
        "models",
        "model_name",
        schema="public",
    )

    mock_conn.execute.assert_awaited_once()

    query = mock_conn.execute.await_args.args[0]

    assert "CREATE TABLE IF NOT EXISTS public.models" in query
    assert "model_name VARCHAR(30)" in query


@pytest.mark.asyncio
async def test_get_data_table(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    mock_conn.fetch.return_value = [
        {"id": 1},
        {"id": 2},
    ]

    rows = await connector.get_data_table(
        "models",
        schema="public",
    )

    assert rows == [
        {"id": 1},
        {"id": 2},
    ]

    mock_conn.fetch.assert_awaited_once_with(
        "SELECT * FROM public.models"
    )


@pytest.mark.asyncio
async def test_get_data_table_in_coloumn(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    mock_conn.fetch.return_value = [
        {"machine": "A"},
    ]

    result = await connector.get_data_table_in_coloumn(
        "models",
        "machine",
        "mach",
        schema="public",
    )

    assert result == [
        {"machine": "A"},
    ]

    mock_conn.fetch.assert_awaited_once_with(
        (
            "SELECT * FROM public.models "
            "WHERE machine LIKE $1"
        ),
        "%mach%",
    )


@pytest.mark.asyncio
async def test_delete_table_agent(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    await connector.delete_table_agent(
        "models",
        schema="public",
    )

    mock_conn.execute.assert_awaited_once_with(
        "TRUNCATE TABLE public.models RESTART IDENTITY;"
    )


@pytest.mark.asyncio
async def test_close_calls(connector):
    mock_conn = make_mock_connection()
    connector._DatabaseConnector__conn = mock_conn

    await connector.close()

    mock_conn.close.assert_awaited_once()

    assert connector._DatabaseConnector__conn is None
