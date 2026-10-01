import pickle
from typing import Optional, Dict, Any, List

import asyncpg
from fastapi import HTTPException


class DatabaseConnector:
    """Подключение к PostgreSQL и выполнение операций с данными."""

    def __init__(
        self,
        server: str,
        port: int,
        database: str,
        user: str,
        password: str,
        equipment: Optional[str] = None,
        equipment_predict: Optional[str] = None,
    ):
        # Проверка параметра перед сохранением.
        if isinstance(port, bool) or not isinstance(port, int):
            raise TypeError("Порт должен быть целым числом")

        if not 1 <= port <= 65535:
            raise ValueError("Порт должен быть от 1 до 65535")

        # Закрытые параметры подключения
        self.__server = server
        self.__port = port
        self.__database = database
        self.__user = user
        self.__password = password

        # Соединение
        self.__conn: Optional[asyncpg.Connection] = None

        self.equipment = equipment
        self.equipment_predict = equipment_predict

    # Свойства только для чтения: сеттеров нет
    @property
    def server(self) -> str:
        return self.__server

    @property
    def port(self) -> int:
        return self.__port

    @property
    def database(self) -> str:
        return self.__database

    @property
    def user(self) -> str:
        return self.__user

    @property
    def is_connected(self) -> bool:
        """Есть ли открытое соединение."""
        return (
            self.__conn is not None
            and not self.__conn.is_closed()
        )

    def __require_connection(self) -> asyncpg.Connection:
        """Возвращает соединение для внутренних операций."""
        if not self.is_connected:
            raise HTTPException(
                status_code=500,
                detail="Соединение с базой данных не установлено",
            )

        return self.__conn

    async def connect(self) -> bool:
        """Устанавливает соединение, не выдавая его внешнему коду."""
        if self.is_connected:
            return True

        try:
            self.__conn = await asyncpg.connect(
                user=self.__user,
                password=self.__password,
                database=self.__database,
                host=self.__server,
                port=self.__port,
            )
            return True

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database connection error: {str(e)}",
            ) from e

    async def check_table_exists(
        self,
        table_name: str,
        schema: str = "public",
    ) -> bool:
        """Проверяет существование таблицы."""
        conn = self.__require_connection()

        try:
            query = (
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = $1 AND table_name = $2)"
            )

            return await conn.fetchval(
                query,
                schema,
                table_name.lower(),
            )

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error checking table existence: {str(e)}",
            ) from e

    async def check_exists_in_table(
        self,
        table_name: str,
        machine_name: str,
        schema: str = "public",
    ) -> bool:
        """Проверяет наличие оборудования в таблице."""
        conn = self.__require_connection()

        try:
            query = (
                f"SELECT EXISTS (SELECT 1 FROM {schema}.{table_name} "
                "WHERE machine LIKE $1);"
            )

            return await conn.fetchval(
                query,
                f"%{machine_name}%",
            )

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error checking machine existence: {str(e)}",
            ) from e

    async def create_model_table(
        self,
        table_name: str,
        table_column_name: str,
        schema: str = "public",
    ) -> None:
        """Создаёт таблицу для хранения моделей."""
        conn = self.__require_connection()

        try:
            query = f"""
                CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                    id SERIAL PRIMARY KEY,
                    machine VARCHAR(30),
                    {table_column_name} VARCHAR(30),
                    model BYTEA,
                    method_param VARCHAR(30),
                    accuracy REAL
                );
            """

            await conn.execute(query)

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error creating table: {str(e)}",
            ) from e

    async def insert_data(
        self,
        table_name: str,
        data: Dict[str, Any],
        schema: str = "public",
    ) -> None:
        """Вставляет данные в таблицу."""
        conn = self.__require_connection()

        # Работаем с копией, чтобы не менять словарь вызывающего кода.
        values = data.copy()

        if "model" in values:
            values["model"] = pickle.dumps(values["model"])

        try:
            columns = ", ".join(values.keys())
            placeholders = ", ".join(
                f"${i + 1}" for i in range(len(values))
            )

            query = (
                f"INSERT INTO {schema}.{table_name} "
                f"({columns}) VALUES ({placeholders});"
            )

            await conn.execute(query, *values.values())

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error inserting data into table: {str(e)}",
            ) from e

    async def get_data_table(
        self,
        table_name: str,
        schema: str = "public",
    ) -> List[Dict[str, Any]]:
        """Получает все данные из таблицы."""
        conn = self.__require_connection()

        try:
            query = f"SELECT * FROM {schema}.{table_name}"
            rows = await conn.fetch(query)

            return [dict(row) for row in rows]

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from table: {str(e)}",
            ) from e

    async def get_data_table_in_coloumn(
        self,
        table_name: str,
        coloumn_name: str,
        machine_name: str,
        schema: str = "public",
    ) -> List[Dict[str, Any]]:
        """Получает данные по значению в заданном столбце."""
        conn = self.__require_connection()

        try:
            query = (
                f"SELECT * FROM {schema}.{table_name} "
                f"WHERE {coloumn_name} LIKE $1"
            )

            rows = await conn.fetch(
                query,
                f"%{machine_name}%",
            )

            return [dict(row) for row in rows]

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from table: {str(e)}",
            ) from e

    async def delete_table_agent(
        self,
        table_name: str,
        schema: str = "public",
    ) -> None:
        """Очищает таблицу и сбрасывает счётчик идентификаторов."""
        conn = self.__require_connection()

        try:
            query = (
                f"TRUNCATE TABLE {schema}.{table_name} "
                "RESTART IDENTITY;"
            )

            await conn.execute(query)

        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting data from table: {str(e)}",
            ) from e

    async def close(self) -> None:
        """Закрывает соединение и очищает внутреннюю ссылку."""
        if self.__conn is None:
            return

        if not self.__conn.is_closed():
            await self.__conn.close()

        self.__conn = None