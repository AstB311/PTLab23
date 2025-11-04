import asyncpg
import pickle
from typing import Optional, Dict, Any, List

from fastapi import HTTPException


class DatabaseConnector:
    """
    Класс для установления соединения с базой данных PostgreSQL
    и выполнения различных операций.
    """

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
        """
        Инициализирует объект DatabaseConnector с параметрами подключения.

        Args:
            server: Адрес сервера базы данных.
            port: Порт сервера базы данных.
            database: Имя базы данных.
            user: Имя пользователя базы данных.
            password: Пароль пользователя базы данных.
            equipment: Опциональное название оборудования.
            equipment_predict: Опциональное название оборудования
                для предсказания.
        """
        self.server = server
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.equipment = equipment
        self.equipment_predict = equipment_predict
        self.conn: Optional[asyncpg.Connection] = None

    async def connect(self) -> asyncpg.Connection:
        """
        Устанавливает соединение с базой данных.
        """
        try:
            self.conn = \
                await asyncpg.connect(
                    user=self.user,
                    password=self.password,
                    database=self.database,
                    host=self.server,
                    port=self.port
                )
            return self.conn
        except asyncpg.PostgresError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Database connection error: {str(e)}",
            )

    async def check_table_exists(
        self, table_name: str, schema: str = "public"
    ) -> bool:
        """
        Проверяет существование таблицы в базе данных.
        """
        try:
            query = (
                "SELECT EXISTS ("
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_schema = $1 AND table_name = $2)"
            )
            exists = await self.conn.fetchval(query, schema,
                                              table_name.lower())
            return exists
        except asyncpg.PostgresError as e:
            print(
                "Таблица не найдена. Повторите попытку и/или введите "
                "правильные данные."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Database connection error: {str(e)}",
            )

    async def check_exists_in_table(
        self,
        table_name: str,
        machine_name: str,
        schema: str = "public",
    ) -> bool:
        """
        Проверяет наличие оборудования (machine) в таблице.
        """
        try:
            query = (
                f"SELECT EXISTS (SELECT 1 FROM {schema}.{table_name} "
                "WHERE machine LIKE $1);"
            )
            exists = await self.conn.fetchval(query, f"%{machine_name}%")
            return exists
        except asyncpg.PostgresError as e:
            print(
                "Ошибка при проверке наличия machine. "
                "Повторите попытку и/или введите правильные данные."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error checking machine existence: {str(e)}",
            )

    async def create_model_table(
        self,
        table_name: str,
        table_column_name: str,
        schema: str = "public",
    ) -> None:
        """
        Создает таблицу для хранения моделей.
        """
        try:
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table_name} (
                id SERIAL PRIMARY KEY,
                machine VARCHAR(30),
                {table_column_name} VARCHAR(30),
                model BYTEA,
                method_param VARCHAR(30),
                accuracy REAL
            );
            """
            await self.conn.execute(create_table_query)
        except asyncpg.PostgresError as e:
            print(
                "Таблица не может быть создана. Повторите попытку и/или "
                "введите правильные данные."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error creating table: {str(e)}",
            )

    async def insert_data(
        self,
        table_name: str,
        data: Dict[str, Any],
        schema: str = "public",
    ) -> None:
        """
        Вставляет данные в таблицу.
        """
        try:
            if "model" in data:
                data["model"] = pickle.dumps(data["model"])

            columns = ", ".join(data.keys())
            placeholders = ", ".join([f"${i + 1}" for i in range(len(data))])
            query = (
                f"INSERT INTO {schema}.{table_name} "
                f"({columns}) VALUES ({placeholders});"
            )
            await self.conn.execute(query, *data.values())
        except asyncpg.PostgresError as e:
            print(f"Ошибка вставки данных: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error inserting data into table: {str(e)}",
            )

    async def get_data_table(
        self, table_name: str, schema: str = "public"
    ) -> List[Dict[str, Any]]:
        """
        Получает все данные из таблицы.
        """
        try:
            query = f"SELECT * FROM {schema}.{table_name}"
            rows = await self.conn.fetch(query)
            return [dict(row) for row in rows]
        except asyncpg.PostgresError as e:
            print(
                "Ошибка при получении данных. Повторите попытку и/или "
                "введите правильные данные."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from table: {str(e)}",
            )

    async def get_data_table_in_coloumn(
        self,
        table_name: str,
        coloumn_name: str,
        machine_name: str,
        schema: str = "public",
    ) -> List[Dict[str, Any]]:
        """
        Получает данные по значению в заданном столбце.
        """
        try:
            query = (
                f"SELECT * FROM {schema}.{table_name} "
                f"WHERE {coloumn_name} LIKE $1"
            )
            rows = await self.conn.fetch(query, f"%{machine_name}%")
            return [dict(row) for row in rows]
        except asyncpg.PostgresError as e:
            print(
                "Ошибка при получении данных. Повторите попытку и/или "
                "введите правильные данные."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching data from table: {str(e)}",
            )

    async def delete_table_agent(
        self, table_name: str, schema: str = "public"
    ) -> None:
        """
        Очищает таблицу (TRUNCATE).
        """
        try:
            query = (
                f"TRUNCATE TABLE {schema}.{table_name} RESTART IDENTITY;"
            )
            await self.conn.execute(query)
        except asyncpg.PostgresError as e:
            print(
                "Удаление данных не может быть выполнено. "
                "Повторите попытку и/или введите правильные данные."
            )
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting data from table: {str(e)}",
            )

    async def close(self) -> None:
        """
        Закрывает соединение с базой данных.
        Закрывает соединение с базой данных.
        """
        if self.conn:
            await self.conn.close()
