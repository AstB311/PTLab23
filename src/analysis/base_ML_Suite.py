class BaseMLSuite:
    """Общие настройки для классов работы с ML-алгоритмами."""

    def __init__(self, random_state: int = 42):
        self.__random_state = random_state

    @property
    def random_state(self) -> int:
        return self.__random_state
