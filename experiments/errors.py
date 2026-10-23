class LLMRequestError(RuntimeError):
    """Запрос к модели не удался после всех попыток.

    Общее исключение для всех режимов: раннеру не нужно знать,
    какой именно пайплайн (zero_shot, RAG) его бросил.
    """

    def __init__(self, model: str, attempts: int, last_error: Exception):
        self.model = model
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{model}: запрос не выполнен за {attempts} попыток; "
            f"последняя ошибка - {type(last_error).__name__}: {last_error}"
        )
