class LLMRequestError(RuntimeError):
    """Запрос к модели не удался после всех попыток.

    Модель не возвращает строку с ошибкой: исключение прерывает прогон,
    поэтому в таблицу не попадает невалидный ответ.
    """

    def __init__(self, model: str, attempts: int, last_error: Exception):
        self.model = model
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{model}: запрос не выполнен за {attempts} попыток; "
            f"последняя ошибка - {type(last_error).__name__}: {last_error}"
        )
