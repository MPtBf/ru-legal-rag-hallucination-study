import logging
import time

import httpx
from gigachat.exceptions import (
    RateLimitError as GigaRateLimitError,
    ServerError as GigaServerError,
)
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError as OpenAIRateLimitError,
)

import config
from llm.errors import LLMRequestError

logger = logging.getLogger(__name__)


class EmptyResponseError(RuntimeError):
    """Модель вернула пустой ответ (None или строка из пробелов)."""


def _short(message: str, limit: int = 300) -> str:
    return message if len(message) <= limit else message[:limit] + "..."


def is_retryable(error: Exception) -> bool:
    """Повторяем только временные сбои: сеть, таймауты, 408, 429 и 5xx."""
    if isinstance(
        error,
        (EmptyResponseError, OpenAIRateLimitError, GigaRateLimitError, GigaServerError),
    ):
        return True
    if isinstance(error, APIStatusError):
        status = error.status_code
        return status in (408, 429) or status >= 500
    return isinstance(
        error,
        (
            APIConnectionError,
            APITimeoutError,
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
        ),
    )


def retry_delay(error: Exception, attempt: int) -> float:
    """Пауза перед повтором: Retry-After сервера, иначе экспоненциальная."""
    retry_after = None
    if isinstance(error, OpenAIRateLimitError):
        try:
            retry_after = error.response.headers.get("retry-after")
        except Exception:
            retry_after = None
    elif isinstance(error, GigaRateLimitError):
        retry_after = getattr(error, "retry_after", None)

    if retry_after:
        try:
            seconds = float(retry_after)
        except (TypeError, ValueError):
            seconds = 0
        if seconds > 0:
            return seconds

    return float(config.RETRY_BASE_DELAY * (2 ** (attempt - 1)))


def call_with_retries(model: str, request):
    """Выполняет запрос с повторами и экспоненциальной паузой."""
    last_error = None
    attempt = 0
    for attempt in range(1, config.RETRY_ATTEMPTS + 1):
        try:
            return request()
        except Exception as error:
            last_error = error
            logger.error(
                "%s: попытка %d/%d не удалась - %s: %s",
                model, attempt, config.RETRY_ATTEMPTS,
                type(error).__name__, _short(str(error)),
            )
            if not is_retryable(error) or attempt == config.RETRY_ATTEMPTS:
                break
            delay = retry_delay(error, attempt)
            logger.info("%s: повтор через %s с", model, delay)
            time.sleep(delay)
    raise LLMRequestError(model, attempt, last_error) from last_error
