import logging
import time
from abc import ABC, abstractmethod

import urllib3
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat

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
from config import allowed_models, DEEPSEEK_MODEL, GIGACHAT_MODEL
from experiments.errors import LLMRequestError

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class EmptyResponseError(RuntimeError):
    """Модель вернула пустой ответ (None или строка из пробелов)."""


def _require_text(model: str, text) -> str:
    """Пустой ответ считаем сбоем, иначе он молча попадёт в таблицу."""
    if text is None or not str(text).strip():
        raise EmptyResponseError(f"{model}: пустой ответ модели")
    return str(text)


def _short(message: str, limit: int = 300) -> str:
    return message if len(message) <= limit else message[:limit] + "..."


def _is_retryable(error: Exception) -> bool:
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
        (APIConnectionError, APITimeoutError, TimeoutError, ConnectionError),
    )


def _retry_delay(error: Exception, attempt: int) -> float:
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


def _call_with_retries(model_name: str, invoke, *args):
    """Выполняет invoke(*args) с повторами и экспоненциальной паузой."""
    last_error = None
    attempt = 0
    for attempt in range(1, config.RETRY_ATTEMPTS + 1):
        try:
            return invoke(*args)
        except Exception as error:
            last_error = error
            logger.error(
                "%s: попытка %d/%d не удалась - %s: %s",
                model_name, attempt, config.RETRY_ATTEMPTS,
                type(error).__name__, _short(str(error)),
            )
            if not _is_retryable(error) or attempt == config.RETRY_ATTEMPTS:
                break
            delay = _retry_delay(error, attempt)
            logger.info("%s: повтор через %s с", model_name, delay)
            time.sleep(delay)
    raise LLMRequestError(model_name, attempt, last_error) from last_error


def _build_messages(prompt: str, system_prompt: str) -> list:
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    return messages

def _check_model(model: str):
    if model not in allowed_models:
        raise ValueError("Укажите доступную модель")

class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        pass


class GigaChatModel(BaseLLM):
    def __init__(self, auth_key: str, model:str = GIGACHAT_MODEL):
        if not auth_key:
            raise ValueError(
                "Не задан GIGACHAT_AUTH_KEY. Скопируй .env.example в .env и заполни ключ."
            )

        if model == "gigachat": model = GIGACHAT_MODEL
        _check_model(model)

        self.auth_key = auth_key
        self.last_model = None
        self.last_fingerprint = None
        self.last_usage = None
        self.client = GigaChat(
            credentials=auth_key,
            model=config.GIGACHAT_MODEL,
            temperature=config.TEMPERATURE,
            verify_ssl_certs=False,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=0,
        )

    def _invoke(self, messages):
        result = self.client.invoke(messages)
        meta = result.response_metadata or {}
        self.last_model = meta.get("model_name") or meta.get("model")
        self.last_usage = meta.get("usage") or getattr(result, "usage_metadata", None)
        return _require_text("GigaChat", result.content)

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = _build_messages(prompt, system_prompt)
        return _call_with_retries("GigaChat", self._invoke, messages)


class DeepSeekModel(BaseLLM):
    def __init__(self, api_key: str, model:str = DEEPSEEK_MODEL):
        if not api_key:
            raise ValueError(
                "Не задан DEEPSEEK_API_KEY. Скопируй .env.example в .env и заполни ключ."
            )

        if model == "deepseek": model = DEEPSEEK_MODEL
        _check_model(model)

        self.api_key = api_key
        self.last_model = None
        self.last_fingerprint = None
        self.last_usage = None
        self.client = ChatDeepSeek(
            api_key=api_key,
            model=config.DEEPSEEK_MODEL,
            temperature=config.TEMPERATURE,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=0,
        )

    def _invoke(self, messages):
        result = self.client.invoke(messages)
        meta = result.response_metadata or {}
        self.last_model = meta.get("model_name")
        self.last_fingerprint = meta.get("system_fingerprint")
        self.last_usage = meta.get("token_usage") or getattr(result, "usage_metadata", None)
        return _require_text("DeepSeek", result.content)

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = _build_messages(prompt, system_prompt)
        return _call_with_retries("DeepSeek", self._invoke, messages)