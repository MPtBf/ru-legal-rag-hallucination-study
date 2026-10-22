import logging
import time
from abc import ABC, abstractmethod

import httpx
import urllib3
from gigachat import GigaChat
from gigachat.exceptions import RateLimitError, ServerError
from gigachat.models import Chat, Messages

import config
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
    if isinstance(error, (RateLimitError, ServerError, EmptyResponseError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in (408, 429) or status >= 500
    return isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError))


def _retry_delay(error: Exception, attempt: int) -> float:
    """Пауза перед повтором: Retry-After сервера, иначе экспоненциальная."""
    if isinstance(error, RateLimitError) and error.retry_after > 0:
        return float(error.retry_after)
    if isinstance(error, httpx.HTTPStatusError):
        value = error.response.headers.get("Retry-After")
        if value:
            try:
                seconds = float(value)
            except ValueError:
                seconds = 0
            if seconds > 0:
                return seconds
    return float(config.RETRY_BASE_DELAY * (2 ** (attempt - 1)))


def _call_with_retries(model: str, request):
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
                model, attempt, config.RETRY_ATTEMPTS, type(error).__name__, _short(str(error)),
            )
            if not _is_retryable(error) or attempt == config.RETRY_ATTEMPTS:
                break
            delay = _retry_delay(error, attempt)
            logger.info("%s: повтор через %s с", model, delay)
            time.sleep(delay)
    raise LLMRequestError(model, attempt, last_error) from last_error


class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        pass


class GigaChatModel(BaseLLM):
    def __init__(self, auth_key: str):
        if not auth_key:
            raise ValueError(
                "Не задан GIGACHAT_AUTH_KEY. Скопируй .env.example в .env и заполни ключ."
            )
        self.auth_key = auth_key
        # Фактическая модель из последнего ответа (например, "GigaChat-2:2.0.30.01").
        self.last_model = None
        # У GigaChat аналога system_fingerprint нет.
        self.last_fingerprint = None
        # Токены последнего ответа: нужны для оценки стоимости прогона.
        self.last_usage = None
        # Клиент переиспользуется между запросами: иначе на каждую попытку
        # уходил бы лишний OAuth-запрос. Токен SDK обновляет сам.
        # max_retries=0: повторами управляет _call_with_retries, а не SDK.
        self.client = GigaChat(
            credentials=auth_key,
            model=config.GIGACHAT_MODEL,
            verify_ssl_certs=False,
            max_retries=0,
            timeout=config.REQUEST_TIMEOUT,
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        # Как и у DeepSeek: инструкция отдельным system-сообщением, вопрос - user.
        messages = []
        if system_prompt:
            messages.append(Messages(role="system", content=system_prompt))
        messages.append(Messages(role="user", content=prompt))

        def request() -> str:
            res = self.client.chat(
                Chat(
                    messages=messages,
                    temperature=config.TEMPERATURE,
                )
            )
            if not res.choices:
                raise EmptyResponseError("GigaChat: пустой список choices")
            self.last_model = res.model
            self.last_usage = res.usage.model_dump() if res.usage else None
            return _require_text("GigaChat", res.choices[0].message.content)

        return _call_with_retries("GigaChat", request)


class DeepSeekModel(BaseLLM):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "Не задан DEEPSEEK_API_KEY. Скопируй .env.example в .env и заполни ключ."
            )
        self.api_key = api_key
        self.url = config.DEEPSEEK_URL
        # Фактическая модель из последнего ответа (например, "deepseek-flash").
        self.last_model = None
        # Отпечаток конфигурации бэкенда: меняется при обновлении весов/настроек.
        self.last_fingerprint = None
        # Токены последнего ответа: нужны для оценки стоимости прогона.
        self.last_usage = None

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": config.TEMPERATURE,
            "tool_choice": "none"
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        def request() -> str:
            with httpx.Client(http2=True, verify=False, timeout=config.REQUEST_TIMEOUT) as client:
                response = client.post(self.url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {_short(response.text)}",
                        request=response.request,
                        response=response,
                    )
                data = response.json()
                choices = data.get("choices") or []
                if not choices:
                    raise EmptyResponseError("DeepSeek: пустой список choices")
                self.last_model = data.get("model")
                self.last_fingerprint = data.get("system_fingerprint")
                self.last_usage = data.get("usage")
                return _require_text("DeepSeek", choices[0].get("message", {}).get("content"))

        return _call_with_retries("DeepSeek", request)
