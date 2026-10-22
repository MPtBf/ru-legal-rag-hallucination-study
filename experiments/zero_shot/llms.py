import logging
import time
from abc import ABC, abstractmethod

import httpx
import urllib3
from gigachat import GigaChat
from gigachat.exceptions import RateLimitError, ServerError
from gigachat.models import Chat, Messages

import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class LLMRequestError(RuntimeError):
    """Запрос к модели не удался после всех попыток."""

    def __init__(self, model: str, attempts: int, last_error: Exception):
        self.model = model
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"{model}: запрос не выполнен за {attempts} попыток; "
            f"последняя ошибка - {type(last_error).__name__}: {last_error}"
        )


def _short(message: str, limit: int = 300) -> str:
    return message if len(message) <= limit else message[:limit] + "..."


def _is_retryable(error: Exception) -> bool:
    """Повторяем только временные сбои: сеть, таймауты, 429 и 5xx."""
    if isinstance(error, (RateLimitError, ServerError)):
        return True
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status == 429 or status >= 500
    return isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError))


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
            delay = config.RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logger.info("%s: повтор через %s с", model, delay)
            time.sleep(delay)
    raise LLMRequestError(model, attempt, last_error)


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

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nВопрос: {prompt}"

        def request() -> str:
            # max_retries=0: повторами управляет _call_with_retries, а не SDK.
            client = GigaChat(
                credentials=self.auth_key,
                model=config.GIGACHAT_MODEL,
                verify_ssl_certs=False,
                max_retries=0,
            )
            with client as cl:
                res = cl.chat(
                    Chat(
                        messages=[Messages(role="user", content=full_prompt)],
                        temperature=config.TEMPERATURE,
                    )
                )
                return res.choices[0].message.content

        return _call_with_retries("GigaChat", request)


class DeepSeekModel(BaseLLM):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "Не задан DEEPSEEK_API_KEY. Скопируй .env.example в .env и заполни ключ."
            )
        self.api_key = api_key
        self.url = config.DEEPSEEK_URL

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
            with httpx.Client(http2=True, verify=False, timeout=30.0) as client:
                response = client.post(self.url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise httpx.HTTPStatusError(
                        f"HTTP {response.status_code}: {_short(response.text)}",
                        request=response.request,
                        response=response,
                    )
                return response.json()["choices"][0]["message"]["content"]

        return _call_with_retries("DeepSeek", request)
