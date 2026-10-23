from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage

from llm.retry import EmptyResponseError, call_with_retries


def _require_text(model: str, text) -> str:
    """Пустой ответ считаем сбоем, иначе он молча попадёт в таблицу."""
    if text is None or not str(text).strip():
        raise EmptyResponseError(f"{model}: пустой ответ модели")
    return str(text)


def _build_messages(prompt: str, system_prompt: str) -> list:
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    return messages


def _content_text(content) -> str:
    """Текст ответа: строка или список блоков (мультимодальный формат)."""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return "" if content is None else str(content)


def _normalize_usage(usage) -> dict | None:
    """Приводим usage LangChain к ключам, которые ждёт раннер."""
    if usage is None:
        return None
    if not isinstance(usage, dict):
        try:
            usage = dict(usage)
        except (TypeError, ValueError):
            return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    result = {}
    if isinstance(prompt, int):
        result["prompt_tokens"] = prompt
    if isinstance(completion, int):
        result["completion_tokens"] = completion
    if isinstance(total, int):
        result["total_tokens"] = total
    return result or None


def _usage_from(result) -> dict | None:
    meta = result.response_metadata or {}
    usage = (
        meta.get("token_usage")
        or meta.get("usage_metadata")
        or meta.get("usage")
        or getattr(result, "usage_metadata", None)
    )
    return _normalize_usage(usage)


class BaseLLM(ABC):
    """Интерфейс модели для экспериментов: текст ответа и его метаданные."""

    # Фактическая модель из последнего ответа (например, "GigaChat-2:2.0.30.01").
    last_model: str | None = None
    # Отпечаток конфигурации бэкенда: меняется при обновлении весов/настроек.
    last_fingerprint: str | None = None
    # Токены последнего ответа: нужны для оценки стоимости прогона.
    last_usage: dict | None = None

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        pass


class ChatModelLLM(BaseLLM):
    """Обёртка над LangChain-клиентом: промпты, метаданные и наши повторы."""

    def __init__(self, client, name: str):
        self.client = client
        # Подпись провайдера для логов и текста ошибок ("GigaChat", "DeepSeek").
        self.name = name
        self.last_model = None
        self.last_fingerprint = None
        self.last_usage = None

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = _build_messages(prompt, system_prompt)

        def request() -> str:
            result = self.client.invoke(messages)
            meta = result.response_metadata or {}
            self.last_model = meta.get("model_name") or meta.get("model")
            self.last_fingerprint = meta.get("system_fingerprint")
            self.last_usage = _usage_from(result)
            return _require_text(self.name, _content_text(result.content))

        return call_with_retries(self.name, request)
