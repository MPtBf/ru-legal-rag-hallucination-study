import sys
from abc import ABC, abstractmethod

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat

import config


def _short(message: str, limit: int = 300) -> str:
    return message if len(message) <= limit else message[:limit] + "..."


def log_error(model: str, message: str) -> None:
    print(f"[ОШИБКА] {model}: {_short(message)}", file=sys.stderr, flush=True)


class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        pass


class GigaChatModel(BaseLLM):
    def __init__(self, auth_key: str, giga_model: str = config.GIGACHAT_MODEL):
        if not auth_key:
            raise ValueError(
                "Не задан GIGACHAT_AUTH_KEY. Скопируй .env.example в .env и заполни ключ."
            )
        self.auth_key = auth_key

        allowed_models = config.allowed_giga_models

        if giga_model not in allowed_models:
            raise ValueError(
                f"Выбранная модель '{giga_model}' не поддерживается. "
                f"Доступные модели: {', '.join(allowed_models)}"
            )

        self.model = giga_model
        self.client = GigaChat(
            credentials=self.auth_key,
            model=self.model,
            temperature=config.TEMPERATURE,
            verify_ssl_certs=False,
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            result = self.client.invoke(messages)
            return result.content
        except Exception as e:
            log_error("GigaChat", f"{type(e).__name__}: {e}")
            return f"[Ошибка GigaChat API]: {_short(f'{type(e).__name__}: {e}')}"


class DeepSeekModel(BaseLLM):
    def __init__(self, api_key: str, deepseek_model: str = config.DEEPSEEK_MODEL):
        if not api_key:
            raise ValueError(
                "Не задан DEEPSEEK_API_KEY. Скопируй .env.example в .env и заполни ключ."
            )
        self.api_key = api_key

        allowed_models = config.allowed_deepseek_models

        if deepseek_model.lower() not in allowed_models:
            raise ValueError(
                f"Выбранная модель '{deepseek_model}' не поддерживается. "
                f"Доступные модели: {', '.join(allowed_models)}"
            )

        self.model = deepseek_model
        self.client = ChatDeepSeek(
            api_key=self.api_key,
            model=self.model,
            temperature=config.TEMPERATURE,
            timeout=30.0,
        )

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        try:
            result = self.client.invoke(messages)
            return result.content
        except Exception as e:
            log_error("DeepSeek", f"{type(e).__name__}: {e}")
            return f"[Ошибка DeepSeek API]: {_short(f'{type(e).__name__}: {e}')}"