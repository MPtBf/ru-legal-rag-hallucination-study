import sys
from abc import ABC, abstractmethod

import httpx
import urllib3
from gigachat import GigaChat

import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _short(message: str, limit: int = 300) -> str:
    return message if len(message) <= limit else message[:limit] + "..."


def log_error(model: str, message: str) -> None:
    """Печатает ошибку запроса в консоль сразу, а не только в итоговой таблице."""
    print(f"[ОШИБКА] {model}: {_short(message)}", file=sys.stderr, flush=True)


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

        client = GigaChat(
            credentials=self.auth_key,
            model=config.GIGACHAT_MODEL,
            verify_ssl_certs=False
        )
        try:
            with client as cl:
                res = cl.chat(full_prompt)
                return res.choices[0].message.content
        except Exception as e:
            log_error("GigaChat", f"{type(e).__name__}: {e}")
            return f"[Ошибка GigaChat API]: {_short(f'{type(e).__name__}: {e}')}"


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

        try:
            with httpx.Client(http2=True, verify=False, timeout=30.0) as client:
                response = client.post(self.url, json=payload, headers=headers)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    log_error("DeepSeek", f"HTTP {response.status_code}: {response.text}")
                    return f"[Ошибка DeepSeek API {response.status_code}]: {_short(response.text)}"
        except Exception as e:
            log_error("DeepSeek", f"{type(e).__name__}: {e}")
            return f"[Ошибка сети DeepSeek]: {_short(f'{type(e).__name__}: {e}')}"
