from abc import ABC, abstractmethod
from gigachat import GigaChat


class BaseLLM(ABC):
    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        pass


class GigaChatModel(BaseLLM):
    def __init__(self, auth_key: str):
        self.auth_key = auth_key

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nВопрос: {prompt}"

        client = GigaChat(
            credentials=self.auth_key,
            model="GigaChat-2",
            verify_ssl_certs=False
        )
        try:
            with client as cl:
                res = cl.chat(full_prompt)
                return res.choices[0].message.content
        except Exception as e:
            return f"[Ошибка GigaChat API]: {str(e)}"

import urllib3

# Отключаем предупреждения об SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import httpx

class DeepSeekModel(BaseLLM):
    """Сущность для стабильной работы с DeepSeek"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.deepseek.com/chat/completions"

    def generate_response(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": 0.1,
            "tool_choice": "none"
        }

        # Базовые заголовки авторизации
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
                    return f"[Ошибка DeepSeek API {response.status_code}]: {response.text}"

        except Exception as e:
            return f"[Ошибка сети DeepSeek]: {str(e)}"
