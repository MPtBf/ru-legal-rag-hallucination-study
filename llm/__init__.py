"""Общий слой моделей: клиенты, повторы и разбор ответа.

Эксперименты (zero-shot, RAG) используют create_llm, чтобы получить
одинаковые ретраи и метаданные. create_chat_model возвращает «сырой»
LangChain-клиент для цепочек и агентов.
"""
from llm.base import BaseLLM, ChatModelLLM
from llm.errors import LLMRequestError
from llm.factory import create_chat_model, create_llm

__all__ = [
    "BaseLLM",
    "ChatModelLLM",
    "LLMRequestError",
    "create_chat_model",
    "create_llm",
]
