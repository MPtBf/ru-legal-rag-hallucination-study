from langchain_deepseek import ChatDeepSeek
from langchain_gigachat import GigaChat

import config
from llm.base import ChatModelLLM


def _model_config(alias: str) -> dict:
    cfg = config.MODELS.get(alias)
    if cfg is None:
        raise ValueError(f"Неизвестная модель: {alias}")
    return cfg


def _build_gigachat(cfg: dict, options: dict):
    if not config.GIGACHAT_AUTH_KEY:
        raise ValueError(
            "Не задан GIGACHAT_AUTH_KEY. Скопируй .env.example в .env и заполни ключ."
        )
    # У GigaChat нет отдельного system_fingerprint, сертификат проверять не нужно.
    options.setdefault("verify_ssl_certs", False)
    return GigaChat(
        credentials=config.GIGACHAT_AUTH_KEY,
        model=cfg["api_model"],
        **options,
    )


def _build_deepseek(cfg: dict, options: dict):
    if not config.DEEPSEEK_API_KEY:
        raise ValueError(
            "Не задан DEEPSEEK_API_KEY. Скопируй .env.example в .env и заполни ключ."
        )
    return ChatDeepSeek(
        api_key=config.DEEPSEEK_API_KEY,
        model=cfg["api_model"],
        **options,
    )


# Новый провайдер: запись в config.MODELS + билдер здесь.
_BUILDERS = {
    "gigachat": _build_gigachat,
    "deepseek": _build_deepseek,
}


def create_chat_model(alias: str, **overrides):
    """Создаёт LangChain-клиент по алиасу из config.MODELS.

    Модель задаётся алиасом, overrides — параметры генерации (temperature,
    timeout и т. п.). Клиентские повторы не включаем: в create_llm их
    заменяет call_with_retries, а в LangChain-цепочках решение остаётся
    за экспериментом.
    """
    cfg = _model_config(alias)
    builder = _BUILDERS.get(cfg["provider"])
    if builder is None:
        raise ValueError(f"Неизвестный провайдер: {cfg['provider']}")

    options = {
        "temperature": config.TEMPERATURE,
        "timeout": config.REQUEST_TIMEOUT,
    }
    options.update(overrides)
    return builder(cfg, options)


def create_llm(alias: str, **overrides) -> ChatModelLLM:
    """Обёртка для простых вызовов: ретраи, текст, usage и response_model.

    Готовый клиент доступен как llm.client — его можно использовать в
    LangChain-цепочках, если нужен доступ к сообщениям напрямую.
    """
    cfg = _model_config(alias)
    options = {"max_retries": 0}
    options.update(overrides)
    return ChatModelLLM(create_chat_model(alias, **options), cfg["label"])
