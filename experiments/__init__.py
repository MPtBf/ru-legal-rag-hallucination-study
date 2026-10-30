from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

# Реестр пайплайнов: mode -> класс.
# Новый режим добавляется декоратором @register_pipeline("имя") в experiments/.
PIPELINES: Dict[str, type] = {}


def register_pipeline(name: str) -> Callable:
    def decorator(cls: type) -> type:
        PIPELINES[name] = cls
        return cls

    return decorator


@dataclass
class Answer:
    """Результат ответа пайплайна на один вопрос."""
    text: str
    docs: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# Импорт подмодулей регистрирует пайплайны в реестре.
import experiments.naive_rag  # noqa: E402,F401
import experiments.zero_shot  # noqa: E402,F401
import experiments.hybrid_rag