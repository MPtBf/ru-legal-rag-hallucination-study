import config
import prompts
from experiments import Answer, register_pipeline
from llm import create_llm


@register_pipeline("zero_shot")
class ZeroShotPipeline:
    """Нулевой шот: вопрос отправляется без контекста."""

    # Название колонки с ответом должно совпадать с уже сгенерированными файлами.
    ANSWER_PREFIX = "Ответ"

    def __init__(self, model: str):
        self.llm = create_llm(model)
        cfg = config.MODELS[model]
        self.model = model
        self.requested_model = cfg["api_model"]
        self.answer_col = f"{self.ANSWER_PREFIX}_{cfg['label']}_ZeroShot"
        self.model = model

    def answer(self, question: str) -> Answer:
        # Вызов модели сохраняем как в исходных скриптах: промпт и та же инструкция.
        text = self.llm.generate_response(prompt=question, system_prompt=prompts.SYSTEM_PROMPT)
        return Answer(
            text=text,
            meta={
                "response_model": self.llm.last_model,
                "system_fingerprint": self.llm.last_fingerprint,
                "usage": self.llm.last_usage,
            },
        )
