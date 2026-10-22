import config
import prompts
from experiments import Answer, register_pipeline
from experiments.zero_shot.llms import DeepSeekModel, GigaChatModel


@register_pipeline("zero_shot")
class ZeroShotPipeline:
    """Нулевой шот: вопрос отправляется без контекста."""

    # Название колонки с ответом должно совпадать с уже сгенерированными файлами.
    ANSWER_PREFIX = "Ответ"

    def __init__(self, model: str):
        if model not in config.MODEL_LABELS:
            raise ValueError(f"Неизвестная модель: {model}")
        if model == "gigachat":
            self.llm = GigaChatModel(auth_key=config.GIGACHAT_AUTH_KEY)
        else:
            self.llm = DeepSeekModel(api_key=config.DEEPSEEK_API_KEY)
        self.model = model
        self.answer_col = f"{self.ANSWER_PREFIX}_{config.MODEL_LABELS[model]}_ZeroShot"

    def answer(self, question: str) -> Answer:
        # Вызов модели сохраняем как в исходных скриптах: промпт и та же инструкция.
        text = self.llm.generate_response(prompt=question, system_prompt=prompts.SYSTEM_PROMPT)
        return Answer(text=text, meta={"response_model": self.llm.last_model})
