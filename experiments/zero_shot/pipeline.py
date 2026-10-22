import config
import prompts
from experiments import Answer, register_pipeline
from llms import DeepSeekModel, GigaChatModel


@register_pipeline("zero_shot")
class ZeroShotPipeline:
    """Нулевой шот: вопрос отправляется без контекста."""

    # Название колонки с ответом должно совпадать с уже сгенерированными файлами.
    ANSWER_PREFIX = "Ответ"

    def __init__(self, model: str):

        if model in config.allowed_giga_models:
            self.llm = GigaChatModel(auth_key=config.GIGACHAT_AUTH_KEY, giga_model=model)

        elif model in config.allowed_deepseek_models:
            self.llm = DeepSeekModel(api_key=config.DEEPSEEK_API_KEY, deepseek_model=model)
        else:
            raise ValueError(f"Неизвестная модель: {model}")

        self.model = model
        self.answer_col = f"{self.ANSWER_PREFIX}_{self.model}_ZeroShot"

    def answer(self, question: str) -> Answer:
        # Вызов модели сохраняем как в исходных скриптах: промпт и та же инструкция.
        text = self.llm.generate_response(prompt=question, system_prompt=prompts.SYSTEM_PROMPT)
        return Answer(text=text)
